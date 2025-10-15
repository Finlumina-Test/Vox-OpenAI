import io
import wave
import base64
import asyncio
import aiohttp
import numpy as np
import time    
from scipy.signal import resample
from typing import Dict, Optional, List, Callable
from config import Config
from services.log_utils import Log


class TranscriptionService:
    """
    Real-time transcription service with smart speaker-aware streaming.
    
    Key Features:
    - Unified queue for first-come-first-serve
    - Smart gap control: only Caller→AI gets 0.5s gap, AI→Caller has no artificial gap
    - Correct sample rates: Caller=8kHz µ-law, AI=24kHz PCM16
    - Resamples both to 24kHz PCM16 for consistent high-quality playback
    """
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    
    # Audio format specs
    CALLER_SAMPLE_RATE = 8000   # Twilio µ-law 8kHz
    AI_SAMPLE_RATE = 24000       # OpenAI Realtime API PCM16 24kHz
    DASHBOARD_SAMPLE_RATE = 24000  # Output 24kHz for dashboard
    
    # Speaker turn detection
    SPEAKER_SILENCE_THRESHOLD = 0.3  # If no chunks for 0.3s, speaker is done
    CALLER_TO_AI_GAP = 0.5           # Gap only when Caller → AI
    AI_TO_CALLER_GAP = 0.0           # No artificial gap for AI → Caller
    
    # Transcription settings
    MIN_AUDIO_DURATION = 0.8
    SILENCE_TIMEOUT = 0.5
    MAX_BUFFER_DURATION = 3.0
    
    def __init__(self):
        # Transcription buffers
        self._caller_buffer: bytearray = bytearray()
        self._ai_buffer: bytearray = bytearray()
        
        # Unified audio queue
        self._unified_audio_queue: asyncio.Queue = asyncio.Queue()
        
        # Streaming task
        self._stream_task: Optional[asyncio.Task] = None
        
        # Speaker tracking for gaps
        self._last_streamed_speaker: Optional[str] = None
        self._last_queued_time_per_speaker: Dict[str, float] = {}
        
        # Transcription timing
        self._caller_last_chunk_time: float = 0
        self._ai_last_chunk_time: float = 0
        self._caller_first_chunk_time: float = 0
        self._ai_first_chunk_time: float = 0
        
        # Processing flags
        self._caller_processing: bool = False
        self._ai_processing: bool = False
        
        # Transcription monitoring
        self._caller_monitor_task: Optional[asyncio.Task] = None
        self._ai_monitor_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.audio_callback: Optional[Callable] = None
        self.transcription_callback: Optional[Callable] = None
        
        # Deduplication
        self._caller_last_transcript: str = ""
        self._ai_last_transcript: str = ""
        
        # Shutdown flag
        self._shutdown: bool = False
    
    def set_audio_callback(self, callback: Callable):
        """Set callback for raw audio chunks."""
        self.audio_callback = callback
        
        if not self._stream_task or self._stream_task.done():
            self._stream_task = asyncio.create_task(self._stream_unified_audio())
    
    def set_word_callback(self, callback: Callable):
        """Set callback for transcription results."""
        self.transcription_callback = callback
    
    def _calculate_chunk_duration(self, audio_bytes: bytes, sample_rate: int) -> float:
        """Calculate audio chunk duration in seconds."""
        num_samples = len(audio_bytes) // 2  # PCM16 = 2 bytes per sample
        duration_seconds = num_samples / sample_rate
        return duration_seconds
    
    def _resample_caller_audio(self, mulaw_bytes: bytes) -> bytes:
        """
        Convert Caller's µ-law 8kHz to PCM16 24kHz.
        """
        try:
            # Convert µ-law to PCM16 at 8kHz
            pcm16_8k = self._mulaw_to_pcm16(mulaw_bytes)
            
            # Resample from 8kHz to 24kHz
            pcm16_24k = self._resample_pcm16(pcm16_8k, self.CALLER_SAMPLE_RATE, self.DASHBOARD_SAMPLE_RATE)
            
            return pcm16_24k.tobytes()
        except Exception as e:
            Log.error(f"[Resample Caller] Error: {e}")
            pcm16_8k = self._mulaw_to_pcm16(mulaw_bytes)
            return pcm16_8k.tobytes()
    
    def _resample_ai_audio(self, pcm16_bytes: bytes) -> bytes:
        """
        AI audio is already PCM16 24kHz from OpenAI Realtime API.
        Just pass through (or resample if needed).
        """
        try:
            # OpenAI sends PCM16 at 24kHz, so we can use it directly
            # If you find it's actually different, adjust AI_SAMPLE_RATE constant
            return pcm16_bytes
        except Exception as e:
            Log.error(f"[Resample AI] Error: {e}")
            return pcm16_bytes
    
    def _create_wav_base64(self, pcm16_bytes: bytes, sample_rate: int) -> str:
        """
        Create a WAV file from PCM16 data and return as base64.
        """
        try:
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(pcm16_bytes)
            
            wav_io.seek(0)
            wav_bytes = wav_io.read()
            return base64.b64encode(wav_bytes).decode('ascii')
        except Exception as e:
            Log.error(f"[WAV] Error creating WAV: {e}")
            return base64.b64encode(pcm16_bytes).decode('ascii')
    
    async def _stream_unified_audio(self):
        """
        Unified streaming with asymmetric gaps:
        - Caller → AI: 0.5s gap (natural conversation flow)
        - AI → Caller: NO gap (processing delay is enough)
        """
        Log.info("[Unified Stream] Started with asymmetric gap control")
        
        while not self._shutdown:
            try:
                audio_data = await self._unified_audio_queue.get()
                
                if audio_data is None:
                    break
                
                speaker = audio_data.get("speaker")
                queued_time = audio_data.get("queued_time", time.time())
                
                # Check if we're switching speakers
                speaker_changed = (
                    self._last_streamed_speaker is not None and 
                    self._last_streamed_speaker != speaker
                )
                
                if speaker_changed:
                    previous_speaker = self._last_streamed_speaker
                    previous_last_queued = self._last_queued_time_per_speaker.get(previous_speaker, 0)
                    
                    # Calculate natural gap between speakers
                    time_gap = queued_time - previous_last_queued if previous_last_queued > 0 else 0
                    previous_finished = time_gap >= self.SPEAKER_SILENCE_THRESHOLD
                    
                    # ASYMMETRIC GAP LOGIC
                    if previous_speaker == "Caller" and speaker == "AI":
                        # Caller → AI: Add gap for natural feel
                        if previous_finished and time_gap < self.CALLER_TO_AI_GAP:
                            remaining_gap = self.CALLER_TO_AI_GAP - time_gap
                            Log.debug(f"[Stream] Caller → AI: adding {remaining_gap:.3f}s gap")
                            await asyncio.sleep(remaining_gap)
                        elif previous_finished:
                            Log.debug(f"[Stream] Caller → AI: natural gap {time_gap:.3f}s sufficient")
                    
                    elif previous_speaker == "AI" and speaker == "Caller":
                        # AI → Caller: NO artificial gap (processing delay is enough)
                        Log.debug(f"[Stream] AI → Caller: no gap (natural: {time_gap:.3f}s)")
                    
                    else:
                        # Same speaker continuing or quick interruption
                        Log.debug(f"[Stream] {previous_speaker} → {speaker}: no gap")
                
                # Update tracking
                self._last_queued_time_per_speaker[speaker] = queued_time
                self._last_streamed_speaker = speaker
                
                # Process and send audio
                if self.audio_callback:
                    try:
                        audio_b64 = audio_data.get("audio", "")
                        raw_bytes = base64.b64decode(audio_b64)
                        
                        # Resample based on speaker
                        if speaker == "Caller":
                            # Caller: µ-law 8kHz → PCM16 24kHz
                            pcm16_bytes = self._resample_caller_audio(raw_bytes)
                            source_rate = self.CALLER_SAMPLE_RATE
                        else:  # AI
                            # AI: Already PCM16 24kHz (or convert if needed)
                            pcm16_bytes = self._resample_ai_audio(raw_bytes)
                            source_rate = self.AI_SAMPLE_RATE
                        
                        # Create WAV
                        wav_b64 = self._create_wav_base64(pcm16_bytes, self.DASHBOARD_SAMPLE_RATE)
                        
                        # Send to dashboard
                        enhanced_payload = {
                            "speaker": speaker,
                            "audio": wav_b64,
                            "timestamp": audio_data.get("timestamp"),
                            "size": len(pcm16_bytes),
                            "format": "wav",
                            "sampleRate": self.DASHBOARD_SAMPLE_RATE,
                            "channels": 1,
                            "bitDepth": 16
                        }
                        
                        await self.audio_callback(enhanced_payload)
                    except Exception as e:
                        Log.error(f"[Stream] callback error: {e}")
                
                # Wait for chunk playback
                try:
                    chunk_duration = self._calculate_chunk_duration(pcm16_bytes, self.DASHBOARD_SAMPLE_RATE)
                    await asyncio.sleep(chunk_duration * 0.95)
                except Exception:
                    await asyncio.sleep(0.02)
                
                self._unified_audio_queue.task_done()
                
            except Exception as e:
                Log.error(f"[Stream] error: {e}")
                await asyncio.sleep(0.01)
    
    async def transcribe_realtime(self, audio_input, source: str = "Unknown") -> str:
        """
        Process incoming audio: queue for streaming + buffer for transcription.
        """
        try:
            # Convert to bytes
            if isinstance(audio_input, str):
                audio_bytes = base64.b64decode(audio_input)
                original_base64 = audio_input
            elif isinstance(audio_input, (bytes, bytearray)):
                audio_bytes = bytes(audio_input)
                original_base64 = base64.b64encode(audio_bytes).decode('ascii')
            else:
                return ""
            
            # Capture exact arrival time
            queued_time = time.time()
            
            # Queue for streaming
            audio_packet = {
                "speaker": source,
                "audio": original_base64,
                "timestamp": int(queued_time),
                "queued_time": queued_time,
                "size": len(audio_bytes)
            }
            
            await self._unified_audio_queue.put(audio_packet)
            
            # Buffer for transcription
            if source == "Caller":
                await self._add_to_caller_buffer(audio_bytes)
            elif source == "AI":
                await self._add_to_ai_buffer(audio_bytes)
            
            return ""
            
        except Exception as e:
            Log.error(f"[{source}] Error: {e}")
            return ""
    
    async def _add_to_caller_buffer(self, audio_bytes: bytes):
        """Add to caller transcription buffer."""
        current_time = time.time()
        
        if len(self._caller_buffer) == 0:
            self._caller_first_chunk_time = current_time
        
        self._caller_buffer.extend(audio_bytes)
        self._caller_last_chunk_time = current_time
        
        if not self._caller_monitor_task or self._caller_monitor_task.done():
            self._caller_monitor_task = asyncio.create_task(
                self._monitor_caller_buffer()
            )
    
    async def _add_to_ai_buffer(self, audio_bytes: bytes):
        """Add to AI transcription buffer."""
        current_time = time.time()
        
        if len(self._ai_buffer) == 0:
            self._ai_first_chunk_time = current_time
        
        self._ai_buffer.extend(audio_bytes)
        self._ai_last_chunk_time = current_time
        
        if not self._ai_monitor_task or self._ai_monitor_task.done():
            self._ai_monitor_task = asyncio.create_task(
                self._monitor_ai_buffer()
            )
    
    async def _monitor_caller_buffer(self):
        """Monitor caller buffer for transcription."""
        while not self._shutdown:
            try:
                await asyncio.sleep(0.1)
                
                buffer_duration = len(self._caller_buffer) / 8000.0
                if buffer_duration < self.MIN_AUDIO_DURATION:
                    continue
                
                time_since_last = time.time() - self._caller_last_chunk_time
                
                should_transcribe = (
                    time_since_last >= self.SILENCE_TIMEOUT or
                    buffer_duration >= self.MAX_BUFFER_DURATION
                )
                
                if should_transcribe and not self._caller_processing:
                    await self._transcribe_buffer("Caller")
                    
            except Exception as e:
                Log.error(f"[Caller monitor] error: {e}")
                break
    
    async def _monitor_ai_buffer(self):
        """Monitor AI buffer for transcription."""
        while not self._shutdown:
            try:
                await asyncio.sleep(0.1)
                
                buffer_duration = len(self._ai_buffer) / 24000.0  # AI is 24kHz
                if buffer_duration < self.MIN_AUDIO_DURATION:
                    continue
                
                time_since_last = time.time() - self._ai_last_chunk_time
                
                should_transcribe = (
                    time_since_last >= self.SILENCE_TIMEOUT or
                    buffer_duration >= self.MAX_BUFFER_DURATION
                )
                
                if should_transcribe and not self._ai_processing:
                    await self._transcribe_buffer("AI")
                    
            except Exception as e:
                Log.error(f"[AI monitor] error: {e}")
                break
    
    async def _transcribe_buffer(self, source: str):
        """Transcribe accumulated buffer."""
        if source == "Caller":
            if self._caller_processing or len(self._caller_buffer) == 0:
                return
            self._caller_processing = True
            
            try:
                audio_data = bytes(self._caller_buffer)
                self._caller_buffer.clear()
                
                transcript = await self._transcribe_audio(audio_data, source, self.CALLER_SAMPLE_RATE)
                
                if transcript and transcript != self._caller_last_transcript:
                    self._caller_last_transcript = transcript
                    
                    if self.transcription_callback:
                        await self.transcription_callback({
                            "speaker": source,
                            "text": transcript,
                            "timestamp": int(time.time())
                        })
                
            finally:
                self._caller_processing = False
                
        elif source == "AI":
            if self._ai_processing or len(self._ai_buffer) == 0:
                return
            self._ai_processing = True
            
            try:
                audio_data = bytes(self._ai_buffer)
                self._ai_buffer.clear()
                
                transcript = await self._transcribe_audio(audio_data, source, self.AI_SAMPLE_RATE)
                
                if transcript and transcript != self._ai_last_transcript:
                    self._ai_last_transcript = transcript
                    
                    if self.transcription_callback:
                        await self.transcription_callback({
                            "speaker": source,
                            "text": transcript,
                            "timestamp": int(time.time())
                        })
                
            finally:
                self._ai_processing = False
    
    async def _transcribe_audio(self, audio_bytes: bytes, source: str, source_sample_rate: int) -> str:
        """
        Convert audio to PCM16 16kHz and transcribe with Whisper.
        """
        try:
            # Convert to PCM16 if needed
            if source == "Caller":
                pcm16 = self._mulaw_to_pcm16(audio_bytes)
            else:  # AI already PCM16
                pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Resample to 16kHz for Whisper
            pcm16_16k = self._resample_pcm16(pcm16, source_sample_rate, 16000)
            
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm16_16k.tobytes())
            wav_io.seek(0)
            
            headers = {"Authorization": f"Bearer {Config.OPENAI_API_KEY}"}
            form = aiohttp.FormData()
            form.add_field("file", wav_io, filename="audio.wav", content_type="audio/wav")
            form.add_field("model", "gpt-4o-mini-transcribe")
            form.add_field("language", "en")
            form.add_field("prompt", 
                "Transcribe using Latin script. "
                "For Urdu words, use Roman Urdu (e.g., 'aapka shukriya'). "
                "For Punjabi words, use Roman Punjabi (e.g., 'ki haal hai'). "
                "Keep English words as-is."
            )
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.OPENAI_API_URL, headers=headers, data=form) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        Log.error(f"[{source}] Transcription failed ({resp.status}): {text}")
                        return ""
                    
                    data = await resp.json()
                    transcript = (data.get("text") or "").strip()
                    
                    if transcript:
                        Log.info(f"[{source}] 📝 {transcript}")
                    
                    return transcript
                    
        except Exception as e:
            Log.error(f"[{source}] Transcription error: {e}")
            return ""
    
    def _mulaw_to_pcm16(self, mulaw_bytes: bytes) -> np.ndarray:
        """Convert µ-law 8-bit to PCM16."""
        mu = np.frombuffer(mulaw_bytes, dtype=np.uint8)
        mu = ~mu
        
        sign = (mu & 0x80).astype(np.int32)
        exponent = ((mu >> 4) & 0x07).astype(np.int32)
        mantissa = (mu & 0x0F).astype(np.int32)
        
        magnitude = ((mantissa << 3) + 0x84) << exponent
        magnitude = np.clip(magnitude, 0, 0x7FFF)
        
        pcm16 = np.where(sign == 0, magnitude, -magnitude)
        return pcm16.astype(np.int16)
    
    def _resample_pcm16(self, pcm_data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Resample PCM16 audio."""
        if src_rate == dst_rate:
            return pcm_data
        num_samples = int(len(pcm_data) * dst_rate / src_rate)
        return resample(pcm_data, num_samples).astype(np.int16)
    
    async def shutdown(self):
        """Gracefully shutdown."""
        try:
            self._shutdown = True
            await self._unified_audio_queue.put(None)
            
            if self._stream_task and not self._stream_task.done():
                await asyncio.wait([self._stream_task], timeout=2.0)
            
            Log.info("TranscriptionService shutdown complete")
                
        except Exception as e:
            Log.error(f"Shutdown error: {e}")
