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
    Real-time transcription service with strict sequential audio delivery.
    
    Key Features:
    - Sequential audio playback (no overlap possible)
    - 0.5s gap ONLY for Caller->AI transitions
    - NO gaps for AI->Caller (natural conversation flow)
    - Silence detection to prevent silent chunk backlog
    - Async lock ensures one chunk completes before next starts
    """
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    
    # Audio format specs
    SAMPLE_RATE = 8000  # µ-law 8kHz from Twilio and OpenAI
    CHUNK_DURATION_MS = 20
    BYTES_PER_20MS = 160
    
    # Speaker turn detection
    SPEAKER_SILENCE_THRESHOLD = 0.3  # If no chunks for 0.3s, speaker is done
    SPEAKER_TRANSITION_DELAY = 0.5   # Gap ONLY for Caller->AI
    
    # Silence detection (for skipping silent chunks)
    SILENCE_THRESHOLD_ENERGY = 500  # Energy threshold for silence (tune as needed)
    SILENCE_THRESHOLD_RMS = 200     # RMS threshold for silence
    
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
        
        # Speaker tracking
        self._last_streamed_speaker: Optional[str] = None
        self._last_chunk_time_per_speaker: Dict[str, float] = {}
        
        # 🔒 CRITICAL: Sequential playback lock
        self._playback_lock: asyncio.Lock = asyncio.Lock()
        
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
    
    def _calculate_chunk_duration(self, audio_bytes: bytes) -> float:
        """Calculate audio chunk duration in seconds (8kHz µ-law)."""
        num_samples = len(audio_bytes)
        duration_seconds = num_samples / self.SAMPLE_RATE
        return duration_seconds
    
    def _is_silence(self, mulaw_bytes: bytes) -> bool:
        """
        Detect if audio chunk is silence using energy and RMS thresholds.
        
        This prevents silent caller chunks (when listening to AI) from
        clogging the queue and adding artificial delay.
        """
        try:
            # Convert µ-law to PCM16 for analysis
            pcm16 = self._mulaw_to_pcm16(mulaw_bytes)
            
            # Calculate energy (sum of squared amplitudes)
            energy = np.sum(pcm16.astype(np.float64) ** 2)
            
            # Calculate RMS (root mean square)
            rms = np.sqrt(np.mean(pcm16.astype(np.float64) ** 2))
            
            # Chunk is silent if both energy and RMS are below thresholds
            is_silent = (energy < self.SILENCE_THRESHOLD_ENERGY and 
                        rms < self.SILENCE_THRESHOLD_RMS)
            
            return is_silent
            
        except Exception as e:
            Log.debug(f"[Silence Detection] Error: {e}")
            return False  # If error, assume not silent (safer)
    
    async def _stream_unified_audio(self):
        """
        Sequential audio streaming with smart speaker transitions.
        
        🔒 Uses async lock to guarantee:
        - ONE chunk plays at a time (no overlap)
        - 0.5s gap for Caller->AI transitions
        - NO gap for AI->Caller transitions
        """
        Log.info("[Stream] Started - SEQUENTIAL with Caller→AI gap only")
        
        while not self._shutdown:
            try:
                # Wait for next chunk
                audio_data = await self._unified_audio_queue.get()
                
                if audio_data is None:
                    break
                
                speaker = audio_data.get("speaker")
                current_time = time.time()
                
                # 🔒 LOCK: Ensures sequential playback
                async with self._playback_lock:
                    
                    # Calculate chunk duration first
                    audio_b64 = audio_data.get("audio", "")
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        chunk_duration = self._calculate_chunk_duration(audio_bytes)
                    except Exception as e:
                        Log.debug(f"[Stream] Duration calc error: {e}")
                        chunk_duration = 0.02
                    
                    # Check if speaker changed
                    speaker_changed = (
                        self._last_streamed_speaker is not None and 
                        self._last_streamed_speaker != speaker
                    )
                    
                    if speaker_changed:
                        previous_speaker = self._last_streamed_speaker
                        previous_last_time = self._last_chunk_time_per_speaker.get(previous_speaker, 0)
                        
                        # Time gap since previous speaker's last chunk
                        time_gap = current_time - previous_last_time if previous_last_time > 0 else 0
                        
                        # Check if previous speaker finished
                        previous_finished = time_gap >= self.SPEAKER_SILENCE_THRESHOLD
                        
                        # ✅ ONLY add gap for Caller → AI
                        if previous_speaker == "Caller" and speaker == "AI" and previous_finished:
                            if time_gap < self.SPEAKER_TRANSITION_DELAY:
                                remaining_gap = self.SPEAKER_TRANSITION_DELAY - time_gap
                                Log.debug(f"[Stream] Caller → AI: +{remaining_gap:.3f}s gap")
                                await asyncio.sleep(remaining_gap)
                            else:
                                Log.debug(f"[Stream] Caller → AI: {time_gap:.3f}s natural")
                        
                        # ✅ AI → Caller: NO gap
                        elif previous_speaker == "AI" and speaker == "Caller":
                            Log.debug(f"[Stream] AI → Caller: NO GAP")
                        
                        # Same speaker: no gap
                        else:
                            Log.debug(f"[Stream] {speaker} continuing")
                    
                    # Update tracking
                    self._last_chunk_time_per_speaker[speaker] = current_time
                    self._last_streamed_speaker = speaker
                    
                    # Send to dashboard
                    if self.audio_callback:
                        try:
                            await self.audio_callback(audio_data)
                        except Exception as e:
                            Log.error(f"[Stream] callback error: {e}")
                    
                    # ⏱️ Wait for chunk to complete playback
                    # This ensures next chunk won't start until current finishes
                    await asyncio.sleep(chunk_duration)
                
                # Lock released here - next chunk can now proceed
                self._unified_audio_queue.task_done()
                
            except Exception as e:
                Log.error(f"[Stream] error: {e}")
                await asyncio.sleep(0.01)
    
    async def transcribe_realtime(self, audio_input, source: str = "Unknown") -> str:
        """
        Process incoming audio: queue for streaming + buffer for transcription.
        
        🎯 KEY FIX: Skip silent caller chunks to prevent queue backlog!
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
            
            # 🎯 CRITICAL: Skip silent caller chunks
            # This prevents queue backlog when caller is listening
            if source == "Caller":
                if self._is_silence(audio_bytes):
                    # Still buffer for transcription (might be pause between words)
                    await self._add_to_caller_buffer(audio_bytes)
                    # But DON'T queue for streaming - this prevents delay!
                    return ""
            
            # Queue for streaming (only non-silent chunks for Caller)
            audio_packet = {
                "speaker": source,
                "audio": original_base64,
                "timestamp": int(time.time()),
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
                
                buffer_duration = len(self._ai_buffer) / 8000.0
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
                
                transcript = await self._transcribe_audio(audio_data, source)
                
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
                
                transcript = await self._transcribe_audio(audio_data, source)
                
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
    
    async def _transcribe_audio(self, mulaw_bytes: bytes, source: str) -> str:
        """
        Convert µ-law 8kHz to PCM16 16kHz and transcribe.
        
        Audio conversion for Whisper:
        - Input: µ-law 8kHz (from Twilio/OpenAI)
        - Output: PCM16 16kHz (for Whisper API)
        """
        try:
            pcm16 = self._mulaw_to_pcm16(mulaw_bytes)
            pcm16_16k = self._resample_pcm16(pcm16, 8000, 16000)
            
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
