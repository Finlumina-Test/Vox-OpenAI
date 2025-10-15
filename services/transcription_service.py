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
    - Detects when a speaker has FINISHED speaking (silence period)
    - Adds 0.5s gap only between different speakers' COMPLETE turns
    - No gaps within same speaker's continuous chunks
    - Resamples µ-law 8kHz to PCM16 16kHz for better browser playback
    """
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    
    # Audio format specs
    SAMPLE_RATE = 8000  # µ-law 8kHz from Twilio and OpenAI
    DASHBOARD_SAMPLE_RATE = 16000  # PCM16 16kHz for dashboard playback
    CHUNK_DURATION_MS = 20
    BYTES_PER_20MS = 160
    
    # Speaker turn detection
    SPEAKER_SILENCE_THRESHOLD = 0.3  # If no chunks for 0.3s, speaker is done
    SPEAKER_TRANSITION_DELAY = 0.5   # Gap between different speakers
    
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
        self._last_chunk_time_per_speaker: Dict[str, float] = {}
        self._last_queued_time_per_speaker: Dict[str, float] = {}  # Track when audio was queued, not streamed
        self._speaker_finished: Dict[str, bool] = {"Caller": False, "AI": False}
        
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
    
    def _calculate_chunk_duration(self, audio_bytes: bytes, sample_rate: int = None) -> float:
        """Calculate audio chunk duration in seconds."""
        if sample_rate is None:
            sample_rate = self.DASHBOARD_SAMPLE_RATE
        num_samples = len(audio_bytes) // 2  # PCM16 = 2 bytes per sample
        duration_seconds = num_samples / sample_rate
        return duration_seconds
    
    def _resample_mulaw_to_pcm16(self, mulaw_bytes: bytes) -> bytes:
        """
        Convert µ-law 8kHz to PCM16 16kHz for better dashboard playback quality.
        
        Args:
            mulaw_bytes: Raw µ-law encoded audio at 8kHz
            
        Returns:
            PCM16 encoded audio at 16kHz as bytes
        """
        try:
            # Convert µ-law to PCM16
            pcm16_8k = self._mulaw_to_pcm16(mulaw_bytes)
            
            # Resample from 8kHz to 16kHz
            pcm16_16k = self._resample_pcm16(pcm16_8k, self.SAMPLE_RATE, self.DASHBOARD_SAMPLE_RATE)
            
            return pcm16_16k.tobytes()
        except Exception as e:
            Log.error(f"[Resample] Error converting audio: {e}")
            # Fallback: just convert µ-law to PCM16 at 8kHz
            try:
                pcm16_8k = self._mulaw_to_pcm16(mulaw_bytes)
                return pcm16_8k.tobytes()
            except:
                return mulaw_bytes
    
    def _create_wav_base64(self, pcm16_bytes: bytes, sample_rate: int = None) -> str:
        """
        Create a WAV file from PCM16 data and return as base64.
        This provides better browser compatibility.
        
        Args:
            pcm16_bytes: PCM16 audio data
            sample_rate: Sample rate (default: 16kHz)
            
        Returns:
            Base64 encoded WAV file
        """
        if sample_rate is None:
            sample_rate = self.DASHBOARD_SAMPLE_RATE
            
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
    
    async def _check_speaker_finished(self, speaker: str) -> bool:
        """
        Check if a speaker has finished their turn (no chunks recently).
        """
        last_time = self._last_chunk_time_per_speaker.get(speaker, 0)
        if last_time == 0:
            return False
        
        time_since_last = time.time() - last_time
        return time_since_last >= self.SPEAKER_SILENCE_THRESHOLD
    
    async def _stream_unified_audio(self):
        """
        Unified streaming task with smart speaker transition detection.
        Only adds gap when transitioning between COMPLETE speaker turns.
        Sends resampled high-quality audio to dashboard.
        """
        Log.info("[Unified Stream] Started with smart speaker detection and resampling")
        
        while not self._shutdown:
            try:
                # Wait for next chunk
                audio_data = await self._unified_audio_queue.get()
                
                if audio_data is None:
                    break
                
                speaker = audio_data.get("speaker")
                queued_time = audio_data.get("queued_time", time.time())  # When it was queued
                current_time = time.time()
                
                # Check if we're switching speakers
                speaker_changed = (
                    self._last_streamed_speaker is not None and 
                    self._last_streamed_speaker != speaker
                )
                
                if speaker_changed:
                    # Use QUEUED time to calculate gap, not processing time
                    previous_speaker = self._last_streamed_speaker
                    previous_last_queued = self._last_queued_time_per_speaker.get(previous_speaker, 0)
                    
                    # Calculate gap between when previous speaker's last chunk was QUEUED 
                    # and when current speaker's first chunk was QUEUED
                    time_gap = queued_time - previous_last_queued if previous_last_queued > 0 else 0
                    
                    # Check if previous speaker finished
                    previous_finished = time_gap >= self.SPEAKER_SILENCE_THRESHOLD
                    
                    if previous_finished and time_gap < self.SPEAKER_TRANSITION_DELAY:
                        # Add remaining gap to reach the desired delay
                        remaining_gap = self.SPEAKER_TRANSITION_DELAY - time_gap
                        Log.debug(f"[Stream] Turn complete: {previous_speaker} → {speaker}, adding {remaining_gap:.3f}s gap (natural: {time_gap:.3f}s)")
                        await asyncio.sleep(remaining_gap)
                    elif previous_finished:
                        # Natural gap already exceeds our desired delay, no artificial gap needed
                        Log.debug(f"[Stream] Turn complete: {previous_speaker} → {speaker}, natural gap {time_gap:.3f}s (no artificial gap needed)")
                    else:
                        # Quick interruption/overlap - no gap
                        Log.debug(f"[Stream] Quick switch {previous_speaker} → {speaker}, no gap (time_gap: {time_gap:.3f}s)")
                
                # Update tracking times
                self._last_chunk_time_per_speaker[speaker] = current_time
                self._last_queued_time_per_speaker[speaker] = queued_time
                
                # Update current speaker
                self._last_streamed_speaker = speaker
                
                # Send to dashboard with resampled audio
                if self.audio_callback:
                    try:
                        # Get the original µ-law audio
                        audio_b64 = audio_data.get("audio", "")
                        mulaw_bytes = base64.b64decode(audio_b64)
                        
                        # Resample to PCM16 16kHz for better quality
                        pcm16_bytes = self._resample_mulaw_to_pcm16(mulaw_bytes)
                        
                        # Create WAV format for better browser compatibility
                        wav_b64 = self._create_wav_base64(pcm16_bytes, self.DASHBOARD_SAMPLE_RATE)
                        
                        # Create enhanced payload
                        enhanced_payload = {
                            "speaker": speaker,
                            "audio": wav_b64,  # High quality resampled audio
                            "timestamp": audio_data.get("timestamp"),
                            "size": len(pcm16_bytes),
                            "format": "wav",  # Indicate WAV format
                            "sampleRate": self.DASHBOARD_SAMPLE_RATE,
                            "channels": 1,
                            "bitDepth": 16
                        }
                        
                        await self.audio_callback(enhanced_payload)
                    except Exception as e:
                        Log.error(f"[Stream] callback error: {e}")
                
                # Wait for chunk playback (use new sample rate for timing)
                try:
                    audio_b64 = audio_data.get("audio", "")
                    mulaw_bytes = base64.b64decode(audio_b64)
                    pcm16_bytes = self._resample_mulaw_to_pcm16(mulaw_bytes)
                    chunk_duration = self._calculate_chunk_duration(pcm16_bytes, self.DASHBOARD_SAMPLE_RATE)
                    await asyncio.sleep(chunk_duration * 0.95)
                except Exception as e:
                    Log.debug(f"[Stream] Duration calc error: {e}")
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
            
            # Capture the EXACT time this audio was received/queued
            queued_time = time.time()
            
            # Queue for streaming (with queued timestamp for accurate gap calculation)
            audio_packet = {
                "speaker": source,
                "audio": original_base64,
                "timestamp": int(queued_time),
                "queued_time": queued_time,  # Critical: when audio actually arrived
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
