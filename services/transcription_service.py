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
    Real-time transcription service with PROPER TIMED audio streaming.
    
    Key Features:
    - Audio chunks are queued and sent with proper 20ms timing
    - Respects actual audio playback duration
    - No overlapping - proper flow maintained
    - Transcription runs independently in background
    - Natural pause between speaker transitions
    """
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    
    # Audio chunk timing (µ-law 8kHz, 20ms chunks = 160 bytes)
    CHUNK_DURATION_MS = 20  # Each chunk is 20ms of audio
    SAMPLE_RATE = 8000  # 8kHz sampling rate
    BYTES_PER_20MS = 160  # 8000 samples/sec * 0.02 sec = 160 bytes
    
    # Speaker transition settings
    SPEAKER_TRANSITION_DELAY = 0.85  # Seconds of silence between speakers (0.7-1.0)
    
    # Transcription accumulation settings
    MIN_AUDIO_DURATION = 0.8
    SILENCE_TIMEOUT = 0.5
    MAX_BUFFER_DURATION = 3.0
    
    def __init__(self):
        # Separate buffers for each speaker (transcription)
        self._caller_buffer: bytearray = bytearray()
        self._ai_buffer: bytearray = bytearray()
        
        # Audio streaming queues (for timed delivery)
        self._caller_audio_queue: asyncio.Queue = asyncio.Queue()
        self._ai_audio_queue: asyncio.Queue = asyncio.Queue()
        
        # Sequential streaming tasks
        self._caller_stream_task: Optional[asyncio.Task] = None
        self._ai_stream_task: Optional[asyncio.Task] = None
        
        # Streaming state
        self._caller_streaming: bool = False
        self._ai_streaming: bool = False
        
        # Timing tracking
        self._caller_last_chunk_time: float = 0
        self._ai_last_chunk_time: float = 0
        
        self._caller_first_chunk_time: float = 0
        self._ai_first_chunk_time: float = 0
        
        # Track last activity per speaker for transition gaps
        self._last_speaker_activity: Dict[str, float] = {
            "Caller": 0,
            "AI": 0
        }
        self._current_speaker: Optional[str] = None
        
        # Processing flags
        self._caller_processing: bool = False
        self._ai_processing: bool = False
        
        # Transcription monitoring tasks
        self._caller_monitor_task: Optional[asyncio.Task] = None
        self._ai_monitor_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.audio_callback: Optional[Callable] = None
        self.transcription_callback: Optional[Callable] = None
        
        # Track last transcription to avoid duplicates
        self._caller_last_transcript: str = ""
        self._ai_last_transcript: str = ""
        
        # Shutdown flag
        self._shutdown: bool = False
    
    def set_audio_callback(self, callback: Callable):
        """Set callback for raw audio chunks (real-time streaming)."""
        self.audio_callback = callback
        
        # Start sequential streaming tasks
        if not self._caller_stream_task or self._caller_stream_task.done():
            self._caller_stream_task = asyncio.create_task(self._stream_caller_audio())
        
        if not self._ai_stream_task or self._ai_stream_task.done():
            self._ai_stream_task = asyncio.create_task(self._stream_ai_audio())
    
    def set_word_callback(self, callback: Callable):
        """Legacy: For compatibility. Now used for transcription results."""
        self.transcription_callback = callback
    
    def _calculate_chunk_duration(self, audio_bytes: bytes) -> float:
        """
        Calculate actual duration of audio chunk in seconds.
        µ-law: 8000 samples/sec, 1 byte per sample
        """
        num_samples = len(audio_bytes)
        duration_seconds = num_samples / self.SAMPLE_RATE
        return duration_seconds
    
    async def _wait_for_speaker_transition(self, new_speaker: str) -> None:
        """
        Add natural pause when switching between speakers.
        Only applies when switching from Caller -> AI or AI -> Caller.
        """
        if self._current_speaker is None:
            # First speaker - no wait
            self._current_speaker = new_speaker
            return
        
        if self._current_speaker == new_speaker:
            # Same speaker continuing - no wait
            return
        
        # Different speaker - check if enough time has passed
        last_activity = self._last_speaker_activity.get(self._current_speaker, 0)
        time_since_last = time.time() - last_activity
        
        if time_since_last < self.SPEAKER_TRANSITION_DELAY:
            # Not enough time passed - wait for the remainder
            remaining_wait = self.SPEAKER_TRANSITION_DELAY - time_since_last
            Log.debug(f"[Speaker Transition] Waiting {remaining_wait:.2f}s before {new_speaker} speaks")
            await asyncio.sleep(remaining_wait)
        
        # Update current speaker
        self._current_speaker = new_speaker
    
    async def _stream_caller_audio(self):
        """
        Sequential audio streaming task for Caller.
        Sends chunks with proper timing based on audio duration.
        """
        Log.info("[Caller Stream] Started")
        
        while not self._shutdown:
            try:
                # Wait for next audio chunk
                audio_data = await self._caller_audio_queue.get()
                
                if audio_data is None:  # Shutdown signal
                    break
                
                # Wait for speaker transition if needed
                await self._wait_for_speaker_transition("Caller")
                
                self._caller_streaming = True
                
                # Send to dashboard
                if self.audio_callback:
                    try:
                        await self.audio_callback(audio_data)
                    except Exception as e:
                        Log.error(f"[Caller stream] callback error: {e}")
                
                # Update last activity time
                self._last_speaker_activity["Caller"] = time.time()
                
                # Calculate how long this chunk should take to play
                audio_b64 = audio_data.get("audio", "")
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    chunk_duration = self._calculate_chunk_duration(audio_bytes)
                    
                    # Wait for the chunk to "play" before sending next one
                    await asyncio.sleep(chunk_duration * 0.95)
                    
                except Exception as e:
                    Log.debug(f"[Caller] Duration calc error: {e}, using default 20ms")
                    await asyncio.sleep(0.02)
                
                self._caller_audio_queue.task_done()
                self._caller_streaming = False
                
            except Exception as e:
                Log.error(f"[Caller stream] error: {e}")
                self._caller_streaming = False
                await asyncio.sleep(0.01)
    
    async def _stream_ai_audio(self):
        """
        Sequential audio streaming task for AI.
        Sends chunks with PRECISE timing based on audio duration.
        Includes natural pause when transitioning from Caller.
        """
        Log.info("[AI Stream] Started")
        
        while not self._shutdown:
            try:
                # Wait for next audio chunk
                audio_data = await self._ai_audio_queue.get()
                
                if audio_data is None:  # Shutdown signal
                    break
                
                # ✅ CRITICAL: Wait for speaker transition (adds 0.7-1s gap)
                await self._wait_for_speaker_transition("AI")
                
                self._ai_streaming = True
                
                # Send to dashboard
                if self.audio_callback:
                    try:
                        await self.audio_callback(audio_data)
                    except Exception as e:
                        Log.error(f"[AI stream] callback error: {e}")
                
                # Update last activity time
                self._last_speaker_activity["AI"] = time.time()
                
                # Calculate actual duration and wait
                audio_b64 = audio_data.get("audio", "")
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    chunk_duration = self._calculate_chunk_duration(audio_bytes)
                    
                    # Wait for exact playback time
                    await asyncio.sleep(chunk_duration * 0.95)
                    
                    Log.debug(f"[AI] Sent chunk ({len(audio_bytes)} bytes, {chunk_duration*1000:.1f}ms)")
                    
                except Exception as e:
                    Log.debug(f"[AI] Duration calc error: {e}, using default 20ms")
                    await asyncio.sleep(0.02)
                
                self._ai_audio_queue.task_done()
                self._ai_streaming = False
                
            except Exception as e:
                Log.error(f"[AI stream] error: {e}")
                self._ai_streaming = False
                await asyncio.sleep(0.01)
    
    async def transcribe_realtime(self, audio_input, source: str = "Unknown") -> str:
        """
        Process incoming audio with TIMED audio streaming + background transcription.
        
        Flow:
        1. Audio chunk arrives
        2. Add to sequential queue with timing info
        3. Background task sends chunks with proper delays
        4. Separate task accumulates and transcribes in background
        """
        try:
            # Convert input to bytes
            if isinstance(audio_input, str):
                audio_bytes = base64.b64decode(audio_input)
                original_base64 = audio_input
            elif isinstance(audio_input, (bytes, bytearray)):
                audio_bytes = bytes(audio_input)
                original_base64 = base64.b64encode(audio_bytes).decode('ascii')
            else:
                return ""
            
            # Queue audio for TIMED streaming
            audio_packet = {
                "speaker": source,
                "audio": original_base64,
                "timestamp": int(time.time()),
                "size": len(audio_bytes)
            }
            
            if source == "Caller":
                await self._caller_audio_queue.put(audio_packet)
            elif source == "AI":
                await self._ai_audio_queue.put(audio_packet)
            
            # Add to transcription buffer (runs independently)
            if source == "Caller":
                await self._add_to_caller_buffer(audio_bytes)
            elif source == "AI":
                await self._add_to_ai_buffer(audio_bytes)
            
            return ""
            
        except Exception as e:
            Log.error(f"[{source}] Transcription error: {e}")
            return ""
    
    async def _add_to_caller_buffer(self, audio_bytes: bytes):
        """Add audio to caller buffer and start monitoring."""
        current_time = time.time()
        
        if len(self._caller_buffer) == 0:
            self._caller_first_chunk_time = current_time
        
        self._caller_buffer.extend(audio_bytes)
        self._caller_last_chunk_time = current_time
        
        # Start monitoring task if not running
        if not self._caller_monitor_task or self._caller_monitor_task.done():
            self._caller_monitor_task = asyncio.create_task(
                self._monitor_caller_buffer()
            )
    
    async def _add_to_ai_buffer(self, audio_bytes: bytes):
        """Add audio to AI buffer and start monitoring."""
        current_time = time.time()
        
        if len(self._ai_buffer) == 0:
            self._ai_first_chunk_time = current_time
        
        self._ai_buffer.extend(audio_bytes)
        self._ai_last_chunk_time = current_time
        
        # Start monitoring task if not running
        if not self._ai_monitor_task or self._ai_monitor_task.done():
            self._ai_monitor_task = asyncio.create_task(
                self._monitor_ai_buffer()
            )
    
    async def _monitor_caller_buffer(self):
        """Monitor caller buffer and transcribe when conditions are met."""
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
        """Monitor AI buffer and transcribe when conditions are met."""
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
        """Transcribe accumulated buffer and send complete phrase."""
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
        """Convert µ-law audio to PCM16 WAV and transcribe with Latin script preference."""
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
        """Convert µ-law 8-bit audio bytes to PCM16."""
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
        """Resample PCM16 audio from src_rate → dst_rate."""
        if src_rate == dst_rate:
            return pcm_data
        num_samples = int(len(pcm_data) * dst_rate / src_rate)
        return resample(pcm_data, num_samples).astype(np.int16)
    
    async def shutdown(self):
        """Gracefully shutdown streaming tasks."""
        try:
            self._shutdown = True
            
            # Signal shutdown to streaming tasks
            await self._caller_audio_queue.put(None)
            await self._ai_audio_queue.put(None)
            
            # Wait for tasks to complete with timeout
            tasks = []
            if self._caller_stream_task and not self._caller_stream_task.done():
                tasks.append(self._caller_stream_task)
            if self._ai_stream_task and not self._ai_stream_task.done():
                tasks.append(self._ai_stream_task)
            
            if tasks:
                await asyncio.wait(tasks, timeout=2.0)
            
            Log.info("TranscriptionService shutdown complete")
                
        except Exception as e:
            Log.error(f"Shutdown error: {e}")
