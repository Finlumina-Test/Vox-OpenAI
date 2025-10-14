import io
import wave
import base64
import asyncio
import aiohttp
import numpy as np
import time    
from scipy.signal import resample
from typing import Dict, Optional, List
from config import Config
from services.log_utils import Log


class TranscriptionService:
    """
    Real-time transcription service with SEPARATE audio streaming.
    
    Key Changes:
    - Audio chunks are sent IMMEDIATELY as they arrive (no pairing with words)
    - Transcription runs independently and sends complete phrases
    - No word-level audio pairing (prevents repetition bug)
    """
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    
    # Transcription accumulation settings
    MIN_AUDIO_DURATION = 0.8  # Accumulate at least 800ms before transcribing
    SILENCE_TIMEOUT = 0.5  # Transcribe after 500ms of silence
    MAX_BUFFER_DURATION = 3.0  # Force transcribe after 3 seconds
    
    def __init__(self):
        # Separate buffers for each speaker
        self._caller_buffer: bytearray = bytearray()
        self._ai_buffer: bytearray = bytearray()
        
        # Timing tracking
        self._caller_last_chunk_time: float = 0
        self._ai_last_chunk_time: float = 0
        
        self._caller_first_chunk_time: float = 0
        self._ai_first_chunk_time: float = 0
        
        # Processing flags
        self._caller_processing: bool = False
        self._ai_processing: bool = False
        
        # Transcription monitoring tasks
        self._caller_monitor_task: Optional[asyncio.Task] = None
        self._ai_monitor_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.audio_callback = None  # For raw audio streaming
        self.transcription_callback = None  # For completed transcriptions
        
        # Track last transcription to avoid duplicates
        self._caller_last_transcript: str = ""
        self._ai_last_transcript: str = ""
    
    def set_audio_callback(self, callback):
        """Set callback for raw audio chunks (real-time streaming)."""
        self.audio_callback = callback
    
    def set_word_callback(self, callback):
        """Legacy: For compatibility. Now used for transcription results."""
        self.transcription_callback = callback
    
    async def transcribe_realtime(self, audio_input, source: str = "Unknown") -> str:
        """
        Process incoming audio with IMMEDIATE audio streaming + background transcription.
        
        Flow:
        1. Audio chunk arrives
        2. IMMEDIATELY send to dashboard via audio_callback (for real-time playback)
        3. Add to buffer for transcription
        4. Background task monitors buffer and transcribes when ready
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
            
            # ✅ IMMEDIATELY send raw audio for playback (no delay, no pairing)
            if self.audio_callback:
                await self.audio_callback({
                    "speaker": source,
                    "audio": original_base64,
                    "timestamp": int(time.time())
                })
            
            # Add to transcription buffer (runs in background)
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
        while True:
            try:
                await asyncio.sleep(0.1)  # Check every 100ms
                
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
        while True:
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
                    
                    # Send transcription update (NOT word-by-word, complete phrase)
                    if self.transcription_callback:
                        await self.transcription_callback({
                            "speaker": source,
                            "text": transcript,  # Complete phrase, not individual words
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
        Convert µ-law audio to PCM16 WAV and transcribe with Latin script preference.
        """
        try:
            # Convert µ-law to PCM16
            pcm16 = self._mulaw_to_pcm16(mulaw_bytes)
            
            # Resample from 8kHz to 16kHz
            pcm16_16k = self._resample_pcm16(pcm16, 8000, 16000)
            
            # Create WAV file in memory
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm16_16k.tobytes())
            wav_io.seek(0)
            
            # ✅ ADD PROMPT FOR LATIN SCRIPT (Roman Urdu/Punjabi)
            headers = {"Authorization": f"Bearer {Config.OPENAI_API_KEY}"}
            form = aiohttp.FormData()
            form.add_field("file", wav_io, filename="audio.wav", content_type="audio/wav")
            form.add_field("model", "gpt-4o-mini-transcribe")
            form.add_field("language", "en")  # Hint for English base
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
