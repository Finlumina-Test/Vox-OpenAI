import io
import wave
import base64
import asyncio
import aiohttp
import numpy as np
from scipy.signal import resample
from typing import Dict, Optional
from config import Config
from services.log_utils import Log


class TranscriptionService:
    """
    Accumulates audio chunks and transcribes complete sentences.
    Waits for silence or timeout before sending to Whisper.
    """
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    
    # Accumulation settings
    MIN_AUDIO_DURATION = 1.0  # At least 1 second of audio
    SILENCE_TIMEOUT = 0.8  # Send after 0.8s of silence
    MAX_BUFFER_DURATION = 10.0  # Force send after 10 seconds
    
    def __init__(self):
        # Separate buffers for caller and AI
        self._caller_buffer: bytearray = bytearray()
        self._ai_buffer: bytearray = bytearray()
        
        # Timers for each speaker
        self._caller_last_chunk_time: float = 0
        self._ai_last_chunk_time: float = 0
        
        self._caller_first_chunk_time: float = 0
        self._ai_first_chunk_time: float = 0
        
        # Processing flags
        self._caller_processing: bool = False
        self._ai_processing: bool = False
        
        # Silence detection tasks
        self._caller_silence_task: Optional[asyncio.Task] = None
        self._ai_silence_task: Optional[asyncio.Task] = None
    
    async def transcribe_realtime(self, audio_input, source: str = "Unknown") -> str:
        """
        Accumulates audio chunks and transcribes when silence detected or buffer full.
        
        Args:
            audio_input: Either base64-encoded string or raw bytes
            source: "Caller" or "AI"
        """
        try:
            # Convert input to bytes
            if isinstance(audio_input, str):
                audio_bytes = base64.b64decode(audio_input)
            elif isinstance(audio_input, (bytes, bytearray)):
                audio_bytes = bytes(audio_input)
            else:
                return ""
            
            # Choose the right buffer
            if source == "Caller":
                return await self._process_caller_chunk(audio_bytes)
            elif source == "AI":
                return await self._process_ai_chunk(audio_bytes)
            
            return ""
            
        except Exception as e:
            Log.error(f"[{source}] Transcription error: {e}")
            return ""
    
    async def _process_caller_chunk(self, audio_bytes: bytes) -> str:
        """Process caller audio chunk."""
        import time
        current_time = time.time()
        
        # Initialize timing if first chunk
        if len(self._caller_buffer) == 0:
            self._caller_first_chunk_time = current_time
        
        # Add to buffer
        self._caller_buffer.extend(audio_bytes)
        self._caller_last_chunk_time = current_time
        
        # Calculate buffer duration (8kHz, 1 byte per sample)
        buffer_duration = len(self._caller_buffer) / 8000.0
        
        # Force send if buffer is too large
        if buffer_duration >= self.MAX_BUFFER_DURATION:
            return await self._flush_caller_buffer()
        
        # Cancel existing silence task
        if self._caller_silence_task:
            self._caller_silence_task.cancel()
        
        # Start new silence detection task
        self._caller_silence_task = asyncio.create_task(
            self._wait_for_caller_silence()
        )
        
        return ""
    
    async def _process_ai_chunk(self, audio_bytes: bytes) -> str:
        """Process AI audio chunk."""
        import time
        current_time = time.time()
        
        # Initialize timing if first chunk
        if len(self._ai_buffer) == 0:
            self._ai_first_chunk_time = current_time
        
        # Add to buffer
        self._ai_buffer.extend(audio_bytes)
        self._ai_last_chunk_time = current_time
        
        # Calculate buffer duration
        buffer_duration = len(self._ai_buffer) / 8000.0
        
        # Force send if buffer is too large
        if buffer_duration >= self.MAX_BUFFER_DURATION:
            return await self._flush_ai_buffer()
        
        # Cancel existing silence task
        if self._ai_silence_task:
            self._ai_silence_task.cancel()
        
        # Start new silence detection task
        self._ai_silence_task = asyncio.create_task(
            self._wait_for_ai_silence()
        )
        
        return ""
    
    async def _wait_for_caller_silence(self):
        """Wait for silence period, then transcribe caller buffer."""
        try:
            await asyncio.sleep(self.SILENCE_TIMEOUT)
            
            # Check if we have enough audio
            buffer_duration = len(self._caller_buffer) / 8000.0
            if buffer_duration >= self.MIN_AUDIO_DURATION:
                return await self._flush_caller_buffer()
        except asyncio.CancelledError:
            pass
        return ""
    
    async def _wait_for_ai_silence(self):
        """Wait for silence period, then transcribe AI buffer."""
        try:
            await asyncio.sleep(self.SILENCE_TIMEOUT)
            
            # Check if we have enough audio
            buffer_duration = len(self._ai_buffer) / 8000.0
            if buffer_duration >= self.MIN_AUDIO_DURATION:
                return await self._flush_ai_buffer()
        except asyncio.CancelledError:
            pass
        return ""
    
    async def _flush_caller_buffer(self) -> str:
        """Transcribe and clear caller buffer."""
        if self._caller_processing or len(self._caller_buffer) == 0:
            return ""
        
        self._caller_processing = True
        
        try:
            # Copy buffer
            audio_data = bytes(self._caller_buffer)
            self._caller_buffer.clear()
            
            # Transcribe
            result = await self._transcribe_audio(audio_data, "Caller")
            return result
            
        finally:
            self._caller_processing = False
    
    async def _flush_ai_buffer(self) -> str:
        """Transcribe and clear AI buffer."""
        if self._ai_processing or len(self._ai_buffer) == 0:
            return ""
        
        self._ai_processing = True
        
        try:
            # Copy buffer
            audio_data = bytes(self._ai_buffer)
            self._ai_buffer.clear()
            
            # Transcribe
            result = await self._transcribe_audio(audio_data, "AI")
            return result
            
        finally:
            self._ai_processing = False
    
    async def _transcribe_audio(self, mulaw_bytes: bytes, source: str) -> str:
        """Convert accumulated µ-law audio to PCM16 WAV and transcribe."""
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
            
            # Send to Whisper API
            headers = {"Authorization": f"Bearer {Config.OPENAI_API_KEY}"}
            form = aiohttp.FormData()
            form.add_field("file", wav_io, filename="audio.wav", content_type="audio/wav")
            form.add_field("model", "gpt-4o-mini-transcribe")
            form.add_field("language", "en")
            
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
