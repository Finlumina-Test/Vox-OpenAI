import io
import wave
import base64
import asyncio
import aiohttp
import numpy as np
from scipy.signal import resample
from typing import Dict, Optional, List, Tuple
from config import Config
from services.log_utils import Log


class WordBoundaryDetector:
    """Detects word boundaries in transcribed text stream."""
    
    def __init__(self):
        self.current_transcript = ""
        self.last_words = []
        
    def detect_new_words(self, new_transcript: str) -> List[str]:
        """
        Detect newly completed words by comparing transcripts.
        Returns list of new complete words.
        """
        if not new_transcript:
            return []
            
        # Clean and split into words
        new_words = new_transcript.strip().split()
        
        # Find new words by comparing with last known state
        if len(new_words) > len(self.last_words):
            # New words added
            new_complete = new_words[len(self.last_words):]
            self.last_words = new_words
            return new_complete
        elif new_words != self.last_words:
            # Transcript changed (possibly correction)
            self.last_words = new_words
            return new_words[-1:] if new_words else []
            
        return []
    
    def reset(self):
        """Reset detector state."""
        self.current_transcript = ""
        self.last_words = []


class TranscriptionService:
    """
    Accumulates audio chunks and transcribes with word-level granularity.
    Sends both audio and text for each word to maintain voice fidelity.
    """
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    
    # Accumulation settings for word detection
    MIN_AUDIO_DURATION = 0.3  # Minimum audio for a word (300ms)
    SILENCE_TIMEOUT = 0.2  # Quick timeout for word boundaries (200ms)
    MAX_BUFFER_DURATION = 2.0  # Force transcribe after 2 seconds
    WORD_CHECK_INTERVAL = 0.1  # Check for words every 100ms
    
    def __init__(self):
        # Separate buffers and detectors for each speaker
        self._caller_buffer: bytearray = bytearray()
        self._ai_buffer: bytearray = bytearray()
        
        self._caller_audio_accumulator: List[bytes] = []  # Store original audio chunks
        self._ai_audio_accumulator: List[bytes] = []
        
        self._caller_word_detector = WordBoundaryDetector()
        self._ai_word_detector = WordBoundaryDetector()
        
        # Timing tracking
        self._caller_last_chunk_time: float = 0
        self._ai_last_chunk_time: float = 0
        
        self._caller_first_chunk_time: float = 0
        self._ai_first_chunk_time: float = 0
        
        # Processing flags
        self._caller_processing: bool = False
        self._ai_processing: bool = False
        
        # Word detection tasks
        self._caller_word_task: Optional[asyncio.Task] = None
        self._ai_word_task: Optional[asyncio.Task] = None
        
        # Callback for word-level updates
        self.word_callback = None
    
    def set_word_callback(self, callback):
        """Set callback function for word-level updates."""
        self.word_callback = callback
    
    async def transcribe_realtime(self, audio_input, source: str = "Unknown") -> str:
        """
        Accumulates audio chunks and detects word boundaries.
        Returns full transcript for backward compatibility.
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
            
            # Process based on speaker
            if source == "Caller":
                return await self._process_caller_chunk(audio_bytes, original_base64)
            elif source == "AI":
                return await self._process_ai_chunk(audio_bytes, original_base64)
            
            return ""
            
        except Exception as e:
            Log.error(f"[{source}] Transcription error: {e}")
            return ""
    
    async def _process_caller_chunk(self, audio_bytes: bytes, original_base64: str) -> str:
        """Process caller audio chunk with word detection."""
        import time
        current_time = time.time()
        
        # Initialize timing if first chunk
        if len(self._caller_buffer) == 0:
            self._caller_first_chunk_time = current_time
        
        # Add to buffers
        self._caller_buffer.extend(audio_bytes)
        self._caller_audio_accumulator.append(original_base64)
        self._caller_last_chunk_time = current_time
        
        # Start word detection task if not running
        if not self._caller_word_task or self._caller_word_task.done():
            self._caller_word_task = asyncio.create_task(
                self._detect_caller_words()
            )
        
        return ""
    
    async def _process_ai_chunk(self, audio_bytes: bytes, original_base64: str) -> str:
        """Process AI audio chunk with word detection."""
        import time
        current_time = time.time()
        
        # Initialize timing if first chunk
        if len(self._ai_buffer) == 0:
            self._ai_first_chunk_time = current_time
        
        # Add to buffers
        self._ai_buffer.extend(audio_bytes)
        self._ai_audio_accumulator.append(original_base64)
        self._ai_last_chunk_time = current_time
        
        # Start word detection task if not running
        if not self._ai_word_task or self._ai_word_task.done():
            self._ai_word_task = asyncio.create_task(
                self._detect_ai_words()
            )
        
        return ""
    
    async def _detect_caller_words(self):
        """Continuously detect word boundaries for caller."""
        while True:
            try:
                await asyncio.sleep(self.WORD_CHECK_INTERVAL)
                
                # Check if we have enough audio
                buffer_duration = len(self._caller_buffer) / 8000.0
                if buffer_duration < self.MIN_AUDIO_DURATION:
                    continue
                
                # Check for silence or max duration
                import time
                time_since_last = time.time() - self._caller_last_chunk_time
                
                should_transcribe = (
                    time_since_last >= self.SILENCE_TIMEOUT or
                    buffer_duration >= self.MAX_BUFFER_DURATION
                )
                
                if should_transcribe and not self._caller_processing:
                    await self._transcribe_and_detect_words("Caller")
                    
            except Exception as e:
                Log.error(f"[Caller word detection] error: {e}")
                break
    
    async def _detect_ai_words(self):
        """Continuously detect word boundaries for AI."""
        while True:
            try:
                await asyncio.sleep(self.WORD_CHECK_INTERVAL)
                
                # Check if we have enough audio
                buffer_duration = len(self._ai_buffer) / 8000.0
                if buffer_duration < self.MIN_AUDIO_DURATION:
                    continue
                
                # Check for silence or max duration
                import time
                time_since_last = time.time() - self._ai_last_chunk_time
                
                should_transcribe = (
                    time_since_last >= self.SILENCE_TIMEOUT or
                    buffer_duration >= self.MAX_BUFFER_DURATION
                )
                
                if should_transcribe and not self._ai_processing:
                    await self._transcribe_and_detect_words("AI")
                    
            except Exception as e:
                Log.error(f"[AI word detection] error: {e}")
                break
    
    async def _transcribe_and_detect_words(self, source: str):
        """Transcribe buffer and detect new words."""
        if source == "Caller":
            if self._caller_processing or len(self._caller_buffer) == 0:
                return
            self._caller_processing = True
            
            try:
                # Copy buffers
                audio_data = bytes(self._caller_buffer)
                audio_chunks = self._caller_audio_accumulator.copy()
                
                # Clear buffers
                self._caller_buffer.clear()
                self._caller_audio_accumulator.clear()
                
                # Transcribe
                transcript = await self._transcribe_audio(audio_data, source)
                
                if transcript:
                    # Detect new words
                    new_words = self._caller_word_detector.detect_new_words(transcript)
                    
                    if new_words and self.word_callback:
                        # Combine audio chunks
                        combined_audio = self._combine_audio_chunks(audio_chunks)
                        
                        # Send word-level update
                        for word in new_words:
                            await self.word_callback({
                                "speaker": source,
                                "word": word,
                                "audio": combined_audio,
                                "timestamp": int(time.time())
                            })
                
            finally:
                self._caller_processing = False
                
        elif source == "AI":
            if self._ai_processing or len(self._ai_buffer) == 0:
                return
            self._ai_processing = True
            
            try:
                # Copy buffers
                audio_data = bytes(self._ai_buffer)
                audio_chunks = self._ai_audio_accumulator.copy()
                
                # Clear buffers
                self._ai_buffer.clear()
                self._ai_audio_accumulator.clear()
                
                # Transcribe
                transcript = await self._transcribe_audio(audio_data, source)
                
                if transcript:
                    # Detect new words
                    new_words = self._ai_word_detector.detect_new_words(transcript)
                    
                    if new_words and self.word_callback:
                        # Combine audio chunks
                        combined_audio = self._combine_audio_chunks(audio_chunks)
                        
                        # Send word-level update
                        for word in new_words:
                            await self.word_callback({
                                "speaker": source,
                                "word": word,
                                "audio": combined_audio,
                                "timestamp": int(time.time())
                            })
                
            finally:
                self._ai_processing = False
    
    def _combine_audio_chunks(self, chunks: List[str]) -> str:
        """Combine multiple base64 audio chunks into one."""
        if not chunks:
            return ""
        
        try:
            # Decode all chunks
            combined_bytes = bytearray()
            for chunk in chunks:
                combined_bytes.extend(base64.b64decode(chunk))
            
            # Re-encode as single base64
            return base64.b64encode(combined_bytes).decode('ascii')
        except Exception as e:
            Log.error(f"Failed to combine audio chunks: {e}")
            return chunks[0] if chunks else ""
    
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
