import io
import wave
import base64
import asyncio
import aiohttp
import numpy as np
import time
from scipy.signal import resample
from typing import Dict, Optional, List, Callable
from collections import deque
from config import Config
from services.log_utils import Log


class TranscriptionService:
    """
    Real-time transcription service with strict Voice Activity Detection.
    
    Key Features:
    - STRICT VAD prevents Whisper hallucinations on silence
    - Sequential audio playback (no overlap)
    - 0.3s gap ONLY for Caller->AI transitions
    - Aggressive silence removal for Caller chunks
    - NO gaps for AI->Caller (natural flow)
    """
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    
    # Audio format specs
    SAMPLE_RATE = 8000  # µ-law 8kHz from Twilio and OpenAI
    CHUNK_DURATION_MS = 20
    BYTES_PER_20MS = 160
    
    # Speaker turn detection
    SPEAKER_SILENCE_THRESHOLD = 0.3  # If no chunks for 0.3s, speaker is done
    SPEAKER_TRANSITION_DELAY = 0.3   # Reduced to 0.3s gap for Caller->AI
    
    # STRICTER Voice Activity Detection to prevent hallucinations
    VAD_ENERGY_THRESHOLD = 2500      # ✅ INCREASED (was 1000)
    VAD_ZCR_THRESHOLD = 0.15         # ✅ INCREASED (was 0.1)
    VAD_CONSECUTIVE_SPEECH = 5       # ✅ INCREASED (was 3) - need more proof
    VAD_CONSECUTIVE_SILENCE = 8      # ✅ INCREASED (was 5) - stop faster
    VAD_LOOKBACK_CHUNKS = 5          # ✅ REDUCED (was 10) - less history noise
    
    # Transcription settings
    MIN_AUDIO_DURATION = 1.2         # ✅ INCREASED (was 0.8) - more speech needed
    SILENCE_TIMEOUT = 0.8            # ✅ INCREASED (was 0.5)
    MAX_BUFFER_DURATION = 3.0
    
    # ✅ NEW: Silence chunk detection
    SILENCE_ENERGY_THRESHOLD = 800   # Below this = pure silence
    MAX_CONSECUTIVE_SILENCE_CHUNKS = 15  # Drop after 15 silent chunks (0.3s)
    
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
        
        # 🔒 Sequential playback lock
        self._playback_lock: asyncio.Lock = asyncio.Lock()
        
        # VAD state for caller
        self._caller_is_speaking: bool = False
        self._caller_speech_chunks_count: int = 0
        self._caller_silence_chunks_count: int = 0
        self._caller_chunk_history: deque = deque(maxlen=self.VAD_LOOKBACK_CHUNKS)
        
        # ✅ NEW: Track consecutive silence for dropping
        self._caller_consecutive_silence: int = 0
        
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
        self._caller_transcript_history: deque = deque(maxlen=5)
        self._ai_transcript_history: deque = deque(maxlen=5)
        
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
    
    def _detect_speech(self, mulaw_bytes: bytes) -> bool:
        """
        STRICT Voice Activity Detection to prevent Whisper hallucinations.
        
        Returns True ONLY if chunk has CLEAR speech characteristics.
        """
        try:
            pcm16 = self._mulaw_to_pcm16(mulaw_bytes)
            
            # Energy calculation
            energy = np.sum(pcm16.astype(np.float64) ** 2)
            
            # Zero-crossing rate (voice frequency indicator)
            zcr = np.sum(np.abs(np.diff(np.sign(pcm16)))) / (2 * len(pcm16))
            
            # ✅ STRICTER thresholds - both must pass
            has_high_energy = energy > self.VAD_ENERGY_THRESHOLD
            has_voice_frequency = zcr > self.VAD_ZCR_THRESHOLD
            
            has_speech = has_high_energy and has_voice_frequency
            
            return has_speech
            
        except Exception as e:
            Log.debug(f"[VAD] Error: {e}")
            return False
    
    def _is_pure_silence(self, mulaw_bytes: bytes) -> bool:
        """
        Detect if chunk is pure silence (background noise/listening).
        Used to DROP chunks during AI speaking.
        """
        try:
            pcm16 = self._mulaw_to_pcm16(mulaw_bytes)
            energy = np.sum(pcm16.astype(np.float64) ** 2)
            return energy < self.SILENCE_ENERGY_THRESHOLD
        except Exception:
            return False
    
    def _update_vad_state(self, has_speech: bool, audio_bytes: bytes) -> bool:
        """
        Update VAD state machine with STRICT requirements.
        
        Returns True if chunk should be streamed to dashboard.
        """
        # Store chunk in history
        self._caller_chunk_history.append(audio_bytes)
        
        if has_speech:
            self._caller_speech_chunks_count += 1
            self._caller_silence_chunks_count = 0
            self._caller_consecutive_silence = 0  # ✅ Reset silence counter
            
            # ✅ Need MORE consecutive speech to start (prevents noise triggering)
            if not self._caller_is_speaking and self._caller_speech_chunks_count >= self.VAD_CONSECUTIVE_SPEECH:
                self._caller_is_speaking = True
                Log.info("[VAD] 🎤 Caller started speaking (strict)")
                return True
        else:
            self._caller_silence_chunks_count += 1
            self._caller_speech_chunks_count = 0
            self._caller_consecutive_silence += 1  # ✅ Track consecutive silence
            
            # ✅ Stop speaking faster on silence
            if self._caller_is_speaking and self._caller_silence_chunks_count >= self.VAD_CONSECUTIVE_SILENCE:
                self._caller_is_speaking = False
                Log.info("[VAD] 🔇 Caller stopped speaking")
        
        # ✅ DROP chunk if pure silence for too long (prevents queue backlog)
        if self._caller_consecutive_silence > self.MAX_CONSECUTIVE_SILENCE_CHUNKS:
            return False  # Don't stream, don't buffer
        
        # Stream if currently in speaking state
        return self._caller_is_speaking
    
    async def _stream_unified_audio(self):
        """
        Sequential audio streaming with smart speaker transitions.
        
        🔒 Guarantees:
        - ONE chunk plays at a time
        - 0.3s gap for Caller->AI only
        - NO gap for AI->Caller
        """
        Log.info("[Stream] Started - SEQUENTIAL with 0.3s Caller→AI gap")
        
        while not self._shutdown:
            try:
                audio_data = await self._unified_audio_queue.get()
                
                if audio_data is None:
                    break
                
                speaker = audio_data.get("speaker")
                current_time = time.time()
                
                # 🔒 LOCK: Sequential playback
                async with self._playback_lock:
                    
                    audio_b64 = audio_data.get("audio", "")
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        chunk_duration = self._calculate_chunk_duration(audio_bytes)
                    except Exception as e:
                        Log.debug(f"[Stream] Duration calc error: {e}")
                        chunk_duration = 0.02
                    
                    speaker_changed = (
                        self._last_streamed_speaker is not None and 
                        self._last_streamed_speaker != speaker
                    )
                    
                    if speaker_changed:
                        previous_speaker = self._last_streamed_speaker
                        previous_last_time = self._last_chunk_time_per_speaker.get(previous_speaker, 0)
                        time_gap = current_time - previous_last_time if previous_last_time > 0 else 0
                        previous_finished = time_gap >= self.SPEAKER_SILENCE_THRESHOLD
                        
                        # ✅ ONLY add 0.3s gap for Caller → AI (reduced from 0.5s)
                        if previous_speaker == "Caller" and speaker == "AI" and previous_finished:
                            if time_gap < self.SPEAKER_TRANSITION_DELAY:
                                remaining_gap = self.SPEAKER_TRANSITION_DELAY - time_gap
                                Log.debug(f"[Stream] Caller → AI: +{remaining_gap:.3f}s gap")
                                await asyncio.sleep(remaining_gap)
                        
                        # ✅ AI → Caller: NO gap
                        elif previous_speaker == "AI" and speaker == "Caller":
                            Log.debug(f"[Stream] AI → Caller: INSTANT")
                    
                    self._last_chunk_time_per_speaker[speaker] = current_time
                    self._last_streamed_speaker = speaker
                    
                    # Send to dashboard
                    if self.audio_callback:
                        try:
                            await self.audio_callback(audio_data)
                        except Exception as e:
                            Log.error(f"[Stream] callback error: {e}")
                    
                    # ⏱️ Wait for chunk playback
                    await asyncio.sleep(chunk_duration)
                
                self._unified_audio_queue.task_done()
                
            except Exception as e:
                Log.error(f"[Stream] error: {e}")
                await asyncio.sleep(0.01)
    
    async def transcribe_realtime(self, audio_input, source: str = "Unknown") -> str:
        """
        Process incoming audio with STRICT VAD and silence removal.
        
        🎯 KEY FIXES:
        1. Stricter VAD prevents Whisper hallucinations
        2. Pure silence chunks are DROPPED (prevents queue backlog)
        3. Only actual speech gets transcribed
        """
        try:
            if isinstance(audio_input, str):
                audio_bytes = base64.b64decode(audio_input)
                original_base64 = audio_input
            elif isinstance(audio_input, (bytes, bytearray)):
                audio_bytes = bytes(audio_input)
                original_base64 = base64.b64encode(audio_bytes).decode('ascii')
            else:
                return ""
            
            should_stream = True
            should_buffer = True  # ✅ NEW: Separate flag for buffering
            
            # 🎯 STRICT VAD + SILENCE DETECTION for caller
            if source == "Caller":
                # Check if pure silence (should be dropped entirely)
                is_silence = self._is_pure_silence(audio_bytes)
                
                if is_silence:
                    # ✅ DROP pure silence chunks (don't stream, don't buffer)
                    self._caller_consecutive_silence += 1
                    if self._caller_consecutive_silence > self.MAX_CONSECUTIVE_SILENCE_CHUNKS:
                        return ""  # Complete drop
                else:
                    self._caller_consecutive_silence = 0
                
                # VAD check for speech
                has_speech = self._detect_speech(audio_bytes)
                should_stream = self._update_vad_state(has_speech, audio_bytes)
                
                # Only buffer if NOT pure silence
                should_buffer = not is_silence
                
                # Buffer for transcription (only if has some content)
                if should_buffer:
                    await self._add_to_caller_buffer(audio_bytes)
                
                # Flush history when starting to speak
                if should_stream and self._caller_speech_chunks_count == self.VAD_CONSECUTIVE_SPEECH:
                    for hist_chunk in list(self._caller_chunk_history)[:-1]:
                        hist_b64 = base64.b64encode(hist_chunk).decode('ascii')
                        hist_packet = {
                            "speaker": source,
                            "audio": hist_b64,
                            "timestamp": int(time.time() * 1000),  # ✅ Milliseconds
                            "size": len(hist_chunk)
                        }
                        await self._unified_audio_queue.put(hist_packet)
            
            # Queue for streaming
            if should_stream:
                audio_packet = {
                    "speaker": source,
                    "audio": original_base64,
                    "timestamp": int(time.time() * 1000),  # ✅ Milliseconds
                    "size": len(audio_bytes)
                }
                await self._unified_audio_queue.put(audio_packet)
            
            # Buffer AI audio (always)
            if source == "AI":
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
        """Transcribe accumulated buffer with deduplication."""
        if source == "Caller":
            if self._caller_processing or len(self._caller_buffer) == 0:
                return
            self._caller_processing = True
            
            try:
                audio_data = bytes(self._caller_buffer)
                self._caller_buffer.clear()
                
                transcript = await self._transcribe_audio(audio_data, source)
                
                # ✅ STRICT deduplication
                if transcript and not self._is_duplicate(transcript, source):
                    self._caller_last_transcript = transcript
                    self._caller_transcript_history.append(transcript.lower())
                    
                    if self.transcription_callback:
                        await self.transcription_callback({
                            "speaker": source,
                            "text": transcript,
                            "timestamp": int(time.time() * 1000)  # ✅ Milliseconds
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
                
                # ✅ STRICT deduplication
                if transcript and not self._is_duplicate(transcript, source):
                    self._ai_last_transcript = transcript
                    self._ai_transcript_history.append(transcript.lower())
                    
                    if self.transcription_callback:
                        await self.transcription_callback({
                            "speaker": source,
                            "text": transcript,
                            "timestamp": int(time.time() * 1000)  # ✅ Milliseconds
                        })
                
            finally:
                self._ai_processing = False
    
    def _is_duplicate(self, transcript: str, source: str) -> bool:
        """Strict duplicate detection."""
        transcript_lower = transcript.lower().strip()
        
        # Ignore very short transcripts (likely noise)
        if len(transcript_lower) < 3:
            return True
        
        if source == "Caller":
            if transcript_lower == self._caller_last_transcript.lower():
                return True
            for prev in self._caller_transcript_history:
                if self._similarity(transcript_lower, prev) > 0.85:
                    return True
        else:
            if transcript_lower == self._ai_last_transcript.lower():
                return True
            for prev in self._ai_transcript_history:
                if self._similarity(transcript_lower, prev) > 0.85:
                    return True
        
        return False
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio."""
        if not s1 or not s2:
            return 0.0
        
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _transcribe_audio(self, mulaw_bytes: bytes, source: str) -> str:
        """
        Convert µ-law 8kHz to PCM16 16kHz and transcribe with Whisper.
        Strict prompt to prevent hallucinations.
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
            form.add_field("model", "whisper-1")
            form.add_field("response_format", "verbose_json")
            # ✅ STRICT prompt to prevent hallucinations
            form.add_field("prompt", 
                "Transcribe ONLY clear speech. "
                "If no speech detected, return empty. "
                "Use English script: pizza, biryani, delivery, order."
            )
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.OPENAI_API_URL, headers=headers, data=form) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        Log.error(f"[{source}] Transcription failed ({resp.status}): {text}")
                        return ""
                    
                    data = await resp.json()
                    transcript = (data.get("text") or "").strip()
                    detected_lang = data.get("language", "unknown")
                    
                    # ✅ Filter out common hallucination patterns
                    hallucination_patterns = [
                        "thank you", "thanks for watching", "subscribe",
                        "you", ".", "...", "मुझे", "आप"
                    ]
                    
                    transcript_lower = transcript.lower()
                    if any(pattern in transcript_lower and len(transcript) < 15 
                           for pattern in hallucination_patterns):
                        Log.debug(f"[{source}] Filtered hallucination: {transcript}")
                        return ""
                    
                    if transcript:
                        Log.info(f"[{source}] 📝 [{detected_lang}] {transcript}")
                        return transcript
                    
                    return ""
                    
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
