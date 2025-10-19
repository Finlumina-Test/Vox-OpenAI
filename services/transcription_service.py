import io
import wave
import base64
import asyncio
import aiohttp
import numpy as np
import time    
from scipy.signal import resample, butter, filtfilt, wiener
from scipy.ndimage import median_filter
from typing import Dict, Optional, List, Callable
from collections import deque
from config import Config
from services.log_utils import Log


class TranscriptionService:
    """
    Real-time transcription service with 4K audio enhancement.
    
    Key Features:
    - 24kHz PCM16 output (3x higher quality than 8kHz)
    - Noise reduction & spectral enhancement
    - Voice Activity Detection (VAD)
    - Sequential audio playback
    - 0.5s gap ONLY for Caller->AI transitions
    - NO gaps for AI->Caller
    """
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    
    # Audio format specs
    INPUT_SAMPLE_RATE = 8000    # µ-law 8kHz from Twilio/OpenAI
    OUTPUT_SAMPLE_RATE = 24000  # 🎵 24kHz for dashboard (3x quality boost!)
    CHUNK_DURATION_MS = 20
    
    # Speaker turn detection
    SPEAKER_SILENCE_THRESHOLD = 0.3
    SPEAKER_TRANSITION_DELAY = 1.0   # 1 second gap for Caller→AI
    
    # Voice Activity Detection
    VAD_ENERGY_THRESHOLD = 1000
    VAD_ZCR_THRESHOLD = 0.1
    VAD_CONSECUTIVE_SPEECH = 3
    VAD_CONSECUTIVE_SILENCE = 5
    VAD_LOOKBACK_CHUNKS = 10
    
    # Audio enhancement settings (START DISABLED for safety)
    ENABLE_ENHANCEMENT = False       # 🔧 Set to True to enable enhancement
    NOISE_REDUCTION_STRENGTH = 0.3  # 0.0-1.0 (lower = safer)
    SPECTRAL_ENHANCEMENT = False    # Disable for now
    DYNAMIC_RANGE_COMPRESSION = False # Disable for now
    
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
        
        # Sequential playback lock
        self._playback_lock: asyncio.Lock = asyncio.Lock()
        
        # VAD state
        self._caller_is_speaking: bool = False
        self._caller_speech_chunks_count: int = 0
        self._caller_silence_chunks_count: int = 0
        self._caller_chunk_history: deque = deque(maxlen=self.VAD_LOOKBACK_CHUNKS)
        
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
        
        # Noise profile estimation (for adaptive noise reduction)
        self._noise_profile: Optional[np.ndarray] = None
        self._noise_samples_count: int = 0
        
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
            sample_rate = self.OUTPUT_SAMPLE_RATE
        # For PCM16, each sample is 2 bytes
        num_samples = len(audio_bytes) // 2
        duration_seconds = num_samples / sample_rate
        return duration_seconds
    
    # ==================== AUDIO ENHANCEMENT ====================
    
    def _enhance_audio(self, pcm16_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        🎵 4K Audio Enhancement Pipeline
        
        Steps:
        1. Noise reduction (spectral subtraction + Wiener filtering)
        2. Bandpass filter (preserve voice frequencies)
        3. Spectral enhancement (clarity boost)
        4. Dynamic range compression (volume normalization)
        """
        try:
            # Convert to float for processing
            audio_float = pcm16_data.astype(np.float32) / 32768.0
            
            # Step 1: Noise Reduction
            audio_float = self._reduce_noise(audio_float, sample_rate)
            
            # Step 2: Bandpass Filter (300Hz - 3400Hz for voice)
            audio_float = self._bandpass_filter(audio_float, sample_rate)
            
            # Step 3: Spectral Enhancement
            if self.SPECTRAL_ENHANCEMENT:
                audio_float = self._spectral_enhancement(audio_float, sample_rate)
            
            # Step 4: Dynamic Range Compression
            if self.DYNAMIC_RANGE_COMPRESSION:
                audio_float = self._compress_dynamic_range(audio_float)
            
            # Convert back to PCM16
            audio_float = np.clip(audio_float, -1.0, 1.0)
            enhanced_pcm16 = (audio_float * 32767).astype(np.int16)
            
            return enhanced_pcm16
            
        except Exception as e:
            Log.debug(f"[Enhancement] Error: {e}, returning original")
            return pcm16_data
    
    def _reduce_noise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Adaptive noise reduction using spectral subtraction.
        """
        try:
            # Estimate noise profile from first few frames if not done
            if self._noise_profile is None and self._noise_samples_count < 5:
                self._update_noise_profile(audio)
                return audio  # Return original for first few chunks
            
            if self._noise_profile is None:
                return audio
            
            # Apply spectral subtraction
            fft_audio = np.fft.rfft(audio)
            magnitude = np.abs(fft_audio)
            phase = np.angle(fft_audio)
            
            # Subtract noise profile
            clean_magnitude = magnitude - (self._noise_profile * self.NOISE_REDUCTION_STRENGTH)
            clean_magnitude = np.maximum(clean_magnitude, magnitude * 0.1)  # Floor to prevent artifacts
            
            # Reconstruct signal
            clean_fft = clean_magnitude * np.exp(1j * phase)
            clean_audio = np.fft.irfft(clean_fft, n=len(audio))
            
            return clean_audio
            
        except Exception as e:
            Log.debug(f"[Noise Reduction] Error: {e}")
            return audio
    
    def _update_noise_profile(self, audio: np.ndarray):
        """Update noise profile from silent/background audio."""
        try:
            fft_audio = np.fft.rfft(audio)
            magnitude = np.abs(fft_audio)
            
            if self._noise_profile is None:
                self._noise_profile = magnitude
            else:
                # Running average
                alpha = 0.95
                self._noise_profile = alpha * self._noise_profile + (1 - alpha) * magnitude
            
            self._noise_samples_count += 1
            
        except Exception as e:
            Log.debug(f"[Noise Profile] Error: {e}")
    
    def _bandpass_filter(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Bandpass filter to isolate human voice frequencies (300Hz - 3400Hz).
        """
        try:
            nyquist = sample_rate / 2
            low_freq = 300 / nyquist
            high_freq = 3400 / nyquist
            
            # Butterworth bandpass filter
            b, a = butter(4, [low_freq, high_freq], btype='band')
            filtered = filtfilt(b, a, audio)
            
            return filtered
            
        except Exception as e:
            Log.debug(f"[Bandpass] Error: {e}")
            return audio
    
    def _spectral_enhancement(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Enhance speech clarity by boosting mid-high frequencies (1kHz-3kHz).
        """
        try:
            # Apply gentle high-shelf filter to enhance clarity
            fft_audio = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
            
            # Boost 1kHz-3kHz range (speech clarity)
            boost_mask = (freqs >= 1000) & (freqs <= 3000)
            boost_factor = 1.3  # 30% boost
            
            fft_audio[boost_mask] *= boost_factor
            
            enhanced = np.fft.irfft(fft_audio, n=len(audio))
            return enhanced
            
        except Exception as e:
            Log.debug(f"[Spectral Enhancement] Error: {e}")
            return audio
    
    def _compress_dynamic_range(self, audio: np.ndarray) -> np.ndarray:
        """
        Soft compression to normalize volume levels.
        """
        try:
            # Calculate RMS and normalize
            rms = np.sqrt(np.mean(audio ** 2))
            
            if rms < 0.01:  # Too quiet
                return audio
            
            target_rms = 0.15
            gain = target_rms / rms
            
            # Limit gain to prevent over-amplification
            gain = min(gain, 3.0)
            
            compressed = audio * gain
            
            # Soft clipping
            compressed = np.tanh(compressed * 1.2) * 0.9
            
            return compressed
            
        except Exception as e:
            Log.debug(f"[Compression] Error: {e}")
            return audio
    
    # ==================== VAD ====================
    
    def _detect_speech(self, mulaw_bytes: bytes) -> bool:
        """Voice Activity Detection using energy and zero-crossing rate."""
        try:
            pcm16 = self._mulaw_to_pcm16(mulaw_bytes)
            
            energy = np.sum(pcm16.astype(np.float64) ** 2)
            zcr = np.sum(np.abs(np.diff(np.sign(pcm16)))) / (2 * len(pcm16))
            
            has_speech = (energy > self.VAD_ENERGY_THRESHOLD and 
                         zcr > self.VAD_ZCR_THRESHOLD)
            
            return has_speech
            
        except Exception as e:
            Log.debug(f"[VAD] Error: {e}")
            return False
    
    def _update_vad_state(self, has_speech: bool, audio_bytes: bytes) -> bool:
        """Update VAD state machine."""
        self._caller_chunk_history.append(audio_bytes)
        
        if has_speech:
            self._caller_speech_chunks_count += 1
            self._caller_silence_chunks_count = 0
            
            if not self._caller_is_speaking and self._caller_speech_chunks_count >= self.VAD_CONSECUTIVE_SPEECH:
                self._caller_is_speaking = True
                Log.info("[VAD] 🎤 Caller started speaking")
                return True
        else:
            self._caller_silence_chunks_count += 1
            self._caller_speech_chunks_count = 0
            
            if self._caller_is_speaking and self._caller_silence_chunks_count >= self.VAD_CONSECUTIVE_SILENCE:
                self._caller_is_speaking = False
                Log.info("[VAD] 🔇 Caller stopped speaking")
        
        return self._caller_is_speaking
    
    # ==================== STREAMING ====================
    
    async def _stream_unified_audio(self):
        """Sequential audio streaming with transitions."""
        Log.info("[Stream] Started - 24kHz PCM16 enhanced audio")
        
        while not self._shutdown:
            try:
                audio_data = await self._unified_audio_queue.get()
                
                if audio_data is None:
                    break
                
                speaker = audio_data.get("speaker")
                current_time = time.time()
                
                async with self._playback_lock:
                    
                    audio_b64 = audio_data.get("audio", "")
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        chunk_duration = self._calculate_chunk_duration(audio_bytes, self.OUTPUT_SAMPLE_RATE)
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
                        
                        if previous_speaker == "Caller" and speaker == "AI" and previous_finished:
                            if time_gap < self.SPEAKER_TRANSITION_DELAY:
                                remaining_gap = self.SPEAKER_TRANSITION_DELAY - time_gap
                                Log.debug(f"[Stream] Caller → AI: +{remaining_gap:.3f}s gap")
                                await asyncio.sleep(remaining_gap)
                        
                        elif previous_speaker == "AI" and speaker == "Caller":
                            Log.debug(f"[Stream] AI → Caller: NO GAP")
                    
                    self._last_chunk_time_per_speaker[speaker] = current_time
                    self._last_streamed_speaker = speaker
                    
                    if self.audio_callback:
                        try:
                            await self.audio_callback(audio_data)
                        except Exception as e:
                            Log.error(f"[Stream] callback error: {e}")
                    
                    await asyncio.sleep(chunk_duration)
                
                self._unified_audio_queue.task_done()
                
            except Exception as e:
                Log.error(f"[Stream] error: {e}")
                await asyncio.sleep(0.01)
    
    async def transcribe_realtime(self, audio_input, source: str = "Unknown") -> str:
        """
        Process incoming audio with VAD and enhancement.
        """
        try:
            # Convert to bytes
            if isinstance(audio_input, str):
                audio_bytes = base64.b64decode(audio_input)
            elif isinstance(audio_input, (bytes, bytearray)):
                audio_bytes = bytes(audio_input)
            else:
                return ""
            
            should_stream = True
            
            # Apply VAD to caller audio
            if source == "Caller":
                has_speech = self._detect_speech(audio_bytes)
                should_stream = self._update_vad_state(has_speech, audio_bytes)
                
                await self._add_to_caller_buffer(audio_bytes)
                
                # Flush history when starting to speak
                if should_stream and self._caller_speech_chunks_count == self.VAD_CONSECUTIVE_SPEECH:
                    for hist_chunk in list(self._caller_chunk_history)[:-1]:
                        await self._queue_enhanced_audio(hist_chunk, source)
            
            # Queue enhanced audio
            if should_stream:
                await self._queue_enhanced_audio(audio_bytes, source)
            
            # Buffer AI audio
            if source == "AI":
                await self._add_to_ai_buffer(audio_bytes)
            
            return ""
            
        except Exception as e:
            Log.error(f"[{source}] Error: {e}")
            return ""
    
    async def _queue_enhanced_audio(self, mulaw_bytes: bytes, source: str):
        """
        🎵 Convert µ-law 8kHz → PCM16 24kHz (with optional enhancement)
        """
        try:
            # Step 1: µ-law → PCM16 8kHz
            pcm16_8k = self._mulaw_to_pcm16(mulaw_bytes)
            
            # Step 2: Upsample to 24kHz (3x quality)
            pcm16_24k = self._resample_pcm16(pcm16_8k, self.INPUT_SAMPLE_RATE, self.OUTPUT_SAMPLE_RATE)
            
            # Step 3: Apply audio enhancement (ONLY if enabled)
            if self.ENABLE_ENHANCEMENT:
                enhanced_pcm16 = self._enhance_audio(pcm16_24k, self.OUTPUT_SAMPLE_RATE)
            else:
                enhanced_pcm16 = pcm16_24k  # 🔧 Raw upsampled audio
            
            # Step 4: Encode to base64
            enhanced_base64 = base64.b64encode(enhanced_pcm16.tobytes()).decode('ascii')
            
            audio_packet = {
                "speaker": source,
                "audio": enhanced_base64,
                "timestamp": int(time.time()),
                "size": len(enhanced_pcm16.tobytes()),
                "format": "pcm16",        # 🎵 High quality PCM16
                "sample_rate": 24000,     # 🎵 24kHz
                "encoding": "pcm16"
            }
            
            await self._unified_audio_queue.put(audio_packet)
            
        except Exception as e:
            Log.error(f"[Enhancement] Error for {source}: {e}")
    
    # ==================== TRANSCRIPTION BUFFERS ====================
    
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
        """Convert µ-law 8kHz to PCM16 16kHz and transcribe."""
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
