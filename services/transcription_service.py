import io
import wave
import base64
import aiohttp
import numpy as np
from scipy.signal import resample
from config import Config
from services.log_utils import Log


class TranscriptionService:
    """
    Converts Twilio µ-law (PCMU, 8 kHz) audio chunks → 16-bit PCM WAV → OpenAI transcription.
    Supports both base64 strings and raw bytes input.
    """
    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
    
    async def transcribe_realtime(self, audio_input, source: str = "Unknown") -> str:
        """
        Takes audio (base64 string or raw bytes), converts and sends to OpenAI.
        
        Args:
            audio_input: Either base64-encoded string or raw bytes
            source: Label for logging (e.g., "Caller" or "AI")
        """
        try:
            # 1️⃣ Handle both base64 and raw bytes input
            if isinstance(audio_input, str):
                mulaw_bytes = base64.b64decode(audio_input)
            elif isinstance(audio_input, (bytes, bytearray)):
                mulaw_bytes = bytes(audio_input)
            else:
                Log.error(f"Invalid audio input type: {type(audio_input)}")
                return ""
            
            # Skip if chunk is too small (avoid API errors)
            if len(mulaw_bytes) < 100:
                return ""
            
            # 2️⃣ Convert µ-law → PCM16
            pcm16 = self._mulaw_to_pcm16(mulaw_bytes)
            
            # 3️⃣ Resample from 8 kHz → 16 kHz
            pcm16_16k = self._resample_pcm16(pcm16, 8000, 16000)
            
            # 4️⃣ Write to in-memory WAV
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm16_16k.tobytes())
            wav_io.seek(0)
            
            # 5️⃣ Send to OpenAI Whisper
            headers = {"Authorization": f"Bearer {Config.OPENAI_API_KEY}"}
            form = aiohttp.FormData()
            form.add_field("file", wav_io, filename="chunk.wav", content_type="audio/wav")
            form.add_field("model", "gpt-4o-mini-transcribe")
            form.add_field("language", "en")  # Optional: specify language for better accuracy
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.OPENAI_API_URL, headers=headers, data=form) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        Log.error(f"[{source}] Transcription failed ({resp.status}): {text}")
                        return ""
                    
                    data = await resp.json()
                    transcript = (data.get("text") or "").strip()
                    
                    if transcript:
                        Log.info(f"[{source}] Transcribed: {transcript}")
                    
                    return transcript
                    
        except Exception as e:
            Log.error(f"[{source}] Transcription error: {e}")
            return ""
    
    # --- Internal conversion helpers ---
    def _mulaw_to_pcm16(self, mulaw_bytes: bytes) -> np.ndarray:
        """Convert µ-law 8-bit audio bytes to PCM16 (NumPy 2.x compatible)."""
        # Convert bytes to uint8 array
        mu = np.frombuffer(mulaw_bytes, dtype=np.uint8)
        
        # Invert bits
        mu = ~mu
        
        # Extract sign, exponent, and mantissa
        sign = (mu & 0x80).astype(np.int32)
        exponent = ((mu >> 4) & 0x07).astype(np.int32)
        mantissa = (mu & 0x0F).astype(np.int32)
        
        # Calculate magnitude using int32 to avoid overflow
        magnitude = ((mantissa << 3) + 0x84) << exponent
        magnitude = np.clip(magnitude, 0, 0x7FFF)
        
        # Apply sign
        pcm16 = np.where(sign == 0, magnitude, -magnitude)
        
        # Convert to int16
        return pcm16.astype(np.int16)
    
    def _resample_pcm16(self, pcm_data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Resample PCM16 audio from src_rate → dst_rate."""
        if src_rate == dst_rate:
            return pcm_data
        num_samples = int(len(pcm_data) * dst_rate / src_rate)
        return resample(pcm_data, num_samples).astype(np.int16)
