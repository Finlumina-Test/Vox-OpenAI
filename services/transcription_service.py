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
    Converts Twilio µ-law (PCMU, 8 kHz, base64) audio chunks → 16-bit PCM WAV → OpenAI transcription.
    """

    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"

    async def transcribe_realtime(self, twilio_payload_b64: str) -> str:
        """
        Takes a base64 µ-law audio payload from Twilio, converts and sends it to OpenAI.
        """
        try:
            # 1️⃣ Decode base64 payload
            mulaw_bytes = base64.b64decode(twilio_payload_b64)

            # 2️⃣ Convert µ-law → PCM16 (manual decode)
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

            # 5️⃣ Send to OpenAI
            headers = {"Authorization": f"Bearer {Config.OPENAI_API_KEY}"}
            form = aiohttp.FormData()
            form.add_field("file", wav_io, filename="chunk.wav", content_type="audio/wav")
            form.add_field("model", "gpt-4o-mini-transcribe")

            async with aiohttp.ClientSession() as session:
                async with session.post(self.OPENAI_API_URL, headers=headers, data=form) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        Log.error(f"Transcription request failed with {resp.status}: {text}")
                        return ""

                    data = await resp.json()
                    return (data.get("text") or data.get("transcript") or "").strip()

        except Exception as e:
            Log.error(f"Transcription error: {e}")
            return ""

    # --- Internal conversion helpers ---

    def _mulaw_to_pcm16(self, mulaw_bytes: bytes) -> np.ndarray:
        """Convert µ-law 8-bit audio bytes to PCM16 (NumPy)."""
        MULAW_MAX = 0x1FFF
        MULAW_BIAS = 0x84

        mu = np.frombuffer(mulaw_bytes, dtype=np.uint8)
        mu = ~mu
        sign = (mu & 0x80)
        exponent = (mu >> 4) & 0x07
        mantissa = mu & 0x0F
        magnitude = ((mantissa << 4) + MULAW_BIAS) << (exponent + 3)
        pcm16 = (magnitude if sign == 0 else -magnitude)
        return pcm16.astype(np.int16)

    def _resample_pcm16(self, pcm_data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Resample PCM16 audio from src_rate → dst_rate."""
        if src_rate == dst_rate:
            return pcm_data
        num_samples = int(len(pcm_data) * dst_rate / src_rate)
        return resample(pcm_data, num_samples).astype(np.int16)
