import io
import wave
import aiohttp
from config import Config
from services.log_utils import Log

# For Python 3.13+, install audioop-lts:
# pip install audioop-lts
class TranscriptionService:
    """Handles Twilio audio chunks → WAV → OpenAI transcription."""

    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"

    async def transcribe_realtime(self, audio_chunk: bytes) -> str:
        try:
            # --- Twilio sends 8kHz µ-law, 8-bit samples ---
            # Convert µ-law to 16-bit PCM
            pcm16 = audioop.ulaw2lin(audio_chunk, 2)

            # Optional: resample to 16kHz for better accuracy
            pcm16 = audioop.ratecv(pcm16, 2, 1, 8000, 16000, None)[0]

            # --- Write proper WAV file in memory ---
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)       # 16-bit samples
                wf.setframerate(16000)
                wf.writeframes(pcm16)
            wav_io.seek(0)

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
