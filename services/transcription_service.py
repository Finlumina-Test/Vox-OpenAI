import io
import wave
import aiohttp
from config import Config
from services.log_utils import Log


class TranscriptionService:
    """
    Handles real-time audio transcription using Whisper-1.
    Converts audio chunks to proper WAV format before sending.
    """

    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"

    async def transcribe_chunk(self, audio_chunk: bytes) -> str:
        """
        Sends a short audio chunk to Whisper for transcription.
        """
        try:
            # Convert raw PCM16 or μ-law bytes to WAV (mono, 16kHz)
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_chunk)
            wav_io.seek(0)

            headers = {
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
            }

            async with aiohttp.ClientSession() as session:
                form = aiohttp.FormData()
                form.add_field("file", wav_io, filename="chunk.wav", content_type="audio/wav")
                form.add_field("model", "whisper-1")

                async with session.post(self.OPENAI_API_URL, headers=headers, data=form) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        Log.error(f"Transcription request failed with {resp.status}: {text}")
                        return ""

                    data = await resp.json()
                    transcript = data.get("text") or data.get("transcript") or ""
                    return transcript.strip()

        except Exception as e:
            Log.error(f"Transcription error: {e}")
            return ""
