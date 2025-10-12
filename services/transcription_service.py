import io
import wave
import aiohttp
import audioop
from config import Config
from services.log_utils import Log


class TranscriptionService:
    """
    Handles real-time audio transcription.
    Converts Twilio μ-law audio chunks to proper 16kHz PCM16 WAV before sending.
    """

    OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"

    async def transcribe_realtime(self, audio_chunk: bytes) -> str:
        """
        Sends a short audio chunk to OpenAI's transcription model.
        """
        try:
            # --- Convert Twilio μ-law (default) to PCM16 mono 16kHz ---
            try:
                pcm_audio = audioop.ulaw2lin(audio_chunk, 2)
            except Exception:
                pcm_audio = audio_chunk  # if it's already PCM16, skip conversion

            # --- Wrap PCM into WAV ---
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm_audio)
            wav_io.seek(0)

            # --- Send to OpenAI ---
            headers = {
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
            }

            async with aiohttp.ClientSession() as session:
                form = aiohttp.FormData()
                form.add_field("file", wav_io, filename="chunk.wav", content_type="audio/wav")
                form.add_field("model", "gpt-4o-mini-transcribe")

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
