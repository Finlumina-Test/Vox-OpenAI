import aiohttp
import base64
from config import Config
from services.log_utils import Log

class TranscriptionService:
    """
    Handles real-time audio transcription using whisper-1.
    Runs asynchronously and returns text from audio bytes.
    """

    # Use the transcription endpoint with whisper-1
    OPENAI_API_URL = "https://api.openai.com/v1/realtime?model=whisper-1"

    async def transcribe_realtime(self, audio_chunk: bytes) -> str:
        """
        Sends a short audio chunk for realtime transcription.
        """
        try:
            audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")

            headers = {
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.OPENAI_API_URL, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        Log.error(f"Transcription request failed with {resp.status}")
                        return ""
                    data = await resp.json()
                    transcript = data.get("text") or data.get("transcript") or ""
                    return transcript.strip()
        except Exception as e:
            Log.error(f"Transcription error: {e}")
            return ""
