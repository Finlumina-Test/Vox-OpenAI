import io
import wave
import base64
import aiohttp
from typing import Optional
from config import Config
from services.log_utils import Log


class TranscriptionService:
    """
    Real-time transcription service that uses the same audio format as audio_service.py
    """
    OPENAI_TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"
    
    async def transcribe_realtime(self, audio_input, source: str = "Unknown") -> str:
        """
        Uses the exact same audio format handling as audio_service.py for consistency.
        """
        try:
            # Use the same format converter logic as audio_service
            if isinstance(audio_input, str):
                # This is already in the format that audio_service uses
                # Just decode and re-encode to ensure it's clean base64
                audio_bytes = base64.b64decode(audio_input)
                clean_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            elif isinstance(audio_input, (bytes, bytearray)):
                # Convert bytes to base64 string (like audio_service does)
                clean_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            else:
                return ""
            
            # Skip if too small
            if len(clean_base64) < 50:
                return ""
            
            # Create WAV from the base64 data using simple approach
            wav_buffer = self._create_simple_wav(clean_base64)
            
            if not wav_buffer:
                return ""
            
            return await self._send_to_transcription_api(wav_buffer, source)
            
        except Exception as e:
            Log.error(f"[{source}] Transcription error: {e}")
            return ""
    
    def _create_simple_wav(self, base64_audio: str) -> Optional[io.BytesIO]:
        """
        Create a simple WAV file from base64 audio data.
        This uses a very basic approach that should work with any valid audio.
        """
        try:
            # Decode base64
            audio_bytes = base64.b64decode(base64_audio)
            
            if len(audio_bytes) < 44:  # WAV header is 44 bytes
                return None
            
            # Create WAV file
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                # Use standard telephony format
                wav_file.setnchannels(1)    # Mono
                wav_file.setsampwidth(2)    # 16-bit
                wav_file.setframerate(8000) # 8kHz
                wav_file.writeframes(audio_bytes)
            
            wav_buffer.seek(0)
            return wav_buffer
            
        except Exception as e:
            Log.error(f"Simple WAV creation error: {e}")
            return None
    
    async def _send_to_transcription_api(self, wav_buffer: io.BytesIO, source: str) -> str:
        """
        Send to OpenAI transcription API.
        """
        try:
            headers = {"Authorization": f"Bearer {Config.OPENAI_API_KEY}"}
            
            form = aiohttp.FormData()
            form.add_field(
                "file",
                wav_buffer,
                filename="audio.wav",
                content_type="audio/wav"
            )
            form.add_field("model", "gpt-4o-mini-transcribe")
            form.add_field("response_format", "json")
            
            timeout = aiohttp.ClientTimeout(total=8)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.OPENAI_TRANSCRIPTION_URL,
                    headers=headers,
                    data=form
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        transcript = result.get("text", "").strip()
                        
                        if transcript:
                            Log.info(f"[{source}] Transcribed: {transcript}")
                            
                        return transcript
                    else:
                        error_text = await response.text()
                        Log.error(f"[{source}] API error {response.status}: {error_text}")
                        return ""
                        
        except Exception as e:
            Log.error(f"[{source}] API request failed: {e}")
            return ""
