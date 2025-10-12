import io
import wave
import base64
import aiohttp
from typing import Optional
from config import Config
from services.log_utils import Log
from services.audio_service import AudioService


class TranscriptionService:
    """
    Real-time transcription service that reuses the working AudioService for format conversion.
    """
    OPENAI_TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"
    
    def __init__(self):
        self.audio_service = AudioService()
    
    async def transcribe_realtime(self, audio_input, source: str = "Unknown") -> str:
        """
        Uses the same audio processing as AudioService to ensure compatibility.
        """
        try:
            # Use AudioService's format handling to get clean audio data
            if isinstance(audio_input, str):
                # This mimics what audio_service does with Twilio payloads
                clean_audio = self.audio_service.format_converter.twilio_to_openai(audio_input)
                audio_bytes = base64.b64decode(clean_audio)
            elif isinstance(audio_input, (bytes, bytearray)):
                # For AI audio, use the same approach as audio_service
                clean_audio = base64.b64encode(audio_input).decode('utf-8')
                audio_bytes = audio_input
            else:
                return ""
            
            # Skip if too small
            if len(audio_bytes) < 100:
                return ""
            
            # Create WAV file using the same parameters as audio_service
            wav_buffer = self._create_compatible_wav(audio_bytes)
            
            if not wav_buffer:
                return ""
            
            return await self._send_to_transcription_api(wav_buffer, source)
            
        except Exception as e:
            Log.error(f"[{source}] Transcription error: {e}")
            return ""
    
    def _create_compatible_wav(self, audio_bytes: bytes) -> Optional[io.BytesIO]:
        """
        Create a WAV file using the same format that works with Twilio/OpenAI real-time.
        """
        try:
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                # Use the exact same format as audio_service
                wav_file.setnchannels(1)    # Mono
                wav_file.setsampwidth(2)    # 16-bit  
                wav_file.setframerate(8000) # 8kHz - same as Twilio
                wav_file.writeframes(audio_bytes)
            
            wav_buffer.seek(0)
            
            # Verify it's a reasonable size
            if wav_buffer.getbuffer().nbytes < 100:
                return None
                
            return wav_buffer
            
        except Exception as e:
            Log.error(f"WAV creation error: {e}")
            return None
    
    async def _send_to_transcription_api(self, wav_buffer: io.BytesIO, source: str) -> str:
        """
        Send to OpenAI transcription API with proper error handling.
        """
        try:
            headers = {
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "User-Agent": "Twilio-Realtime-Agent/1.0"
            }
            
            # Log the WAV size for debugging
            wav_size = wav_buffer.getbuffer().nbytes
            Log.debug(f"[{source}] Sending WAV: {wav_size} bytes")
            
            form = aiohttp.FormData()
            form.add_field(
                "file",
                wav_buffer,
                filename="audio.wav",
                content_type="audio/wav"
            )
            form.add_field("model", "gpt-4o-mini-transcribe")
            form.add_field("response_format", "json")
            # Remove language parameter to let it auto-detect
            
            timeout = aiohttp.ClientTimeout(total=10)
            
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
                        else:
                            Log.debug(f"[{source}] Empty transcription")
                            
                        return transcript
                    else:
                        error_text = await response.text()
                        Log.error(f"[{source}] API error {response.status}: {error_text}")
                        return ""
                        
        except asyncio.TimeoutError:
            Log.error(f"[{source}] Transcription API timeout")
            return ""
        except Exception as e:
            Log.error(f"[{source}] API request failed: {e}")
            return ""
