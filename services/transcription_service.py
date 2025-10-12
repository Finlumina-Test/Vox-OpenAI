import io
import wave
import base64
import aiohttp
import numpy as np
from typing import Optional
from config import Config
from services.log_utils import Log


class TranscriptionService:
    """
    Real-time transcription service using gpt-4o-mini-transcribe model.
    Handles Twilio µ-law audio chunks and converts them for OpenAI transcription API.
    """
    OPENAI_TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"
    
    async def transcribe_realtime(self, audio_input, source: str = "Unknown") -> str:
        """
        Takes audio (base64 string or raw bytes), converts and sends to OpenAI for transcription.
        
        Args:
            audio_input: Either base64-encoded string or raw bytes
            source: Label for logging (e.g., "Caller" or "AI")
        """
        try:
            # 1️⃣ Handle both base64 and raw bytes input
            if isinstance(audio_input, str):
                # It's already base64 encoded, decode to bytes
                audio_bytes = base64.b64decode(audio_input)
            elif isinstance(audio_input, (bytes, bytearray)):
                # It's already bytes, use directly
                audio_bytes = bytes(audio_input)
            else:
                Log.error(f"Invalid audio input type: {type(audio_input)}")
                return ""
            
            # Skip if chunk is too small (avoid API errors)
            if len(audio_bytes) < 100:
                Log.debug(f"[{source}] Audio chunk too small: {len(audio_bytes)} bytes")
                return ""
            
            # 2️⃣ Convert µ-law to PCM16
            pcm_data = self._mulaw_to_pcm16(audio_bytes)
            
            if len(pcm_data) == 0:
                Log.debug(f"[{source}] Empty PCM data after conversion")
                return ""
            
            # 3️⃣ Create proper WAV file in memory (16kHz, 16-bit mono)
            wav_buffer = self._create_wav_file(pcm_data)
            
            if wav_buffer is None:
                Log.error(f"[{source}] Failed to create WAV file")
                return ""
            
            # 4️⃣ Send to OpenAI gpt-4o-mini-transcribe
            transcript = await self._send_to_transcription_api(wav_buffer, source)
            return transcript
            
        except Exception as e:
            Log.error(f"[{source}] Transcription error: {e}")
            return ""
    
    def _mulaw_to_pcm16(self, mulaw_bytes: bytes) -> np.ndarray:
        """
        Convert µ-law audio to 16-bit PCM using standard expansion.
        Twilio sends µ-law (8kHz, 8-bit) and we need PCM (16-bit) for WAV.
        """
        try:
            # Standard µ-law expansion table
            MU_LAW_TABLE = [
                -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
                -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
                -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
                -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
                -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
                -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
                -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
                -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
                -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
                -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
                -876, -844, -812, -780, -748, -716, -684, -652,
                -620, -588, -556, -524, -492, -460, -428, -396,
                -372, -356, -340, -324, -308, -292, -276, -260,
                -244, -228, -212, -196, -180, -164, -148, -132,
                -120, -112, -104, -96, -88, -80, -72, -64,
                -56, -48, -40, -32, -24, -16, -8, 0,
                32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
                23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
                15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
                11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
                7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
                5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
                3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
                2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
                1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
                1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
                876, 844, 812, 780, 748, 716, 684, 652,
                620, 588, 556, 524, 492, 460, 428, 396,
                372, 356, 340, 324, 308, 292, 276, 260,
                244, 228, 212, 196, 180, 164, 148, 132,
                120, 112, 104, 96, 88, 80, 72, 64,
                56, 48, 40, 32, 24, 16, 8, 0
            ]
            
            # Convert each µ-law byte to PCM using lookup table
            pcm_samples = []
            for byte_val in mulaw_bytes:
                if byte_val < 256:
                    pcm_val = MU_LAW_TABLE[byte_val]
                    pcm_samples.append(pcm_val)
            
            return np.array(pcm_samples, dtype=np.int16)
            
        except Exception as e:
            Log.error(f"µ-law to PCM conversion error: {e}")
            return np.array([], dtype=np.int16)
    
    def _create_wav_file(self, pcm_data: np.ndarray) -> Optional[io.BytesIO]:
        """
        Create a WAV file from PCM data optimized for gpt-4o-mini-transcribe.
        Uses 8kHz sample rate since that's what Twilio sends and the model expects.
        """
        if len(pcm_data) == 0:
            return None
            
        try:
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                # Use 8kHz since that's the original Twilio sample rate
                # gpt-4o-mini-transcribe can handle various sample rates
                wav_file.setnchannels(1)    # Mono
                wav_file.setsampwidth(2)    # 16-bit
                wav_file.setframerate(8000) # 8kHz - same as Twilio input
                wav_file.writeframes(pcm_data.tobytes())
            
            wav_buffer.seek(0)
            return wav_buffer
            
        except Exception as e:
            Log.error(f"WAV file creation error: {e}")
            return None
    
    async def _send_to_transcription_api(self, wav_buffer: io.BytesIO, source: str) -> str:
        """
        Send WAV buffer to OpenAI transcription API using gpt-4o-mini-transcribe model.
        """
        try:
            headers = {
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "User-Agent": "Twilio-Realtime-Agent/1.0"
            }
            
            form = aiohttp.FormData()
            form.add_field(
                "file",
                wav_buffer,
                filename="audio_chunk.wav",
                content_type="audio/wav"
            )
            form.add_field("model", "gpt-4o-mini-transcribe")
            form.add_field("language", "en")
            form.add_field("response_format", "json")
            
            # Shorter timeout for real-time (we want fast responses)
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
                        else:
                            Log.debug(f"[{source}] Empty transcription")
                            
                        return transcript
                    else:
                        error_text = await response.text()
                        Log.error(f"[{source}] Transcription API error {response.status}: {error_text}")
                        
                        # Log more details for debugging
                        if response.status == 400:
                            Log.error(f"[{source}] Bad request - audio format may be incompatible")
                        elif response.status == 429:
                            Log.error(f"[{source}] Rate limit exceeded")
                        elif response.status >= 500:
                            Log.error(f"[{source}] OpenAI server error")
                            
                        return ""
                        
        except asyncio.TimeoutError:
            Log.error(f"[{source}] Transcription API timeout")
            return ""
        except Exception as e:
            Log.error(f"[{source}] Transcription API request failed: {e}")
            return ""
