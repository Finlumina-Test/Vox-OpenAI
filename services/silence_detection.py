import base64
import numpy as np
from typing import Optional
from services.log_utils import Log


class SilenceDetector:
    """
    Simple silence detector - filters ONLY absolute silence, not background noise.
    Very permissive to avoid cutting off speech.
    """
    
    # Very low threshold - only catches absolute silence
    SILENCE_THRESHOLD = 50  # Even lower
    GRACE_CHUNKS = 10  # Keep way more chunks
    
    def __init__(self):
        self._consecutive_silence_count = 0
        self._last_was_speech = False
    
    @staticmethod
    def calculate_audio_energy(audio_base64: str) -> float:
        """Calculate RMS energy of µ-law audio chunk."""
        try:
            audio_bytes = base64.b64decode(audio_base64)
            audio_array = np.frombuffer(audio_bytes, dtype=np.uint8)
            
            # For µ-law, silence is around 127-128
            # Any deviation indicates sound
            centered = audio_array.astype(np.float32) - 127.5
            rms = np.sqrt(np.mean(centered ** 2))
            
            return rms
            
        except Exception as e:
            Log.debug(f"[Silence] Energy error: {e}")
            return 1000.0  # Assume speech on error
    
    def is_silence(self, audio_base64: str) -> bool:
        """Check if audio is absolute silence."""
        try:
            energy = self.calculate_audio_energy(audio_base64)
            is_silent = energy < self.SILENCE_THRESHOLD
            
            if is_silent:
                self._consecutive_silence_count += 1
            else:
                self._consecutive_silence_count = 0
            
            previous_was_speech = self._last_was_speech
            self._last_was_speech = not is_silent
            
            # Log only on transitions
            if previous_was_speech and is_silent:
                Log.debug(f"[Silence] Stopped (energy: {energy:.1f})")
            elif not previous_was_speech and not is_silent:
                Log.debug(f"[Silence] Started (energy: {energy:.1f})")
            
            return is_silent
            
        except Exception:
            return False  # Assume speech on error
    
    def should_transmit(self, audio_base64: str, speaker: str) -> bool:
        """
        Determine if audio should be transmitted.
        Very permissive - only filters prolonged absolute silence.
        """
        is_silent = self.is_silence(audio_base64)
        
        # Always transmit speech
        if not is_silent:
            return True
        
        # Keep first few silent chunks to avoid abrupt cutoff
        if self._consecutive_silence_count <= self.GRACE_CHUNKS:
            return True
        
        # Filter only prolonged absolute silence
        return False
    
    def reset(self):
        """Reset detector state."""
        self._consecutive_silence_count = 0
        self._last_was_speech = False
