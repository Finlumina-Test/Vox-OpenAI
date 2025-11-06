import base64
import numpy as np
from typing import Optional
from services.log_utils import Log


class SilenceDetector:
    """
    Simple silence detector with adaptive thresholding for noisy environments.
    Filters ONLY absolute silence, not background noise.
    VERY permissive to avoid cutting off speech.
    """
    
    # 🔥 FIXED: More permissive thresholds
    SILENCE_THRESHOLD = 30  # Lower base threshold (was 50)
    GRACE_CHUNKS = 20  # More grace chunks (was 10)
    ADAPTIVE_WINDOW = 100  # Longer history (was 50)
    MAX_ADAPTIVE_THRESHOLD = 80  # 🔥 Cap the adaptive threshold
    
    def __init__(self):
        self._consecutive_silence_count = 0
        self._last_was_speech = False
        self._energy_history = []  # For adaptive thresholding
        self._adaptive_threshold = self.SILENCE_THRESHOLD
        self._total_chunks = 0
    
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
    
    def _update_adaptive_threshold(self, energy: float):
        """
        Update adaptive threshold based on recent energy history.
        Automatically adjusts to noisy environments (markets, restaurants, etc.)
        🔥 FIXED: Capped threshold to prevent over-filtering
        """
        self._energy_history.append(energy)
        self._total_chunks += 1
        
        # Keep only recent history
        if len(self._energy_history) > self.ADAPTIVE_WINDOW:
            self._energy_history.pop(0)
        
        # 🔥 Only adapt after collecting enough samples
        if len(self._energy_history) >= 20:
            sorted_energies = sorted(self._energy_history)
            percentile_5 = sorted_energies[len(sorted_energies) // 20]  # 5th percentile (was 10th)
            
            # 🔥 FIXED: Cap adaptive threshold to prevent over-filtering
            calculated_threshold = percentile_5 * 1.1  # Less aggressive (was 1.2)
            self._adaptive_threshold = min(
                max(self.SILENCE_THRESHOLD, calculated_threshold),
                self.MAX_ADAPTIVE_THRESHOLD  # 🔥 Never go above this
            )
            
            # Log threshold changes for debugging
            if self._total_chunks % 100 == 0:
                Log.debug(f"[Silence] Adaptive threshold: {self._adaptive_threshold:.1f} (cap: {self.MAX_ADAPTIVE_THRESHOLD})")
    
    def is_silence(self, audio_base64: str) -> bool:
        """Check if audio is absolute silence (with adaptive thresholding)."""
        try:
            energy = self.calculate_audio_energy(audio_base64)
            
            # Update adaptive threshold
            self._update_adaptive_threshold(energy)
            
            # Use adaptive threshold instead of fixed threshold
            is_silent = energy < self._adaptive_threshold
            
            if is_silent:
                self._consecutive_silence_count += 1
            else:
                self._consecutive_silence_count = 0
            
            previous_was_speech = self._last_was_speech
            self._last_was_speech = not is_silent
            
            # Log only on transitions (with adaptive threshold info)
            if previous_was_speech and is_silent:
                Log.debug(f"[Silence] Speech→Silence (energy: {energy:.1f}, threshold: {self._adaptive_threshold:.1f})")
            elif not previous_was_speech and not is_silent:
                Log.debug(f"[Silence] Silence→Speech (energy: {energy:.1f}, threshold: {self._adaptive_threshold:.1f})")
            
            return is_silent
            
        except Exception:
            return False  # Assume speech on error
    
    def should_transmit(self, audio_base64: str, speaker: str) -> bool:
        """
        Determine if audio should be transmitted.
        VERY permissive - only filters prolonged absolute silence.
        """
        is_silent = self.is_silence(audio_base64)
        
        # Always transmit speech
        if not is_silent:
            return True
        
        # 🔥 FIXED: More grace chunks to avoid cutting off
        if self._consecutive_silence_count <= self.GRACE_CHUNKS:
            return True
        
        # Filter only prolonged absolute silence
        # 🔥 Log when we actually filter
        if self._consecutive_silence_count == self.GRACE_CHUNKS + 1:
            Log.debug(f"[Silence] Started filtering {speaker} silence after {self.GRACE_CHUNKS} chunks")
        
        return False
    
    def reset(self):
        """Reset detector state."""
        self._consecutive_silence_count = 0
        self._last_was_speech = False
        self._energy_history.clear()
        self._adaptive_threshold = self.SILENCE_THRESHOLD
        self._total_chunks = 0
        Log.debug("[Silence] Detector reset")
