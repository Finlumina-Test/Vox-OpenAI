import base64
import numpy as np
from typing import Optional
from services.log_utils import Log


class SilenceDetector:
    """
    Detects and filters silence in audio streams.
    Prevents transmission of silent audio chunks to reduce bandwidth and delays.
    """
    
    # Silence detection thresholds
    SILENCE_THRESHOLD = 500  # RMS threshold for μ-law audio (adjust as needed)
    MIN_SPEECH_DURATION_MS = 100  # Minimum speech duration to consider it real
    
    def __init__(self):
        self._consecutive_silence_count = 0
        self._consecutive_speech_count = 0
        self._last_was_speech = False
    
    @staticmethod
    def calculate_audio_energy(audio_base64: str) -> float:
        """
        Calculate the energy (RMS) of an audio chunk.
        
        Args:
            audio_base64: Base64 encoded μ-law audio
            
        Returns:
            RMS energy value
        """
        try:
            # Decode base64 to bytes
            audio_bytes = base64.b64decode(audio_base64)
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.uint8)
            
            # Calculate RMS (Root Mean Square) energy
            # For μ-law, values around 127-128 are silence
            # Deviation from 127-128 indicates sound
            centered = audio_array.astype(np.float32) - 127.5
            rms = np.sqrt(np.mean(centered ** 2))
            
            return rms
            
        except Exception as e:
            Log.debug(f"[Silence] Energy calculation error: {e}")
            # If error, assume it's speech to be safe
            return 1000.0
    
    def is_silence(self, audio_base64: str, speaker: str) -> bool:
        """
        Determine if an audio chunk is silence.
        
        Args:
            audio_base64: Base64 encoded audio
            speaker: "Caller" or "AI"
            
        Returns:
            True if silence, False if speech
        """
        try:
            energy = self.calculate_audio_energy(audio_base64)
            
            is_silent = energy < self.SILENCE_THRESHOLD
            
            if is_silent:
                self._consecutive_silence_count += 1
                self._consecutive_speech_count = 0
            else:
                self._consecutive_speech_count += 1
                self._consecutive_silence_count = 0
            
            # Update state
            previous_was_speech = self._last_was_speech
            self._last_was_speech = not is_silent
            
            # Log only on transitions (speech -> silence or silence -> speech)
            if previous_was_speech and is_silent:
                Log.debug(f"[Silence] {speaker} stopped speaking (energy: {energy:.1f})")
            elif not previous_was_speech and not is_silent:
                Log.debug(f"[Silence] {speaker} started speaking (energy: {energy:.1f})")
            
            return is_silent
            
        except Exception as e:
            Log.error(f"[Silence] Detection error: {e}")
            # On error, assume speech to avoid dropping audio
            return False
    
    def should_transmit(self, audio_base64: str, speaker: str) -> bool:
        """
        Determine if audio chunk should be transmitted to dashboard.
        
        Args:
            audio_base64: Base64 encoded audio
            speaker: "Caller" or "AI"
            
        Returns:
            True if should transmit, False if should drop
        """
        is_silent = self.is_silence(audio_base64, speaker)
        
        # Always transmit speech
        if not is_silent:
            return True
        
        # Drop prolonged silence (more than 3 consecutive silent chunks)
        # Keep first 2-3 silent chunks to avoid cutting off audio abruptly
        if self._consecutive_silence_count <= 3:
            return True
        
        return False
    
    def reset(self):
        """Reset silence detection state."""
        self._consecutive_silence_count = 0
        self._consecutive_speech_count = 0
        self._last_was_speech = False
        Log.debug("[Silence] Detector reset")


class AdaptiveSilenceDetector(SilenceDetector):
    """
    Advanced silence detector with adaptive thresholds.
    Automatically adjusts silence threshold based on background noise.
    """
    
    def __init__(self):
        super().__init__()
        self._noise_floor = 500.0  # Initial noise floor estimate
        self._energy_history = []
        self._max_history = 50  # Keep last 50 samples for noise estimation
    
    def update_noise_floor(self, energy: float):
        """Update noise floor estimate based on recent audio."""
        self._energy_history.append(energy)
        
        # Keep only recent history
        if len(self._energy_history) > self._max_history:
            self._energy_history.pop(0)
        
        # Calculate noise floor as 25th percentile of energy
        if len(self._energy_history) >= 10:
            sorted_energy = sorted(self._energy_history)
            percentile_25 = sorted_energy[len(sorted_energy) // 4]
            
            # Smooth the noise floor update
            self._noise_floor = 0.9 * self._noise_floor + 0.1 * percentile_25
    
    def is_silence(self, audio_base64: str, speaker: str) -> bool:
        """Detect silence with adaptive threshold."""
        try:
            energy = self.calculate_audio_energy(audio_base64)
            
            # Update noise floor
            self.update_noise_floor(energy)
            
            # Adaptive threshold: 2x noise floor + safety margin
            adaptive_threshold = max(self._noise_floor * 2.0, 300.0)
            
            is_silent = energy < adaptive_threshold
            
            if is_silent:
                self._consecutive_silence_count += 1
                self._consecutive_speech_count = 0
            else:
                self._consecutive_speech_count += 1
                self._consecutive_silence_count = 0
            
            # Update state
            previous_was_speech = self._last_was_speech
            self._last_was_speech = not is_silent
            
            # Log transitions
            if previous_was_speech and is_silent:
                Log.debug(f"[Silence] {speaker} stopped (energy: {energy:.1f}, threshold: {adaptive_threshold:.1f})")
            elif not previous_was_speech and not is_silent:
                Log.debug(f"[Silence] {speaker} started (energy: {energy:.1f}, threshold: {adaptive_threshold:.1f})")
            
            return is_silent
            
        except Exception as e:
            Log.error(f"[Silence] Detection error: {e}")
            return False
    
    def reset(self):
        """Reset detector and clear history."""
        super().reset()
        self._energy_history.clear()
        self._noise_floor = 500.0
        Log.debug("[Silence] Adaptive detector reset")
