import base64
import asyncio
import time
from typing import Dict, Optional, Callable
from services.log_utils import Log


class TranscriptionService:
    """
    Real-time audio streaming service with strict sequential delivery.
    
    Key Features:
    - Sequential audio playback (no overlap possible)
    - 1.0s gap ONLY for Caller->AI transitions
    - NO gaps for AI->Caller (natural conversation flow)
    - Async lock ensures one chunk completes before next starts
    
    NOTE: Transcription now handled by OpenAI natively!
    NOTE: Silence detection handled by silence_detection.py module!
    """
    
    # Audio format specs
    SAMPLE_RATE = 8000  # µ-law 8kHz from Twilio and OpenAI
    CHUNK_DURATION_MS = 20
    BYTES_PER_20MS = 160
    
    # Speaker turn detection
    SPEAKER_SILENCE_THRESHOLD = 0.3  # If no chunks for 0.3s, speaker is done
    SPEAKER_TRANSITION_DELAY = 1.0   # 1 SECOND gap for Caller->AI
    
    def __init__(self):
        # Unified audio queue
        self._unified_audio_queue: asyncio.Queue = asyncio.Queue()
        
        # Streaming task
        self._stream_task: Optional[asyncio.Task] = None
        
        # Speaker tracking
        self._last_streamed_speaker: Optional[str] = None
        self._last_chunk_time_per_speaker: Dict[str, float] = {}
        
        # 🔒 CRITICAL: Sequential playback lock
        self._playback_lock: asyncio.Lock = asyncio.Lock()
        
        # Callbacks
        self.audio_callback: Optional[Callable] = None
        
        # Shutdown flag
        self._shutdown: bool = False
    
    def set_audio_callback(self, callback: Callable):
        """Set callback for raw audio chunks."""
        self.audio_callback = callback
        
        if not self._stream_task or self._stream_task.done():
            self._stream_task = asyncio.create_task(self._stream_unified_audio())
    
    def _calculate_chunk_duration(self, audio_bytes: bytes) -> float:
        """Calculate audio chunk duration in seconds (8kHz µ-law)."""
        num_samples = len(audio_bytes)
        duration_seconds = num_samples / self.SAMPLE_RATE
        return duration_seconds
    
    async def _stream_unified_audio(self):
        """
        Sequential audio streaming with smart speaker transitions.
        
        🔒 Uses async lock to guarantee:
        - ONE chunk plays at a time (no overlap)
        - 1.0s gap for Caller->AI transitions
        - NO gap for AI->Caller transitions
        """
        Log.info("[Stream] Started - SEQUENTIAL with 1.0s Caller→AI gap")
        
        while not self._shutdown:
            try:
                audio_data = await self._unified_audio_queue.get()
                
                if audio_data is None:
                    break
                
                speaker = audio_data.get("speaker")
                current_time = time.time()
                
                async with self._playback_lock:
                    
                    audio_b64 = audio_data.get("audio", "")
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        chunk_duration = self._calculate_chunk_duration(audio_bytes)
                    except Exception as e:
                        Log.debug(f"[Stream] Duration calc error: {e}")
                        chunk_duration = 0.02
                    
                    speaker_changed = (
                        self._last_streamed_speaker is not None and 
                        self._last_streamed_speaker != speaker
                    )
                    
                    if speaker_changed:
                        previous_speaker = self._last_streamed_speaker
                        previous_last_time = self._last_chunk_time_per_speaker.get(previous_speaker, 0)
                        time_gap = current_time - previous_last_time if previous_last_time > 0 else 0
                        previous_finished = time_gap >= self.SPEAKER_SILENCE_THRESHOLD
                        
                        # ✅ ONLY add 1.0s gap for Caller → AI
                        if previous_speaker == "Caller" and speaker == "AI" and previous_finished:
                            if time_gap < self.SPEAKER_TRANSITION_DELAY:
                                remaining_gap = self.SPEAKER_TRANSITION_DELAY - time_gap
                                Log.debug(f"[Stream] Caller → AI: +{remaining_gap:.3f}s gap")
                                await asyncio.sleep(remaining_gap)
                            else:
                                Log.debug(f"[Stream] Caller → AI: {time_gap:.3f}s natural")
                        
                        # ✅ AI → Caller: NO gap
                        elif previous_speaker == "AI" and speaker == "Caller":
                            Log.debug(f"[Stream] AI → Caller: NO GAP")
                    
                    self._last_chunk_time_per_speaker[speaker] = current_time
                    self._last_streamed_speaker = speaker
                    
                    # Send to dashboard
                    if self.audio_callback:
                        try:
                            await self.audio_callback(audio_data)
                        except Exception as e:
                            Log.error(f"[Stream] callback error: {e}")
                    
                    # Wait for chunk duration to maintain timing
                    await asyncio.sleep(chunk_duration)
                
                self._unified_audio_queue.task_done()
                
            except Exception as e:
                Log.error(f"[Stream] error: {e}")
                await asyncio.sleep(0.01)
    
    async def stream_audio_chunk(self, audio_input, source: str = "Unknown") -> None:
        """
        Process incoming audio chunk and queue for streaming.
        
        NOTE: Silence detection now handled by silence_detection.py before calling this!
        """
        try:
            if isinstance(audio_input, str):
                audio_bytes = base64.b64decode(audio_input)
                original_base64 = audio_input
            elif isinstance(audio_input, (bytes, bytearray)):
                audio_bytes = bytes(audio_input)
                original_base64 = base64.b64encode(audio_bytes).decode('ascii')
            else:
                return
            
            # Queue for streaming
            audio_packet = {
                "speaker": source,
                "audio": original_base64,
                "timestamp": int(time.time() * 1000),
                "size": len(audio_bytes)
            }
            await self._unified_audio_queue.put(audio_packet)
            
        except Exception as e:
            Log.error(f"[{source}] Audio streaming error: {e}")
    
    def clear_buffers(self):
        """Clear audio queue (used during human takeover)."""
        try:
            while not self._unified_audio_queue.empty():
                try:
                    self._unified_audio_queue.get_nowait()
                    self._unified_audio_queue.task_done()
                except:
                    break
            Log.info("[Stream] Buffers cleared")
        except Exception as e:
            Log.error(f"[Stream] Buffer clear error: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown."""
        try:
            self._shutdown = True
            await self._unified_audio_queue.put(None)
            
            if self._stream_task and not self._stream_task.done():
                await asyncio.wait([self._stream_task], timeout=2.0)
            
            Log.info("TranscriptionService shutdown complete")
                
        except Exception as e:
            Log.error(f"Shutdown error: {e}")
