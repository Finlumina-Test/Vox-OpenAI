import json
import asyncio
import time
from typing import Optional, Dict, Any
from config import Config
from services.log_utils import Log


class OpenAIEventHandler:
    """
    Interprets and processes events received from the OpenAI Realtime API.
    """
    
    @staticmethod
    def should_log_event(event_type: str) -> bool:
        """Check if an event type should be logged."""
        return event_type in Config.LOG_EVENT_TYPES
    
    @staticmethod
    def is_audio_delta_event(event: Dict[str, Any]) -> bool:
        """Check if event is an audio delta from OpenAI."""
        return (event.get('type') == 'response.output_audio.delta' and 
                'delta' in event)
    
    @staticmethod
    def is_speech_started_event(event: Dict[str, Any]) -> bool:
        """Check if event indicates user speech has started."""
        return event.get('type') == 'input_audio_buffer.speech_started'
    
    @staticmethod
    def extract_audio_delta(event: Dict[str, Any]) -> Optional[str]:
        """Extract audio delta from OpenAI event."""
        if OpenAIEventHandler.is_audio_delta_event(event):
            return event.get('delta')
        return None
    
    @staticmethod
    def extract_item_id(event: Dict[str, Any]) -> Optional[str]:
        """Extract item ID from OpenAI event."""
        return event.get('item_id')


from typing import Dict, Any
from config import Config


class OpenAISessionManager:
    """
    Configures and initializes OpenAI Realtime API sessions.
    Ensures transcription stays in English (Roman) script
    for Urdu/Punjabi/English mixed speech — without translation.
    """

    @staticmethod
    def create_session_update() -> Dict[str, Any]:
        """Create a session update message for OpenAI Realtime API."""
        session = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": "gpt-realtime-mini-2025-10-06",
                "output_modalities": ["audio", "text"],  # Include text for AI understanding

                "audio": {
                    "input": {
                        "format": {"type": "audio/pcmu"},
                        "turn_detection": {"type": "server_vad"},
                        "transcription": {
                            "model": "gpt-4o-mini-transcribe",
                            "language": "ur"  # Force Urdu/Punjabi speech mode
                        }
                    },
                    "output": {"format": {"type": "audio/pcmu"}}
                },

                "instructions": (
                    Config.SYSTEM_MESSAGE
                    + "\n\n"
                    "### TRANSCRIPTION RULES ###\n"
                    "You will receive speech in Urdu or Punjabi mixed with English.\n"
                    "Always transcribe it **in English (Roman) script**, preserving the natural sounds.\n"
                    "Do NOT translate or write in Urdu or Punjabi script.\n"
                    "For example:\n"
                    "  - If the caller says 'آج میں نے برگر کھانا ہے', write: 'aaj maine burger khana hai'.\n"
                    "  - If the caller says 'دو زنگر برگر دے دینا', write: 'do zinger burger de dena'.\n"
                    "  - If the caller says 'I want one zinger burger', write exactly that.\n"
                    "If unsure of a word, write what you hear phonetically in Roman letters.\n"
                    "Keep the style casual and conversational — just as it’s spoken."
                ),

                "tools": [
                    {
                        "type": "function",
                        "name": "end_call",
                        "description": (
                            "Politely end the phone call when the caller says goodbye "
                            "or requests to end the conversation."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reason": {
                                    "type": "string",
                                    "description": "Brief reason for ending, e.g., user said bye."
                                }
                            },
                            "required": []
                        }
                    }
                ]
            }
        }
        return session


    @staticmethod
    def create_initial_conversation_item() -> Dict[str, Any]:
        """Create an initial conversation item for AI-first interactions."""
        return {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Greet the user with 'Hello there! I am an AI voice assistant "
                            "powered by Twilio and the OpenAI Realtime API. You can ask me "
                            "for facts, jokes, or anything you can imagine. How can I help you?'"
                        )
                    }
                ]
            }
        }
    
    @staticmethod
    def create_response_trigger() -> Dict[str, Any]:
        """Create a response trigger message."""
        return {"type": "response.create"}


class OpenAIConversationManager:
    """
    Manages conversation flow and interruption logic for OpenAI sessions.
    """
    
    @staticmethod
    def create_truncate_event(item_id: str, audio_end_ms: int) -> Dict[str, Any]:
        """Create a conversation item truncation event."""
        return {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": audio_end_ms
        }
    
    @staticmethod
    def should_handle_interruption(
        last_assistant_item: Optional[str],
        mark_queue: list,
        response_start_timestamp: Optional[int]
    ) -> bool:
        """Determine if an interruption should be processed."""
        return (last_assistant_item is not None and 
                len(mark_queue) > 0 and 
                response_start_timestamp is not None)
    
    @staticmethod
    def calculate_truncation_time(
        current_timestamp: int,
        response_start_timestamp: int
    ) -> int:
        """Calculate the elapsed time for audio truncation."""
        return current_timestamp - response_start_timestamp


class TranscriptFilter:
    """
    Filters out low-quality transcripts from OpenAI's native transcription.
    """
    
    # Common noise patterns that OpenAI might transcribe
    NOISE_PATTERNS = [
        "thank you",  # Often phantom
        "thanks",
        "bye",
        "okay",
        "ok",
        "yeah",
        "yes",
        "no",
        "um",
        "uh",
        "hmm",
        "mhm",
        "ah",
    ]
    
    # Minimum length for valid transcript
    MIN_TRANSCRIPT_LENGTH = 3
    
    # Maximum length for noise (usually short)
    MAX_NOISE_LENGTH = 15
    
    @staticmethod
    def is_valid_transcript(text: str, speaker: str) -> bool:
        """
        Validate if transcript is real speech or just noise.
        
        Rules:
        1. Must be at least 3 characters
        2. If very short (< 15 chars) and matches noise patterns, reject
        3. AI transcripts are always valid (we trust our own output)
        """
        if not text or not isinstance(text, str):
            return False
        
        cleaned = text.strip().lower()
        
        # Empty or too short
        if len(cleaned) < TranscriptFilter.MIN_TRANSCRIPT_LENGTH:
            return False
        
        # AI transcripts are always valid
        if speaker == "AI":
            return True
        
        # For caller: Check if it's a noise pattern
        if len(cleaned) <= TranscriptFilter.MAX_NOISE_LENGTH:
            # Check if it matches any noise pattern
            for pattern in TranscriptFilter.NOISE_PATTERNS:
                if cleaned == pattern or cleaned.startswith(pattern + " ") or cleaned.endswith(" " + pattern):
                    Log.debug(f"[Filter] Rejected noise: '{text}'")
                    return False
        
        # Passed all checks
        return True


class OpenAIService:
    """
    Unified service for OpenAI Realtime API.
    Uses OpenAI's native transcription for BOTH caller and AI.
    No external Whisper calls needed!
    """

    def __init__(self):
        self.session_manager = OpenAISessionManager()
        self.conversation_manager = OpenAIConversationManager()
        self.event_handler = OpenAIEventHandler()
        self.transcript_filter = TranscriptFilter()
        self._pending_tool_calls: Dict[str, Dict[str, Any]] = {}
        self._pending_goodbye: bool = False
        self._goodbye_audio_heard: bool = False
        self._goodbye_item_id: Optional[str] = None
        self._goodbye_watchdog: Optional[asyncio.Task] = None
        
        # Callbacks for transcripts
        self.caller_transcript_callback: Optional[callable] = None
        self.ai_transcript_callback: Optional[callable] = None
        
        # ✅ Track last transcript timestamp per speaker to ensure ordering
        self._last_transcript_time: Dict[str, float] = {"Caller": 0, "AI": 0}

    # --- SESSION & GREETING ---
    async def initialize_session(self, connection_manager) -> None:
        session_update = self.session_manager.create_session_update()
        Log.json('Sending session update', session_update)
        await connection_manager.send_to_openai(session_update)

    async def send_initial_greeting(self, connection_manager) -> None:
        initial_item = self.session_manager.create_initial_conversation_item()
        response_trigger = self.session_manager.create_response_trigger()
        await connection_manager.send_to_openai(initial_item)
        await connection_manager.send_to_openai(response_trigger)

    # --- EVENT LOGGING & TOOL CALLS ---
    def process_event_for_logging(self, event: Dict[str, Any]) -> None:
        if self.event_handler.should_log_event(event.get('type', '')):
            Log.event(f"Received event: {event['type']}", event)

    def is_tool_call(self, event: Dict[str, Any]) -> bool:
        etype = event.get('type')
        if etype in ('response.function_call.arguments.delta', 'response.function_call.completed'):
            return True
        if etype == 'response.done':
            resp = event.get('response') or {}
            output = resp.get('output') or []
            for item in output:
                if isinstance(item, dict) and item.get('type') == 'function_call':
                    return True
        return False

    def accumulate_tool_call(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        etype = event.get('type')
        if etype == 'response.function_call.arguments.delta':
            call_id = event.get('call_id') or event.get('id') or 'default'
            delta = event.get('delta', '')
            buf = self._pending_tool_calls.setdefault(call_id, {"args": "", "name": event.get('name')})
            buf["args"] += delta
            return None
        if etype == 'response.function_call.completed':
            call_id = event.get('call_id') or event.get('id') or 'default'
            payload = self._pending_tool_calls.pop(call_id, None)
            if payload is None:
                return None
            try:
                args = json.loads(payload["args"]) if payload["args"] else {}
            except Exception:
                args = {"_raw": payload["args"]}
            return {"name": payload.get('name') or event.get('name'), "arguments": args}
        if etype == 'response.done':
            resp = event.get('response') or {}
            output = resp.get('output') or []
            for item in output:
                if isinstance(item, dict) and item.get('type') == 'function_call':
                    name = item.get('name')
                    raw_args = item.get('arguments')
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        args = {"_raw": raw_args}
                    return {"name": name, "arguments": args}
        return None

    async def maybe_handle_tool_call(self, connection_manager, tool_call: Dict[str, Any]) -> bool:
        if not tool_call:
            return False
        name = tool_call.get('name')
        if name != 'end_call':
            return False

        args = tool_call.get('arguments') or {}
        reason = args.get('reason') if isinstance(args, dict) else None
        farewell = Config.build_end_call_farewell(reason)

        if self._pending_goodbye:
            Log.info("End-call already pending; ignoring duplicate request")
            return False

        Log.info("Queueing farewell response before hangup")
        await self._send_goodbye_response(connection_manager, farewell)
        self._pending_goodbye = True
        self._goodbye_audio_heard = False
        self._goodbye_item_id = None
        self._start_goodbye_watchdog(connection_manager)
        return True

    async def _send_goodbye_response(self, connection_manager, text: str) -> None:
        try:
            await connection_manager.send_to_openai({
                "type": "response.create",
                "response": {"instructions": text}
            })
        except Exception as e:
            Log.error(f"Failed to queue goodbye response: {e}")
            self._pending_goodbye = True
            self._goodbye_audio_heard = False

    # --- GOODBYE HANDLING ---
    def should_finalize_on_event(self, event: Dict[str, Any]) -> bool:
        if not (self._pending_goodbye and self._goodbye_audio_heard):
            return False
        etype = event.get('type')
        if etype == 'response.output_audio.done':
            return True
        if etype == 'response.done':
            if not self._goodbye_item_id:
                resp = event.get('response') or {}
                for item in (resp.get('output') or []):
                    if isinstance(item, dict) and item.get('type') == 'message' and item.get('role') == 'assistant':
                        for c in (item.get('content') or []):
                            if isinstance(c, dict) and c.get('type') == 'output_audio':
                                return True
                return False
            resp = event.get('response') or {}
            for item in (resp.get('output') or []):
                if isinstance(item, dict) and item.get('id') == self._goodbye_item_id:
                    return True
        return False

    async def finalize_goodbye(self, connection_manager) -> None:
        self._pending_goodbye = False
        self._goodbye_audio_heard = False
        self._goodbye_item_id = None
        self._cancel_goodbye_watchdog()
        try:
            await asyncio.sleep(getattr(Config, 'END_CALL_GRACE_SECONDS', 0.5))
        except Exception:
            pass
        if Config.has_twilio_credentials():
            try:
                from twilio.rest import Client
                client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
                call_sid = getattr(connection_manager.state, 'call_sid', None)
                if call_sid:
                    Log.event("Completing call via Twilio REST", {"callSid": call_sid})
                    client.calls(call_sid).update(status='completed')
            except Exception as e:
                Log.error(f"Optional Twilio REST hangup failed: {e}")
        try:
            await connection_manager.close_twilio_connection(reason="assistant completed")
        except Exception:
            pass

    def is_goodbye_pending(self) -> bool:
        return self._pending_goodbye

    def mark_goodbye_audio_heard(self, item_id: Optional[str]) -> None:
        if self._pending_goodbye:
            self._goodbye_audio_heard = True
            if item_id and not self._goodbye_item_id:
                self._goodbye_item_id = item_id
            self._cancel_goodbye_watchdog()

    def _start_goodbye_watchdog(self, connection_manager) -> None:
        self._cancel_goodbye_watchdog()
        try:
            timeout = getattr(Config, 'END_CALL_WATCHDOG_SECONDS', 4)
            async def _watch():
                try:
                    await asyncio.sleep(timeout)
                    if self._pending_goodbye and not self._goodbye_audio_heard:
                        Log.info("Goodbye audio not detected in time; finalizing call")
                        await self.finalize_goodbye(connection_manager)
                except Exception:
                    pass
            self._goodbye_watchdog = asyncio.create_task(_watch())
        except Exception:
            self._goodbye_watchdog = None

    def _cancel_goodbye_watchdog(self) -> None:
        if self._goodbye_watchdog and not self._goodbye_watchdog.done():
            self._goodbye_watchdog.cancel()
        self._goodbye_watchdog = None

    # --- TRANSCRIPT EXTRACTION (OpenAI Native for BOTH) ---
    async def extract_caller_transcript(self, event: Dict[str, Any]) -> None:
        """
        Extract CALLER transcript from OpenAI's native transcription.
        Event: conversation.item.input_audio_transcription.completed
        
        ✅ FIX: Filter out noise and ensure proper timing
        """
        try:
            etype = event.get("type", "")
            
            if etype == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript")
                
                if not transcript or not isinstance(transcript, str):
                    return
                
                cleaned = transcript.strip()
                
                # ✅ FIX: Filter out noise using our validator
                if not self.transcript_filter.is_valid_transcript(cleaned, "Caller"):
                    Log.debug(f"[Caller] ❌ Filtered out: '{cleaned}'")
                    return
                
                # ✅ FIX: Ensure sequential timing
                current_time = time.time()
                if current_time < self._last_transcript_time.get("Caller", 0):
                    Log.debug(f"[Caller] ⏭️ Skipped out-of-order transcript")
                    return
                
                self._last_transcript_time["Caller"] = current_time
                
                Log.info(f"[Caller] 📝 {cleaned}")
                
                if self.caller_transcript_callback:
                    await self.caller_transcript_callback({
                        "speaker": "Caller",
                        "text": cleaned,
                        "timestamp": int(current_time * 1000)
                    })
                    
        except Exception as e:
            Log.debug(f"[openai] Caller transcript extract error: {e}")

    async def extract_ai_transcript(self, event: Dict[str, Any]) -> None:
        """
        Extract AI transcript from OpenAI's native response.done event.
        
        ✅ FIX: Ensure proper timing
        """
        try:
            etype = event.get("type", "")
            
            if etype != "response.done":
                return
            
            resp = event.get("response") or {}
            output = resp.get("output") or []
            
            for item in output:
                if not isinstance(item, dict):
                    continue
                    
                if item.get("type") == "message" and item.get("role") == "assistant":
                    content = item.get("content") or []
                    
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                            
                        if c.get("type") == "output_audio":
                            transcript = c.get("transcript")
                            
                            if not transcript or not isinstance(transcript, str):
                                continue
                            
                            cleaned = transcript.strip()
                            
                            if not cleaned:
                                continue
                            
                            # ✅ FIX: Ensure sequential timing
                            current_time = time.time()
                            if current_time < self._last_transcript_time.get("AI", 0):
                                Log.debug(f"[AI] ⏭️ Skipped out-of-order transcript")
                                return
                            
                            self._last_transcript_time["AI"] = current_time
                            
                            Log.info(f"[AI] 📝 {cleaned}")
                            
                            if self.ai_transcript_callback:
                                await self.ai_transcript_callback({
                                    "speaker": "AI",
                                    "text": cleaned,
                                    "timestamp": int(current_time * 1000)
                                })
                            
                            return
                                
        except Exception as e:
            Log.debug(f"[openai] AI transcript extract error: {e}")

    # --- AUDIO EVENTS ---
    def extract_audio_response_data(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.event_handler.is_audio_delta_event(event):
            return None
        return {'delta': self.event_handler.extract_audio_delta(event),
                'item_id': self.event_handler.extract_item_id(event)}

    def is_speech_started(self, event: Dict[str, Any]) -> bool:
        return self.event_handler.is_speech_started_event(event)

    # --- INTERRUPTION HANDLING ---
    async def handle_interruption(self, connection_manager, current_timestamp: int, response_start_timestamp: int, last_assistant_item: str) -> None:
        elapsed_time = self.conversation_manager.calculate_truncation_time(current_timestamp, response_start_timestamp)
        if Config.SHOW_TIMING_MATH:
            print(f"Truncating item {last_assistant_item} at {elapsed_time}ms")
        truncate_event = self.conversation_manager.create_truncate_event(last_assistant_item, elapsed_time)
        await connection_manager.send_to_openai(truncate_event)

    def should_process_interruption(self, last_assistant_item: Optional[str], mark_queue: list, response_start_timestamp: Optional[int]) -> bool:
        return self.conversation_manager.should_handle_interruption(last_assistant_item, mark_queue, response_start_timestamp)
