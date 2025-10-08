import json
import asyncio
from typing import Optional, Dict, Any
from config import Config
from services.log_utils import Log


class OpenAIEventHandler:
    """
    Interprets and processes events received from the OpenAI Realtime API.
    
    - Determines which events should be logged.
    - Identifies and extracts audio deltas, speech start events, and item IDs from event payloads.
    
    Used by higher-level services to make sense of incoming OpenAI events and route them appropriately.
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


class OpenAISessionManager:
    """
    Configures and initializes OpenAI Realtime API sessions.
    
    - Generates session update messages specifying model, audio formats, and system instructions.
    - Creates the initial conversation item (for AI-first greetings) and triggers responses.
    
    Ensures consistent and correct session setup for all OpenAI interactions.
    """
    
    @staticmethod
    def create_session_update() -> Dict[str, Any]:
        """
        Create a session update message for OpenAI Realtime API.
        
        Returns:
            Dictionary containing session configuration
        """
        session = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": "gpt-realtime",
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcmu"},
                        "turn_detection": {"type": "server_vad"}
                    },
                    "output": {
                        "format": {"type": "audio/pcmu"}
                    }
                },
                "instructions": Config.SYSTEM_MESSAGE,
                "tools": [
                    {
                        "type": "function",
                        "name": "end_call",
                        "description": "Politely end the phone call when the caller says goodbye or requests to end the conversation.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reason": {"type": "string", "description": "Brief reason for ending, e.g., user said bye."}
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
        """
        Create an initial conversation item for AI-first interactions.
        
        Returns:
            Dictionary containing initial conversation setup
        """
        return {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Greet the user with 'Hello there! I am an AI voice assistant powered by Twilio and the OpenAI Realtime API. You can ask me for facts, jokes, or anything you can imagine. How can I help you?'"
                    }
                ]
            }
        }
    
    @staticmethod
    def create_response_trigger() -> Dict[str, Any]:
        """
        Create a response trigger message.
        
        Returns:
            Dictionary to trigger OpenAI response generation
        """
        return {"type": "response.create"}


class OpenAIConversationManager:
    """
    Manages conversation flow and interruption logic for OpenAI sessions.
    
    - Creates truncation events to interrupt/cut off ongoing AI responses.
    - Determines when interruptions should be processed based on marks and timing.
    - Calculates elapsed time for precise truncation.
    
    Used by the main service to support real-time, interactive voice experiences.
    """
    
    @staticmethod
    def create_truncate_event(item_id: str, audio_end_ms: int) -> Dict[str, Any]:
        """
        Create a conversation item truncation event.
        
        Args:
            item_id: ID of the item to truncate
            audio_end_ms: Timestamp where to truncate the audio
            
        Returns:
            Dictionary containing truncation command
        """
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
        """
        Determine if an interruption should be processed.
        
        Args:
            last_assistant_item: ID of the last assistant response
            mark_queue: Queue of pending marks
            response_start_timestamp: When the current response started
            
        Returns:
            True if interruption should be handled
        """
        return (last_assistant_item is not None and 
                len(mark_queue) > 0 and 
                response_start_timestamp is not None)
    
    @staticmethod
    def calculate_truncation_time(
        current_timestamp: int,
        response_start_timestamp: int
    ) -> int:
        """
        Calculate the elapsed time for audio truncation.
        
        Args:
            current_timestamp: Current media timestamp
            response_start_timestamp: When the response started
            
        Returns:
            Elapsed time in milliseconds
        """
        return current_timestamp - response_start_timestamp


class OpenAIService:
    """
    Main service layer for all OpenAI Realtime API operations in the application.
    
    - Composes the event handler, session manager, and conversation manager.
    - Provides high-level methods to initialize sessions, send greetings, process/log events, extract audio, and handle interruptions.
    
    This is the primary interface for the rest of the application to interact with OpenAI, abstracting away lower-level event and session management details.
    """
    
    def __init__(self):
        self.session_manager = OpenAISessionManager()
        self.conversation_manager = OpenAIConversationManager()
        self.event_handler = OpenAIEventHandler()
        self._pending_tool_calls: Dict[str, Dict[str, Any]] = {}
        self._pending_goodbye: bool = False
        self._goodbye_audio_heard: bool = False
        self._goodbye_item_id: Optional[str] = None
        self._goodbye_watchdog: Optional[asyncio.Task] = None
    
    async def initialize_session(self, connection_manager) -> None:
        """
        Initialize OpenAI session with proper configuration.
        
        Args:
            connection_manager: WebSocket connection manager
        """
        session_update = self.session_manager.create_session_update()
        Log.json('Sending session update', session_update)
        await connection_manager.send_to_openai(session_update)
    
    async def send_initial_greeting(self, connection_manager) -> None:
        """
        Send initial conversation item to make AI greet first.
        
        Args:
            connection_manager: WebSocket connection manager
        """
        initial_item = self.session_manager.create_initial_conversation_item()
        response_trigger = self.session_manager.create_response_trigger()
        
        await connection_manager.send_to_openai(initial_item)
        await connection_manager.send_to_openai(response_trigger)
    
    def process_event_for_logging(self, event: Dict[str, Any]) -> None:
        """
        Process OpenAI event for logging if needed.
        
        Args:
            event: OpenAI event data
        """
        if self.event_handler.should_log_event(event.get('type', '')):
            Log.event(f"Received event: {event['type']}", event)

    def is_tool_call(self, event: Dict[str, Any]) -> bool:
        """Return True if the event is a tool call from the model."""
        etype = event.get('type')
        if etype in ('response.function_call.arguments.delta', 'response.function_call.completed'):
            return True
        # Also detect tool/function calls embedded in response.done payloads
        if etype == 'response.done':
            resp = event.get('response') or {}
            output = resp.get('output') or []
            for item in output:
                if isinstance(item, dict) and item.get('type') == 'function_call':
                    return True
        return False

    def accumulate_tool_call(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Accumulate streamed tool call arguments until completion.
        Returns the completed call payload when finished.
        """
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
        # Handle non-streamed function calls embedded in response.done
        if etype == 'response.done':
            resp = event.get('response') or {}
            output = resp.get('output') or []
            for item in output:
                if isinstance(item, dict) and item.get('type') == 'function_call':
                    name = item.get('name')
                    raw_args = item.get('arguments')
                    args: Dict[str, Any]
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        args = {"_raw": raw_args}
                    return {"name": name, "arguments": args}
        return None

    async def maybe_handle_tool_call(self, connection_manager, tool_call: Dict[str, Any]) -> bool:
        """
        Handle supported tool calls. Returns True if a tool was handled.
        Currently supports: end_call
        """
        if not tool_call:
            return False
        name = tool_call.get('name')
        if name != 'end_call':
            return False

        # 1) Ask the assistant to speak a short farewell before ending.
        args = tool_call.get('arguments') or {}
        reason = args.get('reason') if isinstance(args, dict) else None
        farewell = Config.build_end_call_farewell(reason)

        # Ignore duplicate end_call requests while a goodbye is pending
        if self._pending_goodbye:
            Log.info("End-call already pending; ignoring duplicate request")
            # Return False so other handlers (e.g., finalize check) can proceed on this event
            return False

        Log.info("Queueing farewell response before hangup")
        await self._send_goodbye_response(connection_manager, farewell)
        self._pending_goodbye = True
        self._goodbye_audio_heard = False
        # Clear any previous tracked item id; it will be set when we first hear the goodbye audio
        self._goodbye_item_id = None
        # Start a watchdog to avoid stalling if no audio arrives
        self._start_goodbye_watchdog(connection_manager)
        return True

    async def _send_goodbye_response(self, connection_manager, text: str) -> None:
        """Send a final assistant response (audio) with the provided text before hangup.
        Uses response.create with inline instructions so the model speaks immediately without tool calls.
        """
        try:
            # Note: Recent Realtime API versions expect instructions at the top level
            # of the response.create event. Modalities are already defined in the
            # session (output_modalities=["audio"]). Sending a nested
            # response.modalities triggers an 'unknown_parameter' error.
            await connection_manager.send_to_openai({
                "type": "response.create",
                "response": {
                    "instructions": text
                }
            })
        except Exception as e:
            # If we fail to queue a goodbye, fall back to immediate hangup on next finalize
            Log.error(f"Failed to queue goodbye response: {e}")
            self._pending_goodbye = True
            self._goodbye_audio_heard = False

    def should_finalize_on_event(self, event: Dict[str, Any]) -> bool:
        """Return True if we should finalize hangup after the goodbye audio has completed."""
        if not (self._pending_goodbye and self._goodbye_audio_heard):
            return False
        etype = event.get('type')
        # Primary: finalize when the output_audio stream indicates done (most reliable)
        if etype == 'response.output_audio.done':
            return True
        # Secondary: finalize on response.done for the goodbye message if we can match IDs
        if etype == 'response.done':
            if not self._goodbye_item_id:
                # Fallback: if we can't match IDs, but the response contains an assistant message with audio, allow finalize
                resp = event.get('response') or {}
                for item in (resp.get('output') or []):
                    if isinstance(item, dict) and item.get('type') == 'message' and item.get('role') == 'assistant':
                        for c in (item.get('content') or []):
                            if isinstance(c, dict) and c.get('type') == 'output_audio':
                                return True
                return False
            # If we do have a tracked item id, try to match it to the output item id
            resp = event.get('response') or {}
            for item in (resp.get('output') or []):
                if isinstance(item, dict) and item.get('id') == self._goodbye_item_id:
                    return True
        return False

    async def finalize_goodbye(self, connection_manager) -> None:
        """After goodbye audio is finished, clear/close and optionally complete the call via REST."""
        self._pending_goodbye = False
        self._goodbye_audio_heard = False
        self._goodbye_item_id = None
        self._cancel_goodbye_watchdog()
        # Small grace to allow final frames to play to the caller
        try:
            Log.info(f"Grace sleep before hangup: {getattr(Config, 'END_CALL_GRACE_SECONDS', 0.5)}s")
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
        # Always attempt to close the Twilio WS as a fallback; this ends the stream
        try:
            await connection_manager.close_twilio_connection(reason="assistant completed")
        except Exception:
            pass

    def is_goodbye_pending(self) -> bool:
        """Return True if a farewell has been queued and we await its completion."""
        return self._pending_goodbye

    def mark_goodbye_audio_heard(self, item_id: Optional[str]) -> None:
        """Mark that we've begun receiving audio for the goodbye message and capture its item_id."""
        if self._pending_goodbye:
            self._goodbye_audio_heard = True
            if item_id and not self._goodbye_item_id:
                self._goodbye_item_id = item_id
            # Once audio is heard, watchdog is no longer needed
            self._cancel_goodbye_watchdog()

    def _start_goodbye_watchdog(self, connection_manager) -> None:
        """Start a watchdog that finalizes the call if no goodbye audio starts in time."""
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
    
class OpenAIService:
    """
    Main service layer for all OpenAI Realtime API operations.
    Handles session initialization, greetings, event processing, transcripts,
    audio responses, tool calls, interruptions, and goodbyes.
    """

    def __init__(self):
        self.session_manager = OpenAISessionManager()
        self.conversation_manager = OpenAIConversationManager()
        self.event_handler = OpenAIEventHandler()
        self._pending_tool_calls: Dict[str, Dict[str, Any]] = {}
        self._pending_goodbye: bool = False
        self._goodbye_audio_heard: bool = False
        self._goodbye_item_id: Optional[str] = None
        self._goodbye_watchdog: Optional[asyncio.Task] = None

    # --- SESSION & GREETING ---

    async def initialize_session(self, connection_manager) -> None:
        """Initialize OpenAI session with proper configuration."""
        session_update = self.session_manager.create_session_update()
        Log.json('Sending session update', session_update)
        await connection_manager.send_to_openai(session_update)

    async def send_initial_greeting(self, connection_manager) -> None:
        """Send initial greeting conversation item."""
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

    # --- AUDIO & TRANSCRIPT ---

    def extract_audio_response_data(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.event_handler.is_audio_delta_event(event):
            return None
        return {'delta': self.event_handler.extract_audio_delta(event),
                'item_id': self.event_handler.extract_item_id(event)}

    def is_speech_started(self, event: Dict[str, Any]) -> bool:
        return self.event_handler.is_speech_started_event(event)

    def extract_transcript_text(self, event: Dict[str, Any]) -> Optional[str]:
        try:
            etype = event.get("type", "")
            Log.debug(f"[openai] Received event type for transcript extraction: {etype}")
            if etype in ("response.output_text.delta", "response.output_text.delta.text"):
                delta = event.get("delta")
                if isinstance(delta, str):
                    return delta
                if isinstance(delta, dict):
                    return delta.get("text") or delta.get("value") or None
            if etype == "response.done":
                resp = event.get("response") or {}
                for item in (resp.get("output") or []):
                    if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "assistant":
                        for c in (item.get("content") or []):
                            if isinstance(c, dict) and c.get("type") == "output_text":
                                txt = c.get("text") or c.get("value")
                                if isinstance(txt, str):
                                    return txt
            if isinstance(event.get("text"), str):
                return event.get("text")
        except Exception as e:
            Log.debug("[openai] transcript extract error", e)
        return None

    # --- INTERRUPTION HANDLING ---

    async def handle_interruption(
        self,
        connection_manager,
        current_timestamp: int,
        response_start_timestamp: int,
        last_assistant_item: str
    ) -> None:
        elapsed_time = self.conversation_manager.calculate_truncation_time(
            current_timestamp, response_start_timestamp
        )
        if Config.SHOW_TIMING_MATH:
            print(f"Calculating elapsed time for truncation: {current_timestamp} - {response_start_timestamp} = {elapsed_time}ms")
            print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")
        truncate_event = self.conversation_manager.create_truncate_event(
            last_assistant_item, elapsed_time
        )
        await connection_manager.send_to_openai(truncate_event)

    def should_process_interruption(
        self,
        last_assistant_item: Optional[str],
        mark_queue: list,
        response_start_timestamp: Optional[int]
    ) -> bool:
        return self.conversation_manager.should_handle_interruption(
            last_assistant_item, mark_queue, response_start_timestamp
        )
