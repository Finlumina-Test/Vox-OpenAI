from typing import Dict, Any, Optional
import asyncio
import json
from services.log_utils import Log
from services.openai_session_manager import OpenAISessionManager
from services.openai_conversation_manager import OpenAIConversationManager
from services.openai_event_handler import OpenAIEventHandler
from config import Config

class OpenAIEventHandler:
    """Interprets and processes events received from the OpenAI Realtime API."""

    @staticmethod
    def should_log_event(event_type: str) -> bool:
        return event_type in Config.LOG_EVENT_TYPES

    @staticmethod
    def is_audio_delta_event(event: Dict[str, Any]) -> bool:
        return event.get('type') == 'response.output_audio.delta' and 'delta' in event

    @staticmethod
    def is_speech_started_event(event: Dict[str, Any]) -> bool:
        return event.get('type') == 'input_audio_buffer.speech_started'

    @staticmethod
    def extract_audio_delta(event: Dict[str, Any]) -> Optional[str]:
        if OpenAIEventHandler.is_audio_delta_event(event):
            return event.get('delta')
        return None

    @staticmethod
    def extract_item_id(event: Dict[str, Any]) -> Optional[str]:
        return event.get('item_id')


class OpenAISessionManager:
    @staticmethod
    def create_session_update() -> Dict[str, Any]:
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
                    "output": {"format": {"type": "audio/pcmu"}}
                },
                "instructions": Config.SYSTEM_MESSAGE,
                "tools": [
                    {
                        "type": "function",
                        "name": "end_call",
                        "description": "Politely end the phone call when the caller says goodbye or requests to end the conversation.",
                        "parameters": {
                            "type": "object",
                            "properties": {"reason": {"type": "string", "description": "Brief reason for ending, e.g., user said bye."}},
                            "required": []
                        }
                    }
                ]
            }
        }
        return session

    @staticmethod
    def create_initial_conversation_item() -> Dict[str, Any]:
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
        return {"type": "response.create"}


class OpenAIConversationManager:
    @staticmethod
    def create_truncate_event(item_id: str, audio_end_ms: int) -> Dict[str, Any]:
        return {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": audio_end_ms
        }

    @staticmethod
    def should_handle_interruption(last_assistant_item: Optional[str], mark_queue: list, response_start_timestamp: Optional[int]) -> bool:
        return last_assistant_item is not None and len(mark_queue) > 0 and response_start_timestamp is not None

    @staticmethod
    def calculate_truncation_time(current_timestamp: int, response_start_timestamp: int) -> int:
        return current_timestamp - response_start_timestamp


class OpenAIService:
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
        session_update = self.session_manager.create_session_update()
        Log.json('Sending session update', session_update)
        await connection_manager.send_to_openai(session_update)

    async def send_initial_greeting(self, connection_manager) -> None:
        initial_item = self.session_manager.create_initial_conversation_item()
        response_trigger = self.session_manager.create_response_trigger()
        await connection_manager.send_to_openai(initial_item)
        await connection_manager.send_to_openai(response_trigger)

    def process_event_for_logging(self, event: Dict[str, Any]) -> None:
        if self.event_handler.should_log_event(event.get('type', '')):
            Log.event(f"Received event: {event['type']}", event)

    def is_tool_call(self, event: Dict[str, Any]) -> bool:
        etype = event.get('type')
        if etype in ('response.function_call.arguments.delta', 'response.function_call.completed'):
            return True
        if etype == 'response.done':
            resp = event.get('response') or {}
            for item in (resp.get('output') or []):
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
            for item in (resp.get('output') or []):
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
            await connection_manager.send_to_openai({"type": "response.create", "response": {"instructions": text}})
        except Exception as e:
            Log.error(f"Failed to queue goodbye response: {e}")
            self._pending_goodbye = True
            self._goodbye_audio_heard = False

    def should_finalize_on_event(self, event: Dict[str, Any]) -> bool:
        if not (self._pending_goodbye and self._goodbye_audio_heard):
            return False
        etype = event.get('type')
        if etype in ('response.output_audio.done', 'response.done'):
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
            if self._goodbye_watchdog and not self._goodbye_watchdog.done():
                self._goodbye_watchdog.cancel()

    def _start_goodbye_watchdog(self, connection_manager) -> None:
        async def watchdog():
            try:
                await asyncio.sleep(Config.GOODBYE_TIMEOUT_SECONDS)
                if self._pending_goodbye:
                    Log.info("Goodbye timeout reached; forcing finalization")
                    await self.finalize_goodbye(connection_manager)
            except asyncio.CancelledError:
                pass

        self._goodbye_watchdog = asyncio.create_task(watchdog())

    def _cancel_goodbye_watchdog(self) -> None:
        if self._goodbye_watchdog and not self._goodbye_watchdog.done():
            self._goodbye_watchdog.cancel()
            self._goodbye_watchdog = None

    # ---------- NEW METHODS FOR TRANSCRIPT & AUDIO ----------
    def extract_transcript_text(self, event: Dict[str, Any]) -> Optional[str]:
        """Extract textual transcript from OpenAI realtime events."""
        try:
            etype = event.get("type", "")
            Log.debug(f"[openai] Received event type: {etype}")

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

    def extract_audio_response_data(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract audio delta info from OpenAI realtime events."""
        try:
            if not self.event_handler.is_audio_delta_event(event):
                return None
            return {
                "delta": self.event_handler.extract_audio_delta(event),
                "item_id": self.event_handler.extract_item_id(event)
            }
        except Exception as e:
            Log.debug(f"[openai] extract_audio_response_data error: {e}")
            return None
