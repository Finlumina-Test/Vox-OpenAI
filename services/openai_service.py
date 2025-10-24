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


class RomanScriptConverter:
    """
    Converts Hindi/Urdu script transcripts to Roman (Latin) script.
    Uses GPT to transliterate while preserving pronunciation.
    """
    
    @staticmethod
    async def convert_to_roman(text: str) -> str:
        """
        Convert Urdu/Hindi script text to Roman script.
        
        If text is already in Roman script, returns as-is.
        If text contains Urdu/Hindi script, converts to Roman.
        """
        try:
            # Check if text contains Urdu/Hindi characters
            has_urdu_hindi = any('\u0600' <= char <= '\u06FF' or '\u0900' <= char <= '\u097F' for char in text)
            
            if not has_urdu_hindi:
                # Already in Roman script
                return text
            
            Log.info(f"[Roman] Converting: {text}")
            
            # Use GPT to transliterate
            import aiohttp
            
            system_prompt = """You are a transliteration expert. Convert Urdu/Hindi script text to Roman (Latin) script while preserving the exact pronunciation.

Rules:
- Write EXACTLY how it sounds in Roman letters
- Do NOT translate meanings
- Keep it phonetically accurate
- Use casual/conversational spelling

Examples:
- آج میں نے برگر کھانا ہے → aaj maine burger khana hai
- دو زنگر برگر دے دینا → do zinger burger de dena
- ٹھیک ہے → theek hai"""

            headers = {
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Transliterate to Roman script: {text}"}
                ],
                "temperature": 0.1,
                "max_tokens": 100
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        roman_text = data['choices'][0]['message']['content'].strip()
                        Log.info(f"[Roman] ✅ Converted to: {roman_text}")
                        return roman_text
                    else:
                        Log.error(f"[Roman] API failed: {resp.status}")
                        return text
                        
        except Exception as e:
            Log.error(f"[Roman] Conversion error: {e}")
            return text


class OpenAISessionManager:
    """
    Configures and initializes OpenAI Realtime API sessions.
    """

    @staticmethod
    def create_session_update() -> Dict[str, Any]:
        """Create a session update message for OpenAI Realtime API."""
        session = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": "gpt-realtime-mini-2025-10-06",
                "output_modalities": ["audio"],

                "audio": {
                    "input": {
                        "format": {"type": "audio/pcmu"},
                        "turn_detection": {"type": "server_vad"},
                        "transcription": {
                            "model": "whisper-1",
                        }
                    },
                    "output": {"format": {"type": "audio/pcmu"}}
                },

                "instructions": (
                    Config.SYSTEM_MESSAGE
                    + "\n\nRespond naturally to customer queries about orders, menu items, and delivery."
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
    
    NOISE_PATTERNS = [
        "thank you",
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
    
    MIN_TRANSCRIPT_LENGTH = 3
    MAX_NOISE_LENGTH = 15
    
    @staticmethod
    def is_valid_transcript(text: str, speaker: str) -> bool:
        """Validate if transcript is real speech or just noise."""
        if not text or not isinstance(text, str):
            return False
        
        cleaned = text.strip().lower()
        
        if len(cleaned) < TranscriptFilter.MIN_TRANSCRIPT_LENGTH:
            return False
        
        if speaker == "AI":
            return True
        
        if len(cleaned) <= TranscriptFilter.MAX_NOISE_LENGTH:
            for pattern in TranscriptFilter.NOISE_PATTERNS:
                if cleaned == pattern or cleaned.startswith(pattern + " ") or cleaned.endswith(" " + pattern):
                    Log.debug(f"[Filter] Rejected noise: '{text}'")
                    return False
        
        return True


class OpenAIService:
    """
    Unified service for OpenAI Realtime API with human takeover support.
    """

    def __init__(self):
        self.session_manager = OpenAISessionManager()
        self.conversation_manager = OpenAIConversationManager()
        self.event_handler = OpenAIEventHandler()
        self.transcript_filter = TranscriptFilter()
        self.roman_converter = RomanScriptConverter()
        self._pending_tool_calls: Dict[str, Dict[str, Any]] = {}
        self._pending_goodbye: bool = False
        self._goodbye_audio_heard: bool = False
        self._goodbye_item_id: Optional[str] = None
        self._goodbye_watchdog: Optional[asyncio.Task] = None
        
        # Callbacks
        self.caller_transcript_callback: Optional[callable] = None
        self.ai_transcript_callback: Optional[callable] = None
        
        # Track last transcript timestamp
        self._last_transcript_time: Dict[str, float] = {"Caller": 0, "AI": 0, "Human": 0}
        
        # ✅ NEW: Human takeover state
        self.human_takeover_active: bool = False
        self.human_audio_callback: Optional[callable] = None

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

    # --- HUMAN TAKEOVER ---
    def enable_human_takeover(self) -> None:
        """Enable human takeover mode - AI stops responding."""
        self.human_takeover_active = True
        Log.info("🎤 Human takeover ENABLED")
    
    def disable_human_takeover(self) -> None:
        """Disable human takeover mode - AI resumes."""
        self.human_takeover_active = False
        Log.info("🤖 Human takeover DISABLED - AI resumed")
    
    def is_human_in_control(self) -> bool:
        """Check if human is currently controlling the call."""
        return self.human_takeover_active
    
    async def send_human_audio_to_openai(self, audio_base64: str, connection_manager) -> None:
        """
        Send human agent's audio to OpenAI (so it gets transcribed and processed).
        """
        try:
            if not self.human_takeover_active:
                Log.debug("[Human] Takeover not active, ignoring audio")
                return
            
            # Send audio to OpenAI's input buffer
            audio_message = {
                "type": "input_audio_buffer.append",
                "audio": audio_base64
            }
            await connection_manager.send_to_openai(audio_message)
            
        except Exception as e:
            Log.error(f"[Human] Audio send error: {e}")

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

    # --- TRANSCRIPT EXTRACTION WITH ROMAN CONVERSION ---
    async def extract_caller_transcript(self, event: Dict[str, Any]) -> None:
        """
        Extract CALLER transcript and convert to Roman script if needed.
        """
        try:
            etype = event.get("type", "")
            
            if etype == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript")
                
                if not transcript or not isinstance(transcript, str):
                    return
                
                cleaned = transcript.strip()
                
                if not cleaned:
                    return
                
                # ✅ Convert to Roman script if needed
                roman_text = await self.roman_converter.convert_to_roman(cleaned)
                
                # Filter noise
                if not self.transcript_filter.is_valid_transcript(roman_text, "Caller"):
                    Log.debug(f"[Caller] ❌ Filtered: '{roman_text}'")
                    return
                
                # Ensure sequential timing
                current_time = time.time()
                if current_time < self._last_transcript_time.get("Caller", 0):
                    Log.debug(f"[Caller] ⏭️ Out-of-order")
                    return
                
                self._last_transcript_time["Caller"] = current_time
                
                Log.info(f"[Caller] 📝 {roman_text}")
                
                if self.caller_transcript_callback:
                    await self.caller_transcript_callback({
                        "speaker": "Caller",
                        "text": roman_text,
                        "timestamp": int(current_time * 1000)
                    })
                    
        except Exception as e:
            Log.error(f"[Caller] Transcript error: {e}")

    async def extract_ai_transcript(self, event: Dict[str, Any]) -> None:
        """Extract AI transcript from response.done event."""
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
                            
                            current_time = time.time()
                            if current_time < self._last_transcript_time.get("AI", 0):
                                Log.debug(f"[AI] ⏭️ Out-of-order")
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
            Log.error(f"[AI] Transcript error: {e}")

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
