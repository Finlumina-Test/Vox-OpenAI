import json
from typing import Optional, Dict, Any
from config import Config


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
        return {
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
            }
        }
    
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
    
    async def initialize_session(self, connection_manager) -> None:
        """
        Initialize OpenAI session with proper configuration.
        
        Args:
            connection_manager: WebSocket connection manager
        """
        session_update = self.session_manager.create_session_update()
        print('Sending session update:', json.dumps(session_update))
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
            print(f"Received event: {event['type']}", event)
    
    def extract_audio_response_data(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract relevant data from OpenAI audio response.
        
        Args:
            event: OpenAI event data
            
        Returns:
            Dictionary with audio delta and item ID, or None
        """
        if not self.event_handler.is_audio_delta_event(event):
            return None
            
        return {
            'delta': self.event_handler.extract_audio_delta(event),
            'item_id': self.event_handler.extract_item_id(event)
        }
    
    def is_speech_started(self, event: Dict[str, Any]) -> bool:
        """
        Check if event indicates user speech started (interruption).
        
        Args:
            event: OpenAI event data
            
        Returns:
            True if speech started
        """
        return self.event_handler.is_speech_started_event(event)
    
    async def handle_interruption(
        self,
        connection_manager,
        current_timestamp: int,
        response_start_timestamp: int,
        last_assistant_item: str
    ) -> None:
        """
        Handle conversation interruption by truncating the current response.
        
        Args:
            connection_manager: WebSocket connection manager
            current_timestamp: Current media timestamp
            response_start_timestamp: When the response started
            last_assistant_item: ID of the item to truncate
        """
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
        """
        Determine if an interruption should be processed.
        
        Args:
            last_assistant_item: ID of the last assistant response
            mark_queue: Queue of pending marks
            response_start_timestamp: When the current response started
            
        Returns:
            True if interruption should be handled
        """
        return self.conversation_manager.should_handle_interruption(
            last_assistant_item, mark_queue, response_start_timestamp
        )
