import json
import asyncio
import websockets
from typing import Optional, Callable, Awaitable
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from config import Config


class ConnectionState:
    """
    Tracks the current Twilio stream session and manages connection state.
    
    - Holds the current stream SID for the active Twilio media stream session.
    - Provides methods to reset or clear state when a new stream starts or an interruption occurs.
    
    Audio-related state is now managed by AudioService, so this class focuses only on connection/session state.
    """
    
    def __init__(self):
        self.stream_sid: Optional[str] = None
    
    def reset_stream_state(self) -> None:
        """Reset state when a new stream starts."""
        # Audio-related state is now managed by AudioService
        pass
    
    def clear_response_state(self) -> None:
        """Clear response-related state during interruptions."""
        # Audio-related state is now managed by AudioService
        pass


class WebSocketConnectionManager:
    """
    Orchestrates all WebSocket communication between Twilio and OpenAI for the application.
    
    - Establishes, maintains, and closes WebSocket connections to both Twilio (FastAPI) and OpenAI (websockets).
    - Routes incoming messages to appropriate event handlers for media, start, mark, audio delta, and speech events.
    - Sends and receives messages, manages connection state, and coordinates with AudioService for mark/clear events.
    
    This is the main interface for real-time, bidirectional communication and event-driven processing between the two services.
    """
    
    def __init__(self, twilio_ws: WebSocket):
        self.twilio_ws = twilio_ws
        self.openai_ws: Optional[websockets.WebSocketServerProtocol] = None
        self.state = ConnectionState()
        self._is_connected = False
    
    async def connect_to_openai(self) -> None:
        """Establish connection to OpenAI WebSocket API."""
        try:
            self.openai_ws = await websockets.connect(
                Config.get_openai_websocket_url(),
                additional_headers=Config.get_openai_headers()
            )
            self._is_connected = True
            print("Connected to OpenAI WebSocket API")
        except Exception as e:
            print(f"Failed to connect to OpenAI: {e}")
            raise
    
    async def close_openai_connection(self) -> None:
        """Close the OpenAI WebSocket connection."""
        if self.openai_ws and self._is_connected:
            await self.openai_ws.close()
            self._is_connected = False
            print("Closed OpenAI WebSocket connection")
    
    async def send_to_openai(self, message: dict) -> None:
        """Send a message to OpenAI WebSocket."""
        if self.openai_ws and self._is_connected:
            await self.openai_ws.send(json.dumps(message))
        else:
            raise ConnectionError("OpenAI WebSocket is not connected")
    
    async def send_to_twilio(self, message: dict) -> None:
        """Send a message to Twilio WebSocket."""
        await self.twilio_ws.send_json(message)
    
    async def receive_from_twilio(
        self, 
        on_media: Callable[[dict], Awaitable[None]],
        on_start: Callable[[str], Awaitable[None]],
        on_mark: Callable[[], Awaitable[None]]
    ) -> None:
        """
        Receive messages from Twilio and route them to appropriate handlers.
        
        Args:
            on_media: Handler for media events
            on_start: Handler for stream start events
            on_mark: Handler for mark events
        """
        try:
            async for message in self.twilio_ws.iter_text():
                data = json.loads(message)
                
                if data['event'] == 'media':
                    await on_media(data)
                elif data['event'] == 'start':
                    stream_sid = data['start']['streamSid']
                    self.state.stream_sid = stream_sid
                    self.state.reset_stream_state()
                    await on_start(stream_sid)
                elif data['event'] == 'mark':
                    await on_mark()
                    
        except WebSocketDisconnect:
            print("Twilio WebSocket disconnected")
            await self.close_openai_connection()
            raise
    
    async def receive_from_openai(
        self,
        on_audio_delta: Callable[[dict], Awaitable[None]],
        on_speech_started: Callable[[], Awaitable[None]],
        on_other_event: Callable[[dict], Awaitable[None]]
    ) -> None:
        """
        Receive messages from OpenAI and route them to appropriate handlers.
        
        Args:
            on_audio_delta: Handler for audio delta events
            on_speech_started: Handler for speech started events  
            on_other_event: Handler for other events
        """
        if not self.openai_ws or not self._is_connected:
            raise ConnectionError("OpenAI WebSocket is not connected")
        
        try:
            # Import here to avoid circular imports
            from services.openai_service import OpenAIEventHandler
            
            async for openai_message in self.openai_ws:
                response = json.loads(openai_message)
                
                if OpenAIEventHandler.is_audio_delta_event(response):
                    await on_audio_delta(response)
                elif OpenAIEventHandler.is_speech_started_event(response):
                    await on_speech_started()
                else:
                    await on_other_event(response)
                    
        except Exception as e:
            print(f"Error receiving from OpenAI: {e}")
            raise
    
    async def send_mark_to_twilio(self) -> None:
        """Send a mark event to Twilio using AudioService."""
        if self.state.stream_sid:
            # Import here to avoid circular imports
            from services.audio_service import AudioService
            audio_service = AudioService()
            mark_event = audio_service.create_mark_message(self.state.stream_sid)
            await self.send_to_twilio(mark_event)
    
    async def clear_twilio_audio(self) -> None:
        """Clear audio buffer in Twilio using AudioService."""
        if self.state.stream_sid:
            # Import here to avoid circular imports
            from services.audio_service import AudioService
            audio_service = AudioService()
            clear_event = audio_service.create_clear_message(self.state.stream_sid)
            await self.send_to_twilio(clear_event)
    
    def is_openai_connected(self) -> bool:
        """Check if OpenAI WebSocket is connected and ready."""
        return (self.openai_ws is not None and 
                self._is_connected and 
                self.openai_ws.state.name == 'OPEN')
