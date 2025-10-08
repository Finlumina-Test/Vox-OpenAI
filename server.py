import os
import json
import base64
import asyncio
import websockets
from typing import Set
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream

# Import our centralized configuration and organized services
from config import Config
from services import (
    WebSocketConnectionManager,
    TwilioService, 
    TwilioAudioProcessor,
    OpenAIService,
    AudioService
)
from services.log_utils import Log

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or replace * with your dashboard domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Store connected dashboard clients
dashboard_clients: Set[WebSocket] = set()

@app.websocket("/dashboard-stream")
async def dashboard_stream(websocket: WebSocket):
    """
    WebSocket endpoint for dashboards to receive live transcripts.
    Supports optional callId filtering via query param or first message from client.
    """
    await websocket.accept()
    
    DASHBOARD_TOKEN = os.getenv("DASHBOARD_TOKEN")
    client_call_id = None  # optional filter per dashboard client

    # Token validation
    if DASHBOARD_TOKEN:
        provided = websocket.query_params.get("token") or websocket.headers.get("x-dashboard-token")
        if provided != DASHBOARD_TOKEN:
            await websocket.close(code=4003)
            return

    dashboard_clients.add(websocket)

    try:
        # Optional: wait for first message to get callId
        try:
            msg = await asyncio.wait_for(websocket.receive_text(), timeout=5)
            data = json.loads(msg)
            client_call_id = data.get("callId")  # filter transcripts for this callId only
        except (asyncio.TimeoutError, json.JSONDecodeError, KeyError):
            client_call_id = None

        while True:
            # Keep the connection alive; receive dummy messages or ping every 20s
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=20)
            except asyncio.TimeoutError:
                # send ping
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        dashboard_clients.discard(websocket)


async def broadcast_transcript(transcript_obj: dict):
    """
    Broadcast a new transcript to all connected dashboard clients.
    transcript_obj: {
        "callSid": "...",
        "speaker": "User" | "AI",
        "text": "...",
        "timestamp": "..."
    }
    """
    payload = json.dumps(transcript_obj)
    for client in list(dashboard_clients):
        try:
            # If client has filtered by callId, skip unrelated messages
            if hasattr(client, "call_id") and client.call_id:
                if transcript_obj.get("callSid") != client.call_id:
                    continue
            await client.send_text(payload)
        except Exception:
            dashboard_clients.discard(client)



@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    return TwilioService.create_incoming_call_response(request)


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    Log.header("Client connected")
    await websocket.accept()

    # Create connection manager and services
    connection_manager = WebSocketConnectionManager(websocket)
    openai_service = OpenAIService()
    audio_service = AudioService()
    
    try:
        # Connect to OpenAI and initialize session
        await connection_manager.connect_to_openai()
        await openai_service.initialize_session(connection_manager)

        # Define event handlers for cleaner separation of concerns
        async def handle_media_event(data: dict) -> None:
            """Handle incoming media data from Twilio."""
            if connection_manager.is_openai_connected():
                audio_message = audio_service.process_incoming_audio(data)
                if audio_message:
                    await connection_manager.send_to_openai(audio_message)

            # --- Optional: Broadcast caller transcript if speech-to-text added later ---
            if "text" in data:
                payload_obj = {
                    "id": data.get("id") or str(int(asyncio.get_event_loop().time() * 1000)),
                    "callSid": getattr(connection_manager.state, "call_sid", None),
                    "streamSid": getattr(connection_manager.state, "stream_sid", None),
                    "speaker": "User",
                    "text": data["text"],
                    "timestamp": data.get("timestamp") or (asyncio.get_event_loop().time())
                }
                payload = json.dumps(payload_obj)
                for client in list(dashboard_clients):
                    try:
                        await client.send_text(payload)
                    except Exception:
                        dashboard_clients.discard(client)
            # ---------------------------------------------------------------------------

        async def handle_stream_start(stream_sid: str) -> None:
            """Handle stream start event."""
            Log.event("Twilio stream started", {"streamSid": stream_sid})

        async def handle_mark_event() -> None:
            """Handle mark event from Twilio."""
            audio_service.handle_mark_event()

        async def handle_audio_delta(response: dict) -> None:
            """Handle audio delta from OpenAI."""
            audio_data = openai_service.extract_audio_response_data(response)
            
            # --- Broadcast transcript updates to dashboard ---
            transcript_text = None
            # use extractor if provided by OpenAI service
            if hasattr(openai_service, "extract_transcript_text"):
                try:
                    transcript_text = openai_service.extract_transcript_text(response)
                except Exception:
                    transcript_text = None

            if transcript_text:
                payload_obj = {
                    "id": response.get("id") or str(int(asyncio.get_event_loop().time() * 1000)),
                    "callSid": getattr(connection_manager.state, "call_sid", None),
                    "streamSid": getattr(connection_manager.state, "stream_sid", None),
                    "speaker": "AI",
                    "text": transcript_text,
                    "timestamp": response.get("timestamp") or (asyncio.get_event_loop().time())
                }
                payload = json.dumps(payload_obj)
                for client in list(dashboard_clients):
                    try:
                        await client.send_text(payload)
                    except Exception:
                        dashboard_clients.discard(client)
            # -------------------------------------------------

            if audio_data and connection_manager.state.stream_sid:
                # If we're in a goodbye flow, mark that farewell audio has started and capture its item_id
                if openai_service.is_goodbye_pending():
                    openai_service.mark_goodbye_audio_heard(audio_data.get('item_id'))
                audio_message = audio_service.process_outgoing_audio(
                    response, 
                    connection_manager.state.stream_sid
                )
                if audio_message:
                    await connection_manager.send_to_twilio(audio_message)
                    
                    # Send mark for synchronization
                    mark_message = audio_service.create_mark_message(connection_manager.state.stream_sid)
                    await connection_manager.send_to_twilio(mark_message)

        async def handle_speech_started() -> None:
            """Handle speech started event (interruption)."""
            Log.info("Speech started detected.")
            # Do not interrupt the assistant's final goodbye
            if openai_service.is_goodbye_pending():
                Log.info("Ignoring interruption during goodbye flow")
                return
            current_item_id = audio_service.get_current_item_id()
            if current_item_id:
                Log.info(f"Interrupting response with id: {current_item_id}")
                await handle_speech_started_event(connection_manager, openai_service, audio_service)

        async def handle_other_openai_event(response: dict) -> None:
            """Handle other OpenAI events."""
            # Log events
            openai_service.process_event_for_logging(response)
            # Handle tool calls (e.g., end_call)
            if openai_service.is_tool_call(response):
                tool_call = openai_service.accumulate_tool_call(response)
                if tool_call:
                    handled = await openai_service.maybe_handle_tool_call(connection_manager, tool_call)
                    if handled:
                        return
            # If a goodbye was queued and we've heard its audio, finalize after the response completes
            if openai_service.should_finalize_on_event(response):
                await openai_service.finalize_goodbye(connection_manager)

        # Run Twilio receiver and OpenAI receiver; plus a renewal loop for OpenAI session
        async def openai_receiver():
            await connection_manager.receive_from_openai(
                handle_audio_delta, handle_speech_started, handle_other_openai_event
            )

        async def renew_openai_session():
            # Preemptively reconnect before the 60-minute cap to avoid session_expired drops
            while True:
                await asyncio.sleep(Config.REALTIME_SESSION_RENEW_SECONDS)
                try:
                    print("Preemptive OpenAI session renewal starting…")
                    await connection_manager.close_openai_connection()
                    await connection_manager.connect_to_openai()
                    await openai_service.initialize_session(connection_manager)
                    print("OpenAI session renewed.")
                except Exception as e:
                    print(f"OpenAI session renewal failed: {e}")

        await asyncio.gather(
            connection_manager.receive_from_twilio(handle_media_event, handle_stream_start, handle_mark_event),
            openai_receiver(),
            renew_openai_session(),
        )

    except Exception as e:
        Log.error(f"Error in media stream handler: {e}")
    finally:
        await connection_manager.close_openai_connection()


async def handle_speech_started_event(
    connection_manager: WebSocketConnectionManager, 
    openai_service: OpenAIService,
    audio_service: AudioService
):
    """Handle interruption when the caller's speech starts."""
    Log.subheader("Handling speech started event")
    
    if audio_service.should_handle_interruption():
        elapsed_time = audio_service.calculate_interruption_timing()
        current_item_id = audio_service.get_current_item_id()
        
        if elapsed_time is not None and current_item_id:
            await openai_service.handle_interruption(
                connection_manager,
                audio_service.timing_manager.current_timestamp,
                audio_service.timing_manager.response_start_timestamp,
                current_item_id
            )
            
            # Clear audio and reset state
            clear_message = audio_service.create_clear_message(connection_manager.state.stream_sid)
            await connection_manager.send_to_twilio(clear_message)
            audio_service.reset_interruption_state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT)
