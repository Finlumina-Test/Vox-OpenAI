# server.py (No Whisper - OpenAI Native Transcription Only)
import os
import json
import time
import base64
import asyncio
from typing import Set, Optional, Dict, Any

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from services.order_extraction_service import OrderExtractionService

from config import Config
from services import (
    WebSocketConnectionManager,
    TwilioService,
    OpenAIService,
    AudioService,
)
from services.log_utils import Log


# ---------------------------
# Multi-call dashboard tracking
# ---------------------------
class DashboardClient:
    """Track dashboard websocket with optional call filtering."""
    def __init__(self, websocket: WebSocket, call_sid: Optional[str] = None):
        self.websocket = websocket
        self.call_sid = call_sid  # None = subscribe to ALL calls


dashboard_clients: Set[DashboardClient] = set()


# ---------------------------
# Broadcasting functions with call filtering
# ---------------------------
async def _do_broadcast(payload: Dict[str, Any], call_sid: Optional[str] = None):
    """Broadcast to dashboard clients with call filtering."""
    try:
        if "timestamp" not in payload or payload["timestamp"] is None:
            payload["timestamp"] = int(time.time() * 1000)
        else:
            payload["timestamp"] = int(float(payload["timestamp"]))
    except Exception:
        payload["timestamp"] = int(time.time() * 1000)

    if call_sid and "callSid" not in payload:
        payload["callSid"] = call_sid

    text = json.dumps(payload)
    to_remove = []
    
    for client in list(dashboard_clients):
        try:
            should_send = (
                client.call_sid is None or
                client.call_sid == call_sid
            )
            
            if should_send:
                await client.websocket.send_text(text)
                
        except Exception as e:
            Log.debug(f"Failed to send to client: {e}")
            to_remove.append(client)
    
    for c in to_remove:
        dashboard_clients.discard(c)


def broadcast_to_dashboards_nonblocking(payload: Dict[str, Any], call_sid: Optional[str] = None):
    """Fire-and-forget broadcast with call filtering."""
    asyncio.create_task(_do_broadcast(payload, call_sid))


def log_nonblocking(func, msg):
    """Fire-and-forget logging (non-blocking)."""
    asyncio.create_task(_run_log(func, msg))


async def _run_log(func, msg):
    try:
        func(msg)
    except Exception:
        pass


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Dashboard websocket
# ---------------------------
@app.websocket("/dashboard-stream")
async def dashboard_stream(websocket: WebSocket):
    await websocket.accept()
    DASHBOARD_TOKEN = os.getenv("DASHBOARD_TOKEN")
    client_call_id: Optional[str] = None

    if DASHBOARD_TOKEN:
        provided = websocket.query_params.get("token") or websocket.headers.get("x-dashboard-token")
        if provided != DASHBOARD_TOKEN:
            await websocket.close(code=4003)
            return

    try:
        msg = await asyncio.wait_for(websocket.receive_text(), timeout=5)
        data = json.loads(msg)
        client_call_id = data.get("callId")
        Log.info(f"Dashboard client subscribed to call: {client_call_id or 'ALL'}")
    except (asyncio.TimeoutError, json.JSONDecodeError, KeyError):
        Log.info("Dashboard client subscribed to ALL calls")
        client_call_id = None

    client = DashboardClient(websocket, client_call_id)
    dashboard_clients.add(client)
    Log.info(f"Dashboard connected. Total clients: {len(dashboard_clients)}")
    
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=20.0)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        dashboard_clients.discard(client)
        Log.info(f"Dashboard disconnected. Total clients: {len(dashboard_clients)}")


# ---------------------------
# Simple health endpoint
# ---------------------------
@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}


# ---------------------------
# Twilio incoming-call TwiML
# ---------------------------
@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    return TwilioService.create_incoming_call_response(request)


# ---------------------------
# Media stream: Twilio <-> OpenAI <-> Dashboard
# ---------------------------
@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    Log.header("Client connected")
    await websocket.accept()

    # Each connection gets its own instances (isolated per call)
    connection_manager = WebSocketConnectionManager(websocket)
    openai_service = OpenAIService()
    audio_service = AudioService()
    order_extractor = OrderExtractionService()
    
    # Track call_sid for this connection
    current_call_sid: Optional[str] = None
    
    # Set callback for order updates WITH call_sid
    async def send_order_update(order_data: Dict[str, Any]):
        """Send order updates to dashboard."""
        payload = {
            "messageType": "orderUpdate",
            "orderData": order_data,
            "timestamp": int(time.time() * 1000),
            "callSid": current_call_sid,
        }
        broadcast_to_dashboards_nonblocking(payload, current_call_sid)
    
    order_extractor.set_update_callback(send_order_update)
    
    # ✅ Unified transcription callback (OpenAI native only)
    async def handle_transcription_with_extraction(transcription_data: Dict[str, Any]):
        """
        Handle transcription from OpenAI Realtime API.
        Both Caller and AI transcripts come from OpenAI.
        """
        speaker = transcription_data["speaker"]
        
        # Send to dashboard
        payload = {
            "messageType": "transcription",
            "speaker": speaker,
            "text": transcription_data["text"],
            "timestamp": transcription_data.get("timestamp") or int(time.time() * 1000),
            "callSid": current_call_sid,
        }
        broadcast_to_dashboards_nonblocking(payload, current_call_sid)
        
        # Extract order information
        try:
            order_extractor.add_transcript(
                speaker,
                transcription_data["text"]
            )
        except Exception as e:
            Log.error(f"[OrderExtraction] Error: {e}")
    
    # Set up callbacks for OpenAI native transcription
    openai_service.caller_transcript_callback = handle_transcription_with_extraction  # Caller from OpenAI
    openai_service.ai_transcript_callback = handle_transcription_with_extraction      # AI from OpenAI

    try:
        # Connect to OpenAI
        try:
            await connection_manager.connect_to_openai()
        except Exception as e:
            Log.error(f"OpenAI connection failed: {e}")
            await connection_manager.close_openai_connection()
            return

        try:
            await openai_service.initialize_session(connection_manager)
        except Exception as e:
            Log.error(f"OpenAI session initialization failed: {e}")
            await connection_manager.close_openai_connection()
            return

        # Twilio -> Server handler
        async def handle_media_event(data: dict):
            if data.get("event") == "media":
                if connection_manager.is_openai_connected():
                    try:
                        audio_message = audio_service.process_incoming_audio(data)
                        if audio_message:
                            await connection_manager.send_to_openai(audio_message)
                    except Exception as e:
                        log_nonblocking(Log.error, f"[media] failed to send incoming audio: {e}")

            if "text" in data and isinstance(data["text"], str) and data["text"].strip():
                txt_obj = {
                    "messageType": "text",
                    "speaker": "Caller",
                    "text": data["text"].strip(),
                    "timestamp": data.get("timestamp") or int(time.time() * 1000),
                    "callSid": current_call_sid,
                }
                broadcast_to_dashboards_nonblocking(txt_obj, current_call_sid)

        # OpenAI -> Twilio handler
        async def handle_audio_delta(response: dict):
            try:
                audio_data = openai_service.extract_audio_response_data(response) or {}
                delta = audio_data.get("delta")
                
                if delta and getattr(connection_manager.state, "stream_sid", None):
                    try:
                        audio_message = audio_service.process_outgoing_audio(
                            response, connection_manager.state.stream_sid
                        )
                        if audio_message:
                            await connection_manager.send_to_twilio(audio_message)
                            mark_msg = audio_service.create_mark_message(
                                connection_manager.state.stream_sid
                            )
                            await connection_manager.send_to_twilio(mark_msg)
                    except Exception as e:
                        log_nonblocking(Log.error, f"[audio->twilio] failed: {e}")

            except Exception as e:
                log_nonblocking(Log.error, f"[audio-delta] failed: {e}")

        async def handle_speech_started():
            try:
                await connection_manager.send_mark_to_twilio()
            except Exception:
                pass

        async def handle_other_openai_event(response: dict):
            openai_service.process_event_for_logging(response)
            
            # ✅ Extract transcripts from OpenAI native events
            await openai_service.extract_caller_transcript(response)  # Caller transcript
            await openai_service.extract_ai_transcript(response)       # AI transcript

        async def openai_receiver():
            await connection_manager.receive_from_openai(
                handle_audio_delta,
                handle_speech_started,
                handle_other_openai_event,
            )

        async def renew_openai_session():
            while True:
                await asyncio.sleep(getattr(Config, "REALTIME_SESSION_RENEW_SECONDS", 1200))
                try:
                    Log.info("Renewing OpenAI session…")
                    await connection_manager.close_openai_connection()
                    await connection_manager.connect_to_openai()
                    await openai_service.initialize_session(connection_manager)
                    Log.info("Session renewed successfully.")
                except Exception as e:
                    Log.error(f"Session renewal failed: {e}")

        async def on_start_cb(stream_sid: str):
            nonlocal current_call_sid
            current_call_sid = getattr(connection_manager.state, 'call_sid', stream_sid)
            Log.event("Twilio Start", {"
