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

# keep your existing service imports (assuming services.__init__ exposes them)
from config import Config
from services import (
    WebSocketConnectionManager,
    TwilioService,
    OpenAIService,
    AudioService
)
from services.log_utils import Log

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# connected dashboard clients
dashboard_clients: Set[WebSocket] = set()


# ---------------------------
# Dashboard websocket (UI)
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

    dashboard_clients.add(websocket)

    try:
        try:
            # allow client to send optional initial payload (e.g., callId)
            msg = await asyncio.wait_for(websocket.receive_text(), timeout=5)
            data = json.loads(msg)
            client_call_id = data.get("callId")
        except (asyncio.TimeoutError, json.JSONDecodeError, KeyError):
            client_call_id = None

        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=20.0)
            except asyncio.TimeoutError:
                # keep-alive
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        dashboard_clients.discard(websocket)


# ---------------------------
# Broadcast helpers
# ---------------------------
async def broadcast_to_dashboards(payload: Dict[str, Any]):
    """Send JSON payload to all connected dashboards. Remove dead clients."""
    try:
        if "timestamp" not in payload or payload["timestamp"] is None:
            payload["timestamp"] = int(time.time())
        else:
            payload["timestamp"] = int(float(payload["timestamp"]))
    except Exception:
        payload["timestamp"] = int(time.time())

    text = json.dumps(payload)
    to_remove = []
    for client in list(dashboard_clients):
        try:
            await client.send_text(text)
        except Exception:
            to_remove.append(client)
    for c in to_remove:
        dashboard_clients.discard(c)


async def broadcast_transcript(transcript_obj: dict):
    await broadcast_to_dashboards(transcript_obj)


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
# Media stream
# ---------------------------
@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    Log.header("Client connected")
    await websocket.accept()

    connection_manager = WebSocketConnectionManager(websocket)
    openai_service = OpenAIService()
    audio_service = AudioService()

    try:
        # --- Connect to OpenAI safely ---
        connected = await connection_manager.connect_to_openai()
        if not connected:
            Log.error("OpenAI connection failed — cannot start media stream.")
            return

        initialized = await openai_service.initialize_session(connection_manager)
        if initialized is None:
            Log.error("OpenAI session initialization failed — aborting stream.")
            return

        # ---------------------------
        # Twilio incoming media/audio
        # ---------------------------
        async def handle_media_event(data: dict):
            if data.get("event") == "media":
                media = data.get("media") or {}
                payload_b64 = media.get("payload")
                if payload_b64:
                    aud = {
                        "messageType": "audio",
                        "speaker": "Caller",
                        "audio": payload_b64,
                        "encoding": "base64",
                        "timestamp": data.get("timestamp") or int(time.time())
                    }
                    await broadcast_to_dashboards(aud)

            if connection_manager.is_openai_connected():
                audio_message = audio_service.process_incoming_audio(data)
                if audio_message:
                    await connection_manager.send_to_openai(audio_message)

            if "text" in data and isinstance(data["text"], str):
                txt_obj = {
                    "messageType": "text",
                    "speaker": "Caller",
                    "text": data["text"].strip(),
                    "timestamp": data.get("timestamp") or int(time.time())
                }
                await broadcast_to_dashboards(txt_obj)

        # ---------------------------
        # OpenAI Audio + Text output
        # ---------------------------
        async def handle_audio_delta(response: dict):
            audio_data = openai_service.extract_audio_response_data(response) or {}
            delta = audio_data.get("delta")
            transcript_text = None

            # Try extracting assistant transcript text
            if hasattr(openai_service, "extract_transcript_text"):
                try:
                    transcript_text = openai_service.extract_transcript_text(response)
                except Exception:
                    transcript_text = None

            if transcript_text:
                bot_text_obj = {
                    "messageType": "text",
                    "speaker": "AI",
                    "text": transcript_text,
                    "timestamp": response.get("timestamp") or int(time.time())
                }
                await broadcast_to_dashboards(bot_text_obj)

            # Send AI audio to dashboard + Twilio
            if delta:
                try:
                    delta_b64 = base64.b64encode(delta).decode("ascii") if isinstance(delta, bytes) else delta
                    ai_audio = {
                        "messageType": "audio",
                        "speaker": "AI",
                        "audio": delta_b64,
                        "encoding": "base64",
                        "timestamp": response.get("timestamp") or int(time.time())
                    }
                    await broadcast_to_dashboards(ai_audio)

                    if connection_manager.state.stream_sid:
                        audio_message = audio_service.process_outgoing_audio(response, connection_manager.state.stream_sid)
                        if audio_message:
                            await connection_manager.send_to_twilio(audio_message)
                            mark_message = audio_service.create_mark_message(connection_manager.state.stream_sid)
                            await connection_manager.send_to_twilio(mark_message)
                except Exception as e:
                    Log.error(f"[audio->twilio] failed to send audio back to Twilio: {e}")

        async def handle_speech_started():
            Log.info("Speech started detected.")
            if openai_service.is_goodbye_pending():
                return
            try:
                await connection_manager.send_mark_to_twilio()
            except Exception:
                pass

        async def handle_other_openai_event(response: dict):
            openai_service.process_event_for_logging(response)

        async def openai_receiver():
            await connection_manager.receive_from_openai(
                handle_audio_delta, handle_speech_started, handle_other_openai_event
            )

        async def renew_openai_session():
            while True:
                await asyncio.sleep(getattr(Config, "REALTIME_SESSION_RENEW_SECONDS", 1200))
                try:
                    Log.info("Renewing OpenAI session…")
                    await connection_manager.close_openai_connection()
                    ok = await connection_manager.connect_to_openai()
                    if ok:
                        await openai_service.initialize_session(connection_manager)
                        Log.info("Session renewed successfully.")
                    else:
                        Log.error("Session renewal failed: unable to reconnect.")
                except Exception as e:
                    Log.error(f"Session renewal failed: {e}")

        await asyncio.gather(
            connection_manager.receive_from_twilio(
                handle_media_event,
                lambda sid: Log.event("Twilio stream start", {"sid": sid}),
                lambda: audio_service.handle_mark_event()
            ),
            openai_receiver(),
            renew_openai_session(),
        )

    except Exception as e:
        Log.error(f"Error in media stream handler: {e}")
    finally:
        await connection_manager.close_openai_connection()
