# server.py (non-blocking dashboard + logging)
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

from config import Config
from services import (
    WebSocketConnectionManager,
    TwilioService,
    OpenAIService,
    AudioService,
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

dashboard_clients: Set[WebSocket] = set()


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

    dashboard_clients.add(websocket)
    try:
        try:
            msg = await asyncio.wait_for(websocket.receive_text(), timeout=5)
            data = json.loads(msg)
            client_call_id = data.get("callId")
        except (asyncio.TimeoutError, json.JSONDecodeError, KeyError):
            client_call_id = None

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
        dashboard_clients.discard(websocket)


# ---------------------------
# Non-blocking broadcast
# ---------------------------
async def _do_broadcast(payload: Dict[str, Any]):
    """Internal coroutine to broadcast dashboard updates."""
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


def broadcast_to_dashboards_nonblocking(payload: Dict[str, Any]):
    """Fire-and-forget broadcast (non-blocking)."""
    asyncio.create_task(_do_broadcast(payload))


def log_nonblocking(func, msg):
    """Fire-and-forget logging (non-blocking)."""
    asyncio.create_task(_run_log(func, msg))


async def _run_log(func, msg):
    try:
        func(msg)
    except Exception:
        pass


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

    connection_manager = WebSocketConnectionManager(websocket)
    openai_service = OpenAIService()
    audio_service = AudioService()

    try:
        # --- Connect to OpenAI ---
        try:
            await connection_manager.connect_to_openai()
            await openai_service.initialize_session(connection_manager)
        except Exception as e:
            Log.error(f"OpenAI setup failed: {e}")
            return

        # ---------------------------
        # Dashboard Broadcast Helper (non-blocking)
        # ---------------------------
        async def dashboard_broadcast(payload: dict):
            asyncio.create_task(broadcast_to_dashboards(payload))

        # ---------------------------
        # Twilio Media Input Handler (Caller side)
        # ---------------------------
        async def handle_media_event(data: dict):
            # Handle Twilio audio packets
            if data.get("event") == "media":
                payload_b64 = (data.get("media") or {}).get("payload")
                if payload_b64:
                    asyncio.create_task(broadcast_to_dashboards({
                        "messageType": "audio",
                        "speaker": "Caller",
                        "audio": payload_b64,
                        "encoding": "base64",
                        "timestamp": data.get("timestamp") or int(time.time())
                    }))

            # Handle Twilio text (if any)
            if "text" in data and isinstance(data["text"], str) and data["text"].strip():
                asyncio.create_task(broadcast_to_dashboards({
                    "messageType": "text",
                    "speaker": "Caller",
                    "text": data["text"].strip(),
                    "timestamp": data.get("timestamp") or int(time.time())
                }))

            # Forward caller audio to OpenAI
            if connection_manager.is_openai_connected():
                try:
                    audio_message = audio_service.process_incoming_audio(data)
                    if audio_message:
                        await connection_manager.send_to_openai(audio_message)
                except Exception as e:
                    Log.debug(f"[media] incoming audio processing failed: {e}")

        # ---------------------------
        # OpenAI Stream Event Handler (AI side)
        # ---------------------------
        async def handle_audio_delta(response: dict):
            # --- Extract Caller Text (if any) ---
            try:
                caller_txt = openai_service.extract_caller_transcript(response)
                if caller_txt:
                    asyncio.create_task(broadcast_to_dashboards({
                        "messageType": "text",
                        "speaker": "Caller",
                        "text": caller_txt,
                        "timestamp": int(time.time())
                    }))
            except Exception:
                pass

            # --- Extract Assistant Text (if any) ---
            try:
                transcript_text = openai_service.extract_transcript_text(response)
                if transcript_text:
                    asyncio.create_task(broadcast_to_dashboards({
                        "messageType": "text",
                        "speaker": "AI",
                        "text": transcript_text,
                        "timestamp": int(time.time())
                    }))
            except Exception:
                pass

            # --- Handle Audio Response ---
            audio_data = openai_service.extract_audio_response_data(response) or {}
            delta = audio_data.get("delta")

            if delta:
                try:
                    delta_b64 = (
                        base64.b64encode(delta).decode("ascii")
                        if isinstance(delta, (bytes, bytearray))
                        else delta
                    )
                    # Send to dashboard (non-blocking)
                    asyncio.create_task(broadcast_to_dashboards({
                        "messageType": "audio",
                        "speaker": "AI",
                        "audio": delta_b64,
                        "encoding": "base64",
                        "timestamp": response.get("timestamp") or int(time.time())
                    }))
                    # Send to Twilio
                    if connection_manager.state.stream_sid:
                        audio_message = audio_service.process_outgoing_audio(response, connection_manager.state.stream_sid)
                        if audio_message:
                            await connection_manager.send_to_twilio(audio_message)
                            mark_msg = audio_service.create_mark_message(connection_manager.state.stream_sid)
                            await connection_manager.send_to_twilio(mark_msg)
                except Exception as e:
                    Log.error(f"[audio->twilio] AI audio send failed: {e}")

        # ---------------------------
        # Speech Start Handler
        # ---------------------------
        async def handle_speech_started():
            Log.info("Speech started detected (OpenAI)")
            if openai_service.is_goodbye_pending():
                return
            try:
                await connection_manager.send_mark_to_twilio()
            except Exception:
                pass

        # ---------------------------
        # Other OpenAI Event Handler
        # ---------------------------
        async def handle_other_openai_event(response: dict):
            openai_service.process_event_for_logging(response)

        # ---------------------------
        # Receivers
        # ---------------------------
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

        # ---------------------------
        # Start concurrent tasks
        # ---------------------------
        async def on_start_cb(stream_sid: str):
            Log.event("Twilio Stream Start", {"streamSid": stream_sid})

        async def on_mark_cb():
            try:
                audio_service.handle_mark_event()
            except Exception:
                pass

        await asyncio.gather(
            connection_manager.receive_from_twilio(
                handle_media_event,
                on_start_cb,
                on_mark_cb,
            ),
            openai_receiver(),
            renew_openai_session(),
        )

    except Exception as e:
        Log.error(f"Error in media stream handler: {e}")
    finally:
        try:
            await connection_manager.close_openai_connection()
        except Exception:
            pass

