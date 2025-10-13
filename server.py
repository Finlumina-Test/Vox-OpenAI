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

async def handle_word_update(word_data: Dict[str, Any]):
    """Handle word-level updates from transcription service."""
    payload = {
        "messageType": "word",
        "speaker": word_data["speaker"],
        "word": word_data["word"],
        "audio": word_data["audio"],
        "timestamp": word_data["timestamp"]
    }
    broadcast_to_dashboards_nonblocking(payload)

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


async def send_transcription_to_dashboard(text: str, speaker: str, timestamp: int):
    """Send transcription text to dashboard."""
    if text and text.strip():
        payload = {
            "messageType": "transcription",
            "speaker": speaker,
            "text": text.strip(),
            "timestamp": timestamp,
        }
        broadcast_to_dashboards_nonblocking(payload)


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
    
    # Set up word-level callback
    openai_service.whisper_service.set_word_callback(handle_word_update)

    try:
        # --- Connect to OpenAI ---
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

        # ---------------------------
        # Twilio -> Server handler
        # ---------------------------
        async def handle_media_event(data: dict):
            if data.get("event") == "media":
                media = data.get("media") or {}
                payload_b64 = media.get("payload")
                if payload_b64:
                    timestamp = data.get("timestamp") or int(time.time())
                    
                    # Send audio to dashboard
                    aud = {
                        "messageType": "audio",
                        "speaker": "Caller",
                        "audio": payload_b64,
                        "encoding": "base64",
                        "timestamp": timestamp,
                    }
                    broadcast_to_dashboards_nonblocking(aud)

                    # --- Parallel Whisper transcription (non-blocking, safe) ---
                    async def transcribe_caller():
                        try:
                            transcript = await openai_service.whisper_service.transcribe_realtime(
                                payload_b64, 
                                source="Caller"
                            )
                            if transcript:
                                await send_transcription_to_dashboard(transcript, "Caller", timestamp)
                        except Exception as e:
                            log_nonblocking(Log.error, f"[Caller transcription] failed: {e}")
                    
                    asyncio.create_task(transcribe_caller())

                    # Send to OpenAI for conversation
                    if connection_manager.is_openai_connected():
                        try:
                            audio_message = audio_service.process_incoming_audio(data)
                            if audio_message:
                                await connection_manager.send_to_openai(audio_message)
                        except Exception as e:
                            log_nonblocking(Log.error, f"[media] failed to send incoming audio: {e}")

            # Twilio may include text directly
            if "text" in data and isinstance(data["text"], str) and data["text"].strip():
                txt_obj = {
                    "messageType": "text",
                    "speaker": "Caller",
                    "text": data["text"].strip(),
                    "timestamp": data.get("timestamp") or int(time.time()),
                }
                broadcast_to_dashboards_nonblocking(txt_obj)

        # ---------------------------
        # OpenAI -> Twilio handler
        # ---------------------------
        async def handle_audio_delta(response: dict):
            try:
                # first, if OpenAI produced caller transcript events, send them to dashboard
                caller_txt = openai_service.extract_caller_transcript(response)
                if caller_txt:
                    broadcast_to_dashboards_nonblocking({
                        "messageType": "text",
                        "speaker": "Caller",
                        "text": caller_txt,
                        "timestamp": response.get("timestamp") or int(time.time()),
                    })
            except Exception:
                # ignore transcript extraction errors to keep main flow running
                pass

            audio_data = openai_service.extract_audio_response_data(response) or {}
            delta = audio_data.get("delta")

            if delta is not None:
                try:
                    timestamp = response.get("timestamp") or int(time.time())
                    
                    # ensure base64 for dashboard (OpenAI might send bytes or base64)
                    if isinstance(delta, (bytes, bytearray)):
                        delta_b64 = base64.b64encode(delta).decode("ascii")
                        delta_bytes = bytes(delta)
                    elif isinstance(delta, str):
                        delta_b64 = delta
                        try:
                            delta_bytes = base64.b64decode(delta)
                        except Exception:
                            delta_bytes = None
                    else:
                        # fallback: stringify then base64
                        raw = json.dumps(delta).encode("utf-8")
                        delta_b64 = base64.b64encode(raw).decode("ascii")
                        delta_bytes = raw

                    # Send audio to dashboard
                    broadcast_to_dashboards_nonblocking({
                        "messageType": "audio",
                        "speaker": "AI",
                        "audio": delta_b64,
                        "encoding": "base64",
                        "timestamp": timestamp,
                    })

                    # --- Parallel Whisper transcription for AI audio (non-blocking) ---
                    if delta_bytes:
                        async def transcribe_ai():
                            try:
                                transcript = await openai_service.whisper_service.transcribe_realtime(
                                    delta_bytes,
                                    source="AI"
                                )
                                if transcript:
                                    await send_transcription_to_dashboard(transcript, "AI", timestamp)
                            except Exception as e:
                                log_nonblocking(Log.error, f"[AI transcription] failed: {e}")
                        
                        asyncio.create_task(transcribe_ai())

                    # send audio to Twilio so caller hears AI
                    if audio_data and getattr(connection_manager.state, "stream_sid", None):
                        audio_message = audio_service.process_outgoing_audio(
                            response, connection_manager.state.stream_sid
                        )
                        if audio_message:
                            await connection_manager.send_to_twilio(audio_message)
                            mark_msg = audio_service.create_mark_message(connection_manager.state.stream_sid)
                            await connection_manager.send_to_twilio(mark_msg)
                except Exception as e:
                    log_nonblocking(Log.error, f"[audio->twilio] failed to send audio: {e}")

            # assistant transcript text (if OpenAI sent it already)
            transcript_text = None
            if hasattr(openai_service, "extract_transcript_text"):
                try:
                    transcript_text = openai_service.extract_transcript_text(response)
                except Exception:
                    transcript_text = None

            if transcript_text:
                broadcast_to_dashboards_nonblocking({
                    "messageType": "text",
                    "speaker": "AI",
                    "text": transcript_text,
                    "timestamp": response.get("timestamp") or int(time.time()),
                })

        async def handle_speech_started():
            try:
                await connection_manager.send_mark_to_twilio()
            except Exception:
                pass

        async def handle_other_openai_event(response: dict):
            openai_service.process_event_for_logging(response)

        # ---------------------------
        # Background receivers
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

        async def on_start_cb(stream_sid: str):
            Log.event("Twilio Start", {"streamSid": stream_sid})

        async def on_mark_cb():
            try:
                audio_service.handle_mark_event()
            except Exception:
                pass

        await asyncio.gather(
            connection_manager.receive_from_twilio(handle_media_event, on_start_cb, on_mark_cb),
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

# ---------------------------
# Proper entry point for Render + production
# ---------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", getattr(Config, "PORT", 8000))),
        log_level="info",
        reload=False,  # True only for dev
    )
