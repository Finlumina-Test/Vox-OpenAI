# server.py
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

# service imports (your project)
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
            # optional initial payload
            msg = await asyncio.wait_for(websocket.receive_text(), timeout=5)
            data = json.loads(msg)
            client_call_id = data.get("callId")
        except (asyncio.TimeoutError, json.JSONDecodeError, KeyError):
            client_call_id = None

        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=20.0)
            except asyncio.TimeoutError:
                # keep-alive ping
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


# legacy wrapper for compatibility
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
        # --- Connect to OpenAI (don't expect a boolean return) ---
        try:
            await connection_manager.connect_to_openai()
        except Exception as e:
            Log.error(f"OpenAI connection failed: {e}")
            # close/cleanup and return early
            try:
                await connection_manager.close_openai_connection()
            except Exception:
                pass
            return

        # --- Initialize OpenAI session (may return None) ---
        try:
            await openai_service.initialize_session(connection_manager)
        except Exception as e:
            Log.error(f"OpenAI session initialization failed: {e}")
            try:
                await connection_manager.close_openai_connection()
            except Exception:
                pass
            return

        # ---------------------------
        # Twilio incoming media/audio handler (caller side)
        # ---------------------------
        async def handle_media_event(data: dict):
            # broadcast raw Twilio media 'media' events to dashboard (base64)
            if data.get("event") == "media":
                media = data.get("media") or {}
                payload_b64 = media.get("payload") or None
                if not payload_b64 and isinstance(media.get("payloads"), list) and media.get("payloads"):
                    payload_b64 = media.get("payloads")[0].get("payload")
                if payload_b64:
                    aud = {
                        "messageType": "audio",
                        "speaker": "Caller",
                        "audio": payload_b64,
                        "encoding": "base64",
                        "callSid": getattr(connection_manager.state, "call_sid", None),
                        "streamSid": getattr(connection_manager.state, "stream_sid", None),
                        "timestamp": data.get("timestamp") or int(time.time())
                    }
                    Log.event("[dashboard] caller audio chunk", {"len_b64": len(payload_b64)})
                    await broadcast_to_dashboards(aud)

            # forward incoming audio to OpenAI (via AudioService) if connection available
            if connection_manager.is_openai_connected():
                try:
                    audio_message = audio_service.process_incoming_audio(data)
                    if audio_message:
                        await connection_manager.send_to_openai(audio_message)
                except Exception as e:
                    Log.debug(f"[media] failed to process/send incoming audio: {e}")

            # Twilio may also send transcription text (if enabled)
            if "text" in data and isinstance(data["text"], str) and data["text"].strip():
                txt_obj = {
                    "messageType": "text",
                    "speaker": "Caller",
                    "text": data["text"].strip(),
                    "callSid": getattr(connection_manager.state, "call_sid", None),
                    "streamSid": getattr(connection_manager.state, "stream_sid", None),
                    "timestamp": data.get("timestamp") or int(time.time())
                }
                Log.event("[dashboard] Caller text (twilio)", {"text": txt_obj["text"]})
                await broadcast_to_dashboards(txt_obj)

        # ---------------------------
        # OpenAI -> server handler (AI audio/text & caller transcripts)
        # ---------------------------
        async def handle_audio_delta(response: dict):
            # caller transcript emitted by OpenAI
            try:
                caller_txt = openai_service.extract_caller_transcript(response)
                if caller_txt:
                    caller_obj = {
                        "messageType": "text",
                        "speaker": "Caller",
                        "text": caller_txt,
                        "callSid": getattr(connection_manager.state, "call_sid", None),
                        "streamSid": getattr(connection_manager.state, "stream_sid", None),
                        "timestamp": response.get("timestamp") or int(time.time())
                    }
                    Log.event("[dashboard] Caller text (openai)", {"text": caller_txt})
                    await broadcast_to_dashboards(caller_obj)
            except Exception as e:
                Log.debug("[openai] caller transcript extract error: " + str(e))

            # audio delta from OpenAI (may be base64 or bytes)
            audio_data = openai_service.extract_audio_response_data(response) or {}
            delta = audio_data.get("delta")

            if delta is not None:
                try:
                    if isinstance(delta, (bytes, bytearray)):
                        delta_b64 = base64.b64encode(delta).decode("ascii")
                    elif isinstance(delta, str):
                        delta_b64 = delta
                    else:
                        delta_b64 = base64.b64encode(json.dumps(delta).encode("utf-8")).decode("ascii")

                    ai_audio = {
                        "messageType": "audio",
                        "speaker": "AI",
                        "audio": delta_b64,
                        "encoding": "base64",
                        "callSid": getattr(connection_manager.state, "call_sid", None),
                        "streamSid": getattr(connection_manager.state, "stream_sid", None),
                        "timestamp": response.get("timestamp") or int(time.time())
                    }
                    Log.event("[dashboard] AI audio chunk", {"len_b64": len(delta_b64)})
                    await broadcast_to_dashboards(ai_audio)
                except Exception as e:
                    Log.debug(f"[dashboard] failed to broadcast AI audio chunk: {e}")

            # assistant text transcript (if present)
            transcript_text = None
            if hasattr(openai_service, "extract_transcript_text"):
                try:
                    transcript_text = openai_service.extract_transcript_text(response)
                except Exception as e:
                    Log.debug("[openai] assistant transcript extract error: " + str(e))
                    transcript_text = None

            if transcript_text:
                bot_text_obj = {
                    "messageType": "text",
                    "speaker": "AI",
                    "text": transcript_text,
                    "callSid": getattr(connection_manager.state, "call_sid", None),
                    "streamSid": getattr(connection_manager.state, "stream_sid", None),
                    "timestamp": response.get("timestamp") or int(time.time())
                }
                Log.event("[dashboard] AI text", {"text": transcript_text})
                await broadcast_to_dashboards(bot_text_obj)

            # send actual audio back to Twilio (so caller hears AI)
            if audio_data and getattr(connection_manager.state, "stream_sid", None):
                try:
                    audio_message = audio_service.process_outgoing_audio(response, connection_manager.state.stream_sid)
                    if audio_message:
                        await connection_manager.send_to_twilio(audio_message)
                        mark_msg = audio_service.create_mark_message(connection_manager.state.stream_sid)
                        await connection_manager.send_to_twilio(mark_msg)
                except Exception as e:
                    Log.error(f"[audio->twilio] failed to send audio back to Twilio: {e}")

        # When OpenAI detects speech started
        async def handle_speech_started():
            Log.info("Speech started detected (OpenAI).")
            if openai_service.is_goodbye_pending():
                Log.info("Ignoring interruption during goodbye flow")
                return
            # send a mark to Twilio for sync (wrapped as awaitable)
            try:
                await connection_manager.send_mark_to_twilio()
            except Exception:
                pass

        async def handle_other_openai_event(response: dict):
            openai_service.process_event_for_logging(response)
            if openai_service.is_tool_call(response):
                tool_call = openai_service.accumulate_tool_call(response)
                if tool_call:
                    handled = await openai_service.maybe_handle_tool_call(connection_manager, tool_call)
                    if handled:
                        return
            if openai_service.should_finalize_on_event(response):
                await openai_service.finalize_goodbye(connection_manager)

        # ---------------------------
        # receiver coroutines
        # ---------------------------
        async def openai_receiver():
            await connection_manager.receive_from_openai(
                handle_audio_delta, handle_speech_started, handle_other_openai_event
            )

        async def renew_openai_session():
            while True:
                await asyncio.sleep(getattr(Config, "REALTIME_SESSION_RENEW_SECONDS", 1200))
                try:
                    Log.info("Preemptive OpenAI session renewal starting…")
                    await connection_manager.close_openai_connection()
                    await connection_manager.connect_to_openai()
                    await openai_service.initialize_session(connection_manager)
                    Log.info("OpenAI session renewed.")
                except Exception as e:
                    Log.error(f"OpenAI session renewal failed: {e}")

        # ---------------------------
        # IMPORTANT: on_start/on_mark passed to receive_from_twilio must be awaitable
        # ---------------------------
        async def on_start_cb(stream_sid: str):
            # synchronous Log.event is fine inside async wrapper
            Log.event("Twilio Start", {"streamSid": stream_sid})

        async def on_mark_cb():
            # audio_service.handle_mark_event() is synchronous; wrap it
            try:
                audio_service.handle_mark_event()
            except Exception:
                pass

        # Run Twilio receiver (which awaits callbacks), OpenAI receiver, renewer concurrently
        await asyncio.gather(
            connection_manager.receive_from_twilio(
                handle_media_event,
                on_start_cb,
                on_mark_cb
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


# Handle interruptions (kept from original)
async def handle_speech_started_event(
    connection_manager: WebSocketConnectionManager,
    openai_service: OpenAIService,
    audio_service: AudioService
):
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

            clear_message = audio_service.create_clear_message(connection_manager.state.stream_sid)
            await connection_manager.send_to_twilio(clear_message)
            audio_service.reset_interruption_state()


# ---------------------------
# Start uvicorn when running directly
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Finlumina server...")
    port = int(os.getenv("PORT", getattr(Config, "PORT", 8000) if hasattr(Config, "PORT") else 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
