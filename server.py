# server.py (multi-call capable with dual transcription system)
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
# Timed audio streaming with proper delays
# ---------------------------
async def handle_audio_stream(audio_data: Dict[str, Any], call_sid: str):
    """Handle raw audio chunks with proper timing."""
    payload = {
        "messageType": "audio",
        "speaker": audio_data["speaker"],
        "audio": audio_data["audio"],
        "timestamp": int(time.time() * 1000),  # ✅ CHANGED: milliseconds
        "callSid": call_sid,
    }
    await _do_broadcast(payload, call_sid)


async def handle_transcription_update(transcription_data: Dict[str, Any], call_sid: str):
    """Handle completed transcription phrases."""
    payload = {
        "messageType": "transcription",
        "speaker": transcription_data["speaker"],
        "text": transcription_data["text"],
        "timestamp": transcription_data.get("timestamp") or int(time.time() * 1000),
        "callSid": call_sid,
    }
    broadcast_to_dashboards_nonblocking(payload, call_sid)


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
# Broadcasting functions with call filtering
# ---------------------------
async def _do_broadcast(payload: Dict[str, Any], call_sid: Optional[str] = None):
    """Broadcast to dashboard clients with call filtering."""
    try:
        if "timestamp" not in payload or payload["timestamp"] is None:
            payload["timestamp"] = int(time.time() * 1000)
        else:
            ts = float(payload["timestamp"])
            if ts < 32503680000:
                payload["timestamp"] = int(ts * 1000)
            else:
                payload["timestamp"] = int(ts)
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
    asyncio.create_task(_do_broadcast(payload, call_sid))


def log_nonblocking(func, msg):
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
    order_extractor = OrderExtractionService()

    current_call_sid: Optional[str] = None

    async def send_order_update(order_data: Dict[str, Any]):
        payload = {
            "messageType": "orderUpdate",
            "orderData": order_data,
            "timestamp": int(time.time() * 1000),
            "callSid": current_call_sid,
        }
        broadcast_to_dashboards_nonblocking(payload, current_call_sid)

    order_extractor.set_update_callback(send_order_update)

    async def handle_openai_transcript(transcription_data: Dict[str, Any]):
        if not transcription_data or not isinstance(transcription_data, dict):
            return
        speaker = transcription_data.get("speaker")
        text = transcription_data.get("text")
        if not speaker or not text:
            return
        payload = {
            "messageType": "transcription",
            "speaker": speaker,
            "text": text,
            "timestamp": transcription_data.get("timestamp") or int(time.time() * 1000),
            "callSid": current_call_sid,
        }
        broadcast_to_dashboards_nonblocking(payload, current_call_sid)
        try:
            order_extractor.add_transcript(speaker, text)
        except Exception as e:
            Log.error(f"[OrderExtraction] Error: {e}")

    async def handle_whisper_for_orders(transcription_data: Dict[str, Any]):
        if not transcription_data or not isinstance(transcription_data, dict):
            return
        speaker = transcription_data.get("speaker")
        text = transcription_data.get("text")
        if not speaker or not text:
            return
        try:
            normalized_speaker = "AI" if speaker == "AI_whisper" else speaker
            order_extractor.add_transcript(normalized_speaker, text)
        except Exception as e:
            Log.error(f"[OrderExtraction] Error: {e}")

    async def handle_audio_with_call_id(audio_data: Dict[str, Any]):
        if current_call_sid:
            await handle_audio_stream(audio_data, current_call_sid)

    openai_service.whisper_service.set_audio_callback(handle_audio_with_call_id)
    openai_service.whisper_service.set_word_callback(handle_whisper_for_orders)
    openai_service.transcript_callback = handle_openai_transcript

    try:
        await connection_manager.connect_to_openai()
        await openai_service.initialize_session(connection_manager)

        async def handle_media_event(data: dict):
            if data.get("event") == "media":
                media = data.get("media") or {}
                payload_b64 = media.get("payload")
                if payload_b64:
                    try:
                        asyncio.create_task(
                            openai_service.whisper_service.transcribe_realtime(
                                payload_b64, source="Caller"
                            )
                        )
                    except Exception as e:
                        log_nonblocking(Log.error, f"[Caller processing] failed: {e}")

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

        async def handle_audio_delta(response: dict):
            try:
                if response.get('type') == 'response.audio.delta':
                    Log.info(f"[DEBUG] 🔊 Audio delta received! Size: {len(response.get('delta', ''))}")

                audio_data = openai_service.extract_audio_response_data(response) or {}
                delta = audio_data.get("delta")
                if delta:
                    Log.info(f"[DEBUG] 🎵 Extracted delta, size: {len(delta) if isinstance(delta, str) else 'bytes'}")

                    if isinstance(delta, (bytes, bytearray)):
                        delta_bytes = bytes(delta)
                    elif isinstance(delta, str):
                        try:
                            delta_bytes = base64.b64decode(delta)
                            Log.info(f"[DEBUG] ✅ Decoded delta to {len(delta_bytes)} bytes")
                        except Exception as e:
                            Log.error(f"[DEBUG] ❌ Failed to decode delta: {e}")
                            delta_bytes = None
                    else:
                        delta_bytes = None

                    if delta_bytes:
                        asyncio.create_task(
                            openai_service.whisper_service.transcribe_realtime(
                                delta_bytes, source="AI_whisper"
                            )
                        )

                        if getattr(connection_manager.state, "stream_sid", None):
                            try:
                                audio_message = audio_service.process_outgoing_audio(
                                    response, connection_manager.state.stream_sid
                                )
                                if audio_message:
                                    Log.info(f"[DEBUG] 📤 Sending audio to Twilio")
                                    await connection_manager.send_to_twilio(audio_message)
                                    mark_msg = audio_service.create_mark_message(
                                        connection_manager.state.stream_sid
                                    )
                                    await connection_manager.send_to_twilio(mark_msg)
                            except Exception as e:
                                log_nonblocking(Log.error, f"[audio->twilio] failed: {e}")
                    else:
                        Log.error("[DEBUG] ❌ No delta_bytes to send!")
            except Exception as e:
                log_nonblocking(Log.error, f"[audio-delta] failed: {e}")

        async def handle_other_openai_event(response: dict):
            openai_service.process_event_for_logging(response)
            openai_service.track_item_creation(response)
            await openai_service.extract_and_emit_caller_transcript(response)
            await openai_service.extract_and_emit_ai_transcript(response)

        async def openai_receiver():
            await connection_manager.receive_from_openai(
                handle_audio_delta,
                None,
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
            Log.event("Twilio Start", {"streamSid": stream_sid, "callSid": current_call_sid})

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
            final_summary = order_extractor.get_order_summary()
            Log.info(f"\n{final_summary}")
            final_order = order_extractor.get_current_order()
            if any(final_order.values()):
                broadcast_to_dashboards_nonblocking({
                    "messageType": "orderComplete",
                    "orderData": final_order,
                    "summary": final_summary,
                    "timestamp": int(time.time() * 1000),
                    "callSid": current_call_sid,
                }, current_call_sid)
        except Exception:
            pass

        try:
            await order_extractor.shutdown()
        except Exception:
            pass

        try:
            await openai_service.whisper_service.shutdown()
        except Exception:
            pass

        try:
            await connection_manager.close_openai_connection()
        except Exception:
            pass


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", getattr(Config, "PORT", 8000))),
        log_level="info",
        reload=False,
    )
