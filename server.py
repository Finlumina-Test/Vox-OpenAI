# server.py (multi-call capable with call isolation)
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
        "timestamp": audio_data["timestamp"],
        "callSid": call_sid,  # ✅ Add call_sid to every message
    }
    
    await _do_broadcast(payload, call_sid)


async def handle_transcription_update(transcription_data: Dict[str, Any], call_sid: str):
    """Handle completed transcription phrases."""
    payload = {
        "messageType": "transcription",
        "speaker": transcription_data["speaker"],
        "text": transcription_data["text"],
        "timestamp": transcription_data["timestamp"],
        "callSid": call_sid,  # ✅ Add call_sid
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

    # Try to get call_sid from initial message
    try:
        msg = await asyncio.wait_for(websocket.receive_text(), timeout=5)
        data = json.loads(msg)
        client_call_id = data.get("callId")
        Log.info(f"Dashboard client subscribed to call: {client_call_id or 'ALL'}")
    except (asyncio.TimeoutError, json.JSONDecodeError, KeyError):
        Log.info("Dashboard client subscribed to ALL calls")
        client_call_id = None

    # Create client object with call filter
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
    """
    Broadcast to dashboard clients with call filtering.
    
    Args:
        payload: Message to broadcast
        call_sid: If provided, only send to clients subscribed to this call (or ALL)
    """
    try:
        if "timestamp" not in payload or payload["timestamp"] is None:
            payload["timestamp"] = int(time.time())
        else:
            payload["timestamp"] = int(float(payload["timestamp"]))
    except Exception:
        payload["timestamp"] = int(time.time())

    # Ensure call_sid is in payload
    if call_sid and "callSid" not in payload:
        payload["callSid"] = call_sid

    text = json.dumps(payload)
    to_remove = []
    
    for client in list(dashboard_clients):
        try:
            # Send if: client wants ALL calls OR client wants THIS specific call
            should_send = (
                client.call_sid is None or  # Client wants all calls
                client.call_sid == call_sid  # Client wants this call
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

    # ✅ Each connection gets its own instances (isolated)
    connection_manager = WebSocketConnectionManager(websocket)
    openai_service = OpenAIService()
    audio_service = AudioService()
    order_extractor = OrderExtractionService()
    
    # Track call_sid for this connection
    current_call_sid: Optional[str] = None
    
    # ✅ Set callback for order updates WITH call_sid
    async def send_order_update(order_data: Dict[str, Any]):
        """Send order updates to dashboard."""
        payload = {
            "messageType": "orderUpdate",
            "orderData": order_data,
            "timestamp": int(time.time()),
            "callSid": current_call_sid,  # ✅ Include call_sid
        }
        broadcast_to_dashboards_nonblocking(payload, current_call_sid)
    
    order_extractor.set_update_callback(send_order_update)
    
    # ✅ Enhanced transcription callback WITH call_sid
    async def handle_transcription_with_extraction(transcription_data: Dict[str, Any]):
        """Handle transcription AND extract order info."""
        # Send transcription to dashboard
        payload = {
            "messageType": "transcription",
            "speaker": transcription_data["speaker"],
            "text": transcription_data["text"],
            "timestamp": transcription_data["timestamp"],
            "callSid": current_call_sid,  # ✅ Include call_sid
        }
        broadcast_to_dashboards_nonblocking(payload, current_call_sid)
        
        # Extract order information
        try:
            order_extractor.add_transcript(
                transcription_data["speaker"],
                transcription_data["text"]
            )
        except Exception as e:
            Log.error(f"[OrderExtraction] Error: {e}")
    
    # ✅ Audio callback with call_sid
    async def handle_audio_with_call_id(audio_data: Dict[str, Any]):
        """Wrapper to add call_sid to audio."""
        if current_call_sid:
            await handle_audio_stream(audio_data, current_call_sid)
    
    # Set up callbacks
    openai_service.whisper_service.set_audio_callback(handle_audio_with_call_id)
    openai_service.whisper_service.set_word_callback(handle_transcription_with_extraction)

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
                    "timestamp": data.get("timestamp") or int(time.time()),
                    "callSid": current_call_sid,  # ✅ Include call_sid
                }
                broadcast_to_dashboards_nonblocking(txt_obj, current_call_sid)

        # OpenAI -> Twilio handler
        async def handle_audio_delta(response: dict):
            try:
                audio_data = openai_service.extract_audio_response_data(response) or {}
                delta = audio_data.get("delta")
                if delta:
                    if isinstance(delta, (bytes, bytearray)):
                        delta_bytes = bytes(delta)
                    elif isinstance(delta, str):
                        try:
                            delta_bytes = base64.b64decode(delta)
                        except Exception:
                            delta_bytes = None
                    else:
                        delta_bytes = None

                    if delta_bytes:
                        asyncio.create_task(
                            openai_service.whisper_service.transcribe_realtime(
                                delta_bytes, source="AI"
                            )
                        )

                    if getattr(connection_manager.state, "stream_sid", None):
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

                if hasattr(openai_service, "extract_transcript_text"):
                    try:
                        transcript_text = openai_service.extract_transcript_text(response)
                        if transcript_text:
                            broadcast_to_dashboards_nonblocking({
                                "messageType": "text",
                                "speaker": "AI",
                                "text": transcript_text,
                                "timestamp": response.get("timestamp") or int(time.time()),
                                "callSid": current_call_sid,  # ✅ Include call_sid
                            }, current_call_sid)
                    except Exception:
                        pass

            except Exception as e:
                log_nonblocking(Log.error, f"[audio-delta] failed: {e}")

        async def handle_speech_started():
            try:
                await connection_manager.send_mark_to_twilio()
            except Exception:
                pass

        async def handle_other_openai_event(response: dict):
            openai_service.process_event_for_logging(response)

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
            # ✅ Capture call_sid when stream starts
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
        # ✅ Log final order summary
        try:
            final_summary = order_extractor.get_order_summary()
            Log.info(f"\n{final_summary}")
            
            # Send final order
            final_order = order_extractor.get_current_order()
            if any(final_order.values()):
                broadcast_to_dashboards_nonblocking({
                    "messageType": "orderComplete",
                    "orderData": final_order,
                    "summary": final_summary,
                    "timestamp": int(time.time()),
                    "callSid": current_call_sid,  # ✅ Include call_sid
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
# Proper entry point for Render + production
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
