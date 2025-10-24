# server.py - Fixed Human Takeover Implementation
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
from services.transcription_service import TranscriptionService
from services.log_utils import Log


# ---------------------------
# Multi-call dashboard tracking
# ---------------------------
class DashboardClient:
    """Track dashboard websocket with optional call filtering."""
    def __init__(self, websocket: WebSocket, call_sid: Optional[str] = None):
        self.websocket = websocket
        self.call_sid = call_sid

active_calls: Dict[str, Dict[str, Any]] = {}
dashboard_clients: Set[DashboardClient] = set()


# ---------------------------
# Broadcasting functions
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


@app.websocket("/human-audio/{call_sid}")
async def human_audio_stream(websocket: WebSocket, call_sid: str):
    """
    WebSocket endpoint for human agent audio.
    Streams directly to Twilio for real-time caller experience.
    """
    await websocket.accept()
    
    Log.info(f"[HumanAudio] Connected for call {call_sid}")
    
    if call_sid not in active_calls:
        Log.error(f"[HumanAudio] Call {call_sid} not found in active_calls")
        await websocket.close(code=4004, reason="Call not found")
        return
    
    openai_service = active_calls[call_sid].get("openai_service")
    connection_manager = active_calls[call_sid].get("connection_manager")
    
    if not openai_service or not connection_manager:
        Log.error(f"[HumanAudio] Services not available for call {call_sid}")
        await websocket.close(code=4005, reason="Services not available")
        return
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get("type") == "audio":
                audio_base64 = data.get("audio")
                
                if audio_base64 and openai_service.is_human_in_control():
                    # ✅ Send directly to Twilio for real-time audio
                    stream_sid = getattr(connection_manager.state, 'stream_sid', None)
                    if stream_sid:
                        twilio_message = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": audio_base64
                            }
                        }
                        await connection_manager.send_to_twilio(twilio_message)
                        Log.debug(f"[HumanAudio] Sent audio to Twilio")
                    
                    # Also send to OpenAI for transcription/context
                    await openai_service.send_human_audio_to_openai(
                        audio_base64,
                        connection_manager
                    )
                    
                    # Broadcast to dashboard for monitoring
                    broadcast_to_dashboards_nonblocking({
                        "messageType": "audio",
                        "speaker": "Human",
                        "audio": audio_base64,
                        "timestamp": int(time.time() * 1000),
                        "callSid": call_sid
                    }, call_sid)
                    
    except WebSocketDisconnect:
        Log.info(f"[HumanAudio] Disconnected for call {call_sid}")
    except Exception as e:
        Log.error(f"[HumanAudio] Error: {e}")
    finally:
        # ✅ Notify OpenAI to resume when human disconnects
        if openai_service and openai_service.is_human_in_control():
            openai_service.disable_human_takeover()
            
            # Clear OpenAI's audio buffer to prevent stale audio
            try:
                await connection_manager.send_to_openai({
                    "type": "input_audio_buffer.clear"
                })
            except Exception:
                pass
            
            broadcast_to_dashboards_nonblocking({
                "messageType": "takeoverStatus",
                "active": False,
                "callSid": call_sid
            }, call_sid)


# ---------------------------
# Health endpoint
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


@app.api_route("/takeover", methods=["POST"])
async def handle_takeover(request: Request):
    """Handle human takeover requests from dashboard."""
    try:
        data = await request.json()
        call_sid = data.get("callSid")
        action = data.get("action")
        
        Log.info(f"[Takeover] Request: {action} for call {call_sid}")
        
        if not call_sid or action not in ["enable", "disable"]:
            return JSONResponse({"error": "Invalid request"}, status_code=400)
        
        if call_sid not in active_calls:
            Log.error(f"[Takeover] Call {call_sid} not found")
            return JSONResponse({"error": "Call not found"}, status_code=404)
        
        openai_service = active_calls[call_sid].get("openai_service")
        connection_manager = active_calls[call_sid].get("connection_manager")
        
        if not openai_service or not connection_manager:
            return JSONResponse({"error": "Service not available"}, status_code=500)
        
        if action == "enable":
            openai_service.enable_human_takeover()
            
            # ✅ Cancel any ongoing AI responses
            try:
                await connection_manager.send_to_openai({
                    "type": "response.cancel"
                })
                Log.info(f"[Takeover] Cancelled AI response for call {call_sid}")
            except Exception as e:
                Log.error(f"Failed to cancel AI response: {e}")
            
            Log.info(f"[Takeover] ✅ ENABLED for call {call_sid}")
            
            broadcast_to_dashboards_nonblocking({
                "messageType": "takeoverStatus",
                "active": True,
                "callSid": call_sid
            }, call_sid)
            
            return JSONResponse({"success": True, "message": "Takeover enabled"})
        else:
            openai_service.disable_human_takeover()
            
            # ✅ Clear audio buffer and resume AI
            try:
                await connection_manager.send_to_openai({
                    "type": "input_audio_buffer.clear"
                })
                Log.info(f"[Takeover] Cleared audio buffer for call {call_sid}")
            except Exception as e:
                Log.error(f"Failed to clear buffer: {e}")
            
            Log.info(f"[Takeover] ✅ DISABLED for call {call_sid}")
            
            broadcast_to_dashboards_nonblocking({
                "messageType": "takeoverStatus",
                "active": False,
                "callSid": call_sid
            }, call_sid)
            
            return JSONResponse({"success": True, "message": "Takeover disabled"})
            
    except Exception as e:
        Log.error(f"[Takeover] Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


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
    transcription_service = TranscriptionService()
    
    current_call_sid: Optional[str] = None

    # Audio streaming callback
    async def handle_audio_stream(audio_data: Dict[str, Any]):
        if current_call_sid:
            payload = {
                "messageType": "audio",
                "speaker": audio_data["speaker"],
                "audio": audio_data["audio"],
                "timestamp": audio_data.get("timestamp", int(time.time() * 1000)),
                "callSid": current_call_sid,
            }
            await _do_broadcast(payload, current_call_sid)

    transcription_service.set_audio_callback(handle_audio_stream)

    # OpenAI transcript callback
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

    openai_service.caller_transcript_callback = handle_openai_transcript
    openai_service.ai_transcript_callback = handle_openai_transcript

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

        async def handle_media_event(data: dict):
            """Handle incoming media from Twilio."""
            if data.get("event") == "media":
                media = data.get("media") or {}
                payload_b64 = media.get("payload")
                if payload_b64:
                    # ✅ Don't send caller audio to OpenAI during human takeover
                    if not openai_service.is_human_in_control():
                        if connection_manager.is_openai_connected():
                            try:
                                audio_message = audio_service.process_incoming_audio(data)
                                if audio_message:
                                    await connection_manager.send_to_openai(audio_message)
                            except Exception as e:
                                Log.error(f"[media] failed to send to OpenAI: {e}")
                    else:
                        Log.debug("[media] Skipping caller audio - human takeover active")
                    
                    # Stream caller audio to dashboard
                    try:
                        await transcription_service.stream_audio_chunk(payload_b64, source="Caller")
                    except Exception as e:
                        Log.error(f"[media] failed to stream caller audio: {e}")

        async def handle_audio_delta(response: dict):
            """Handle audio response from OpenAI."""
            try:
                # ✅ Skip AI audio if human has taken over
                if openai_service.is_human_in_control():
                    Log.debug("[Audio] Skipping AI audio - human takeover active")
                    return
                
                audio_data = openai_service.extract_audio_response_data(response) or {}
                delta = audio_data.get("delta")
                
                if delta:
                    # Send to Twilio
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
                            Log.error(f"[audio->twilio] failed: {e}")
                    
                    # Stream AI audio to dashboard
                    try:
                        await transcription_service.stream_audio_chunk(delta, source="AI")
                    except Exception as e:
                        Log.error(f"[audio->dashboard] failed: {e}")
                        
            except Exception as e:
                Log.error(f"[audio-delta] failed: {e}")

        async def handle_speech_started():
            """Handle user speech interruption."""
            try:
                # ✅ Don't interrupt if human is in control
                if not openai_service.is_human_in_control():
                    await connection_manager.send_mark_to_twilio()
            except Exception:
                pass

        async def handle_other_openai_event(response: dict):
            """Handle other OpenAI events."""
            openai_service.process_event_for_logging(response)
            await openai_service.extract_caller_transcript(response)
            await openai_service.extract_ai_transcript(response)
       
        async def openai_receiver():
            """Receive and process OpenAI events."""
            await connection_manager.receive_from_openai(
                handle_audio_delta,
                handle_speech_started,
                handle_other_openai_event,
            )

        async def renew_openai_session():
            """Periodically renew OpenAI session."""
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
            """Handle Twilio stream start."""
            nonlocal current_call_sid
            current_call_sid = getattr(connection_manager.state, 'call_sid', stream_sid)
            Log.event("Twilio Start", {"streamSid": stream_sid, "callSid": current_call_sid})
            
            # Register this call
            active_calls[current_call_sid] = {
                "openai_service": openai_service,
                "connection_manager": connection_manager,
                "audio_service": audio_service,
                "transcription_service": transcription_service,
                "order_extractor": order_extractor
            }
            Log.info(f"[ActiveCalls] Registered call {current_call_sid}")

            async def send_order_update(order_data: Dict[str, Any]):
                payload = {
                    "messageType": "orderUpdate",
                    "orderData": order_data,
                    "timestamp": int(time.time() * 1000),
                    "callSid": current_call_sid,
                }
                broadcast_to_dashboards_nonblocking(payload, current_call_sid)
            
            order_extractor.set_update_callback(send_order_update)

        async def on_mark_cb():
            """Handle Twilio mark event."""
            try:
                audio_service.handle_mark_event()
            except Exception:
                pass

        # Run all tasks concurrently
        await asyncio.gather(
            connection_manager.receive_from_twilio(handle_media_event, on_start_cb, on_mark_cb),
            openai_receiver(),
            renew_openai_session(),
        )

    except Exception as e:
        Log.error(f"Error in media stream handler: {e}")
    finally:
        # Final order summary
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

        # Cleanup
        if current_call_sid and current_call_sid in active_calls:
            del active_calls[current_call_sid]
            Log.info(f"[Cleanup] Removed call {current_call_sid} from active_calls")

        try:
            await transcription_service.shutdown()
        except Exception:
            pass

        try:
            await order_extractor.shutdown()
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
