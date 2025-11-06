# server.py - FINAL WORKING VERSION
# Just copy this entire file and replace your server.py

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
from services.silence_detection import SilenceDetector


class DashboardClient:
    def __init__(self, websocket: WebSocket, call_sid: Optional[str] = None):
        self.websocket = websocket
        self.call_sid = call_sid

active_calls: Dict[str, Dict[str, Any]] = {}
dashboard_clients: Set[DashboardClient] = set()


async def _do_broadcast(payload: Dict[str, Any], call_sid: Optional[str] = None):
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
    try:
        asyncio.create_task(_do_broadcast(payload, call_sid))
    except Exception as e:
        Log.error(f"[Broadcast] Failed to create broadcast task: {e}")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}


@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    return TwilioService.create_incoming_call_response(request)


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
    
    active_calls[call_sid]["human_audio_ws"] = websocket
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get("type") == "audio":
                audio_base64 = data.get("audio")
                
                if audio_base64 and openai_service.is_human_in_control():
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
                    
                    await openai_service.send_human_audio_to_openai(
                        audio_base64,
                        connection_manager
                    )
                    
    except WebSocketDisconnect:
        Log.info(f"[HumanAudio] Disconnected for call {call_sid}")
    except Exception as e:
        Log.error(f"[HumanAudio] Error: {e}")
    finally:
        if call_sid in active_calls and "human_audio_ws" in active_calls[call_sid]:
            del active_calls[call_sid]["human_audio_ws"]
        
        if openai_service and openai_service.is_human_in_control():
            openai_service.disable_human_takeover()
            
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


@app.api_route("/takeover", methods=["POST"])
async def handle_takeover(request: Request):
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
            
            # Try to cancel (ignore if no response active - this is normal)
            try:
                await connection_manager.send_to_openai({
                    "type": "response.cancel"
                })
                Log.info(f"[Takeover] Cancelled AI response")
            except Exception:
                Log.debug(f"[Takeover] No active response to cancel (normal)")
            
            try:
                await connection_manager.send_to_openai({
                    "type": "input_audio_buffer.clear"
                })
            except Exception:
                pass
            
            Log.info(f"[Takeover] ✅ ENABLED for call {call_sid}")
            
            broadcast_to_dashboards_nonblocking({
                "messageType": "takeoverStatus",
                "active": True,
                "callSid": call_sid
            }, call_sid)
            
            return JSONResponse({"success": True, "message": "Takeover enabled"})
        else:
            openai_service.disable_human_takeover()
            
            try:
                await connection_manager.send_to_openai({
                    "type": "response.cancel"
                })
            except Exception:
                pass
            
            try:
                await connection_manager.send_to_openai({
                    "type": "input_audio_buffer.clear"
                })
            except Exception:
                pass
            
            await asyncio.sleep(0.3)
            
            try:
                await connection_manager.send_to_openai({
                    "type": "input_audio_buffer.commit"
                })
            except Exception:
                pass
            
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


@app.api_route("/end-call", methods=["POST"])
async def handle_end_call(request: Request):
    try:
        data = await request.json()
        call_sid = data.get("callSid")
        
        Log.info(f"[EndCall] Request to end call {call_sid}")
        
        if not call_sid:
            return JSONResponse({"error": "Invalid request"}, status_code=400)
        
        if call_sid not in active_calls:
            Log.warning(f"[EndCall] Call {call_sid} not in active_calls (might have ended)")
            # Still try to end it via Twilio
        
        openai_service = active_calls.get(call_sid, {}).get("openai_service")
        
        if openai_service and openai_service.is_human_in_control():
            openai_service.disable_human_takeover()
        
        if Config.has_twilio_credentials():
            try:
                from twilio.rest import Client
                from twilio.base.exceptions import TwilioRestException
                
                client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
                client.calls(call_sid).update(status='completed')
                
                Log.info(f"[EndCall] ✅ Call {call_sid} ended successfully")
                
                broadcast_to_dashboards_nonblocking({
                    "messageType": "callEnded",
                    "callSid": call_sid,
                    "timestamp": int(time.time() * 1000)
                }, call_sid)
                
                return JSONResponse({
                    "success": True, 
                    "message": "Call ended successfully"
                })
                
            except TwilioRestException as e:
                if e.status == 404:
                    Log.info(f"[EndCall] Call already ended - this is fine")
                    
                    broadcast_to_dashboards_nonblocking({
                        "messageType": "callEnded",
                        "callSid": call_sid,
                        "timestamp": int(time.time() * 1000)
                    }, call_sid)
                    
                    return JSONResponse({
                        "success": True, 
                        "message": "Call already ended"
                    })
                else:
                    Log.error(f"[EndCall] Twilio error: {e}")
                    return JSONResponse({
                        "error": f"Twilio error: {str(e)}"
                    }, status_code=500)
            except Exception as e:
                Log.error(f"[EndCall] Error: {e}")
                return JSONResponse({
                    "error": str(e)
                }, status_code=500)
        else:
            return JSONResponse({
                "error": "Twilio credentials not configured"
            }, status_code=500)
            
    except Exception as e:
        Log.error(f"[EndCall] Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    Log.header("Client connected")
    await websocket.accept()

    connection_manager = WebSocketConnectionManager(websocket)
    openai_service = OpenAIService()
    audio_service = AudioService()
    order_extractor = OrderExtractionService()
    transcription_service = TranscriptionService()
    
    caller_silence_detector = SilenceDetector()
    ai_silence_detector = SilenceDetector()
    
    current_call_sid: Optional[str] = None
    
    ai_audio_queue = asyncio.Queue()
    ai_stream_task = None
    shutdown_flag = False
    
    ai_currently_speaking = False
    last_speech_started_time = 0

    async def ai_audio_streamer():
        nonlocal ai_currently_speaking
        Log.info("[AI Streamer] 🎵 Started")
        
        while not shutdown_flag:
            try:
                audio_data = await ai_audio_queue.get()
                
                if audio_data is None:
                    break
                
                audio_b64 = audio_data.get("audio", "")
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    duration_seconds = len(audio_bytes) / 8000.0
                except Exception as e:
                    duration_seconds = 0.02
                
                ai_currently_speaking = True
                
                if current_call_sid:
                    broadcast_to_dashboards_nonblocking({
                        "messageType": "audio",
                        "speaker": "AI",
                        "audio": audio_b64,
                        "timestamp": audio_data.get("timestamp", int(time.time() * 1000)),
                        "callSid": current_call_sid,
                    }, current_call_sid)
                
                await asyncio.sleep(duration_seconds)
                ai_audio_queue.task_done()
                
            except Exception as e:
                Log.error(f"[AI Streamer] Error: {e}")
                await asyncio.sleep(0.01)
        
        Log.info("[AI Streamer] 🛑 Stopped")
    
    async def handle_ai_audio(audio_data: Dict[str, Any]):
        await ai_audio_queue.put(audio_data)

    async def handle_openai_transcript(transcription_data: Dict[str, Any]):
        if not transcription_data or not isinstance(transcription_data, dict):
            return
        
        speaker = transcription_data.get("speaker")
        text = transcription_data.get("text")
        if not speaker or not text:
            return
        
        if speaker == "AI" and openai_service.is_human_in_control():
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
            if data.get("event") == "media":
                media = data.get("media") or {}
                payload_b64 = media.get("payload")
                if payload_b64:
                    # Send ALL audio (silence detection disabled for testing)
                    should_send_to_dashboard = True
                    
                    if openai_service.is_human_in_control():
                        if current_call_sid and current_call_sid in active_calls:
                            human_ws = active_calls[current_call_sid].get("human_audio_ws")
                            if human_ws:
                                try:
                                    await human_ws.send_text(json.dumps({
                                        "type": "caller_audio",
                                        "audio": payload_b64,
                                        "timestamp": int(time.time() * 1000)
                                    }))
                                except Exception as e:
                                    Log.error(f"[media] Failed to send to human: {e}")
                        
                        if should_send_to_dashboard:
                            broadcast_to_dashboards_nonblocking({
                                "messageType": "audio",
                                "speaker": "Caller",
                                "audio": payload_b64,
                                "timestamp": int(time.time() * 1000),
                                "callSid": current_call_sid
                            }, current_call_sid)
                    else:
                        if connection_manager.is_openai_connected():
                            try:
                                audio_message = audio_service.process_incoming_audio(data)
                                if audio_message:
                                    await connection_manager.send_to_openai(audio_message)
                            except Exception as e:
                                Log.error(f"[media] failed to send to OpenAI: {e}")
                        
                        if should_send_to_dashboard:
                            broadcast_to_dashboards_nonblocking({
                                "messageType": "audio",
                                "speaker": "Caller",
                                "audio": payload_b64,
                                "timestamp": int(time.time() * 1000),
                                "callSid": current_call_sid
                            }, current_call_sid)

        async def handle_audio_delta(response: dict):
            try:
                if openai_service.is_human_in_control():
                    return
                
                audio_data = openai_service.extract_audio_response_data(response) or {}
                delta = audio_data.get("delta")
                
                if delta:
                    # Send ALL AI audio (silence detection disabled)
                    should_send_to_dashboard = True
                    
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
                    
                    if should_send_to_dashboard:
                        await handle_ai_audio({
                            "audio": delta,
                            "timestamp": int(time.time() * 1000)
                        })
                        
            except Exception as e:
                Log.error(f"[audio-delta] failed: {e}")

        async def handle_speech_started():
            nonlocal ai_currently_speaking, last_speech_started_time
            
            try:
                if not openai_service.is_human_in_control():
                    Log.info("🛑 [Interruption] USER SPEAKING")
                    
                    last_speech_started_time = time.time()
                    
                    try:
                        stream_sid = getattr(connection_manager.state, 'stream_sid', None)
                        if stream_sid:
                            clear_message = {
                                "event": "clear",
                                "streamSid": stream_sid
                            }
                            await connection_manager.send_to_twilio(clear_message)
                    except Exception:
                        pass
                    
                    try:
                        await connection_manager.send_to_openai({
                            "type": "response.cancel"
                        })
                    except Exception:
                        pass
                    
                    cleared_count = 0
                    while not ai_audio_queue.empty():
                        try:
                            ai_audio_queue.get_nowait()
                            ai_audio_queue.task_done()
                            cleared_count += 1
                        except:
                            break
                    
                    ai_currently_speaking = False
                    
                    await connection_manager.send_mark_to_twilio()
                    
            except Exception as e:
                Log.error(f"[Interruption] Error: {e}")

        async def handle_other_openai_event(response: dict):
            openai_service.process_event_for_logging(response)
            await openai_service.extract_caller_transcript(response)
            
            if not openai_service.is_human_in_control():
                await openai_service.extract_ai_transcript(response)
       
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
            nonlocal current_call_sid, ai_stream_task
            current_call_sid = getattr(connection_manager.state, 'call_sid', stream_sid)
            Log.event("Twilio Start", {"streamSid": stream_sid, "callSid": current_call_sid})
            
            caller_silence_detector.reset()
            ai_silence_detector.reset()
            
            if ai_stream_task is None or ai_stream_task.done():
                ai_stream_task = asyncio.create_task(ai_audio_streamer())
                Log.info("[AI Streamer] Task started")
            
            active_calls[current_call_sid] = {
                "openai_service": openai_service,
                "connection_manager": connection_manager,
                "audio_service": audio_service,
                "transcription_service": transcription_service,
                "order_extractor": order_extractor,
                "human_audio_ws": None
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
        shutdown_flag = True
        try:
            await ai_audio_queue.put(None)
            if ai_stream_task and not ai_stream_task.done():
                await asyncio.wait([ai_stream_task], timeout=2.0)
        except Exception:
            pass
        
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

        if current_call_sid and current_call_sid in active_calls:
            del active_calls[current_call_sid]

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", getattr(Config, "PORT", 8000))),
        log_level="info",
        reload=False,
    )
