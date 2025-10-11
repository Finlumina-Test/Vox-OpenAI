import os
import json
import base64
import time
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

# Store all connected dashboard clients
dashboard_clients: Set[WebSocket] = set()


@app.websocket("/dashboard-stream")
async def dashboard_stream(websocket: WebSocket):
    await websocket.accept()
    DASHBOARD_TOKEN = os.getenv("DASHBOARD_TOKEN")
    client_call_id = None

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


async def broadcast_transcript(transcript_obj: dict):
    """Legacy global helper (still usable). Sends JSON to connected dashboards."""
    payload = json.dumps(transcript_obj)
    for client in list(dashboard_clients):
        try:
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
    return TwilioService.create_incoming_call_response(request)


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    Log.header("Client connected")
    await websocket.accept()

    connection_manager = WebSocketConnectionManager(websocket)
    openai_service = OpenAIService()
    audio_service = AudioService()

    try:
        await connection_manager.connect_to_openai()
        await openai_service.initialize_session(connection_manager)

        # ---------------------------
        # Broadcast Helper (local)
        # ---------------------------
        async def broadcast_to_dashboards(payload_obj: dict):
            """Send messages (audio/text) to all connected dashboard clients."""
            # Normalize timestamp to integer epoch seconds
            try:
                if "timestamp" not in payload_obj or payload_obj["timestamp"] is None:
                    payload_obj["timestamp"] = int(time.time())
                else:
                    ts = float(payload_obj["timestamp"])
                    payload_obj["timestamp"] = int(ts)
            except Exception:
                payload_obj["timestamp"] = int(time.time())

            payload = json.dumps(payload_obj)
            for client in list(dashboard_clients):
                try:
                    await client.send_text(payload)
                except Exception:
                    dashboard_clients.discard(client)

        # ---------------------------
        # Handle incoming audio/media events from Twilio (Caller side)
        # ---------------------------
        async def handle_media_event(data: dict) -> None:
            # Debug log the incoming structure from Twilio
            try:
                Log.debug("[media] incoming event keys: " + ", ".join(list(data.keys())))
            except Exception:
                Log.debug("[media] incoming event (unable to list keys)")

            # If Twilio sends a 'media' event, broadcast its payload (base64) to dashboards
            # Typical Twilio media event: {"event":"media","media":{"payload":"<base64>"},"start":{...}}
            if data.get("event") == "media":
                media = data.get("media") or {}
                payload_b64 = media.get("payload") or media.get("payloads") or None
                # support slightly different payload shapes
                if not payload_b64 and isinstance(media.get("payloads"), list) and media.get("payloads"):
                    payload_b64 = media.get("payloads")[0].get("payload")
                if payload_b64:
                    audio_msg = {
                        "messageType": "audio",
                        "speaker": "Caller",
                        "audio": payload_b64,
                        "encoding": "base64",
                        "callSid": getattr(connection_manager.state, "call_sid", None),
                        "streamSid": getattr(connection_manager.state, "stream_sid", None),
                        "timestamp": data.get("timestamp") or int(time.time())
                    }
                    Log.event("[dashboard] Broadcasting caller audio chunk", {"len_base64": len(payload_b64)})
                    await broadcast_to_dashboards(audio_msg)

            # If OpenAI connection is open, convert/process incoming audio and forward
            if connection_manager.is_openai_connected():
                audio_message = audio_service.process_incoming_audio(data)
                Log.debug(f"[media] audio_message produced? {bool(audio_message)}")
                if audio_message:
                    await connection_manager.send_to_openai(audio_message)
                    Log.debug("[media] forwarded audio chunk to OpenAI")

            # Twilio may also send direct transcription text (if enabled)
            if "text" in data and isinstance(data.get("text"), str) and data["text"].strip():
                transcript_obj = {
                    "messageType": "text",
                    "speaker": "Caller",
                    "text": data["text"].strip(),
                    "callSid": getattr(connection_manager.state, "call_sid", None),
                    "streamSid": getattr(connection_manager.state, "stream_sid", None),
                    "timestamp": data.get("timestamp") or int(time.time())
                }
                Log.event("[dashboard] Caller (from Twilio) says", {"text": transcript_obj["text"]})
                await broadcast_to_dashboards(transcript_obj)

        async def handle_stream_start(stream_sid: str) -> None:
            Log.event("Twilio stream started", {"streamSid": stream_sid})

        async def handle_mark_event() -> None:
            audio_service.handle_mark_event()

        # ---------------------------
        # Handle OpenAI events (assistant audio/text and also potential caller transcripts)
        # ---------------------------
        async def handle_audio_delta(response: dict) -> None:
            # 1) If OpenAI emits a caller transcript event, broadcast as text
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
                    Log.event("[dashboard] Caller (from OpenAI) says", {"text": caller_txt})
                    await broadcast_to_dashboards(caller_obj)
            except Exception as e:
                Log.debug("[openai] caller transcript extract error: " + str(e))

            # 2) Assistant audio deltas (output audio) -> broadcast to dashboard as base64
            audio_data = openai_service.extract_audio_response_data(response)
            if audio_data and audio_data.get("delta") is not None:
                delta = audio_data.get("delta")
                # If delta is bytes, encode to base64. If string, assume it's already base64 or text.
                try:
                    if isinstance(delta, (bytes, bytearray)):
                        delta_b64 = base64.b64encode(delta).decode("ascii")
                    elif isinstance(delta, str):
                        delta_b64 = delta
                    else:
                        # fallback: JSON-serialize whatever it is (rare)
                        delta_b64 = base64.b64encode(json.dumps(delta).encode("utf-8")).decode("ascii")
                except Exception:
                    delta_b64 = None

                if delta_b64:
                    ai_audio_obj = {
                        "messageType": "audio",
                        "speaker": "AI",
                        "audio": delta_b64,
                        "encoding": "base64",
                        "callSid": getattr(connection_manager.state, "call_sid", None),
                        "streamSid": getattr(connection_manager.state, "stream_sid", None),
                        "timestamp": response.get("timestamp") or int(time.time())
                    }
                    Log.event("[dashboard] Broadcasting AI audio chunk", {"len_base64": len(delta_b64)})
                    await broadcast_to_dashboards(ai_audio_obj)

            # 3) Assistant text transcripts (if any)
            transcript_text = None
            if hasattr(openai_service, "extract_transcript_text"):
                try:
                    transcript_text = openai_service.extract_transcript_text(response)
                except Exception as e:
                    transcript_text = None
                    Log.debug("[openai] failed to extract assistant transcript: " + str(e))

            if transcript_text:
                transcript_obj = {
                    "messageType": "text",
                    "speaker": "AI",
                    "text": transcript_text,
                    "callSid": getattr(connection_manager.state, "call_sid", None),
                    "streamSid": getattr(connection_manager.state, "stream_sid", None),
                    "timestamp": response.get("timestamp") or int(time.time())
                }
                Log.event("[dashboard] AI says", {"text": transcript_text})
                await broadcast_to_dashboards(transcript_obj)

            # 4) Continue existing flow: send audio back to Twilio if available
            if audio_data and connection_manager.state.stream_sid:
                if openai_service.is_goodbye_pending():
                    openai_service.mark_goodbye_audio_heard(audio_data.get('item_id'))
                audio_message = audio_service.process_outgoing_audio(
                    response,
                    connection_manager.state.stream_sid
                )
                if audio_message:
                    await connection_manager.send_to_twilio(audio_message)
                    mark_message = audio_service.create_mark_message(connection_manager.state.stream_sid)
                    await connection_manager.send_to_twilio(mark_message)

        async def handle_speech_started() -> None:
            Log.info("Speech started detected.")
            if openai_service.is_goodbye_pending():
                Log.info("Ignoring interruption during goodbye flow")
                return
            current_item_id = audio_service.get_current_item_id()
            if current_item_id:
                Log.info(f"Interrupting response with id: {current_item_id}")
                await handle_speech_started_event(connection_manager, openai_service, audio_service)

        async def handle_other_openai_event(response: dict) -> None:
            openai_service.process_event_for_logging(response)

            # Also check for caller transcripts in other event types
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
                    Log.event("[dashboard] Caller (other event) says", {"text": caller_txt})
                    await broadcast_to_dashboards(caller_obj)
            except Exception:
                pass

            if openai_service.is_tool_call(response):
                tool_call = openai_service.accumulate_tool_call(response)
                if tool_call:
                    handled = await openai_service.maybe_handle_tool_call(connection_manager, tool_call)
                    if handled:
                        return
            if openai_service.should_finalize_on_event(response):
                await openai_service.finalize_goodbye(connection_manager)

        # ---------------------------
        # Listeners
        # ---------------------------
        async def openai_receiver():
            await connection_manager.receive_from_openai(
                handle_audio_delta, handle_speech_started, handle_other_openai_event
            )

        async def renew_openai_session():
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

        # Run all background coroutines
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


# Start uvicorn when running the file directly (Render typically runs `python server.py`)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", getattr(Config, "PORT", 8000)))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
