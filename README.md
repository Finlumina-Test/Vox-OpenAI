# Twilio Speech Assistant (Python) — OpenAI Realtime API + Twilio Voice

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Twilio](https://img.shields.io/badge/Twilio-Voice%20API-F22F46?style=flat-square&logo=twilio)](https://twilio.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-Realtime%20API-412991?style=flat-square&logo=openai)](https://platform.openai.com)

A production-ready, cleanly-architected voice assistant that connects a Twilio phone call to OpenAI’s Realtime API over WebSockets for true two-way, low-latency audio conversations.

## Highlights

- Clean architecture with Single Responsibility Principle (SRP)
- Real-time audio: Twilio Media Streams ⇄ OpenAI Realtime API
- Interruption handling (user can interrupt the AI mid-response)
- Modular, testable services with clear boundaries
- Simple local setup and Twilio webhook wiring
- Graceful call termination: model triggers an end_call tool → assistant plays a short farewell → call hangs up automatically

## Project Structure

```
├── main.py                      # Application entrypoint (FastAPI + orchestration)
├── config.py                    # Centralized configuration (env/env vars)
├── services/                    # SRP-aligned service modules
│   ├── __init__.py              # Package exports
│   ├── audio_service.py         # Audio conversion, timing, buffering, marks
│   ├── connection_manager.py    # WebSocket plumbing (Twilio ⇄ OpenAI)
│   ├── log_utils.py             # Structured logging helpers
│   ├── openai_service.py        # Session + events + conversation control
│   └── twilio_service.py        # TwiML + Twilio payload helpers
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── CODE_OF_CONDUCT.md           # Contribution/code of conduct
├── LICENSE                      # MIT license
├── Promptingrealtimemodels.md   # Realtime prompting notes
└── README.md                    # This file
```

## Prerequisites

- Python 3.9+
- Twilio account and a phone number with Voice capability
- OpenAI API key with Realtime API access
- A tunneling tool (e.g. ngrok) to expose your local server

## Quick Start

1) Create a virtual environment and install dependencies

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

2) Configure environment

```
cp .env.example .env
# open .env and set your values
```

Required variables (see more in “Configuration”):

- OPENAI_API_KEY=...
- PORT=5050 (default)
- TEMPERATURE=0.8 (default)
- SHOW_TIMING_MATH=false (default)

3) Run the app locally

```
python main.py
```

You should see Uvicorn running on 0.0.0.0:5050 by default.

4) Expose your server with ngrok (or similar)

```
ngrok http 5050
```

Copy the HTTPS forwarding URL for use in the next step.

5) Point your Twilio phone number to the webhook

In Twilio Console → Phone Numbers → Manage → Active Numbers → your number:

- “A call comes in” → Webhook → https://YOUR-NGROK-SUBDOMAIN.ngrok.app/incoming-call
- Save

Call your number and start talking to the assistant.

## Configuration

Configuration is centralized in `config.py` and loaded from environment variables.

| Variable                    | Description                                         | Default |
|-----------------------------|-----------------------------------------------------|---------|
| OPENAI_API_KEY              | OpenAI API key                                      | —       |
| PORT                        | HTTP server port                                    | 5050    |
| TEMPERATURE                 | Model temperature (0–1)                             | 0.8     |
| SHOW_TIMING_MATH            | Verbose timing logs for interruption math           | false   |
| COMPANY_NAME                | Brand used in prompts                               | Acme Realty |
| VOICE                       | OpenAI voice id (e.g., alloy)                       | alloy   |
| END_CALL_GRACE_SECONDS      | Pause after farewell before hangup (seconds)        | 3       |
| END_CALL_WATCHDOG_SECONDS   | Timeout if farewell audio doesn’t start (seconds)   | 4       |
| REALTIME_SESSION_RENEW_SECONDS | Preemptive session renewal cadence (seconds)     | 3300    |
| TWILIO_ACCOUNT_SID          | Optional: enable REST hangup                        | —       |
| TWILIO_AUTH_TOKEN           | Optional: enable REST hangup                        | —       |

`config.py` builds the OpenAI Realtime WS URL and headers dynamically using these values.

## How It Works

- Twilio hits `/incoming-call` and receives TwiML that connects the call to a WebSocket media stream (`/media-stream`).
- `WebSocketConnectionManager` opens a second WebSocket to OpenAI Realtime.
- `AudioService` converts/labels audio, tracks timestamps, buffers, and manages “marks”.
- `OpenAIService` initializes the session, processes events, and handles truncation when the caller speaks.
- The system streams audio in both directions with minimal latency.
- When the model calls the `end_call` tool, the app queues a brief farewell via `response.create`, ignores user interruptions during the goodbye, then ends the call using Twilio REST (if credentials are set) or by closing the Twilio media stream WebSocket.

### Call hangup behavior

- Trigger: the model issues a function call `end_call` when the user says goodbye or requests to end.
- Farewell: the app sends a `response.create` with `response.instructions` to have the assistant speak one short goodbye.
- Finalize: after a short grace period, the call ends via Twilio REST or, if not configured, by closing the Twilio WS stream.
- Watchdog: if farewell audio doesn’t begin within `END_CALL_WATCHDOG_SECONDS`, the app finalizes anyway to avoid stalling.

## Customization Tips

- Change the AI personality in `config.py` (the `SYSTEM_MESSAGE`).
- Adjust timing behavior in `AudioTimingManager` (inside `audio_service.py`).
- Swap audio formats in `AudioFormatConverter` if your integration needs change.

## Troubleshooting

- WebSocket errors: verify `OPENAI_API_KEY`, network egress, and that Realtime API access is enabled for your key.
- No audio: ensure your Twilio number is configured to call the correct tunnel URL and that the tunnel is active.
- Interruption isn’t working: set `SHOW_TIMING_MATH=true` and watch server logs to confirm timestamps/marks.
- Realtime schema errors (e.g., `unknown_parameter`): the app uses `{"type":"response.create","response":{"instructions":"..."}}` and sets modalities in the session. If you change session config, align payloads with the latest Realtime docs.

## License

MIT — see `LICENSE`.
# Twilio-speech-assistant-openai-realtime-api-python
A professional speech assistant built with Python, Twilio Voice, and OpenAI Realtime API. Features clean architecture with Single Responsibility Principle, organized service modules, and enterprise-level code organization.
