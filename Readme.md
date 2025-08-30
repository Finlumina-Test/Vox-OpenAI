# Speech Assistant with Twilio Voice and OpenAI Realtime API (Python)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-Realtime%20API-orange?style=flat-square&logo=openai)](https://platform.openai.com)
[![Twilio](https://img.shields.io/badge/Twilio-Voice%20API-red?style=flat-square&logo=twilio)](https://twilio.com)
[![Architecture](https://img.shields.io/badge/Architecture-Clean%20%26%20SRP-brightgreen?style=flat-square)](https://github.com)

> **A professional, production-ready speech assistant** that demonstrates clean architecture principles, enterprise-level code organization, and seamless integration between Twilio Voice and OpenAI's Realtime API.

This application enables **real-time voice conversations** with an AI assistant by establishing WebSocket connections between OpenAI's Realtime API and Twilio Voice, creating a seamless two-way audio communication experience.

## 🌟 **Why This Project Stands Out**

- **🏗️ Enterprise Architecture**: Clean separation of concerns with Single Responsibility Principle
- **📦 Professional Organization**: Modular service-based structure following Python best practices  
- **🔧 Production Ready**: Comprehensive error handling, logging, and configuration management
- **🧪 Highly Testable**: Independent services designed for easy unit testing
- **📚 Well Documented**: Extensive documentation and inline code comments
- **🚀 Easily Extensible**: Add new features without affecting existing functionality

## ✨ Architecture Highlights

This codebase follows **clean architecture principles** with strict **separation of concerns** and **single responsibility principle (SRP)**:

### 📁 Project Structure
```
├── main.py                    # 🎯 Application orchestration and FastAPI routes
├── config.py                  # ⚙️  Centralized configuration management
├── services/                  # 📦 Service modules organized by responsibility
│   ├── __init__.py           #     Package initialization and exports
│   ├── connection_manager.py #     🔌 WebSocket connection handling
│   ├── twilio_service.py     #     📞 Twilio-specific operations and TwiML
│   ├── openai_service.py     #     🤖 OpenAI Realtime API integration
│   └── audio_service.py      #     🔊 Audio processing and format conversion
├── requirements.txt           # 📦 Project dependencies
├── .env.example              # 🔧 Environment variables template
└── README.md                 # 📖 Project documentation
```

### 🏗️ Service Architecture

- **Config Service**: Manages environment variables and application settings
- **Connection Manager**: Handles WebSocket lifecycle and message routing
- **Twilio Service**: Encapsulates Twilio Voice API operations and TwiML generation
- **OpenAI Service**: Manages OpenAI Realtime API sessions and conversations
- **Audio Service**: Processes audio format conversions, timing, and synchronization

### 🎯 Benefits

- **🔧 Maintainable**: Changes are isolated to specific service layers
- **🧪 Testable**: Each service can be unit tested independently
- **📖 Readable**: Clear separation makes code self-documenting
- **♻️ Reusable**: Services can be extended or reused easily
- **🐛 Debuggable**: Issues are isolated to specific components

## 🎬 **Quick Demo**

1. **Call** your Twilio phone number
2. **Listen** to the AI greeting  
3. **Speak** naturally with the AI assistant
4. **Experience** real-time voice conversation with interruption handling

## Features

This application uses the following Twilio products in conjunction with OpenAI's Realtime API:
- Voice (and TwiML, Media Streams)
- Phone Numbers

## Prerequisites

To use the app, you will  need:

- **Python 3.9+** We used \`3.9.13\` for development; download from [here](https://www.python.org/downloads/).
- **A Twilio account.** You can sign up for a free trial [here](https://www.twilio.com/try-twilio).
- **A Twilio number with _Voice_ capabilities.** [Here are instructions](https://help.twilio.com/articles/223135247-How-to-Search-for-and-Buy-a-Twilio-Phone-Number-from-Console) to purchase a phone number.
- **An OpenAI account and an OpenAI API Key.** You can sign up [here](https://platform.openai.com/).
  - **OpenAI Realtime API access.**

## Local Setup

There are 4 required steps and 1 optional step to get the app up-and-running locally for development and testing:
1. Run ngrok or another tunneling solution to expose your local server to the internet for testing. Download ngrok [here](https://ngrok.com/).
2. (optional) Create and use a virtual environment
3. Install the packages
4. Twilio setup
5. Update the .env file

### Open an ngrok tunnel
When developing & testing locally, you'll need to open a tunnel to forward requests to your local development server. These instructions use ngrok.

Open a Terminal and run:
```
ngrok http 5050
```
Once the tunnel has been opened, copy the `Forwarding` URL. It will look something like: `https://[your-ngrok-subdomain].ngrok.app`. You will
need this when configuring your Twilio number setup.

Note that the `ngrok` command above forwards to a development server running on port `5050`, which is the default port configured in this application. If
you override the `PORT` defined in the `.env` file, you will need to update the `ngrok` command accordingly.

Keep in mind that each time you run the `ngrok http` command, a new URL will be created, and you'll need to update it everywhere it is referenced below.

### (Optional) Create and use a virtual environment

To reduce cluttering your global Python environment on your machine, you can create a virtual environment. On your command line, enter:

```
python3 -m venv env
source env/bin/activate
```

### Install required packages

In the terminal (with the virtual environment, if you set it up) run:
```
pip install -r requirements.txt
```

### Twilio setup

#### Point a Phone Number to your ngrok URL
In the [Twilio Console](https://console.twilio.com/), go to **Phone Numbers** > **Manage** > **Active Numbers** and click on the additional phone number you purchased for this app in the **Prerequisites**.

In your Phone Number configuration settings, update the first **A call comes in** dropdown to **Webhook**, and paste your ngrok forwarding URL (referenced above), followed by `/incoming-call`. For example, `https://[your-ngrok-subdomain].ngrok.app/incoming-call`. Then, click **Save configuration**.

### Update the .env file

Create a `.env` file, or copy the `.env.example` file to `.env`:

```bash
cp .env.example .env
```

In the .env file, update the `OPENAI_API_KEY` to your OpenAI API key from the **Prerequisites**:

```env
OPENAI_API_KEY=your_openai_api_key_here
PORT=5050
TEMPERATURE=0.8
SHOW_TIMING_MATH=false
```

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | - | ✅ Yes |
| `PORT` | Server port | `5050` | ❌ No |
| `TEMPERATURE` | OpenAI response creativity (0-1) | `0.8` | ❌ No |
| `SHOW_TIMING_MATH` | Enable timing debug logs | `false` | ❌ No |

## Run the app

Once ngrok is running, dependencies are installed, Twilio is configured properly, and the `.env` is set up, run the development server:

```bash
python main.py
```

The application will start and you should see:
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5050 (Press CTRL+C to quit)
```

## Test the app

With the development server running, call the phone number you purchased in the **Prerequisites**. After the introduction, you should be able to talk to the AI Assistant. Have fun!

## Advanced Configuration

### 🎙️ AI Personality Customization

The AI assistant's personality is configured in `config.py`. You can modify the `SYSTEM_MESSAGE` to change how the AI behaves:

```python
SYSTEM_MESSAGE = (
    "You are a helpful and bubbly AI assistant who loves to chat about "
    "anything the user is interested in and is prepared to offer them facts. "
    "You have a penchant for dad jokes, owl jokes, and rickrolling – subtly. "
    "Always stay positive, but work in a joke when appropriate."
)
```

### 🔊 Audio Configuration

Audio processing settings can be adjusted in the `AudioService` class in `services/audio_service.py`:

- **Format conversion**: Modify `AudioFormatConverter` for different audio formats
- **Timing sensitivity**: Adjust timing calculations in `AudioTimingManager`
- **Buffer management**: Configure buffering behavior in `AudioBufferManager`

### 🔌 Service Extension

The modular architecture makes it easy to extend functionality:

1. **Add new services**: Create new service classes in the `services/` directory following the same SRP pattern
2. **Modify existing services**: Each service is independent and can be modified without affecting others
3. **Custom integrations**: Add new external service integrations by creating dedicated service classes in `services/`

### 📦 Package Organization

The `services/` package is designed for easy importing:

```python
# Import individual services
from services import AudioService, TwilioService, OpenAIService

# Import specific components
from services.audio_service import AudioMetadata, AudioFormatConverter
from services.openai_service import OpenAIEventHandler
```

## Special Features

### Have the AI speak first

To have the AI voice assistant talk before the user, you can use the OpenAI service method:

```python
# In your connection handler, uncomment:
await openai_service.send_initial_greeting(connection_manager)
```

The initial greeting message can be customized in the `OpenAISessionManager.create_initial_conversation_item()` method.

### Interrupt handling/AI preemption

The application includes sophisticated interruption handling:

- **Speech detection**: When the user speaks, OpenAI sends `input_audio_buffer.speech_started`
- **Buffer clearing**: The system clears the Twilio Media Streams buffer  
- **Response truncation**: Sends OpenAI `conversation.item.truncate` to stop the current response
- **Timing calculations**: Precise timing ensures smooth conversation flow

The interruption logic is encapsulated in the `AudioService` and `OpenAIService` classes, making it easy to customize the behavior.

### Audio Processing Pipeline

The application includes a sophisticated audio processing pipeline:

1. **Format Conversion**: Seamless conversion between Twilio and OpenAI audio formats
2. **Timing Management**: Precise timestamp tracking for interruption handling
3. **Buffer Synchronization**: Mark-based synchronization between audio streams
4. **Quality Validation**: Audio payload validation and error handling

## Development & Testing

### Running Tests

Each service can be tested independently:

```python
# Example: Testing the AudioService
from audio_service import AudioService

audio_service = AudioService()
# Test audio processing...
```

### Adding New Features

1. **Identify the appropriate service** in the `services/` directory for your feature
2. **Follow SRP**: If the feature doesn't fit existing services, create a new service module
3. **Update package exports**: Add new services to `services/__init__.py`
4. **Test independently**: Test your service in isolation

### Example: Adding a New Service

```python
# services/notification_service.py
class NotificationService:
    """Handle user notifications and alerts."""
    
    def send_sms_notification(self, message: str) -> None:
        # Implementation here
        pass

# services/__init__.py - Add to exports
from .notification_service import NotificationService

__all__ = [
    # ... existing exports
    'NotificationService',
]
```

## Troubleshooting

### Common Issues

1. **WebSocket Connection Errors**: Check your OpenAI API key and network connectivity
2. **Audio Quality Issues**: Verify audio format conversions in `AudioService`
3. **Timing Problems**: Enable debug logging with `SHOW_TIMING_MATH=true`
4. **Service Integration**: Check service interfaces and dependency injection

### Debug Mode

Enable detailed logging by setting environment variables:

```env
SHOW_TIMING_MATH=true
```

This will provide detailed timing information for audio processing and interruption handling.

## Contributing

When contributing to this project:

1. **Follow the SRP**: Each class/function should have a single responsibility
2. **Maintain separation of concerns**: Keep services independent
3. **Add tests**: Test new services independently
4. **Update documentation**: Keep the README and code comments current

## License

This project is licensed under the MIT License - see the LICENSE file for details.
