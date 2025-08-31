import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# OpenAI Realtime System Instructions Structure
# Role & Objective        — who you are and what “success” means
# Personality & Tone      — the voice and style to maintain
# Context                 — retrieved context, relevant info
# Reference Pronunciations — phonetic guides for tricky words
# Tools                   — names, usage rules, and preambles
# Instructions / Rules    — do’s, don’ts, and approach
# Conversation Flow       — states, goals, and transitions
# Safety & Escalation     — fallback and handoff logic
# https://platform.openai.com/docs/guides/realtime-models-prompting

# VOICE: alloy, ash, ballad, coral, echo, sage, shimmer, and verse
class Config:
    """
    Configuration class that handles all application settings.
    Follows SRP by being responsible only for configuration management.
    """
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')
    TEMPERATURE: float = float(os.getenv('TEMPERATURE', 0.8))
    VOICE: str = 'alloy' 
    COMPANY_NAME: str = os.getenv('COMPANY_NAME', 'Acme Realty')
    
    # Server Configuration
    PORT: int = int(os.getenv('PORT', 5050))
    
    # Twilio REST (optional, required for programmatic hangup)
    TWILIO_ACCOUNT_SID: str | None = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN: str | None = os.getenv('TWILIO_AUTH_TOKEN')

    # AI Assistant Configuration
    SYSTEM_MESSAGE: str = (
        "You are a professional real-estate voice assistant for "
        f"{COMPANY_NAME}. Your goal is to qualify leads, answer property "
        "questions, and schedule tours—while capturing clean contact details.\n\n"

        "Voice & pacing: Speak clearly with short sentences suited for phone "
        "audio. Be concise, friendly, and confident. Ask one question at a "
        "time. Avoid long lists. If interrupted, stop immediately and listen.\n\n"

        "Safety & compliance: Follow the U.S. Fair Housing Act. Do not make or "
        "agree with statements about protected classes (race, color, religion, "
        "sex, sexual orientation, gender identity, national origin, familial "
        "status, disability, age). Avoid steering. Focus on objective property "
        "features and location facts. Do not provide legal, financial, or tax "
        "advice—suggest speaking with a licensed professional if asked.\n\n"

        "Information to collect (ask naturally, one item at a time, and "
        "confirm): name, phone or email, buy-or-rent, desired areas/neighborhoods, "
        "budget (price or monthly), property type (house/condo/townhome/apartment), "
        "beds/baths, timeline (move-in or closing), must-haves (parking, yard, pets), "
        "school/commute needs, pre-approval (if buying). When the user provides details, "
        "acknowledge briefly and summarize back.\n\n"

        "Behavioral rules:\n"
        "- Greet once and quickly move to a helpful question.\n"
        "- If the user asks about a specific listing, answer with known facts; if unsure, say so and offer to have an agent follow up.\n"
        "- Keep responses under ~2 short sentences unless the user asks for more.\n"
        "- Periodically confirm understanding with a brief summary.\n"
        "- When appropriate, propose scheduling a tour and collect a preferred time window.\n"
        "- If the user asks for a human, confirm best contact and promise a quick callback by our team.\n\n"

        "Escalation: When a lead is serious or requests an agent, politely collect "
        "their name, phone, email, property of interest, and best time to talk; then "
        "state that you'll pass the details to a licensed agent for follow‑up.\n\n"

        "Tone: professional, warm, and brand-safe. No jokes or emojis."
    )
    
    # Logging and Debug Configuration
    LOG_EVENT_TYPES: List[str] = [
        'error', 'response.content.done', 'rate_limits.updated',
        'response.done', 'input_audio_buffer.committed',
        'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
        'session.created', 'session.updated'
    ]
    SHOW_TIMING_MATH: bool = False

    # End-call farewell configuration
    # Farewell instruction template: ask the model to generate the goodbye itself
    END_CALL_FAREWELL_TEMPLATE: str = (
        "Please deliver a brief, polite goodbye to the caller on behalf of {company}. "
        "Keep it to one short sentence. Do not call any tools; speak the goodbye now."
    )
    END_CALL_GRACE_SECONDS: float = float(os.getenv('END_CALL_GRACE_SECONDS', 3)) # 3
    # Watchdog: if no goodbye audio starts within this window, finalize anyway
    END_CALL_WATCHDOG_SECONDS: float = float(os.getenv('END_CALL_WATCHDOG_SECONDS', 4))
    # Realtime session renewal (preemptive reconnect before 60-minute cap)
    REALTIME_SESSION_RENEW_SECONDS: int = int(os.getenv('REALTIME_SESSION_RENEW_SECONDS', 55 * 60))

    @staticmethod
    def build_end_call_farewell(reason: str | None = None) -> str:
        """Return an instruction prompting the model to generate the goodbye itself.
        If a reason is provided, the instruction asks to briefly acknowledge it.
        """
        company = getattr(Config, 'COMPANY_NAME', None) or 'our team'
        has_reason = isinstance(reason, str) and reason.strip()
        base = Config.END_CALL_FAREWELL_TEMPLATE.format(company=company)
        if has_reason:
            return base + " Acknowledge that the caller requested to end the call."
        return base
    
    @classmethod
    def validate_required_config(cls) -> None:
        """
        Validates that all required configuration values are present.
        Raises ValueError if any required configuration is missing.
        """
        if not cls.OPENAI_API_KEY:
            raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')
    
    @classmethod
    def get_openai_websocket_url(cls) -> str:
        """
        Constructs the OpenAI WebSocket URL with configuration parameters.
        """
        return (
            f"wss://api.openai.com/v1/realtime"
            f"?model=gpt-realtime"
            f"&temperature={cls.TEMPERATURE}"
            f"&voice={cls.VOICE}"
        )
    
    @classmethod
    def get_openai_headers(cls) -> dict:
        """
        Returns the headers needed for OpenAI API authentication.
        """
        return {
            "Authorization": f"Bearer {cls.OPENAI_API_KEY}"
        }

    @classmethod
    def has_twilio_credentials(cls) -> bool:
        """Return True if Twilio credentials are configured."""
        return bool(cls.TWILIO_ACCOUNT_SID and cls.TWILIO_AUTH_TOKEN)


# Initialize and validate configuration when module is imported
Config.validate_required_config()
