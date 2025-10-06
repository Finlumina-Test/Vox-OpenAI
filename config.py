import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """
    Central configuration for the Pakistani multilingual voice assistant.
    Handles OpenAI, Twilio, server, logging, and assistant behavior.
    """
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')
    TEMPERATURE: float = float(os.getenv('TEMPERATURE', 0.8))
    VOICE: str = 'alloy'  # Options: alloy, ash, ballad, coral, echo, sage, shimmer, verse
    OPENAI_REALTIME_MODEL: str = os.getenv('OPENAI_REALTIME_MODEL', 'gpt-realtime')
    COMPANY_NAME: str = os.getenv('COMPANY_NAME', 'Finlumina-Vox')
    
    # Server Configuration
    PORT: int = int(os.getenv('PORT', 5050))
    
    # Twilio REST credentials (optional; used for programmatic hangup)
    TWILIO_ACCOUNT_SID: str | None = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN: str | None = os.getenv('TWILIO_AUTH_TOKEN')

    # AI Assistant Configuration
    SYSTEM_MESSAGE: str = (
        "You are a professional Pakistani voice assistant for "
        f"{COMPANY_NAME}. Your role is to take orders, answer questions, "
        "and guide customers politely.\n\n"

        "Language & Tone:\n"
        "- Speak primarily in English and Urdu; optionally use Punjabi when requested.\n"
        "- Avoid Hindi words. Use words familiar to Pakistani callers.\n"
        "- Warm, polite, and conversational; like a real call center agent.\n"
        "- Short, clear sentences; avoid long lists.\n\n"

        "Behavior:\n"
        "- Greet the caller once, then move quickly to ask relevant questions.\n"
        "- Listen actively; if interrupted, pause immediately and respond naturally.\n"
        "- Confirm understanding occasionally by summarizing back details.\n"
        "- Ask for order details step by step (e.g., item, quantity, delivery info).\n"
        "- If unsure, politely ask for clarification.\n"
        "- If a human agent is requested, collect name, phone, and reason, then assure callback.\n\n"

        "Example context (initial placeholders for early testing):\n"
        "- Customer: 'Mujhe 2 burgers chahiye aur 1 coke.'\n"
        "- Agent: 'Sure! 2 burgers aur 1 coke. Kya aap delivery address bata denge?'\n"
        "- Customer: 'I want a paneer pizza.'\n"
        "- Agent: 'Paneer pizza, got it! Kya aap pick-up karenge ya delivery chahiye?'\n\n"

        "Rules:\n"
        "- Stepwise questions; one at a time.\n"
        "- Polite Urdu/English code-switch.\n"
        "- Responses under ~2 short sentences unless user asks for more.\n"
        "- Escalate to human politely if requested.\n"
    )

    # Logging / Debug
    LOG_EVENT_TYPES: List[str] = [
        'error', 'response.content.done', 'rate_limits.updated',
        'response.done', 'input_audio_buffer.committed',
        'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
        'session.created', 'session.updated'
    ]
    SHOW_TIMING_MATH: bool = False

    # End-call / farewell configuration
    END_CALL_FAREWELL_TEMPLATE: str = (
        "Please deliver a brief, polite goodbye to the caller on behalf of {company}. "
        "Keep it to one short sentence. Do not call any tools; speak the goodbye now."
    )
    END_CALL_GRACE_SECONDS: float = float(os.getenv('END_CALL_GRACE_SECONDS', 3))
    END_CALL_WATCHDOG_SECONDS: float = float(os.getenv('END_CALL_WATCHDOG_SECONDS', 4))
    
    # Realtime session renewal (preemptive reconnect before 60-minute cap)
    REALTIME_SESSION_RENEW_SECONDS: int = int(os.getenv('REALTIME_SESSION_RENEW_SECONDS', 55 * 60))

    # Methods
    @staticmethod
    def build_end_call_farewell(reason: str | None = None) -> str:
        company = getattr(Config, 'COMPANY_NAME', None) or 'our team'
        base = Config.END_CALL_FAREWELL_TEMPLATE.format(company=company)
        if isinstance(reason, str) and reason.strip():
            return base + " Acknowledge that the caller requested to end the call."
        return base

    @classmethod
    def validate_required_config(cls) -> None:
        if not cls.OPENAI_API_KEY:
            raise ValueError('Missing OpenAI API key in .env file.')

    @classmethod
    def get_openai_websocket_url(cls) -> str:
        return (
            f"wss://api.openai.com/v1/realtime"
            f"?model={cls.OPENAI_REALTIME_MODEL}"
            f"&temperature={cls.TEMPERATURE}"
            f"&voice={cls.VOICE}"
        )

    @classmethod
    def get_openai_headers(cls) -> dict:
        return {
            "Authorization": f"Bearer {cls.OPENAI_API_KEY}"
        }

    @classmethod
    def has_twilio_credentials(cls) -> bool:
        return bool(cls.TWILIO_ACCOUNT_SID and cls.TWILIO_AUTH_TOKEN)


# Initialize and validate
Config.validate_required_config()