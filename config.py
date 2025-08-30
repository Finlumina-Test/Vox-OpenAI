import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """
    Configuration class that handles all application settings.
    Follows SRP by being responsible only for configuration management.
    """
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')
    TEMPERATURE: float = float(os.getenv('TEMPERATURE', 0.8))
    VOICE: str = 'alloy'
    
    # Server Configuration
    PORT: int = int(os.getenv('PORT', 5050))
    
    # AI Assistant Configuration
    SYSTEM_MESSAGE: str = (
        "You are a helpful and bubbly AI assistant who loves to chat about "
        "anything the user is interested in and is prepared to offer them facts. "
        "You have a penchant for dad jokes, owl jokes, and rickrolling – subtly. "
        "Always stay positive, but work in a joke when appropriate."
    )
    
    # Logging and Debug Configuration
    LOG_EVENT_TYPES: List[str] = [
        'error', 'response.content.done', 'rate_limits.updated',
        'response.done', 'input_audio_buffer.committed',
        'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
        'session.created', 'session.updated'
    ]
    SHOW_TIMING_MATH: bool = False
    
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


# Initialize and validate configuration when module is imported
Config.validate_required_config()
