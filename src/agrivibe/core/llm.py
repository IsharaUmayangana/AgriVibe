from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model

from agrivibe.config import Settings


def create_chat_model(settings: Settings) -> BaseChatModel:

    """Creates the chat mode named by 'CHAT_MODEL'.
    
    Args:
        settings: Application settings with the mode name and OpenAI API key.

    Returns:
        the chat model the agent use.
    """

    return init_chat_model(
        settings.chat_model,
        api_key=settings.openai_api_key.get_secret_value(),
        use_responses_api=True
    )