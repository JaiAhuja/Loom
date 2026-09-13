from langchain_ollama import ChatOllama, OllamaEmbeddings

from config.settings import settings


def get_llm(model: str = None, temperature: float = None, require_json: bool = False) -> ChatOllama:
    """Create an explicitly owned chat client.

    Client instances are deliberately not cached at module scope.  A caller
    that wants reuse owns that lifecycle (for example Streamlit's resource
    cache or a service object), which keeps tests and separate app sessions
    isolated.
    """
    resolved_model = model or settings.OLLAMA_MODEL
    resolved_temp = temperature if temperature is not None else settings.OLLAMA_TEMPERATURE
    settings.validate_runtime(require_ollama=True)
    if not isinstance(resolved_model, str) or not resolved_model.strip():
        raise ValueError("OLLAMA_MODEL must be non-empty")

    kwargs = {
        "model": resolved_model,
        "streaming": False,
        "base_url": settings.OLLAMA_BASE_URL,
        "temperature": resolved_temp,
        "num_ctx": settings.OLLAMA_NUM_CTX,
    }

    if require_json:
        kwargs["format"] = "json"

    return ChatOllama(**kwargs)


def get_embeddings(model: str = None) -> OllamaEmbeddings:
    """Create an explicitly owned Ollama embeddings client.

    Args:
        model: Embedding model name. Defaults to settings.OLLAMA_EMBEDDING_MODEL.
    """
    resolved_model = model or settings.OLLAMA_EMBEDDING_MODEL
    settings.validate_runtime(require_ollama=True, require_embeddings=True)
    if not isinstance(resolved_model, str) or not resolved_model.strip():
        raise ValueError("OLLAMA_EMBEDDING_MODEL must be non-empty")

    return OllamaEmbeddings(
        model=resolved_model,
        base_url=settings.OLLAMA_BASE_URL,
    )
