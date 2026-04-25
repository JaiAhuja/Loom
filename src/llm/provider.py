from langchain_ollama import ChatOllama, OllamaEmbeddings

from config.settings import settings

# Module-level caches to avoid recreating instances with identical config
_llm_cache: dict[tuple, ChatOllama] = {}
_embeddings_cache: dict[str, OllamaEmbeddings] = {}


def get_llm(model: str = None, temperature: float = None) -> ChatOllama:
    """Get (or create) a cached ChatOllama LLM instance.

    Args:
        model: Ollama model name. Defaults to settings.OLLAMA_MODEL.
        temperature: Sampling temperature. Defaults to settings.OLLAMA_TEMPERATURE.
    """
    resolved_model = model or settings.OLLAMA_MODEL
    resolved_temp = temperature if temperature is not None else settings.OLLAMA_TEMPERATURE
    key = (resolved_model, resolved_temp)

    if key not in _llm_cache:
        _llm_cache[key] = ChatOllama(
            model=resolved_model,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=resolved_temp,
        )
    return _llm_cache[key]


def get_embeddings(model: str = None) -> OllamaEmbeddings:
    """Get (or create) a cached OllamaEmbeddings instance.

    Args:
        model: Embedding model name. Defaults to settings.OLLAMA_EMBEDDING_MODEL.
    """
    resolved_model = model or settings.OLLAMA_EMBEDDING_MODEL

    if resolved_model not in _embeddings_cache:
        _embeddings_cache[resolved_model] = OllamaEmbeddings(
            model=resolved_model,
            base_url=settings.OLLAMA_BASE_URL,
        )
    return _embeddings_cache[resolved_model]
