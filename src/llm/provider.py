from langchain_ollama import ChatOllama, OllamaEmbeddings

from config.settings import settings

# Module-level caches to avoid recreating instances with identical config
_llm_cache: dict[tuple, ChatOllama] = {}
_embeddings_cache: dict[str, OllamaEmbeddings] = {}


def get_llm(
    model: str = None, 
    temperature: float = None,  
    require_json: bool = False
) -> ChatOllama:
    
    resolved_model = model or settings.OLLAMA_MODEL
    resolved_temp = temperature if temperature is not None else settings.OLLAMA_TEMPERATURE
    
    key = (resolved_model, resolved_temp, require_json)

    if key not in _llm_cache:
        kwargs = {
            "model": resolved_model,
            "streaming": True,
            "base_url": settings.OLLAMA_BASE_URL,
            "temperature": resolved_temp,
            "num_ctx": 32768,  # Max context for granite4:tiny-h; adjust if using a different model
        }
        
        if require_json:
            kwargs["format"] = "json"

        _llm_cache[key] = ChatOllama(**kwargs)
        
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
