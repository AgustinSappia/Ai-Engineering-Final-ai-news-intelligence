"""
app/llm.py
----------
Fábrica de clientes de modelos (lo que en el Módulo 1 era el
`AsyncLLMManager`).

Los clientes de LangChain para OpenAI y Anthropic usan por dentro los SDK
ASÍNCRONOS oficiales cuando se los llama con `ainvoke` / `aembed_*`, así
que no hay llamadas de red síncronas escondidas. Además se configura:

- `timeout`: ninguna llamada queda colgada para siempre.
- `max_retries`: el SDK reintenta con backoff exponencial ante errores
  transitorios (red caída, 429 por rate limit, 5xx). Es lo mismo que
  implementamos a mano en la Pre-entrega 1, ahora delegado en el SDK.

Nombres de modelos, temperatura y timeouts salen de `Settings` (variables
de entorno), nunca de constantes en el código.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from app.config import Settings


def _clave(api_key: SecretStr | None) -> dict:
    """Solo pasa la key si está definida: si no, cada SDK la busca en el
    entorno por su cuenta (y da un error claro si tampoco está ahí)."""
    return {"api_key": api_key} if api_key is not None else {}


def crear_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.llm_temperatura,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_reintentos,
            **_clave(settings.openai_api_key),
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.anthropic_model,
        temperature=settings.llm_temperatura,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_reintentos,
        **_clave(settings.anthropic_api_key),
    )


def crear_embeddings(settings: Settings) -> Embeddings:
    """Embeddings para el RAG. Se usan con `aembed_documents` /
    `aembed_query`, que llaman al endpoint de OpenAI de forma asíncrona."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embeddings_model,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_reintentos,
        **_clave(settings.openai_api_key),
    )
