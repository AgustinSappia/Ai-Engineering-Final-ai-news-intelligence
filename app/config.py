"""
app/config.py
-------------
Configuración centralizada y TIPADA del sistema.

Todo lo que puede cambiar entre entornos (modelos, URLs, nombres de
colecciones, umbrales, timeouts) se lee de variables de entorno o del
archivo `.env`. Ningún módulo lee `os.getenv` por su cuenta ni tiene un
nombre de modelo o de colección escrito a mano: todos piden
`obtener_settings()`.

`pydantic-settings` valida los valores al arrancar. Si alguien pone
`RAG_K=cero` o `LLM_PROVIDER=gemini`, la aplicación falla en el arranque
con un mensaje claro, en lugar de fallar más tarde en medio de un job.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Además de leer el .env para `Settings`, se carga en `os.environ`: las
# librerías de LangSmith y de los proveedores de LLM leen sus variables
# directamente del entorno del proceso.
load_dotenv(override=False)

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ----------------------------------------------------------------
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    llm_temperatura: float = Field(0.0, ge=0.0, le=1.0)
    llm_timeout_s: float = Field(60.0, gt=0, description="Timeout de cada llamada al LLM.")
    llm_max_reintentos: int = Field(3, ge=0, le=10, description="Reintentos con backoff ante errores de red/429.")

    # --- Embeddings (siempre OpenAI: Anthropic no ofrece embeddings) --------
    embeddings_model: str = "text-embedding-3-small"

    # --- Infraestructura ------------------------------------------------------
    redis_url: str = "redis://localhost:6379"
    redis_socket_timeout_s: float = Field(
        30.0, gt=0, description="Timeout de lectura del cliente Redis (redis-py 8 trae 5 s por defecto)."
    )
    chroma_host: str = "localhost"
    chroma_port: int = Field(8000, ge=1, le=65535)
    chroma_coleccion: str = "noticias_ia"

    # --- RAG ------------------------------------------------------------------
    corpus_dir: Path = RAIZ_PROYECTO / "data" / "corpus"
    rag_chunk_tokens: int = Field(160, ge=50, le=2000)
    rag_chunk_solapamiento: int = Field(40, ge=0, le=500)
    rag_k: int = Field(5, ge=1, le=10, description="Noticias que devuelve una búsqueda (presupuesto de contexto).")
    rag_candidatos: int = Field(20, ge=1, le=200, description="Candidatos por método antes de fusionar.")
    rag_rrf_k: int = Field(60, ge=1, description="Constante de Reciprocal Rank Fusion.")
    rag_umbral_similitud: float = Field(
        0.39,
        ge=-1.0,
        le=1.0,
        description="Similitud coseno mínima para considerar relevante un fragmento (calibrada con evaluar_rag).",
    )

    # --- Orquestación ---------------------------------------------------------
    max_pasos: int = Field(10, ge=3, le=50, description="Techo anti 'Supervisor Infinito'.")
    max_intentos_investigacion: int = Field(2, ge=1, le=5)
    timeout_job_segundos: float = Field(300.0, gt=0)
    job_ttl_segundos: int = Field(7 * 24 * 3600, gt=0)
    sse_keepalive_s: float = Field(10.0, gt=0, description="Cada cuánto el stream SSE manda un keep-alive.")
    checkpoint_ttl_minutos: int = Field(24 * 60, gt=0)

    # --- Observabilidad -------------------------------------------------------
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "ai-news-intelligence"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _validar_coherencia(self) -> "Settings":
        if self.rag_chunk_solapamiento >= self.rag_chunk_tokens:
            raise ValueError("RAG_CHUNK_SOLAPAMIENTO tiene que ser menor que RAG_CHUNK_TOKENS")
        if self.rag_candidatos < self.rag_k:
            raise ValueError("RAG_CANDIDATOS tiene que ser mayor o igual que RAG_K")
        if self.sse_keepalive_s >= self.redis_socket_timeout_s:
            # El SSE espera eventos con XREAD BLOCK durante sse_keepalive_s: si el
            # cliente Redis corta antes, la lectura falla y el stream se rompe.
            raise ValueError("SSE_KEEPALIVE_S tiene que ser menor que REDIS_SOCKET_TIMEOUT_S")
        return self

    @property
    def modelo_llm(self) -> str:
        return self.openai_model if self.llm_provider == "openai" else self.anthropic_model


@lru_cache
def obtener_settings() -> Settings:
    """Una sola instancia por proceso (se lee el entorno una vez)."""
    return Settings()
