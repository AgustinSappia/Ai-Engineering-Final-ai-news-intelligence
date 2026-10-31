"""
app/observability.py
--------------------
Trazabilidad: trazas en LangSmith + logs estructurados.

Trazas (LangSmith): LangGraph y LangChain ya traen la instrumentación. Con
`LANGSMITH_TRACING=true` y una API key válida, cada ejecución del grafo se
envía como un árbol de spans: cada nodo, cada llamada al LLM (con tokens y
costo), la herramienta de búsqueda y los pasos del RAG. Lo que agrega este
módulo:

- `inicializar_observabilidad()` + `verificar_conexion()`: al arrancar,
  comprueban que la key exista, que no sea el valor de ejemplo de
  .env.example y que LangSmith la acepte (consulta real). Si algo falla,
  apagan el tracing y /health dice por qué. (Lección de la Pre-entrega 7:
  con el valor de ejemplo, el tracing parecía activo y las trazas se
  perdían en silencio.)
- `config_de_ejecucion()`: metadata y tags por job (job_id, etapa, modelo),
  para buscar una ejecución puntual y agrupar en un mismo "Thread" la
  ejecución inicial y la reanudación después de la aprobación humana.
- `configurar_logs()`: logs con formato uniforme; cada línea del worker
  lleva el job_id.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from typing import Any

from app.config import Settings

logger = logging.getLogger("observabilidad")


def configurar_logs(nivel: str) -> None:
    logging.basicConfig(
        level=nivel.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    # Librerías muy verbosas en INFO.
    for ruidosa in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(ruidosa).setLevel(logging.WARNING)


def _desactivar_tracing() -> None:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    try:
        import langsmith
        from langsmith import utils as ls_utils

        ls_utils.get_env_var.cache_clear()  # LangSmith cachea las variables de entorno
        langsmith.configure(enabled=False)
    except Exception:
        logger.exception("No se pudo desactivar el tracing de LangSmith")


def _parece_valor_de_ejemplo(api_key: str) -> bool:
    return "..." in api_key or len(api_key) < 20


def inicializar_observabilidad(settings: Settings) -> dict[str, Any]:
    """Revisa la configuración (sin red). Devuelve el estado para /health."""
    if not settings.langsmith_tracing:
        _desactivar_tracing()
        return {"plataforma": "langsmith", "activo": False, "motivo": "LANGSMITH_TRACING no está en true"}

    api_key = settings.langsmith_api_key.get_secret_value().strip() if settings.langsmith_api_key else ""
    if not api_key:
        _desactivar_tracing()
        logger.warning("LANGSMITH_TRACING=true pero falta LANGSMITH_API_KEY: se desactiva el tracing.")
        return {"plataforma": "langsmith", "activo": False, "motivo": "falta LANGSMITH_API_KEY"}
    if _parece_valor_de_ejemplo(api_key):
        _desactivar_tracing()
        logger.warning("LANGSMITH_API_KEY tiene el valor de ejemplo de .env.example: se desactiva el tracing.")
        return {
            "plataforma": "langsmith",
            "activo": False,
            "motivo": "LANGSMITH_API_KEY tiene el valor de ejemplo de .env.example: poné tu key real",
        }

    # LangSmith lee estas variables del entorno del proceso.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    return {
        "plataforma": "langsmith",
        "activo": True,
        "proyecto": settings.langsmith_project,
        "conexion": "sin verificar",
    }


def _consultar_langsmith(cliente) -> None:
    next(iter(cliente.list_projects(limit=1)), None)


async def verificar_conexion(
    estado: dict[str, Any], crear_cliente: Callable[[], Any] | None = None, limite_s: float = 8.0
) -> dict[str, Any]:
    """Confirma que LangSmith acepta la key. El cliente de LangSmith es
    síncrono: se corre en otro thread para no bloquear el event loop."""
    if not estado.get("activo"):
        return estado
    if crear_cliente is None:
        from langsmith import Client

        crear_cliente = Client
    from langsmith.utils import LangSmithAuthError

    estado = dict(estado)
    try:
        await asyncio.wait_for(asyncio.to_thread(_consultar_langsmith, crear_cliente()), timeout=limite_s)
    except LangSmithAuthError:
        _desactivar_tracing()
        logger.error("LangSmith rechazó LANGSMITH_API_KEY: se desactiva el tracing.")
        estado.update(activo=False, conexion="rechazada", motivo="LangSmith rechazó LANGSMITH_API_KEY (401/403)")
        return estado
    except Exception as exc:
        logger.warning("No se pudo verificar la conexión con LangSmith: %s", exc)
        estado["conexion"] = f"sin verificar ({type(exc).__name__})"
        return estado

    estado["conexion"] = "ok"
    logger.info("LangSmith activo y verificado: proyecto '%s'.", estado["proyecto"])
    return estado


def config_de_ejecucion(job_id: str, etapa: str, settings: Settings) -> dict[str, Any]:
    """Config de UNA invocación del grafo. `thread_id` le dice al
    checkpointer dónde guardar/leer; el resto son etiquetas para LangSmith."""
    return {
        "configurable": {"thread_id": job_id},
        "run_name": f"sistema-inteligencia.{etapa}",
        "tags": ["entrega-final", etapa],
        "metadata": {
            "job_id": job_id,
            "thread_id": job_id,
            "etapa": etapa,
            "llm_provider": settings.llm_provider,
            "llm_modelo": settings.modelo_llm,
            "embeddings_modelo": settings.embeddings_model,
        },
    }


def cerrar_observabilidad() -> None:
    """Las trazas se envían en lotes en segundo plano: al apagar, se espera
    a que salgan las últimas."""
    if os.environ.get("LANGSMITH_TRACING", "").lower() != "true":
        return
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception:
        logger.exception("No se pudieron vaciar las trazas pendientes")
