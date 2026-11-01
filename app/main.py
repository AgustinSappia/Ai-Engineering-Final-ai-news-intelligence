"""
app/main.py
-----------
Punto de entrada de la API (FastAPI). Documentación interactiva en /docs.

Al arrancar (`lifespan`), en este orden:
1. Configuración validada (`Settings`) y logs.
2. LangSmith: valida la key contra el servicio (si está activo).
3. Redis (estado de jobs, eventos y checkpoints) y Chroma (base vectorial),
   con reintentos: en Docker Compose pueden tardar unos segundos en estar
   listos.
4. Ingesta idempotente del corpus (solo genera embeddings si cambió algo).
5. Índice híbrido (BM25 en memoria a partir de lo que hay en Chroma).
6. Checkpointer de Redis y grafo de agentes compilado (uno solo para toda
   la API: cada job usa su propio thread_id).
7. Recuperación: si la API se había caído con jobs a medio camino, los
   retoma desde su último checkpoint (en segundo plano: no demora el
   arranque).
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api import aprobacion, conocimiento, sistema, tareas
from app.config import Settings, obtener_settings
from app.graph import checkpointer_redis, construir_grafo
from app.jobs import JobStore, crear_cliente_redis
from app.llm import crear_chat_model, crear_embeddings
from app.observability import cerrar_observabilidad, configurar_logs, inicializar_observabilidad, verificar_conexion
from app.rag.almacen import AlmacenChroma
from app.rag.ingesta import asegurar_indice
from app.rag.recuperador import RecuperadorHibrido
from app.worker import recuperar_job

logger = logging.getLogger("api")


async def con_reintentos[T](
    nombre: str, accion: Callable[[], Awaitable[T]], intentos: int = 30, espera_s: float = 2.0
) -> T:
    """Reintenta una conexión mientras el servicio termina de arrancar."""
    for intento in range(1, intentos + 1):
        try:
            return await accion()
        except Exception as exc:
            if intento == intentos:
                raise RuntimeError(f"{nombre} no respondió después de {intentos} intentos") from exc
            logger.info("Esperando a %s (%d/%d): %s", nombre, intento, intentos, exc)
            await asyncio.sleep(espera_s)
    raise AssertionError("inalcanzable")


def explicar_error_de_proveedor(exc: BaseException) -> str | None:
    """Traduce los errores de credenciales de OpenAI a un mensaje accionable.
    Sin esto, el motivo real queda al final de un traceback largo."""
    import openai

    if isinstance(exc, openai.AuthenticationError):
        return (
            "OpenAI rechazó OPENAI_API_KEY (vencida, revocada o inválida): "
            f"{getattr(exc, 'message', exc)} -> Generá una nueva en https://platform.openai.com/api-keys, "
            "actualizá el .env y recreá el contenedor (docker compose up -d --force-recreate api)."
        )
    if isinstance(exc, openai.PermissionDeniedError | openai.RateLimitError):
        return (
            f"OpenAI no permite usar la key ({type(exc).__name__}): {getattr(exc, 'message', exc)} -> "
            "Revisá el saldo/límites de la cuenta y los permisos del proyecto en https://platform.openai.com."
        )
    return None


async def iniciar_recuperacion(app: FastAPI, jobs: JobStore, settings: Settings) -> list[asyncio.Task]:
    """Lanza en segundo plano la recuperación de los jobs interrumpidos."""
    interrumpidos = await jobs.en_curso()
    if interrumpidos:
        logger.warning("La API se había detenido con %d job(s) en curso: se retoman.", len(interrumpidos))
    tareas = [
        asyncio.create_task(recuperar_job(j.job_id, grafo=app.state.grafo, jobs=jobs, settings=settings))
        for j in interrumpidos
    ]
    app.state.recuperaciones = tareas  # referencia fuerte: si no, el GC puede cortar la tarea
    return tareas


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = obtener_settings()
    configurar_logs(settings.log_level)
    app.state.settings = settings
    app.state.observabilidad = await verificar_conexion(inicializar_observabilidad(settings))

    redis = crear_cliente_redis(settings.redis_url, settings.redis_socket_timeout_s)
    await con_reintentos("Redis", redis.ping)

    async def conectar_chroma() -> AlmacenChroma:
        almacen = await AlmacenChroma.conectar(settings.chroma_host, settings.chroma_port, settings.chroma_coleccion)
        if not await almacen.ping():
            raise ConnectionError("heartbeat fallido")
        return almacen

    almacen = await con_reintentos("Chroma", conectar_chroma)
    embeddings = crear_embeddings(settings)
    try:
        app.state.ingesta = await asegurar_indice(almacen, embeddings, settings)
    except Exception as exc:
        if explicacion := explicar_error_de_proveedor(exc):
            logger.error(explicacion)
            raise RuntimeError(explicacion) from exc
        raise

    recuperador = RecuperadorHibrido(
        almacen,
        embeddings,
        candidatos=settings.rag_candidatos,
        rrf_k=settings.rag_rrf_k,
        umbral_similitud=settings.rag_umbral_similitud,
    )
    await recuperador.inicializar()

    jobs = JobStore(redis, ttl_segundos=settings.job_ttl_segundos)
    async with checkpointer_redis(settings.redis_url, settings.checkpoint_ttl_minutos) as checkpointer:
        app.state.jobs = jobs
        app.state.recuperador = recuperador
        app.state.grafo = construir_grafo(
            llm=crear_chat_model(settings),
            recuperador=recuperador,
            checkpointer=checkpointer,
            publicar=jobs.publicar_reporte,
            max_pasos=settings.max_pasos,
            max_intentos_investigacion=settings.max_intentos_investigacion,
            k=settings.rag_k,
        )
        logger.info(
            "API lista: %d noticias indexadas, LLM %s/%s.",
            recuperador.cantidad_noticias,
            settings.llm_provider,
            settings.modelo_llm,
        )
        recuperaciones = await iniciar_recuperacion(app, jobs, settings)
        yield
        for tarea in recuperaciones:  # quedan en RUNNING y se retoman en el próximo arranque
            tarea.cancel()

    await redis.aclose()
    cerrar_observabilidad()


def crear_app(lifespan_fn=lifespan) -> FastAPI:
    """Fábrica de la app. Los tests la crean con `lifespan_fn=None` y
    cargan dependencias falsas en `app.state`."""
    app = FastAPI(
        title="AI News Intelligence System",
        description=(
            "Sistema de inteligencia sobre noticias de IA: RAG híbrido + agentes con Supervisor + "
            "Human-in-the-loop, con persistencia en Redis y trazas en LangSmith."
        ),
        version="1.0.0",
        lifespan=lifespan_fn,
    )

    @app.middleware("http")
    async def registrar_request(request: Request, call_next):
        """Cada request lleva un X-Request-ID (el que manda el cliente o uno
        nuevo) y deja una línea de log con método, ruta, estado y duración."""
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        inicio = time.perf_counter()
        respuesta = await call_next(request)
        respuesta.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %d (%.1f ms) [req %s]",
            request.method,
            request.url.path,
            respuesta.status_code,
            (time.perf_counter() - inicio) * 1000,
            request_id,
        )
        return respuesta

    app.include_router(tareas.router)
    app.include_router(aprobacion.router)
    app.include_router(conocimiento.router)
    app.include_router(sistema.router)
    return app


app = crear_app()
