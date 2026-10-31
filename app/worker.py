"""
app/worker.py
-------------
Corre el grafo EN SEGUNDO PLANO (BackgroundTask de FastAPI) y va dejando
en Redis el estado del job y un evento por cada nodo.

- Usa `grafo.astream(..., stream_mode="updates")`: en lugar de esperar al
  final, recibe la salida de cada nodo apenas termina. Con eso actualiza
  `paso_actual`, la bitácora de decisiones y publica eventos que el cliente
  puede seguir en vivo por SSE (GET /tasks/{id}/events).
- Al terminar lee el estado final desde el checkpointer: si el grafo quedó
  pausado en un `interrupt()`, el job pasa a AWAITING_APPROVAL.
- REGLA: ninguna excepción se escapa sin dejar el job en FAILED (y hay un
  timeout global). Si no, el cliente quedaría haciendo polling para siempre.
"""

import asyncio
import logging
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.config import Settings
from app.graph import RECURSION_LIMIT
from app.jobs import EstadoJob, JobStore
from app.observability import config_de_ejecucion
from app.state import DecisionHumana, estado_inicial

logger = logging.getLogger("worker")


async def ejecutar_job(job_id: str, *, grafo: CompiledStateGraph, jobs: JobStore, settings: Settings) -> None:
    try:
        job = await jobs.actualizar(job_id, status=EstadoJob.RUNNING)
        logger.info("[job %s] inicia: %s", job_id, job.solicitud.tema)
        await _correr(
            job_id, estado_inicial(job.solicitud), grafo=grafo, jobs=jobs, settings=settings, etapa="ejecucion"
        )
    except Exception as exc:
        await _marcar_fallido(job_id, exc, jobs, settings)


async def reanudar_job(
    job_id: str, decision: DecisionHumana, *, grafo: CompiledStateGraph, jobs: JobStore, settings: Settings
) -> None:
    """El endpoint de aprobación ya pasó el job a RUNNING (atómicamente)."""
    try:
        logger.info("[job %s] se reanuda (aprobado=%s, revisor=%s)", job_id, decision.aprobado, decision.revisor)
        await _correr(
            job_id,
            Command(resume=decision.model_dump()),
            grafo=grafo,
            jobs=jobs,
            settings=settings,
            etapa="reanudacion",
        )
    except Exception as exc:
        await _marcar_fallido(job_id, exc, jobs, settings)


# ---------------------------------------------------------------------------
async def _correr(
    job_id: str, entrada: Any, *, grafo: CompiledStateGraph, jobs: JobStore, settings: Settings, etapa: str
) -> None:
    config = {**config_de_ejecucion(job_id, etapa, settings), "recursion_limit": RECURSION_LIMIT}

    async def recorrer() -> None:
        async for actualizacion in grafo.astream(entrada, config=config, stream_mode="updates"):
            for nodo, salida in actualizacion.items():
                if nodo.startswith("__"):  # p. ej. "__interrupt__"
                    continue
                salida = salida or {}
                resumen = resumir_nodo(nodo, salida)
                await jobs.registrar_evento(job_id, "nodo", {"nodo": nodo, **resumen})
                cambios: dict[str, Any] = {"paso_actual": nodo}
                if nodo == "Supervisor" and salida.get("decisiones"):
                    actual = await jobs.obtener(job_id)
                    cambios["decisiones"] = (actual.decisiones if actual else []) + salida["decisiones"]
                await jobs.actualizar(job_id, **cambios)
                logger.info("[job %s] %s %s", job_id, nodo, resumen)

    await asyncio.wait_for(recorrer(), timeout=settings.timeout_job_segundos)

    snapshot = await grafo.aget_state(config)
    if snapshot.interrupts:
        pendiente = snapshot.interrupts[0].value
        await jobs.actualizar(
            job_id, status=EstadoJob.AWAITING_APPROVAL, aprobacion_pendiente=pendiente, paso_actual="AprobacionHumana"
        )
        await jobs.registrar_evento(job_id, "aprobacion_requerida", {"motivo": pendiente.get("motivo")})
        logger.info("[job %s] pausado esperando aprobación humana", job_id)
        return

    valores = snapshot.values
    aprobacion = valores.get("aprobacion")
    status = EstadoJob.REJECTED if aprobacion is not None and not aprobacion.aprobado else EstadoJob.DONE
    await jobs.actualizar(job_id, status=status, resultado=resumir_resultado(valores), aprobacion_pendiente=None)
    logger.info("[job %s] terminado: %s", job_id, status.value)


def resumir_nodo(nodo: str, salida: dict) -> dict[str, Any]:
    """Resumen corto de lo que hizo un nodo (para eventos y logs)."""
    if nodo == "Supervisor":
        return {"decision": salida.get("next_agent"), "razon": (salida.get("decisiones") or [""])[-1]}
    if nodo == "Investigador" and salida.get("investigacion"):
        b = salida["investigacion"].busqueda
        return {
            "intento": salida["investigacion"].intento,
            "consulta": b.parametros.consulta,
            "noticias": [f.doc_id for f in b.fragmentos],
            "descartados_por_umbral": b.descartados_por_umbral,
        }
    if nodo == "Analista" and salida.get("analisis"):
        a = salida["analisis"]
        return {"sentimiento_general": a.sentimiento_general, "score_promedio": a.score_promedio}
    if nodo == "Redactor" and salida.get("informe"):
        i = salida["informe"]
        return {
            "titulo": i.titulo,
            "hallazgos": len(i.hallazgos),
            "citas_descartadas": i.citas_descartadas,
            "informacion_suficiente": i.hay_informacion_suficiente,
        }
    if nodo == "AprobacionHumana" and salida.get("aprobacion"):
        a = salida["aprobacion"]
        return {"requerida": a.requerida, "aprobado": a.aprobado, "revisor": a.revisor}
    if nodo == "Publicador" and salida.get("reporte"):
        return {"publicado_en": salida["reporte"].publicado_en}
    return {}


def resumir_resultado(valores: dict) -> dict[str, Any]:
    def volcar(v):
        return v.model_dump(mode="json") if v is not None else None

    inv = valores.get("investigacion")
    return {
        "tema": valores["solicitud"].tema,
        "pasos": valores.get("pasos"),
        "intentos_investigacion": valores.get("intentos_investigacion"),
        "consulta_usada": inv.busqueda.parametros.consulta if inv else None,
        "noticias_recuperadas": [f.doc_id for f in inv.busqueda.fragmentos] if inv else [],
        "analisis": volcar(valores.get("analisis")),
        "informe": volcar(valores.get("informe")),
        "aprobacion": volcar(valores.get("aprobacion")),
        "publicado": valores.get("reporte") is not None,
    }


async def _marcar_fallido(job_id: str, exc: BaseException, jobs: JobStore, settings: Settings) -> None:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        mensaje = f"Timeout: el grafo tardó más de {settings.timeout_job_segundos:.0f} s"
    else:
        mensaje = f"{type(exc).__name__}: {exc}"
    logger.exception("[job %s] falló: %s", job_id, mensaje)
    try:
        await jobs.actualizar(job_id, status=EstadoJob.FAILED, error=mensaje, aprobacion_pendiente=None)
    except Exception:
        logger.exception("[job %s] no se pudo marcar como FAILED", job_id)
