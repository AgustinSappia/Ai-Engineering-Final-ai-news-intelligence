"""
app/api/tareas.py
-----------------
Endpoints de las tareas (jobs):

    POST /tasks                    encola y devuelve el job_id al instante (202)
    GET  /tasks                    últimos jobs
    GET  /tasks/{id}               estado del job (para polling)
    GET  /tasks/{id}/events        progreso EN VIVO (Server-Sent Events)
    GET  /tasks/{id}/checkpoint    lo que el checkpointer tiene guardado de ese hilo
"""

import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.config import Settings
from app.dependencies import obtener_grafo, obtener_jobs, obtener_settings_app
from app.jobs import ESTADOS_FINALES, Job, JobStore
from app.state import SolicitudInforme
from app.worker import ejecutar_job

router = APIRouter(tags=["tareas"])


class TareaCreada(BaseModel):
    job_id: str
    status: str
    links: dict[str, str]


class EstadoCheckpoint(BaseModel):
    job_id: str
    existe: bool
    checkpoints_guardados: int = Field(0, description="Cuántos checkpoints hay para este hilo (uno por paso).")
    ultimo_checkpoint_id: str | None = None
    guardado_en: str | None = None
    siguiente_nodo: list[str] = Field(default_factory=list, description="Dónde se retomaría el grafo.")
    pausado_en_interrupt: bool = False
    pasos: int | None = None
    decisiones: list[str] = Field(default_factory=list)


@router.post(
    "/tasks",
    response_model=TareaCreada,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Encolar un informe de inteligencia",
)
async def crear_tarea(
    solicitud: SolicitudInforme,
    background_tasks: BackgroundTasks,
    jobs: JobStore = Depends(obtener_jobs),
    grafo: CompiledStateGraph = Depends(obtener_grafo),
    settings: Settings = Depends(obtener_settings_app),
) -> TareaCreada:
    """Hace UNA escritura en Redis y responde. El grafo corre después, en
    segundo plano: la API nunca espera al LLM para responder."""
    job = await jobs.crear(solicitud)
    background_tasks.add_task(ejecutar_job, job.job_id, grafo=grafo, jobs=jobs, settings=settings)
    return TareaCreada(
        job_id=job.job_id,
        status=job.status.value,
        links={
            "estado": f"/tasks/{job.job_id}",
            "eventos": f"/tasks/{job.job_id}/events",
            "aprobar": f"/tasks/{job.job_id}/approve",
            "checkpoint": f"/tasks/{job.job_id}/checkpoint",
        },
    )


@router.get("/tasks", response_model=list[Job], summary="Últimos jobs")
async def listar_tareas(limite: int = Query(20, ge=1, le=100), jobs: JobStore = Depends(obtener_jobs)) -> list[Job]:
    return await jobs.listar(limite)


@router.get(
    "/tasks/{job_id}", response_model=Job, summary="Estado de un job", responses={404: {"description": "No existe"}}
)
async def estado_tarea(job_id: str, jobs: JobStore = Depends(obtener_jobs)) -> Job:
    job = await jobs.obtener(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}")
    return job


@router.get(
    "/tasks/{job_id}/events",
    summary="Progreso en vivo (Server-Sent Events)",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}, 404: {"description": "No existe"}},
)
async def eventos_tarea(
    job_id: str,
    request: Request,
    jobs: JobStore = Depends(obtener_jobs),
    settings: Settings = Depends(obtener_settings_app),
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """Un evento por cada nodo del grafo y por cada cambio de estado. Se
    puede seguir con `curl -N`. Si la conexión se corta, el cliente puede
    reconectarse mandando `Last-Event-ID` y retoma desde ahí (los eventos
    quedan guardados en un Redis Stream). El stream se cierra cuando el job
    llega a un estado final."""
    if await jobs.obtener(job_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}")

    finales = {e.value for e in ESTADOS_FINALES}
    limite = time.monotonic() + settings.timeout_job_segundos + 3600  # nunca un stream eterno

    async def generar() -> AsyncIterator[str]:
        ultimo = last_event_id or "0"
        while time.monotonic() < limite:
            if await request.is_disconnected():
                return
            eventos = await jobs.leer_eventos(job_id, desde=ultimo, bloquear_ms=15_000)
            if not eventos:
                yield ": keep-alive\n\n"  # comentario SSE: mantiene viva la conexión
                continue
            for evento in eventos:
                ultimo = evento.id
                yield f"id: {evento.id}\nevent: {evento.tipo}\ndata: {json.dumps(evento.datos, ensure_ascii=False)}\n\n"
                if evento.tipo == "estado" and evento.datos.get("status") in finales:
                    return

    return StreamingResponse(
        generar(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.get(
    "/tasks/{job_id}/checkpoint",
    response_model=EstadoCheckpoint,
    summary="Lo que el checkpointer de LangGraph tiene guardado para el job",
)
async def checkpoint_tarea(job_id: str, grafo: CompiledStateGraph = Depends(obtener_grafo)) -> EstadoCheckpoint:
    """Lee directamente del checkpointer (Redis). Sirve para comprobar la
    persistencia: después de reiniciar la API, este endpoint sigue
    mostrando el hilo, y un job pausado se puede aprobar igual."""
    config = {"configurable": {"thread_id": job_id}}
    snapshot = await grafo.aget_state(config)
    if not snapshot.values:
        return EstadoCheckpoint(job_id=job_id, existe=False)

    cantidad = 0
    async for _ in grafo.aget_state_history(config, limit=200):
        cantidad += 1
    return EstadoCheckpoint(
        job_id=job_id,
        existe=True,
        checkpoints_guardados=cantidad,
        ultimo_checkpoint_id=snapshot.config["configurable"].get("checkpoint_id"),
        guardado_en=snapshot.created_at,
        siguiente_nodo=list(snapshot.next),
        pausado_en_interrupt=bool(snapshot.interrupts),
        pasos=snapshot.values.get("pasos"),
        decisiones=snapshot.values.get("decisiones", []),
    )
