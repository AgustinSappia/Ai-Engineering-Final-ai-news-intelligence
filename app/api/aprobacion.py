"""
app/api/aprobacion.py
---------------------
POST /tasks/{job_id}/approve: la decisión humana del flujo HITL
(la lógica del nodo de aprobación está en app/hitl.py).
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from langgraph.graph.state import CompiledStateGraph

from app.config import Settings
from app.dependencies import obtener_grafo, obtener_jobs, obtener_settings_app
from app.jobs import EstadoJob, Job, JobStore
from app.state import DecisionHumana
from app.worker import reanudar_job

router = APIRouter(tags=["human-in-the-loop"])


@router.post(
    "/tasks/{job_id}/approve",
    response_model=Job,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Aprobar o rechazar la publicación de un informe pausado",
    responses={404: {"description": "El job no existe"}, 409: {"description": "El job no está esperando aprobación"}},
)
async def aprobar_tarea(
    job_id: str,
    decision: DecisionHumana,
    background_tasks: BackgroundTasks,
    jobs: JobStore = Depends(obtener_jobs),
    grafo: CompiledStateGraph = Depends(obtener_grafo),
    settings: Settings = Depends(obtener_settings_app),
) -> Job:
    """Reanuda el grafo en segundo plano con la decisión. La transición
    AWAITING_APPROVAL -> RUNNING es atómica: si llegan dos aprobaciones a la
    vez, solo una reanuda el grafo y la otra recibe 409."""
    job = await jobs.transicionar(job_id, desde={EstadoJob.AWAITING_APPROVAL}, hacia=EstadoJob.RUNNING)
    if job is None:
        existente = await jobs.obtener(job_id)
        if existente is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}")
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"El job está en estado {existente.status.value}: solo se puede aprobar si está en AWAITING_APPROVAL",
        )
    background_tasks.add_task(reanudar_job, job_id, decision, grafo=grafo, jobs=jobs, settings=settings)
    return job
