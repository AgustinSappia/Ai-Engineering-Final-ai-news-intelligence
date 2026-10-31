"""
app/api/sistema.py
------------------
    GET /health     estado de Redis, Chroma, el índice y LangSmith
    GET /reportes   reportes publicados (el efecto secundario que protege el HITL)
"""

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from app.dependencies import obtener_jobs, obtener_recuperador
from app.jobs import JobStore
from app.rag.recuperador import RecuperadorHibrido
from app.state import ReportePublicado

router = APIRouter(tags=["sistema"])


class Health(BaseModel):
    status: str
    redis: bool
    chroma: bool
    indice_listo: bool
    noticias_indexadas: int
    observabilidad: dict | None


@router.get("/health", response_model=Health, summary="Estado de los servicios")
async def health(
    request: Request,
    jobs: JobStore = Depends(obtener_jobs),
    recuperador: RecuperadorHibrido = Depends(obtener_recuperador),
) -> Health:
    try:
        redis_ok = bool(await jobs.redis.ping())
    except Exception:
        redis_ok = False
    chroma_ok = await recuperador.almacen.ping()
    todo_ok = redis_ok and chroma_ok and recuperador.listo
    return Health(
        status="ok" if todo_ok else "degradado",
        redis=redis_ok,
        chroma=chroma_ok,
        indice_listo=recuperador.listo,
        noticias_indexadas=recuperador.cantidad_noticias,
        observabilidad=getattr(request.app.state, "observabilidad", None),
    )


@router.get("/reportes", response_model=list[ReportePublicado], summary="Reportes publicados")
async def reportes(
    limite: int = Query(20, ge=1, le=100), jobs: JobStore = Depends(obtener_jobs)
) -> list[ReportePublicado]:
    return await jobs.reportes_publicados(limite)
