"""
app/api/conocimiento.py
-----------------------
Acceso directo a la capa de conocimiento (RAG), sin pasar por los agentes:
sirve para inspeccionar qué recupera la búsqueda híbrida para una consulta
y con qué puntajes.

    GET  /conocimiento          estado del índice
    POST /conocimiento/buscar   búsqueda híbrida
"""

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.config import Settings
from app.dependencies import obtener_recuperador, obtener_settings_app
from app.rag.recuperador import RecuperadorHibrido
from app.rag.schemas import ParametrosBusqueda, ResultadoBusqueda

router = APIRouter(prefix="/conocimiento", tags=["conocimiento (RAG)"])


class EstadoConocimiento(BaseModel):
    noticias: int
    chunks: int
    coleccion: str
    huella_corpus: str | None
    modelo_embeddings: str
    umbral_similitud: float
    candidatos_por_metodo: int


@router.get("", response_model=EstadoConocimiento, summary="Estado del índice")
async def estado(
    request: Request,
    recuperador: RecuperadorHibrido = Depends(obtener_recuperador),
    settings: Settings = Depends(obtener_settings_app),
) -> EstadoConocimiento:
    ingesta = getattr(request.app.state, "ingesta", None)
    return EstadoConocimiento(
        noticias=recuperador.cantidad_noticias,
        chunks=recuperador.cantidad_chunks,
        coleccion=settings.chroma_coleccion,
        huella_corpus=ingesta.huella if ingesta else None,
        modelo_embeddings=settings.embeddings_model,
        umbral_similitud=recuperador.umbral_similitud,
        candidatos_por_metodo=recuperador.candidatos,
    )


@router.post("/buscar", response_model=ResultadoBusqueda, summary="Búsqueda híbrida (semántica + BM25 + RRF)")
async def buscar(
    params: ParametrosBusqueda, recuperador: RecuperadorHibrido = Depends(obtener_recuperador)
) -> ResultadoBusqueda:
    return await recuperador.buscar(params)
