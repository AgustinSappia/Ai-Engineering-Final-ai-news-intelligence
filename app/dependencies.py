"""
app/dependencies.py
-------------------
Dependencias de FastAPI. Los recursos se crean UNA vez al arrancar (ver
`lifespan` en main.py) y se guardan en `app.state`; los endpoints los piden
con `Depends(...)`. En los tests se cargan versiones falsas en `app.state`.
"""

from fastapi import Request
from langgraph.graph.state import CompiledStateGraph

from app.config import Settings
from app.jobs import JobStore
from app.rag.recuperador import RecuperadorHibrido


def obtener_jobs(request: Request) -> JobStore:
    return request.app.state.jobs


def obtener_grafo(request: Request) -> CompiledStateGraph:
    return request.app.state.grafo


def obtener_settings_app(request: Request) -> Settings:
    return request.app.state.settings


def obtener_recuperador(request: Request) -> RecuperadorHibrido:
    return request.app.state.recuperador
