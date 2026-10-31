"""
app/state.py
------------
Estado compartido del grafo y artefactos (Pydantic) que produce cada agente.

Cada agente escribe su aporte en SU campo tipado (`investigacion`,
`analisis`, `informe`), no en un historial de texto libre. El Supervisor
decide mirando esos campos, y un dato que no cumple el contrato (un score
fuera de 0..1, una cita sin fuente) no puede entrar al estado.

Este estado es lo que el checkpointer de LangGraph guarda en Redis después
de cada nodo: por eso al final está la lista blanca de clases que se
pueden reconstruir al leer un checkpoint.
"""

import operator
from datetime import date
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator

from app.rag.schemas import ResultadoBusqueda

Sentimiento = Literal["positivo", "neutral", "negativo"]
IdNoticia = Annotated[str, Field(pattern=r"^N\d{2,4}$")]


# ---------------------------------------------------------------------------
# Entrada: lo que pide el usuario (también es el body de POST /tasks)
# ---------------------------------------------------------------------------
class SolicitudInforme(BaseModel):
    tema: str = Field(min_length=2, max_length=200, examples=["incidentes de seguridad con agentes de IA"])
    fecha_desde: date | None = Field(None, description="Solo noticias desde esta fecha.")
    fecha_hasta: date | None = Field(None, description="Solo noticias hasta esta fecha.")
    fuentes: list[Annotated[str, Field(min_length=2, max_length=80)]] = Field(
        default_factory=list, max_length=10, description="Limitar a estas fuentes."
    )
    forzar_aprobacion: bool = Field(False, description="Si es true, la publicación SIEMPRE pasa por aprobación humana.")

    @field_validator("tema")
    @classmethod
    def _limpiar(cls, valor: str) -> str:
        return " ".join(valor.split())

    @model_validator(mode="after")
    def _fechas_en_orden(self) -> "SolicitudInforme":
        if self.fecha_desde and self.fecha_hasta and self.fecha_desde > self.fecha_hasta:
            raise ValueError("fecha_desde no puede ser posterior a fecha_hasta")
        return self


# ---------------------------------------------------------------------------
# Investigador
# ---------------------------------------------------------------------------
class ResultadoInvestigacion(BaseModel):
    intento: int = Field(ge=1)
    busqueda: ResultadoBusqueda
    args_del_llm_validos: bool = True  # False si el LLM propuso argumentos inválidos y se usó el plan B

    @property
    def encontro_resultados(self) -> bool:
        return self.busqueda.encontro_resultados


# ---------------------------------------------------------------------------
# Analista
# ---------------------------------------------------------------------------
class AnalisisNoticia(BaseModel):
    doc_id: IdNoticia
    sentimiento: Sentimiento
    score: float = Field(ge=0.0, le=1.0, description="0 = muy negativo, 1 = muy positivo.")
    justificacion: str = Field(min_length=3, max_length=400, description="Por qué, en una oración.")


class AnalisisLLM(BaseModel):
    """Salida estructurada que se le pide al LLM del Analista."""

    noticias: list[AnalisisNoticia] = Field(description="Un elemento por cada noticia recibida.")


class ResultadoAnalisis(BaseModel):
    sentimiento_general: Literal["positivo", "neutral", "negativo", "sin_datos"]
    score_promedio: float = Field(ge=0.0, le=1.0)
    detalle: list[AnalisisNoticia] = Field(default_factory=list)
    descartados: int = Field(0, ge=0, description="Análisis del LLM sobre noticias que no le pasamos.")
    suficiente: bool


# ---------------------------------------------------------------------------
# Redactor
# ---------------------------------------------------------------------------
class HallazgoLLM(BaseModel):
    texto: str = Field(min_length=10, max_length=600)
    citas: list[str] = Field(
        min_length=1, max_length=5, description="IDs de las noticias que respaldan el hallazgo, p. ej. N07."
    )


class InformeLLM(BaseModel):
    """Salida estructurada que se le pide al LLM del Redactor."""

    titulo: str = Field(min_length=5, max_length=150)
    resumen: str = Field(min_length=20, max_length=1500)
    hallazgos: list[HallazgoLLM] = Field(min_length=1, max_length=8)
    conclusion: str = Field(min_length=10, max_length=1000)


class Cita(BaseModel):
    doc_id: IdNoticia
    titular: str
    fuente: str
    fecha: date


class Hallazgo(BaseModel):
    texto: str
    citas: list[IdNoticia] = Field(min_length=1)


class InformeInteligencia(BaseModel):
    """El producto final del sistema."""

    titulo: str
    resumen: str
    hallazgos: list[Hallazgo] = Field(default_factory=list)
    conclusion: str
    fuentes: list[Cita] = Field(default_factory=list)
    sentimiento_general: Literal["positivo", "neutral", "negativo", "sin_datos"]
    score_promedio: float = Field(ge=0.0, le=1.0)
    hay_informacion_suficiente: bool
    citas_descartadas: int = Field(0, ge=0, description="Citas del LLM a noticias que no fueron recuperadas.")
    hallazgos_descartados: int = Field(0, ge=0, description="Hallazgos que quedaron sin ninguna cita válida.")


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------
Destino = Literal["Investigador", "Analista", "Redactor", "FINALIZAR"]


class DecisionSupervisor(BaseModel):
    next: Destino = Field(description="Quién actúa a continuación, o FINALIZAR.")
    razon: str = Field(min_length=3, max_length=400, description="Por qué, en una o dos oraciones.")


# ---------------------------------------------------------------------------
# Human-in-the-loop y publicación
# ---------------------------------------------------------------------------
class DecisionHumana(BaseModel):
    aprobado: bool
    comentario: str = Field("", max_length=1000)
    revisor: str = Field("anonimo", min_length=1, max_length=80)


class RegistroAprobacion(BaseModel):
    requerida: bool
    motivo: str
    aprobado: bool
    comentario: str = ""
    revisor: str = ""


class ReportePublicado(BaseModel):
    tema: str
    titulo: str
    sentimiento_general: str
    cantidad_fuentes: int = Field(ge=0)
    aprobado_por: str
    publicado_en: str  # ISO-8601 UTC
    informe: InformeInteligencia


# ---------------------------------------------------------------------------
# Estado del grafo
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    solicitud: SolicitudInforme
    next_agent: str
    pasos: int
    intentos_investigacion: int
    motivo_refinamiento: str | None
    # Bitácora de decisiones del Supervisor: el reducer `operator.add`
    # hace que cada nodo AGREGUE líneas en vez de reemplazar la lista.
    decisiones: Annotated[list[str], operator.add]
    investigacion: ResultadoInvestigacion | None
    analisis: ResultadoAnalisis | None
    informe: InformeInteligencia | None
    aprobacion: RegistroAprobacion | None
    reporte: ReportePublicado | None


def estado_inicial(solicitud: SolicitudInforme) -> AgentState:
    return {
        "solicitud": solicitud,
        "next_agent": "",
        "pasos": 0,
        "intentos_investigacion": 0,
        "motivo_refinamiento": None,
        "decisiones": [],
        "investigacion": None,
        "analisis": None,
        "informe": None,
        "aprobacion": None,
        "reporte": None,
    }


# ---------------------------------------------------------------------------
# Lista blanca para el checkpointer
# ---------------------------------------------------------------------------
# Reconstruir objetos Python a partir de lo guardado en Redis es un punto
# sensible: LangGraph solo reconstruye las clases permitidas. Se listan en
# los DOS formatos que usa el checkpointer de Redis (msgpack y JSON). En la
# Pre-entrega 7 faltaba el formato JSON y, al reanudar un job, los objetos
# volvían como dict.
TIPOS_SERIALIZABLES: list[tuple[str, str]] = [
    ("app.state", nombre)
    for nombre in (
        "SolicitudInforme",
        "ResultadoInvestigacion",
        "AnalisisNoticia",
        "ResultadoAnalisis",
        "Cita",
        "Hallazgo",
        "InformeInteligencia",
        "RegistroAprobacion",
        "ReportePublicado",
    )
] + [("app.rag.schemas", nombre) for nombre in ("ParametrosBusqueda", "Fragmento", "ResultadoBusqueda")]

TIPOS_SERIALIZABLES_JSON: list[tuple[str, ...]] = [(*modulo.split("."), clase) for modulo, clase in TIPOS_SERIALIZABLES]
