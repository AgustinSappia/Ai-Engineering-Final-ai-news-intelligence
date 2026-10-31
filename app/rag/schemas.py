"""
app/rag/schemas.py
------------------
Contratos de datos (Pydantic) de la capa de conocimiento.

- `NoticiaCorpus`: una noticia del corpus, validada al cargarla. Un archivo
  mal formado (sin fecha, sin titular) falla en la ingesta, no en una
  búsqueda.
- `Chunk`: un pedazo de noticia listo para indexar.
- `ParametrosBusqueda`: los argumentos de la herramienta `buscar_noticias`
  del agente Investigador. El LLM los completa y Pydantic los valida antes
  de tocar la base vectorial (por ejemplo, `k` entre 1 y 10, fechas en
  orden).
- `Fragmento` y `ResultadoBusqueda`: lo que devuelve la búsqueda híbrida,
  con los puntajes de cada método para poder auditar el ranking.
"""

from datetime import date
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

from app.rag.texto import normalizar

Etiqueta = Annotated[str, Field(min_length=2, max_length=40)]


class NoticiaCorpus(BaseModel):
    doc_id: str = Field(pattern=r"^N\d{2,4}$")
    titular: str = Field(min_length=10, max_length=200)
    fuente: str = Field(min_length=2, max_length=80)
    fecha: date
    etiquetas: list[Etiqueta] = Field(default_factory=list)
    cuerpo: str = Field(min_length=100)


class Chunk(BaseModel):
    id: str  # p. ej. "N07#1"
    doc_id: str
    orden: int = Field(ge=0)
    texto: str = Field(min_length=1)
    titular: str
    fuente: str
    fecha: date
    etiquetas: list[str] = Field(default_factory=list)

    @property
    def fecha_num(self) -> int:
        """Fecha como entero AAAAMMDD: Chroma solo compara números, así que
        los filtros por rango de fechas se hacen sobre este valor."""
        return int(self.fecha.strftime("%Y%m%d"))

    def metadatos(self) -> dict[str, str | int]:
        """Metadatos planos (solo str/int), como los acepta Chroma."""
        return {
            "doc_id": self.doc_id,
            "orden": self.orden,
            "titular": self.titular,
            "fuente": self.fuente,
            "fuente_norm": normalizar(self.fuente),
            "fecha": self.fecha.isoformat(),
            "fecha_num": self.fecha_num,
            "etiquetas": ",".join(self.etiquetas),
        }

    @classmethod
    def desde_metadatos(cls, chunk_id: str, texto: str, meta: dict) -> "Chunk":
        return cls(
            id=chunk_id,
            doc_id=meta["doc_id"],
            orden=int(meta["orden"]),
            texto=texto,
            titular=meta["titular"],
            fuente=meta["fuente"],
            fecha=date.fromisoformat(meta["fecha"]),
            etiquetas=[e for e in str(meta.get("etiquetas", "")).split(",") if e],
        )


class ParametrosBusqueda(BaseModel):
    """Argumentos de la herramienta de búsqueda (y del endpoint de búsqueda)."""

    consulta: str = Field(
        min_length=2,
        max_length=300,
        description="Qué buscar, en lenguaje natural. Conviene incluir sinónimos y términos clave.",
    )
    fecha_desde: date | None = Field(None, description="Solo noticias publicadas desde esta fecha (AAAA-MM-DD).")
    fecha_hasta: date | None = Field(None, description="Solo noticias publicadas hasta esta fecha (AAAA-MM-DD).")
    fuentes: list[str] = Field(
        default_factory=list, max_length=10, description="Limitar a estas fuentes. Vacío = todas."
    )
    etiquetas: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Limitar a noticias con alguna de estas etiquetas. Vacío = todas.",
    )
    k: int = Field(5, ge=1, le=10, description="Cantidad máxima de noticias a devolver.")

    @field_validator("consulta")
    @classmethod
    def _sin_espacios_sobrantes(cls, valor: str) -> str:
        return " ".join(valor.split())

    @model_validator(mode="after")
    def _fechas_en_orden(self) -> "ParametrosBusqueda":
        if self.fecha_desde and self.fecha_hasta and self.fecha_desde > self.fecha_hasta:
            raise ValueError("fecha_desde no puede ser posterior a fecha_hasta")
        return self

    @property
    def tiene_filtros(self) -> bool:
        return bool(self.fecha_desde or self.fecha_hasta or self.fuentes or self.etiquetas)


class Fragmento(BaseModel):
    """Una noticia recuperada, con el texto de los chunks que matchearon."""

    doc_id: str
    titular: str
    fuente: str
    fecha: date
    etiquetas: list[str] = Field(default_factory=list)
    texto: str
    chunk_ids: list[str]
    similitud: float = Field(ge=-1.0, le=1.0, description="Similitud coseno (búsqueda semántica).")
    score_bm25: float = Field(ge=0.0, description="Puntaje BM25 (búsqueda por palabras clave).")
    score_rrf: float = Field(ge=0.0, description="Puntaje final de la fusión (Reciprocal Rank Fusion).")


class ResultadoBusqueda(BaseModel):
    parametros: ParametrosBusqueda
    fragmentos: list[Fragmento] = Field(default_factory=list)
    candidatos_evaluados: int = Field(0, ge=0)
    descartados_por_umbral: int = Field(0, ge=0)
    umbral_similitud: float

    @property
    def encontro_resultados(self) -> bool:
        return len(self.fragmentos) > 0
