"""
app/agents/analista.py
----------------------
Agente Analista: evalúa el sentimiento de las noticias recuperadas.

Es una cadena LCEL (Módulo 2): prompt -> LLM con salida estructurada
(`AnalisisLLM`, Pydantic) -> validación. `.with_retry()` reintenta si el
LLM devuelve algo que no cumple el esquema.

Después, el código (no el LLM) hace dos cosas:
- descarta análisis de noticias que NO le pasamos (un `doc_id` inventado);
- calcula el promedio y el sentimiento general con reglas fijas, para que
  la decisión de "¿es crítico?" del Human-in-the-loop sea reproducible.
"""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.rag.schemas import Fragmento
from app.state import AnalisisLLM, ResultadoAnalisis, ResultadoInvestigacion

logger = logging.getLogger("agentes.analista")

PROMPT_ANALISTA = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos el agente Analista. Evaluás el sentimiento (positivo, neutral o negativo) de cada noticia, "
            "desde el punto de vista del impacto para la sociedad y la industria.\n"
            "- Devolvé exactamente un análisis por cada noticia, usando su ID entre corchetes (por ejemplo N07).\n"
            "- score: 0 = muy negativo, 0.5 = neutral, 1 = muy positivo.\n"
            "- Basate SOLO en el texto de cada noticia. No uses conocimiento externo.",
        ),
        ("human", "Noticias a analizar:\n\n{noticias}"),
    ]
)

UMBRAL_NEGATIVO = 0.40
UMBRAL_POSITIVO = 0.60


def formatear_fragmentos(fragmentos: list[Fragmento]) -> str:
    return "\n\n".join(f"[{f.doc_id}] {f.titular} ({f.fuente}, {f.fecha.isoformat()})\n{f.texto}" for f in fragmentos)


def sentimiento_desde_score(score: float) -> str:
    if score < UMBRAL_NEGATIVO:
        return "negativo"
    if score > UMBRAL_POSITIVO:
        return "positivo"
    return "neutral"


def _sin_datos() -> ResultadoAnalisis:
    return ResultadoAnalisis(sentimiento_general="sin_datos", score_promedio=0.5, detalle=[], suficiente=False)


async def analizar(llm: BaseChatModel, investigacion: ResultadoInvestigacion) -> ResultadoAnalisis:
    fragmentos = investigacion.busqueda.fragmentos
    if not fragmentos:
        return _sin_datos()  # nada que analizar: no se gasta una llamada al LLM

    cadena = (PROMPT_ANALISTA | llm.with_structured_output(AnalisisLLM, method="function_calling")).with_retry(
        stop_after_attempt=2
    )
    salida: AnalisisLLM = await cadena.ainvoke(
        {"noticias": formatear_fragmentos(fragmentos)}, config={"run_name": "analista.sentimiento"}
    )

    recibidas = {f.doc_id for f in fragmentos}
    vistos: set[str] = set()
    validos = []
    for item in salida.noticias:
        if item.doc_id in recibidas and item.doc_id not in vistos:
            validos.append(item)
            vistos.add(item.doc_id)
    descartados = len(salida.noticias) - len(validos)
    if descartados:
        logger.warning("El Analista devolvió %d análisis de noticias que no recibió: se descartan.", descartados)

    if not validos:
        return _sin_datos()

    promedio = round(sum(i.score for i in validos) / len(validos), 3)
    return ResultadoAnalisis(
        sentimiento_general=sentimiento_desde_score(promedio),
        score_promedio=promedio,
        detalle=validos,
        descartados=descartados,
        suficiente=True,
    )
