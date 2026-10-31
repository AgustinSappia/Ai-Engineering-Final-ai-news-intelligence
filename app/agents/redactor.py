"""
app/agents/redactor.py
----------------------
Agente Redactor: escribe el informe final FUNDAMENTADO en las noticias
recuperadas, con citas.

Defensa contra alucinaciones en tres capas:
1. El prompt le da al LLM solo los fragmentos recuperados y le exige citar
   el ID de la noticia en cada hallazgo.
2. Salida estructurada (`InformeLLM`): cada hallazgo DEBE traer al menos
   una cita (lo valida Pydantic).
3. Verificación en código (el "filtro de veracidad" de la Pre-entrega 3):
   se descartan las citas a noticias que no fueron recuperadas y los
   hallazgos que se quedan sin ninguna cita válida. Lo descartado se cuenta
   y queda en el informe, visible en la API y en las trazas.

Si no hay noticias recuperadas, NO se llama al LLM: se devuelve un informe
que dice honestamente que no hay información suficiente.
"""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.agents.analista import formatear_fragmentos
from app.state import (
    Cita,
    Hallazgo,
    InformeInteligencia,
    InformeLLM,
    ResultadoAnalisis,
    ResultadoInvestigacion,
    SolicitudInforme,
)

logger = logging.getLogger("agentes.redactor")

PROMPT_REDACTOR = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos el agente Redactor de un sistema de inteligencia sobre noticias de IA. Escribís un informe breve "
            "en español sobre el tema pedido.\n"
            "Reglas estrictas:\n"
            "- Usá SOLO la información de las noticias provistas. No agregues datos externos.\n"
            "- Cada hallazgo debe citar el o los IDs de las noticias que lo respaldan (por ejemplo N07).\n"
            "- Si las noticias no alcanzan para algo, decilo; no lo completes con suposiciones.\n"
            "- Tené en cuenta el análisis de sentimiento para el tono de la conclusión.",
        ),
        (
            "human",
            "Tema: {tema}\n\nAnálisis de sentimiento: {sentimiento} (score promedio {score})\n\n"
            "Noticias disponibles:\n\n{noticias}",
        ),
    ]
)


def informe_sin_informacion(solicitud: SolicitudInforme, motivo: str) -> InformeInteligencia:
    filtros = " con los filtros pedidos" if solicitud.fecha_desde or solicitud.fecha_hasta or solicitud.fuentes else ""
    return InformeInteligencia(
        titulo=f"Sin información suficiente sobre: {solicitud.tema}",
        resumen=(
            "La base de conocimiento no contiene noticias relevantes para este tema"
            f"{filtros}. "
            "Para no inventar información, el sistema no genera un informe."
        ),
        hallazgos=[],
        conclusion=motivo,
        fuentes=[],
        sentimiento_general="sin_datos",
        score_promedio=0.5,
        hay_informacion_suficiente=False,
    )


async def redactar(
    llm: BaseChatModel,
    solicitud: SolicitudInforme,
    investigacion: ResultadoInvestigacion,
    analisis: ResultadoAnalisis,
) -> InformeInteligencia:
    fragmentos = investigacion.busqueda.fragmentos
    if not fragmentos:
        return informe_sin_informacion(solicitud, "No se encontraron noticias que superaran el umbral de relevancia.")

    cadena = (PROMPT_REDACTOR | llm.with_structured_output(InformeLLM, method="function_calling")).with_retry(
        stop_after_attempt=2
    )
    borrador: InformeLLM = await cadena.ainvoke(
        {
            "tema": solicitud.tema,
            "sentimiento": analisis.sentimiento_general,
            "score": analisis.score_promedio,
            "noticias": formatear_fragmentos(fragmentos),
        },
        config={"run_name": "redactor.informe"},
    )

    # --- Filtro de veracidad: solo valen citas a noticias recuperadas ---
    por_id = {f.doc_id: f for f in fragmentos}
    hallazgos: list[Hallazgo] = []
    citas_descartadas = 0
    hallazgos_descartados = 0
    for h in borrador.hallazgos:
        citas = list(dict.fromkeys(c.strip().strip("[]").upper() for c in h.citas))
        validas = [c for c in citas if c in por_id]
        citas_descartadas += len(citas) - len(validas)
        if validas:
            hallazgos.append(Hallazgo(texto=h.texto, citas=validas))
        else:
            hallazgos_descartados += 1

    if citas_descartadas or hallazgos_descartados:
        logger.warning(
            "Filtro de veracidad: %d citas y %d hallazgos descartados.", citas_descartadas, hallazgos_descartados
        )

    if not hallazgos:
        return informe_sin_informacion(
            solicitud, "El borrador del informe no pudo fundamentarse en las noticias recuperadas."
        ).model_copy(update={"citas_descartadas": citas_descartadas, "hallazgos_descartados": hallazgos_descartados})

    citadas = list(dict.fromkeys(c for h in hallazgos for c in h.citas))
    return InformeInteligencia(
        titulo=borrador.titulo,
        resumen=borrador.resumen,
        hallazgos=hallazgos,
        conclusion=borrador.conclusion,
        fuentes=[
            Cita(doc_id=d, titular=por_id[d].titular, fuente=por_id[d].fuente, fecha=por_id[d].fecha) for d in citadas
        ],
        sentimiento_general=analisis.sentimiento_general,
        score_promedio=analisis.score_promedio,
        hay_informacion_suficiente=True,
        citas_descartadas=citas_descartadas,
        hallazgos_descartados=hallazgos_descartados,
    )
