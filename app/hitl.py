"""
app/hitl.py
-----------
Human-in-the-loop: antes de PUBLICAR un informe (una acción con efectos
secundarios que no se puede deshacer), el grafo se detiene si la tarea es
crítica y espera la decisión de una persona.

Cuándo es crítica (`evaluar_criticidad`, función pura):
- el sentimiento general del informe es NEGATIVO (riesgo reputacional), o
- quien creó la tarea pidió revisión obligatoria (`forzar_aprobacion`).

Cómo se pausa: `interrupt(payload)` guarda un checkpoint (en Redis) y corta
la ejecución. POST /tasks/{id}/approve reanuda el mismo hilo con
`Command(resume=decision)`: LangGraph recupera el estado del checkpoint y
vuelve a ejecutar este nodo desde el principio; esta vez `interrupt()`
devuelve la decisión. Por eso lo que está antes del `interrupt()` no tiene
efectos secundarios: se ejecuta dos veces.

El endpoint HTTP de aprobación está en app/api/aprobacion.py.
"""

from langgraph.types import interrupt

from app.state import AgentState, DecisionHumana, RegistroAprobacion


def evaluar_criticidad(state: AgentState) -> tuple[bool, str]:
    informe = state.get("informe")
    solicitud = state.get("solicitud")

    if solicitud is not None and solicitud.forzar_aprobacion:
        return True, "Quien creó la tarea pidió revisión humana obligatoria (forzar_aprobacion=true)."

    if informe is not None and informe.sentimiento_general == "negativo":
        return True, (
            f"El informe tiene sentimiento general NEGATIVO (score promedio {informe.score_promedio}). "
            "Publicarlo tiene riesgo reputacional: requiere aprobación humana."
        )

    sentimiento = informe.sentimiento_general if informe else "desconocido"
    return False, f"Sentimiento '{sentimiento}' (no negativo): se publica sin intervención humana."


def _propuesta(state: AgentState) -> dict:
    """Lo que la persona necesita ver para decidir (serializable a JSON)."""
    informe = state["informe"]
    return {
        "tema": state["solicitud"].tema,
        "titulo": informe.titulo,
        "resumen": informe.resumen,
        "sentimiento_general": informe.sentimiento_general,
        "score_promedio": informe.score_promedio,
        "hallazgos": [h.texto for h in informe.hallazgos],
        "fuentes": [f"[{c.doc_id}] {c.titular} ({c.fuente}, {c.fecha.isoformat()})" for c in informe.fuentes],
    }


async def nodo_aprobacion_humana(state: AgentState) -> dict:
    es_critica, motivo = evaluar_criticidad(state)
    if not es_critica:
        return {"aprobacion": RegistroAprobacion(requerida=False, motivo=motivo, aprobado=True, revisor="sistema")}

    # La primera vez, acá el grafo se DETIENE. Al reanudar, devuelve la decisión.
    respuesta = interrupt(
        {
            "tipo": "aprobacion_publicacion",
            "motivo": motivo,
            "propuesta": _propuesta(state),
            "instrucciones": (
                'POST /tasks/{job_id}/approve con {"aprobado": true|false, "comentario": "...", "revisor": "..."}'
            ),
        }
    )
    decision = DecisionHumana.model_validate(respuesta)  # no se confía en lo que llega de afuera
    return {
        "aprobacion": RegistroAprobacion(
            requerida=True,
            motivo=motivo,
            aprobado=decision.aprobado,
            comentario=decision.comentario,
            revisor=decision.revisor,
        )
    }


def rutear_tras_aprobacion(state: AgentState) -> str:
    aprobacion = state.get("aprobacion")
    return "Publicador" if aprobacion is not None and aprobacion.aprobado else "FIN"
