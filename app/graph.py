"""
app/graph.py
------------
El núcleo del sistema: el grafo de agentes (LangGraph) con Supervisor.

    INICIO -> Supervisor -> Investigador (RAG) ─┐
                ^  ^  ^ <─ Analista ────────────┤  (los especialistas siempre
                │  │  └─<─ Redactor ────────────┘   vuelven al Supervisor)
                │  └── FINALIZAR con informe -> AprobacionHumana -> Publicador -> FIN
                └───── FINALIZAR sin información -----------------------------> FIN

El Supervisor decide en dos niveles:
1. Transiciones OBLIGATORIAS, resueltas en código: si todavía no hay
   investigación, va el Investigador; si hay noticias pero no análisis, el
   Analista; si hay análisis pero no informe, el Redactor; si la búsqueda no
   encontró nada, se refina (hasta `max_intentos`). No necesitan "criterio" y
   así no dependen de que el LLM interprete bien una regla (lección de la
   Pre-entrega 6).
2. Decisión de CRITERIO, con un LLM y salida estructurada: con el informe
   ya escrito, ¿alcanza (FINALIZAR) o conviene refinar la investigación?
   Guardrails en código: no se puede refinar más de `max_intentos` veces y
   hay un tope de pasos contra el "Supervisor Infinito".

Cada decisión queda en `decisiones` (la bitácora), visible en la API y en
las trazas.

Todo lo externo se inyecta en `construir_grafo()` (LLM, recuperador,
checkpointer, publicador, función de decisión): así los tests corren el
grafo completo sin API keys ni servicios.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.analista import analizar
from app.agents.investigador import crear_herramienta_busqueda, investigar
from app.agents.redactor import redactar
from app.hitl import nodo_aprobacion_humana, rutear_tras_aprobacion
from app.rag.recuperador import RecuperadorHibrido
from app.state import (
    TIPOS_SERIALIZABLES,
    TIPOS_SERIALIZABLES_JSON,
    AgentState,
    DecisionSupervisor,
    ReportePublicado,
)

RECURSION_LIMIT = 60

DecisorType = Callable[[AgentState], Awaitable[DecisionSupervisor]]
PublicadorType = Callable[[ReportePublicado], Awaitable[None]]

PROMPT_SUPERVISOR = """Sos el Supervisor de un sistema de inteligencia sobre noticias de IA.
Coordinás a tres especialistas: Investigador (busca en la base de conocimiento), Analista (sentimiento)
y Redactor (informe con citas). Ya hay un informe escrito; tu tarea es evaluarlo y decidir:

- FINALIZAR: el informe responde el tema pedido con hallazgos respaldados por fuentes.
- Investigador: conviene una nueva búsqueda, más amplia o reformulada. Hacelo si el informe se apoya en
  una sola fuente cuando el tema pide más, si el filtro de veracidad descartó citas, o si las noticias
  encontradas no tratan el tema pedido.

No elijas Analista ni Redactor: ya actuaron. Explicá tu razón en una o dos oraciones."""


def resumen_estado(state: AgentState) -> str:
    """Resumen COMPACTO para el Supervisor (no todo el estado): evita la
    "contaminación de contexto"."""
    s = state["solicitud"]
    lineas = [
        f"Tema: {s.tema}",
        f"Filtros: desde={s.fecha_desde} hasta={s.fecha_hasta} fuentes={s.fuentes or 'todas'}",
        f"Intentos de investigación: {state['intentos_investigacion']}",
    ]
    inv = state.get("investigacion")
    if inv:
        b = inv.busqueda
        lineas.append(
            f"Búsqueda: consulta='{b.parametros.consulta}', {len(b.fragmentos)} noticias relevantes, "
            f"{b.descartados_por_umbral} candidatos descartados por baja relevancia."
        )
        lineas += [f"  - [{f.doc_id}] {f.titular} (similitud {f.similitud})" for f in b.fragmentos]
    ana = state.get("analisis")
    if ana:
        lineas.append(f"Análisis: sentimiento={ana.sentimiento_general}, score={ana.score_promedio}")
    inf = state.get("informe")
    if inf:
        lineas.append(
            f"Informe: '{inf.titulo}', {len(inf.hallazgos)} hallazgos, {len(inf.fuentes)} fuentes citadas, "
            f"{inf.citas_descartadas} citas descartadas y {inf.hallazgos_descartados} hallazgos descartados "
            "por el filtro de veracidad."
        )
        lineas += [f"  - {h.texto} {h.citas}" for h in inf.hallazgos]
    return "\n".join(lineas)


def crear_decisor_llm(llm: BaseChatModel) -> DecisorType:
    estructurado = llm.with_structured_output(DecisionSupervisor, method="function_calling").with_retry(
        stop_after_attempt=2
    )

    async def decidir(state: AgentState) -> DecisionSupervisor:
        return await estructurado.ainvoke(
            [SystemMessage(content=PROMPT_SUPERVISOR), HumanMessage(content=resumen_estado(state))],
            config={"run_name": "supervisor.evaluar_informe"},
        )

    return decidir


def transicion_obligatoria(state: AgentState, max_intentos: int) -> tuple[str, str] | None:
    """Transiciones que no requieren criterio. Devuelve (destino, razón) o
    None si hace falta la decisión del LLM."""
    inv = state.get("investigacion")
    if inv is None:
        return "Investigador", "Todavía no se investigó el tema."
    if not inv.encontro_resultados and state["intentos_investigacion"] < max_intentos:
        return "Investigador", "La búsqueda no encontró noticias relevantes: se reformula la consulta."
    if state.get("analisis") is None:
        if inv.encontro_resultados:
            return "Analista", "Hay resultados de búsqueda sin analizar."
        return (
            "Analista",
            f"Sin noticias relevantes tras {state['intentos_investigacion']} intentos: se registra que no hay datos.",
        )
    if state.get("informe") is None:
        return "Redactor", "Hay análisis pero todavía no hay informe."
    if not state["informe"].hay_informacion_suficiente:
        return "FINALIZAR", "No hay información suficiente en la base: se informa sin publicar."
    return None


async def _publicar_en_log(reporte: ReportePublicado) -> None:  # pragma: no cover - solo sin API
    print(f"[Publicador] {reporte.titulo}")


def construir_grafo(
    *,
    llm: BaseChatModel,
    recuperador: RecuperadorHibrido,
    checkpointer: BaseCheckpointSaver | None = None,
    publicar: PublicadorType | None = None,
    decidir: DecisorType | None = None,
    max_pasos: int = 10,
    max_intentos_investigacion: int = 2,
    k: int = 5,
) -> CompiledStateGraph:
    herramienta = crear_herramienta_busqueda(recuperador)
    decidir = decidir or crear_decisor_llm(llm)
    publicar = publicar or _publicar_en_log

    async def supervisor(state: AgentState) -> dict:
        pasos = state["pasos"] + 1

        if pasos > max_pasos:
            destino, razon = "FINALIZAR", f"Tope de {max_pasos} pasos alcanzado (protección contra ciclos infinitos)."
        elif obligatoria := transicion_obligatoria(state, max_intentos_investigacion):
            destino, razon = obligatoria
        else:
            decision = await decidir(state)
            destino, razon = decision.next, decision.razon
            if destino in ("Analista", "Redactor"):
                destino, razon = "FINALIZAR", f"(guardrail) El LLM eligió {decision.next}, que ya actuó. {razon}"
            elif destino == "Investigador" and state["intentos_investigacion"] >= max_intentos_investigacion:
                destino, razon = "FINALIZAR", f"(guardrail) Sin más intentos de investigación. El LLM sugería: {razon}"

        cambios: dict = {"pasos": pasos, "next_agent": destino, "decisiones": [f"[paso {pasos}] {destino}: {razon}"]}
        if destino == "Investigador" and state.get("investigacion") is not None:
            cambios["motivo_refinamiento"] = razon
        return cambios

    async def investigador(state: AgentState) -> dict:
        intento = state["intentos_investigacion"] + 1
        resultado = await investigar(
            llm,
            herramienta,
            state["solicitud"],
            intento=intento,
            k=k,
            motivo_refinamiento=state.get("motivo_refinamiento"),
        )
        # Una investigación nueva invalida el análisis y el informe anteriores.
        return {"investigacion": resultado, "intentos_investigacion": intento, "analisis": None, "informe": None}

    async def analista(state: AgentState) -> dict:
        return {"analisis": await analizar(llm, state["investigacion"])}

    async def redactor(state: AgentState) -> dict:
        informe = await redactar(llm, state["solicitud"], state["investigacion"], state["analisis"])
        return {"informe": informe}

    async def publicador(state: AgentState) -> dict:
        informe = state["informe"]
        reporte = ReportePublicado(
            tema=state["solicitud"].tema,
            titulo=informe.titulo,
            sentimiento_general=informe.sentimiento_general,
            cantidad_fuentes=len(informe.fuentes),
            aprobado_por=state["aprobacion"].revisor,
            publicado_en=datetime.now(UTC).isoformat(),
            informe=informe,
        )
        await publicar(reporte)
        return {"reporte": reporte}

    def rutear_desde_supervisor(state: AgentState) -> str:
        destino = state["next_agent"]
        if destino != "FINALIZAR":
            return destino
        informe = state.get("informe")
        return "AprobacionHumana" if informe is not None and informe.hay_informacion_suficiente else "FIN"

    grafo = StateGraph(AgentState)
    grafo.add_node("Supervisor", supervisor)
    grafo.add_node("Investigador", investigador)
    grafo.add_node("Analista", analista)
    grafo.add_node("Redactor", redactor)
    grafo.add_node("AprobacionHumana", nodo_aprobacion_humana)
    grafo.add_node("Publicador", publicador)

    grafo.set_entry_point("Supervisor")
    grafo.add_conditional_edges(
        "Supervisor",
        rutear_desde_supervisor,
        {
            "Investigador": "Investigador",
            "Analista": "Analista",
            "Redactor": "Redactor",
            "AprobacionHumana": "AprobacionHumana",
            "FIN": END,
        },
    )
    for especialista in ("Investigador", "Analista", "Redactor"):
        grafo.add_edge(especialista, "Supervisor")
    grafo.add_conditional_edges("AprobacionHumana", rutear_tras_aprobacion, {"Publicador": "Publicador", "FIN": END})
    grafo.add_edge("Publicador", END)

    return grafo.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Checkpointer de Redis
# ---------------------------------------------------------------------------
def serializador_redis():
    """Serializador con la lista blanca de clases propias, en los dos
    formatos que usa el checkpointer de Redis (msgpack y JSON)."""
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    return JsonPlusRedisSerializer(
        allowed_msgpack_modules=TIPOS_SERIALIZABLES,
        allowed_json_modules=TIPOS_SERIALIZABLES_JSON,
    )


@asynccontextmanager
async def checkpointer_redis(redis_url: str, ttl_minutos: int) -> AsyncIterator[BaseCheckpointSaver]:
    """`AsyncRedisSaver`: guarda un checkpoint del estado después de cada
    nodo, identificado por `thread_id` (= job_id). Si la API se reinicia,
    los jobs pausados se retoman desde Redis. Necesita Redis 8 (módulos de
    búsqueda y JSON incluidos), que es la imagen del docker-compose.yml."""
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    async with AsyncRedisSaver.from_conn_string(
        redis_url, ttl={"default_ttl": ttl_minutos, "refresh_on_read": True}
    ) as saver:
        saver.serde = serializador_redis()
        await saver.asetup()
        yield saver
