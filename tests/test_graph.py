"""Grafo completo: Supervisor, refinamiento, guardrails, HITL, publicación y persistencia del hilo."""

import pytest
from langgraph.types import Command
from pydantic import ValidationError

from app.graph import construir_grafo, transicion_obligatoria
from app.hitl import evaluar_criticidad
from app.state import DecisionSupervisor, SolicitudInforme, estado_inicial
from tests.fakes import ModeloGuionado


class Publicador:
    def __init__(self):
        self.publicados = []

    async def __call__(self, reporte):
        self.publicados.append(reporte)


def _grafo(recuperador, checkpointer, publicador=None, llm=None, **kwargs):
    return construir_grafo(
        llm=llm or ModeloGuionado(),
        recuperador=recuperador,
        checkpointer=checkpointer,
        publicar=publicador or Publicador(),
        **kwargs,
    )


def _cfg(hilo: str) -> dict:
    return {"configurable": {"thread_id": hilo}}


# ---------------------------------------------------------------------------
# Caminos principales
# ---------------------------------------------------------------------------
async def test_tema_positivo_se_publica_sin_pausa(recuperador, checkpointer):
    publicador = Publicador()
    grafo = _grafo(recuperador, checkpointer, publicador)
    r = await grafo.ainvoke(estado_inicial(SolicitudInforme(tema="chips de hardware eficientes")), _cfg("pos"))

    assert "__interrupt__" not in r
    assert r["informe"].sentimiento_general == "positivo"
    assert r["aprobacion"].requerida is False
    assert len(publicador.publicados) == 1
    assert [d.split("]")[1].split(":")[0].strip() for d in r["decisiones"]] == [
        "Investigador",
        "Analista",
        "Redactor",
        "FINALIZAR",
    ]


async def test_tema_negativo_pausa_y_se_publica_al_aprobar(recuperador, checkpointer):
    publicador = Publicador()
    grafo = _grafo(recuperador, checkpointer, publicador)
    cfg = _cfg("neg")
    r = await grafo.ainvoke(estado_inicial(SolicitudInforme(tema="incidentes de seguridad con agentes")), cfg)

    assert "__interrupt__" in r
    payload = r["__interrupt__"][0].value
    assert payload["propuesta"]["sentimiento_general"] == "negativo"
    assert payload["propuesta"]["fuentes"]
    assert publicador.publicados == []
    assert (await grafo.aget_state(cfg)).next == ("AprobacionHumana",)

    r = await grafo.ainvoke(Command(resume={"aprobado": True, "revisor": "rayo"}), cfg)
    assert r["aprobacion"].requerida and r["aprobacion"].aprobado
    assert r["reporte"].aprobado_por == "rayo"
    assert len(publicador.publicados) == 1


async def test_rechazo_no_publica(recuperador, checkpointer):
    publicador = Publicador()
    grafo = _grafo(recuperador, checkpointer, publicador)
    cfg = _cfg("rech")
    await grafo.ainvoke(estado_inicial(SolicitudInforme(tema="incidentes de seguridad con agentes")), cfg)
    r = await grafo.ainvoke(Command(resume={"aprobado": False, "comentario": "No"}), cfg)
    assert r["aprobacion"].aprobado is False
    assert r["reporte"] is None
    assert publicador.publicados == []


async def test_forzar_aprobacion(recuperador, checkpointer):
    grafo = _grafo(recuperador, checkpointer)
    r = await grafo.ainvoke(
        estado_inicial(SolicitudInforme(tema="chips de hardware eficientes", forzar_aprobacion=True)), _cfg("forz")
    )
    assert "__interrupt__" in r


async def test_tema_fuera_de_la_base_responde_sin_informacion(recuperador, checkpointer):
    publicador = Publicador()
    llm = ModeloGuionado()
    grafo = _grafo(recuperador, checkpointer, publicador, llm=llm)
    r = await grafo.ainvoke(
        estado_inicial(SolicitudInforme(tema="resultados del campeonato de fútbol")), _cfg("trampa")
    )

    assert r["intentos_investigacion"] == 2  # se refinó una vez antes de rendirse
    assert r["informe"].hay_informacion_suficiente is False
    assert r["aprobacion"] is None and r["reporte"] is None
    assert publicador.publicados == []
    assert "DecisionSupervisor" not in llm.llamadas  # no hizo falta el criterio del LLM
    assert "InformeLLM" not in llm.llamadas  # no se le pide al LLM que escriba sobre nada


async def test_decision_humana_invalida_falla(recuperador, checkpointer):
    grafo = _grafo(recuperador, checkpointer)
    cfg = _cfg("inval")
    await grafo.ainvoke(estado_inicial(SolicitudInforme(tema="incidentes de seguridad con agentes")), cfg)
    with pytest.raises(ValidationError):
        await grafo.ainvoke(Command(resume={"comentario": "falta aprobado"}), cfg)


# ---------------------------------------------------------------------------
# Supervisor: criterio del LLM + guardrails
# ---------------------------------------------------------------------------
async def test_el_supervisor_puede_pedir_refinar_y_el_guardrail_lo_limita(recuperador, checkpointer):
    """El LLM pide refinar SIEMPRE: el grafo refina una vez y después el
    guardrail de intentos fuerza FINALIZAR (no hay ciclo infinito)."""
    pedidos = []

    async def siempre_refinar(state):
        pedidos.append(state["intentos_investigacion"])
        return DecisionSupervisor(next="Investigador", razon="Quiero más fuentes.")

    grafo = _grafo(recuperador, checkpointer, decidir=siempre_refinar, max_intentos_investigacion=2)
    r = await grafo.ainvoke(estado_inicial(SolicitudInforme(tema="chips de hardware eficientes")), _cfg("refinar"))

    assert r["intentos_investigacion"] == 2
    assert pedidos == [1, 2]
    assert "(guardrail) Sin más intentos" in r["decisiones"][-1]
    assert r["reporte"] is not None


async def test_guardrail_si_el_llm_elige_un_agente_que_ya_actuo(recuperador, checkpointer):
    async def elige_analista(state):
        return DecisionSupervisor(next="Analista", razon="De nuevo.")

    grafo = _grafo(recuperador, checkpointer, decidir=elige_analista)
    r = await grafo.ainvoke(estado_inicial(SolicitudInforme(tema="chips de hardware eficientes")), _cfg("guard"))
    assert "(guardrail) El LLM eligió Analista" in r["decisiones"][-1]
    assert r["next_agent"] == "FINALIZAR"


async def test_tope_de_pasos(recuperador, checkpointer):
    async def siempre_refinar(state):
        return DecisionSupervisor(next="Investigador", razon="Más.")

    grafo = _grafo(recuperador, checkpointer, decidir=siempre_refinar, max_pasos=3, max_intentos_investigacion=5)
    r = await grafo.ainvoke(estado_inicial(SolicitudInforme(tema="chips de hardware eficientes")), _cfg("tope"))
    assert r["pasos"] == 4
    assert "Tope de 3 pasos" in r["decisiones"][-1]


async def test_el_refinamiento_le_pasa_el_motivo_al_investigador(recuperador, checkpointer):
    decisiones = iter(
        [
            DecisionSupervisor(next="Investigador", razon="Solo una fuente."),
            DecisionSupervisor(next="FINALIZAR", razon="Ok."),
        ]
    )

    async def decidir(state):
        return next(decisiones)

    grafo = _grafo(recuperador, checkpointer, decidir=decidir)
    r = await grafo.ainvoke(estado_inicial(SolicitudInforme(tema="chips de hardware eficientes")), _cfg("motivo"))
    assert r["motivo_refinamiento"] == "Solo una fuente."
    assert r["investigacion"].intento == 2


def test_transiciones_obligatorias():
    base = estado_inicial(SolicitudInforme(tema="tema de prueba"))
    assert transicion_obligatoria(base, 2)[0] == "Investigador"


def test_criticidad():
    from app.state import InformeInteligencia

    def informe(sentimiento):
        return InformeInteligencia(
            titulo="t",
            resumen="r",
            conclusion="c",
            sentimiento_general=sentimiento,
            score_promedio=0.5,
            hay_informacion_suficiente=True,
        )

    sol = SolicitudInforme(tema="tema")
    assert evaluar_criticidad({"solicitud": sol, "informe": informe("negativo")})[0] is True
    assert evaluar_criticidad({"solicitud": sol, "informe": informe("positivo")})[0] is False
    forzada = SolicitudInforme(tema="tema", forzar_aprobacion=True)
    assert evaluar_criticidad({"solicitud": forzada, "informe": informe("positivo")})[0] is True


# ---------------------------------------------------------------------------
# Persistencia del hilo (checkpointer)
# ---------------------------------------------------------------------------
async def test_reanudar_con_otra_instancia_del_grafo(recuperador, checkpointer):
    """Simula un reinicio de la API entre la pausa y la aprobación: el grafo
    que reanuda es otra instancia y todo lo que sabe sale del checkpointer."""
    cfg = _cfg("reinicio")
    antes = _grafo(recuperador, checkpointer)
    await antes.ainvoke(estado_inicial(SolicitudInforme(tema="incidentes de seguridad con agentes")), cfg)

    publicador = Publicador()
    llm_nuevo = ModeloGuionado()
    despues = _grafo(recuperador, checkpointer, publicador, llm=llm_nuevo)
    r = await despues.ainvoke(Command(resume={"aprobado": True}), cfg)

    assert r["investigacion"].busqueda.fragmentos  # el estado vino del checkpoint
    assert len(publicador.publicados) == 1
    assert llm_nuevo.llamadas == []  # no se re-ejecutó nada que ya estaba hecho


async def test_hilos_concurrentes_no_se_mezclan(recuperador, checkpointer):
    import asyncio

    grafo = _grafo(recuperador, checkpointer)
    temas = [
        "chips de hardware eficientes",
        "incidentes de seguridad con agentes",
        "resultados del campeonato de fútbol",
    ]
    resultados = await asyncio.gather(
        *[grafo.ainvoke(estado_inicial(SolicitudInforme(tema=t)), _cfg(f"c{i}")) for i, t in enumerate(temas)]
    )
    assert [r["solicitud"].tema for r in resultados] == temas
