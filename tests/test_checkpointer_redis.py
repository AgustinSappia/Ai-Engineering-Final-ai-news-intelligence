"""
Regresión: el estado tiene que sobrevivir la conversión a JSON del
checkpointer de Redis (`AsyncRedisSaver`) y volver como objetos Pydantic.

En la Pre-entrega 7, el serializador solo permitía las clases propias en
modo msgpack: al reanudar un job desde Redis, los artefactos volvían como
`dict` y el grafo fallaba. Este test usa el código REAL de `AsyncRedisSaver`
que convierte el estado a JSON y lo vuelve a armar, con el serializador de
producción. No necesita un Redis levantado.
"""

from datetime import date

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.types import Command

from app.graph import construir_grafo, serializador_redis
from app.hitl import evaluar_criticidad
from app.rag.schemas import Fragmento, ResultadoBusqueda
from app.state import InformeInteligencia, ReportePublicado, ResultadoInvestigacion, SolicitudInforme, estado_inicial
from tests.fakes import ModeloGuionado


def _ida_y_vuelta(valores: dict) -> dict:
    saver = AsyncRedisSaver(redis_url="redis://localhost:6379")  # no se conecta hasta usarlo
    saver.serde = serializador_redis()
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = valores
    guardado = saver._dump_checkpoint(checkpoint)
    return saver._deserialize_channel_values(guardado["channel_values"])


async def _estado_final(recuperador, checkpointer, aprobar: bool):
    grafo = construir_grafo(llm=ModeloGuionado(), recuperador=recuperador, checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": "redis"}}
    solicitud = SolicitudInforme(tema="incidentes de seguridad con agentes", fecha_desde=date(2026, 1, 1))
    await grafo.ainvoke(estado_inicial(solicitud), cfg)
    if aprobar:
        await grafo.ainvoke(Command(resume={"aprobado": True, "revisor": "rayo"}), cfg)
    return (await grafo.aget_state(cfg)).values


async def test_estado_pausado_vuelve_como_objetos(recuperador, checkpointer):
    original = await _estado_final(recuperador, checkpointer, aprobar=False)
    leido = _ida_y_vuelta(original)

    assert isinstance(leido["solicitud"], SolicitudInforme)
    assert leido["solicitud"].fecha_desde == date(2026, 1, 1)
    assert isinstance(leido["investigacion"], ResultadoInvestigacion)
    assert isinstance(leido["investigacion"].busqueda, ResultadoBusqueda)
    assert all(isinstance(f, Fragmento) for f in leido["investigacion"].busqueda.fragmentos)
    assert isinstance(leido["informe"], InformeInteligencia)
    assert leido["informe"] == original["informe"]
    # Lo que fallaba en la Pre-entrega 7: el nodo de aprobación usa atributos.
    assert evaluar_criticidad(leido)[0] is True


async def test_estado_final_con_reporte_vuelve_como_objetos(recuperador, checkpointer):
    leido = _ida_y_vuelta(await _estado_final(recuperador, checkpointer, aprobar=True))
    assert isinstance(leido["reporte"], ReportePublicado)
    assert leido["aprobacion"].revisor == "rayo"
    assert leido["decisiones"]  # la bitácora también persiste
