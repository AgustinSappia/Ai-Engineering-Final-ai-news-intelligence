"""Worker en segundo plano: estados finales, eventos y manejo de errores (siempre FAILED, nunca colgado)."""

import asyncio
from types import SimpleNamespace

from app import worker
from app.graph import construir_grafo
from app.jobs import EstadoJob, JobStore
from app.state import DecisionHumana, SolicitudInforme
from tests.fakes import ModeloGuionado


def _grafo(recuperador, checkpointer, jobs):
    return construir_grafo(
        llm=ModeloGuionado(), recuperador=recuperador, checkpointer=checkpointer, publicar=jobs.publicar_reporte
    )


class GrafoQueExplota:
    async def astream(self, entrada, config=None, stream_mode=None):
        raise RuntimeError("el proveedor del LLM devolvió 500")
        yield  # pragma: no cover

    async def aget_state(self, config):  # pragma: no cover
        return SimpleNamespace(interrupts=(), values={})


class GrafoLento:
    async def astream(self, entrada, config=None, stream_mode=None):
        await asyncio.sleep(5)
        yield {}

    async def aget_state(self, config):  # pragma: no cover
        return SimpleNamespace(interrupts=(), values={})


async def test_job_positivo_termina_done_y_publica(recuperador, checkpointer, jobs: JobStore, settings):
    job = await jobs.crear(SolicitudInforme(tema="chips de hardware eficientes"))
    await worker.ejecutar_job(job.job_id, grafo=_grafo(recuperador, checkpointer, jobs), jobs=jobs, settings=settings)

    final = await jobs.obtener(job.job_id)
    assert final.status == EstadoJob.DONE
    assert final.resultado["publicado"] is True
    assert final.resultado["informe"]["fuentes"]
    assert len(final.decisiones) == 4
    assert len(await jobs.reportes_publicados()) == 1


async def test_eventos_por_nodo(recuperador, checkpointer, jobs: JobStore, settings):
    job = await jobs.crear(SolicitudInforme(tema="chips de hardware eficientes"))
    await worker.ejecutar_job(job.job_id, grafo=_grafo(recuperador, checkpointer, jobs), jobs=jobs, settings=settings)
    nodos = [e.datos["nodo"] for e in await jobs.leer_eventos(job.job_id) if e.tipo == "nodo"]
    assert nodos == [
        "Supervisor",
        "Investigador",
        "Supervisor",
        "Analista",
        "Supervisor",
        "Redactor",
        "Supervisor",
        "AprobacionHumana",
        "Publicador",
    ]


async def test_job_critico_espera_y_al_aprobar_termina(recuperador, checkpointer, jobs: JobStore, settings):
    grafo = _grafo(recuperador, checkpointer, jobs)
    job = await jobs.crear(SolicitudInforme(tema="incidentes de seguridad con agentes"))
    await worker.ejecutar_job(job.job_id, grafo=grafo, jobs=jobs, settings=settings)

    pausado = await jobs.obtener(job.job_id)
    assert pausado.status == EstadoJob.AWAITING_APPROVAL
    assert pausado.paso_actual == "AprobacionHumana"
    assert pausado.aprobacion_pendiente["propuesta"]["sentimiento_general"] == "negativo"

    await jobs.transicionar(job.job_id, {EstadoJob.AWAITING_APPROVAL}, EstadoJob.RUNNING)
    await worker.reanudar_job(
        job.job_id, DecisionHumana(aprobado=True, revisor="rayo"), grafo=grafo, jobs=jobs, settings=settings
    )
    final = await jobs.obtener(job.job_id)
    assert final.status == EstadoJob.DONE
    assert final.resultado["aprobacion"]["revisor"] == "rayo"


async def test_rechazo_termina_rejected(recuperador, checkpointer, jobs: JobStore, settings):
    grafo = _grafo(recuperador, checkpointer, jobs)
    job = await jobs.crear(SolicitudInforme(tema="incidentes de seguridad con agentes"))
    await worker.ejecutar_job(job.job_id, grafo=grafo, jobs=jobs, settings=settings)
    await jobs.transicionar(job.job_id, {EstadoJob.AWAITING_APPROVAL}, EstadoJob.RUNNING)
    await worker.reanudar_job(job.job_id, DecisionHumana(aprobado=False), grafo=grafo, jobs=jobs, settings=settings)
    final = await jobs.obtener(job.job_id)
    assert final.status == EstadoJob.REJECTED
    assert final.resultado["publicado"] is False


async def test_sin_informacion_termina_done_sin_publicar(recuperador, checkpointer, jobs: JobStore, settings):
    job = await jobs.crear(SolicitudInforme(tema="resultados del campeonato de fútbol"))
    await worker.ejecutar_job(job.job_id, grafo=_grafo(recuperador, checkpointer, jobs), jobs=jobs, settings=settings)
    final = await jobs.obtener(job.job_id)
    assert final.status == EstadoJob.DONE
    assert final.resultado["informe"]["hay_informacion_suficiente"] is False
    assert final.resultado["publicado"] is False


async def test_excepcion_marca_failed(jobs: JobStore, settings):
    job = await jobs.crear(SolicitudInforme(tema="agentes"))
    await worker.ejecutar_job(job.job_id, grafo=GrafoQueExplota(), jobs=jobs, settings=settings)
    final = await jobs.obtener(job.job_id)
    assert final.status == EstadoJob.FAILED
    assert "RuntimeError" in final.error and final.es_final


async def test_timeout_marca_failed(jobs: JobStore, settings):
    lento = settings.model_copy(update={"timeout_job_segundos": 0.05})
    job = await jobs.crear(SolicitudInforme(tema="agentes"))
    await worker.ejecutar_job(job.job_id, grafo=GrafoLento(), jobs=jobs, settings=lento)
    final = await jobs.obtener(job.job_id)
    assert final.status == EstadoJob.FAILED
    assert final.error.startswith("Timeout")


async def test_excepcion_al_reanudar_marca_failed(jobs: JobStore, settings):
    job = await jobs.crear(SolicitudInforme(tema="agentes"))
    await jobs.actualizar(job.job_id, status=EstadoJob.RUNNING)
    await worker.reanudar_job(
        job.job_id, DecisionHumana(aprobado=True), grafo=GrafoQueExplota(), jobs=jobs, settings=settings
    )
    assert (await jobs.obtener(job.job_id)).status == EstadoJob.FAILED


async def test_cinco_jobs_concurrentes_con_el_mismo_grafo(recuperador, checkpointer, jobs: JobStore, settings):
    grafo = _grafo(recuperador, checkpointer, jobs)
    temas = [
        "chips de hardware eficientes",
        "incidentes de seguridad con agentes",
        "inversión en robótica",
        "privacidad y multas por datos",
        "resultados del campeonato de fútbol",
    ]
    creados = [await jobs.crear(SolicitudInforme(tema=t)) for t in temas]
    await asyncio.gather(*[worker.ejecutar_job(j.job_id, grafo=grafo, jobs=jobs, settings=settings) for j in creados])

    for creado in creados:
        final = await jobs.obtener(creado.job_id)
        assert final.status in (EstadoJob.DONE, EstadoJob.AWAITING_APPROVAL), final.error
        if final.status == EstadoJob.DONE:
            assert final.resultado["tema"] == creado.solicitud.tema  # cada hilo con lo suyo
        else:
            assert final.aprobacion_pendiente["propuesta"]["tema"] == creado.solicitud.tema


# ---------------------------------------------------------------------------
# Recuperación tras un reinicio de la API
# ---------------------------------------------------------------------------
async def _cortar_a_mitad(grafo, job_id: str, solicitud: SolicitudInforme, pasos: int) -> None:
    """Simula que el proceso muere después de `pasos` nodos: el checkpoint
    queda guardado hasta ahí y el job queda en RUNNING."""
    from app.state import estado_inicial

    config = {"configurable": {"thread_id": job_id}}
    vistos = 0
    async for _ in grafo.astream(estado_inicial(solicitud), config=config, stream_mode="updates"):
        vistos += 1
        if vistos == pasos:
            break


async def test_job_interrumpido_se_retoma_desde_el_checkpoint(recuperador, checkpointer, jobs: JobStore, settings):
    """Caso real de la demo: el contenedor se reinició durante el Analista y
    el job quedaba en RUNNING para siempre."""
    solicitud = SolicitudInforme(tema="chips de hardware eficientes")
    job = await jobs.crear(solicitud)
    await jobs.actualizar(job.job_id, status=EstadoJob.RUNNING)
    grafo = _grafo(recuperador, checkpointer, jobs)
    await _cortar_a_mitad(grafo, job.job_id, solicitud, pasos=3)  # Supervisor, Investigador, Supervisor

    # "Reinicio": un grafo NUEVO sobre el mismo checkpointer
    accion = await worker.recuperar_job(
        job.job_id, grafo=_grafo(recuperador, checkpointer, jobs), jobs=jobs, settings=settings
    )

    final = await jobs.obtener(job.job_id)
    assert accion == "retomado"
    assert final.status == EstadoJob.DONE and final.resultado["publicado"] is True
    assert final.recuperaciones == 1
    eventos = await jobs.leer_eventos(job.job_id)
    recuperado = next(e for e in eventos if e.tipo == "recuperado")
    assert recuperado.datos["retoma_en"] == ["Analista"]
    nodos = [e.datos["nodo"] for e in eventos if e.tipo == "nodo"]
    assert nodos[0] == "Analista"  # no repite la búsqueda ya hecha
    assert "Investigador" not in nodos


async def test_job_pendiente_que_nunca_corrio_arranca_de_cero(recuperador, checkpointer, jobs: JobStore, settings):
    job = await jobs.crear(SolicitudInforme(tema="chips de hardware eficientes"))
    accion = await worker.recuperar_job(
        job.job_id, grafo=_grafo(recuperador, checkpointer, jobs), jobs=jobs, settings=settings
    )
    assert accion == "reiniciado"
    assert (await jobs.obtener(job.job_id)).status == EstadoJob.DONE


async def test_no_entra_en_un_bucle_de_reinicios(recuperador, checkpointer, jobs: JobStore, settings):
    job = await jobs.crear(SolicitudInforme(tema="agentes"))
    await jobs.actualizar(job.job_id, status=EstadoJob.RUNNING, recuperaciones=1)
    accion = await worker.recuperar_job(
        job.job_id, grafo=_grafo(recuperador, checkpointer, jobs), jobs=jobs, settings=settings
    )
    final = await jobs.obtener(job.job_id)
    assert accion == "abandonado"
    assert final.status == EstadoJob.FAILED and "se reinició 2 veces" in final.error


async def test_job_terminado_o_pausado_no_se_toca(recuperador, checkpointer, jobs: JobStore, settings):
    grafo = _grafo(recuperador, checkpointer, jobs)
    job = await jobs.crear(SolicitudInforme(tema="incidentes de seguridad con agentes"))
    await worker.ejecutar_job(job.job_id, grafo=grafo, jobs=jobs, settings=settings)
    assert (await jobs.obtener(job.job_id)).status == EstadoJob.AWAITING_APPROVAL
    assert await worker.recuperar_job(job.job_id, grafo=grafo, jobs=jobs, settings=settings) == "nada"
    assert [j.job_id for j in await jobs.en_curso()] == []


async def test_aprobacion_perdida_en_el_reinicio_vuelve_a_pedirse(recuperador, checkpointer, jobs: JobStore, settings):
    """Llegó el approve (job en RUNNING) pero el proceso murió antes de
    reanudar: el grafo sigue en el interrupt, así que se vuelve a pedir."""
    grafo = _grafo(recuperador, checkpointer, jobs)
    job = await jobs.crear(SolicitudInforme(tema="incidentes de seguridad con agentes"))
    await worker.ejecutar_job(job.job_id, grafo=grafo, jobs=jobs, settings=settings)
    await jobs.transicionar(job.job_id, desde={EstadoJob.AWAITING_APPROVAL}, hacia=EstadoJob.RUNNING)

    assert await worker.recuperar_job(job.job_id, grafo=grafo, jobs=jobs, settings=settings) == "cerrado"
    final = await jobs.obtener(job.job_id)
    assert final.status == EstadoJob.AWAITING_APPROVAL and final.aprobacion_pendiente


async def test_el_arranque_de_la_api_lanza_la_recuperacion(recuperador, checkpointer, jobs: JobStore, settings):
    from app.main import crear_app, iniciar_recuperacion

    app = crear_app(lifespan_fn=None)
    app.state.grafo = _grafo(recuperador, checkpointer, jobs)
    job = await jobs.crear(SolicitudInforme(tema="chips de hardware eficientes"))
    await jobs.actualizar(job.job_id, status=EstadoJob.RUNNING)

    tareas = await iniciar_recuperacion(app, jobs, settings)
    await asyncio.gather(*tareas)
    assert len(tareas) == 1
    assert (await jobs.obtener(job.job_id)).status == EstadoJob.DONE
