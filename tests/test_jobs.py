"""Estado de los jobs y eventos en Redis (sobre fakeredis)."""

import asyncio

from app.jobs import EstadoJob, JobStore
from app.state import SolicitudInforme


async def test_crear_y_leer(jobs: JobStore):
    job = await jobs.crear(SolicitudInforme(tema="agentes"))
    leido = await jobs.obtener(job.job_id)
    assert leido.status == EstadoJob.PENDING
    assert leido.solicitud.tema == "agentes"
    assert await jobs.obtener("no-existe") is None


async def test_ttl(jobs: JobStore, redis_falso):
    job = await jobs.crear(SolicitudInforme(tema="agentes"))
    assert 0 < await redis_falso.ttl(f"job:{job.job_id}") <= jobs.ttl_segundos
    assert 0 < await redis_falso.ttl(f"job:{job.job_id}:eventos") <= jobs.ttl_segundos


async def test_tiempos_del_ciclo_de_vida(jobs: JobStore):
    job = await jobs.crear(SolicitudInforme(tema="agentes"))
    corriendo = await jobs.actualizar(job.job_id, status=EstadoJob.RUNNING)
    assert corriendo.iniciado_en and not corriendo.finalizado_en
    final = await jobs.actualizar(job.job_id, status=EstadoJob.DONE)
    assert final.es_final and final.duracion_segundos is not None


async def test_doble_aprobacion_simultanea_solo_gana_una(jobs: JobStore):
    job = await jobs.crear(SolicitudInforme(tema="agentes"))
    await jobs.actualizar(job.job_id, status=EstadoJob.AWAITING_APPROVAL)
    resultados = await asyncio.gather(
        *[jobs.transicionar(job.job_id, {EstadoJob.AWAITING_APPROVAL}, EstadoJob.RUNNING) for _ in range(5)]
    )
    assert sum(r is not None for r in resultados) == 1


async def test_transicionar_desde_un_estado_no_permitido(jobs: JobStore):
    job = await jobs.crear(SolicitudInforme(tema="agentes"))
    assert await jobs.transicionar(job.job_id, {EstadoJob.AWAITING_APPROVAL}, EstadoJob.RUNNING) is None
    assert await jobs.transicionar("no-existe", {EstadoJob.PENDING}, EstadoJob.RUNNING) is None


async def test_eventos_en_orden_y_desde_un_id(jobs: JobStore):
    job = await jobs.crear(SolicitudInforme(tema="agentes"))
    await jobs.registrar_evento(job.job_id, "nodo", {"nodo": "Supervisor"})
    await jobs.actualizar(job.job_id, status=EstadoJob.RUNNING)

    eventos = await jobs.leer_eventos(job.job_id)
    assert [(e.tipo, e.datos.get("status") or e.datos.get("nodo")) for e in eventos] == [
        ("estado", "PENDING"),
        ("nodo", "Supervisor"),
        ("estado", "RUNNING"),
    ]
    siguientes = await jobs.leer_eventos(job.job_id, desde=eventos[0].id)
    assert len(siguientes) == 2


async def test_listar(jobs: JobStore):
    a = await jobs.crear(SolicitudInforme(tema="primero"))
    await asyncio.sleep(0.01)
    b = await jobs.crear(SolicitudInforme(tema="segundo"))
    assert [j.job_id for j in await jobs.listar()] == [b.job_id, a.job_id]
