"""
app/jobs.py
-----------
Estado de los jobs y sus eventos, en Redis (cliente `redis.asyncio`).

Ojo con la diferencia entre lo que se guarda en Redis:
- `job:{id}` (este módulo): el resumen que ve el cliente de la API: en qué
  estado está, por qué nodo va, el resultado o el error.
- `job:{id}:eventos` (este módulo): un Redis Stream con un evento por cada
  nodo que se ejecuta. Lo consume GET /tasks/{id}/events (streaming SSE).
- `checkpoint:*` (`AsyncRedisSaver`, ver graph.py): la "foto" interna del
  grafo para pausar y reanudar.

Ciclo de vida:

    PENDING ─> RUNNING ─┬─> DONE
                        ├─> AWAITING_APPROVAL ─(approve)─> RUNNING ─> DONE / REJECTED
                        └─> FAILED   (excepción o timeout en el worker)
"""

import json
import time
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.state import ReportePublicado, SolicitudInforme

CLAVE_REPORTES = "reportes:publicados"
CLAVE_INDICE_JOBS = "jobs:indice"
MAX_EVENTOS_POR_JOB = 500


class EstadoJob(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    DONE = "DONE"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


ESTADOS_FINALES = frozenset({EstadoJob.DONE, EstadoJob.REJECTED, EstadoJob.FAILED})


def ahora_iso() -> str:
    return datetime.now(UTC).isoformat()


class Job(BaseModel):
    job_id: str
    status: EstadoJob
    solicitud: SolicitudInforme
    creado_en: str = Field(default_factory=ahora_iso)
    actualizado_en: str = Field(default_factory=ahora_iso)
    iniciado_en: str | None = None
    finalizado_en: str | None = None
    duracion_segundos: float | None = None
    paso_actual: str | None = Field(None, description="Último nodo del grafo que se ejecutó.")
    decisiones: list[str] = Field(default_factory=list, description="Bitácora de decisiones del Supervisor.")
    aprobacion_pendiente: dict[str, Any] | None = None
    resultado: dict[str, Any] | None = None
    error: str | None = None

    @property
    def es_final(self) -> bool:
        return self.status in ESTADOS_FINALES


class Evento(BaseModel):
    id: str  # id del Redis Stream, p. ej. "1730000000000-0"
    tipo: str
    datos: dict[str, Any]


class JobStore:
    """Todo el acceso a Redis para jobs, eventos y reportes publicados."""

    def __init__(self, redis: Redis, ttl_segundos: int = 7 * 24 * 3600):
        self.redis = redis
        self.ttl_segundos = ttl_segundos

    @staticmethod
    def _clave(job_id: str) -> str:
        return f"job:{job_id}"

    @staticmethod
    def _clave_eventos(job_id: str) -> str:
        return f"job:{job_id}:eventos"

    # -- Jobs -----------------------------------------------------------------
    async def crear(self, solicitud: SolicitudInforme) -> Job:
        job = Job(job_id=str(uuid.uuid4()), status=EstadoJob.PENDING, solicitud=solicitud)
        await self._guardar(job)
        await self.redis.zadd(CLAVE_INDICE_JOBS, {job.job_id: time.time()})
        await self.registrar_evento(job.job_id, "estado", {"status": job.status.value})
        return job

    async def obtener(self, job_id: str) -> Job | None:
        crudo = await self.redis.get(self._clave(job_id))
        return Job.model_validate_json(crudo) if crudo else None

    async def listar(self, limite: int = 20) -> list[Job]:
        ids = await self.redis.zrevrange(CLAVE_INDICE_JOBS, 0, limite - 1)
        jobs = [await self.obtener(i) for i in ids]
        return [j for j in jobs if j is not None]

    async def actualizar(self, job_id: str, **cambios: Any) -> Job:
        """Lo usa el worker, que es el único que escribe mientras el job corre."""
        job = await self.obtener(job_id)
        if job is None:
            raise KeyError(f"No existe el job {job_id}")
        anterior = job.status
        job = self._aplicar(job, cambios)
        await self._guardar(job)
        if job.status != anterior:
            await self.registrar_evento(job_id, "estado", {"status": job.status.value, "error": job.error})
        return job

    async def transicionar(self, job_id: str, desde: set[EstadoJob], hacia: EstadoJob, **cambios: Any) -> Job | None:
        """Cambia el estado SOLO si el actual está en `desde`, de forma
        atómica (WATCH/MULTI/EXEC). Si dos personas aprueban a la vez, solo
        una gana: la otra recibe None (y la API responde 409)."""
        clave = self._clave(job_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    await pipe.watch(clave)
                    crudo = await pipe.get(clave)
                    if not crudo:
                        return None
                    job = Job.model_validate_json(crudo)
                    if job.status not in desde:
                        return None
                    job = self._aplicar(job, {"status": hacia, **cambios})
                    pipe.multi()
                    pipe.set(clave, job.model_dump_json(), ex=self.ttl_segundos)
                    await pipe.execute()
                    break
                except WatchError:
                    continue
        await self.registrar_evento(job_id, "estado", {"status": hacia.value})
        return job

    # -- Eventos (Redis Streams) ----------------------------------------------
    async def registrar_evento(self, job_id: str, tipo: str, datos: dict[str, Any]) -> None:
        clave = self._clave_eventos(job_id)
        await self.redis.xadd(
            clave,
            {"tipo": tipo, "datos": json.dumps(datos, ensure_ascii=False, default=str)},
            maxlen=MAX_EVENTOS_POR_JOB,
            approximate=True,
        )
        await self.redis.expire(clave, self.ttl_segundos)

    async def leer_eventos(self, job_id: str, desde: str = "0", bloquear_ms: int | None = None) -> list[Evento]:
        """Eventos posteriores a `desde`. Con `bloquear_ms`, espera (sin
        bloquear el event loop: es un `await`) hasta que llegue alguno."""
        respuesta = await self.redis.xread({self._clave_eventos(job_id): desde}, block=bloquear_ms, count=100)
        eventos: list[Evento] = []
        for _clave, entradas in respuesta or []:
            for evento_id, campos in entradas:
                eventos.append(Evento(id=evento_id, tipo=campos["tipo"], datos=json.loads(campos["datos"])))
        return eventos

    # -- Reportes publicados (el efecto secundario que protege el HITL) -------
    async def publicar_reporte(self, reporte: ReportePublicado) -> None:
        await self.redis.lpush(CLAVE_REPORTES, reporte.model_dump_json())

    async def reportes_publicados(self, limite: int = 20) -> list[ReportePublicado]:
        crudos = await self.redis.lrange(CLAVE_REPORTES, 0, limite - 1)
        return [ReportePublicado.model_validate_json(c) for c in crudos]

    # -- Internos ----------------------------------------------------------------
    async def _guardar(self, job: Job) -> None:
        await self.redis.set(self._clave(job.job_id), job.model_dump_json(), ex=self.ttl_segundos)

    @staticmethod
    def _aplicar(job: Job, cambios: dict[str, Any]) -> Job:
        datos = job.model_dump()
        datos.update(cambios)
        datos["actualizado_en"] = ahora_iso()
        estado = EstadoJob(datos["status"])
        if estado == EstadoJob.RUNNING and not datos.get("iniciado_en"):
            datos["iniciado_en"] = datos["actualizado_en"]
        if estado in ESTADOS_FINALES:
            datos["finalizado_en"] = datos["actualizado_en"]
            if datos.get("iniciado_en"):
                inicio = datetime.fromisoformat(datos["iniciado_en"])
                fin = datetime.fromisoformat(datos["finalizado_en"])
                datos["duracion_segundos"] = round((fin - inicio).total_seconds(), 3)
        return Job.model_validate(datos)
