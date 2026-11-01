"""
Contra un Redis REAL (lo que fakeredis no reproduce: los timeouts de red).

Se saltea si no hay Redis disponible. Para correrlo:

    docker compose up -d redis
    REDIS_TEST_URL=redis://localhost:6379 pytest tests/test_redis_integracion.py
"""

import os
import time
import uuid

import pytest

from app.config import Settings
from app.jobs import JobStore, crear_cliente_redis
from app.state import SolicitudInforme

URL = os.getenv("REDIS_TEST_URL")

pytestmark = pytest.mark.skipif(not URL, reason="definir REDIS_TEST_URL para correr contra un Redis real")


async def test_la_espera_del_sse_supera_los_5_segundos_sin_cortarse():
    """Bug real de la demo: con el cliente por defecto de redis-py 8
    (socket_timeout=5), un XREAD BLOCK más largo fallaba con TimeoutError."""
    s = Settings(_env_file=None, sse_keepalive_s=6.5)
    redis = crear_cliente_redis(URL, s.redis_socket_timeout_s)
    jobs = JobStore(redis, ttl_segundos=60)
    try:
        job = await jobs.crear(SolicitudInforme(tema=f"prueba {uuid.uuid4().hex[:6]}"))
        ultimo = (await jobs.leer_eventos(job.job_id))[-1].id
        inicio = time.monotonic()
        eventos = await jobs.leer_eventos(job.job_id, desde=ultimo, bloquear_ms=int(s.sse_keepalive_s * 1000))
        assert eventos == []
        assert time.monotonic() - inicio >= 6
    finally:
        await redis.aclose()
