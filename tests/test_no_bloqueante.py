"""
Con un servidor HTTP REAL (uvicorn): la API no bloquea el event loop.

Un grafo falso tarda 1 s por tarea (espera ASÍNCRONA, como la espera a un
LLM). Se lanzan 5 POST /tasks a la vez y se verifica que:
1. los 5 responden enseguida (no esperan al grafo);
2. /health responde rápido mientras las 5 tareas corren;
3. las 5 terminan en ~1 s en total (concurrencia), no en ~5 s.
"""

import asyncio
import socket
import time
from types import SimpleNamespace

import httpx
import uvicorn

from app.main import crear_app

DURACION = 1.0


class GrafoLento:
    async def astream(self, entrada, config=None, stream_mode=None):
        await asyncio.sleep(DURACION)
        yield {"Supervisor": {"next_agent": "FINALIZAR", "decisiones": ["[paso 1] FINALIZAR: prueba"]}}

    async def aget_state(self, config):
        return SimpleNamespace(
            interrupts=(),
            values={"solicitud": SimpleNamespace(tema="t"), "pasos": 1, "intentos_investigacion": 0},
        )


def _puerto_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_cinco_peticiones_concurrentes_no_bloquean(jobs, recuperador, settings):
    app = crear_app(lifespan_fn=None)
    app.state.settings, app.state.jobs, app.state.recuperador = settings, jobs, recuperador
    app.state.grafo, app.state.observabilidad = GrafoLento(), None

    puerto = _puerto_libre()
    servidor = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=puerto, log_level="warning", lifespan="off"))
    tarea = asyncio.create_task(servidor.serve())
    while not servidor.started:  # noqa: ASYNC110 -- uvicorn no expone un evento de "arrancado"
        await asyncio.sleep(0.01)

    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{puerto}") as c:
            inicio = time.perf_counter()

            async def post(tema):
                t0 = time.perf_counter()
                r = await c.post("/tasks", json={"tema": tema})
                return r, time.perf_counter() - t0

            respuestas = await asyncio.gather(*[post(f"tema {i}") for i in range(5)])
            for r, latencia in respuestas:
                assert r.status_code == 202
                assert latencia < DURACION / 2, f"POST tardó {latencia:.2f} s: la API esperó al grafo"

            t0 = time.perf_counter()
            assert (await c.get("/health")).status_code == 200
            assert time.perf_counter() - t0 < 0.3

            ids = [r.json()["job_id"] for r, _ in respuestas]
            while True:
                estados = [(await c.get(f"/tasks/{i}")).json()["status"] for i in ids]
                if all(e == "DONE" for e in estados):
                    break
                assert time.perf_counter() - inicio < 5 * DURACION, f"no corren en paralelo: {estados}"
                await asyncio.sleep(0.05)
            assert time.perf_counter() - inicio < 2.5 * DURACION
    finally:
        servidor.should_exit = True
        await tarea
