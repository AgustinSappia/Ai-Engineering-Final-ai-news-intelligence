"""
Endpoints HTTP con `httpx.ASGITransport` (la app en memoria, sin servidor).

Con ASGITransport las BackgroundTasks terminan antes de que el cliente
reciba la respuesta, así que un GET justo después del POST ya ve el estado
final. Que la API real NO espera al grafo lo prueba test_no_bloqueante.py.
"""

import httpx
import pytest

from app.graph import construir_grafo
from app.main import crear_app
from tests.fakes import ModeloGuionado


class GrafoQueExplota:
    async def astream(self, entrada, config=None, stream_mode=None):
        raise ValueError("respuesta inválida del LLM")
        yield  # pragma: no cover


@pytest.fixture
def app_test(recuperador, checkpointer, jobs, settings):
    app = crear_app(lifespan_fn=None)
    app.state.settings = settings
    app.state.jobs = jobs
    app.state.recuperador = recuperador
    app.state.observabilidad = {"plataforma": "langsmith", "activo": False}
    app.state.grafo = construir_grafo(
        llm=ModeloGuionado(), recuperador=recuperador, checkpointer=checkpointer, publicar=jobs.publicar_reporte
    )
    return app


@pytest.fixture
async def cliente(app_test):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_test), base_url="http://test") as c:
        yield c


async def test_post_devuelve_202_con_job_id_y_links(cliente):
    r = await cliente.post("/tasks", json={"tema": "chips de hardware eficientes"})
    assert r.status_code == 202
    cuerpo = r.json()
    assert cuerpo["status"] == "PENDING"
    assert set(cuerpo["links"]) == {"estado", "eventos", "aprobar", "checkpoint"}
    assert r.headers["X-Request-ID"]


async def test_request_id_del_cliente_se_respeta(cliente):
    r = await cliente.get("/health", headers={"X-Request-ID": "mi-id-123"})
    assert r.headers["X-Request-ID"] == "mi-id-123"


async def test_flujo_completo_tema_positivo(cliente):
    job_id = (await cliente.post("/tasks", json={"tema": "chips de hardware eficientes"})).json()["job_id"]
    job = (await cliente.get(f"/tasks/{job_id}")).json()
    assert job["status"] == "DONE"
    assert job["resultado"]["informe"]["fuentes"]
    assert len((await cliente.get("/reportes")).json()) == 1


async def test_flujo_hitl_por_http(cliente):
    job_id = (await cliente.post("/tasks", json={"tema": "incidentes de seguridad con agentes"})).json()["job_id"]
    job = (await cliente.get(f"/tasks/{job_id}")).json()
    assert job["status"] == "AWAITING_APPROVAL"
    assert job["aprobacion_pendiente"]["propuesta"]["fuentes"]

    cp = (await cliente.get(f"/tasks/{job_id}/checkpoint")).json()
    assert cp["existe"] and cp["pausado_en_interrupt"] and cp["siguiente_nodo"] == ["AprobacionHumana"]
    assert cp["checkpoints_guardados"] > 5

    r = await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "revisor": "rayo"})
    assert r.status_code == 202
    assert (await cliente.get(f"/tasks/{job_id}")).json()["status"] == "DONE"
    assert (await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True})).status_code == 409


async def test_rechazo_por_http(cliente):
    job_id = (await cliente.post("/tasks", json={"tema": "incidentes de seguridad con agentes"})).json()["job_id"]
    await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": False, "comentario": "No"})
    assert (await cliente.get(f"/tasks/{job_id}")).json()["status"] == "REJECTED"
    assert (await cliente.get("/reportes")).json() == []


async def test_eventos_sse(cliente):
    job_id = (await cliente.post("/tasks", json={"tema": "chips de hardware eficientes"})).json()["job_id"]
    async with cliente.stream("GET", f"/tasks/{job_id}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        texto = "".join([trozo async for trozo in r.aiter_text()])
    assert texto.count("event: nodo") == 9
    assert '"status": "DONE"' in texto  # el stream se cierra en el estado final


async def test_errores_404_y_409(cliente):
    assert (await cliente.get("/tasks/no-existe")).status_code == 404
    assert (await cliente.get("/tasks/no-existe/events")).status_code == 404
    assert (await cliente.post("/tasks/no-existe/approve", json={"aprobado": True})).status_code == 404
    job_id = (await cliente.post("/tasks", json={"tema": "chips de hardware eficientes"})).json()["job_id"]
    r = await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True})
    assert r.status_code == 409 and "DONE" in r.json()["detail"]
    assert (await cliente.get("/tasks/no-existe/checkpoint")).json()["existe"] is False


@pytest.mark.parametrize(
    "cuerpo",
    [
        {},
        {"tema": "x"},
        {"tema": "agentes", "fecha_desde": "2026-06-01", "fecha_hasta": "2026-01-01"},
        {"tema": "agentes", "fecha_desde": "no-es-fecha"},
    ],
)
async def test_validacion_de_entrada_422(cliente, cuerpo):
    assert (await cliente.post("/tasks", json=cuerpo)).status_code == 422


async def test_aprobacion_invalida_422(cliente):
    assert (await cliente.post("/tasks/x/approve", json={"comentario": "sin campo aprobado"})).status_code == 422


async def test_error_del_grafo_se_ve_como_failed(app_test):
    app_test.state.grafo = GrafoQueExplota()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_test), base_url="http://test") as c:
        job_id = (await c.post("/tasks", json={"tema": "agentes"})).json()["job_id"]
        job = (await c.get(f"/tasks/{job_id}")).json()
    assert job["status"] == "FAILED"
    assert "ValueError" in job["error"]


async def test_busqueda_directa_en_el_rag(cliente):
    r = await cliente.post("/conocimiento/buscar", json={"consulta": "salud", "fuentes": ["Portal Ciencia Abierta"]})
    assert r.status_code == 200
    assert [f["doc_id"] for f in r.json()["fragmentos"]] == ["N09"]
    assert (await cliente.post("/conocimiento/buscar", json={"consulta": "ia", "k": 50})).status_code == 422


async def test_estado_del_conocimiento_y_health(cliente):
    conocimiento = (await cliente.get("/conocimiento")).json()
    assert conocimiento["noticias"] == 30
    health = (await cliente.get("/health")).json()
    assert health == {
        "status": "ok",
        "redis": True,
        "chroma": True,
        "indice_listo": True,
        "noticias_indexadas": 30,
        "observabilidad": {"plataforma": "langsmith", "activo": False},
    }


async def test_listar_tareas(cliente):
    job_id = (await cliente.post("/tasks", json={"tema": "agentes de IA"})).json()["job_id"]
    assert [j["job_id"] for j in (await cliente.get("/tasks")).json()] == [job_id]
    assert (await cliente.get("/tasks", params={"limite": 0})).status_code == 422


def test_todos_los_endpoints_tienen_modelo_de_respuesta(app_test):
    """Toda salida de la API está tipada con Pydantic (salvo el stream SSE)."""
    for ruta in app_test.routes:
        if getattr(ruta, "path", "").startswith(("/openapi", "/docs", "/redoc")) or not hasattr(ruta, "response_model"):
            continue
        if ruta.path.endswith("/events"):
            continue
        assert ruta.response_model is not None, ruta.path


def _error_openai(clase, estado: int, mensaje: str):
    respuesta = httpx.Response(estado, request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"))
    return clase(mensaje, response=respuesta, body=None)


def test_error_de_key_vencida_se_explica_claramente():
    """Caso real de la primera corrida: la key de OpenAI estaba vencida y el
    motivo quedaba al final de un traceback largo."""
    import openai

    from app.main import explicar_error_de_proveedor

    texto = explicar_error_de_proveedor(_error_openai(openai.AuthenticationError, 401, "Your API key has expired."))
    assert "OPENAI_API_KEY" in texto and "expired" in texto and "platform.openai.com/api-keys" in texto
    sin_saldo = explicar_error_de_proveedor(_error_openai(openai.RateLimitError, 429, "insufficient_quota"))
    assert "saldo" in sin_saldo
    assert explicar_error_de_proveedor(ValueError("otra cosa")) is None
