"""Validación de LangSmith al arrancar (clientes falsos: sin red ni key real) y config de trazas."""

import os

import langsmith
import pytest
from langsmith.utils import LangSmithAuthError

from app import observability
from app.config import Settings

KEY_VALIDA = "lsv2_pt_" + "a" * 40


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch):
    for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    yield
    langsmith.configure(enabled=None)  # _desactivar_tracing() apaga el tracing global: se restaura


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


class Acepta:
    def list_projects(self, limit=1):
        return iter([{}])


class Rechaza:
    def list_projects(self, limit=1):
        raise LangSmithAuthError("401")


class SinRed:
    def list_projects(self, limit=1):
        raise ConnectionError("sin red")


def test_tracing_apagado():
    assert observability.inicializar_observabilidad(_settings())["activo"] is False


def test_falta_la_key():
    estado = observability.inicializar_observabilidad(_settings(langsmith_tracing=True))
    assert estado["activo"] is False and "falta" in estado["motivo"]


def test_valor_de_ejemplo_no_cuenta_como_activo():
    estado = observability.inicializar_observabilidad(_settings(langsmith_tracing=True, langsmith_api_key="lsv2_..."))
    assert estado["activo"] is False and "ejemplo" in estado["motivo"]
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_key_valida_exporta_variables_para_langsmith():
    estado = observability.inicializar_observabilidad(
        _settings(langsmith_tracing=True, langsmith_api_key=KEY_VALIDA, langsmith_project="mi-proyecto")
    )
    assert estado["activo"] is True and estado["conexion"] == "sin verificar"
    assert os.environ["LANGSMITH_PROJECT"] == "mi-proyecto"


@pytest.mark.parametrize(
    ("cliente", "activo", "conexion"),
    [(Acepta, True, "ok"), (Rechaza, False, "rechazada"), (SinRed, True, "sin verificar (ConnectionError)")],
)
async def test_verificar_conexion(cliente, activo, conexion):
    estado = observability.inicializar_observabilidad(_settings(langsmith_tracing=True, langsmith_api_key=KEY_VALIDA))
    estado = await observability.verificar_conexion(estado, crear_cliente=cliente)
    assert estado["activo"] is activo and estado["conexion"] == conexion


def test_config_de_ejecucion():
    config = observability.config_de_ejecucion("job-1", "ejecucion", _settings())
    assert config["configurable"]["thread_id"] == "job-1"
    assert config["metadata"]["thread_id"] == "job-1"  # agrupa ejecución y reanudación en un Thread
    assert config["metadata"]["llm_modelo"] == "gpt-4o-mini"
    assert "ejecucion" in config["tags"]
