"""Configuración: valores por defecto, validaciones y que no haya nada fijo en el código."""

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_valores_por_defecto():
    s = Settings(_env_file=None)
    assert s.llm_provider == "openai"
    assert s.modelo_llm == s.openai_model
    assert s.rag_k <= s.rag_candidatos


def test_lee_variables_de_entorno(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("CHROMA_COLECCION", "otra_coleccion")
    monkeypatch.setenv("RAG_K", "7")
    s = Settings(_env_file=None)
    assert s.openai_model == "gpt-4.1-mini"
    assert s.chroma_coleccion == "otra_coleccion"
    assert s.rag_k == 7


@pytest.mark.parametrize(
    "cambios",
    [
        {"llm_provider": "gemini"},
        {"rag_k": 0},
        {"rag_chunk_tokens": 100, "rag_chunk_solapamiento": 100},
        {"rag_k": 10, "rag_candidatos": 5},
        {"chroma_port": 70000},
        {"rag_k": 15},  # más que el máximo de la herramienta de búsqueda
        {"sse_keepalive_s": 30, "redis_socket_timeout_s": 5},  # el SSE se cortaría (bug real)
    ],
)
def test_configuracion_invalida_falla_al_arrancar(cambios):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **cambios)


def test_las_claves_no_se_imprimen():
    s = Settings(_env_file=None, openai_api_key="sk-secreta")
    assert "sk-secreta" not in repr(s)
    assert s.openai_api_key.get_secret_value() == "sk-secreta"


def test_el_cliente_redis_espera_mas_que_el_sse():
    """Regresión: redis-py 8 corta las lecturas a los 5 s por defecto y el
    XREAD BLOCK del stream SSE fallaba si un nodo tardaba más que eso."""
    from app.jobs import crear_cliente_redis

    s = Settings(_env_file=None)
    cliente = crear_cliente_redis(s.redis_url, s.redis_socket_timeout_s)
    assert cliente.connection_pool.connection_kwargs["socket_timeout"] == s.redis_socket_timeout_s
    assert s.sse_keepalive_s < s.redis_socket_timeout_s
