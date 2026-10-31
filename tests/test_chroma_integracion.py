"""
Integración con un servidor de Chroma REAL (cliente asíncrono).

Se saltea si no hay un Chroma disponible. Para correrlo:

    docker compose up -d chroma
    CHROMA_TEST_HOST=localhost CHROMA_TEST_PORT=8000 pytest tests/test_chroma_integracion.py
"""

import os
import uuid
from datetime import date

import pytest

from app.rag.almacen import AlmacenChroma
from app.rag.ingesta import asegurar_indice
from app.rag.recuperador import RecuperadorHibrido
from app.rag.schemas import ParametrosBusqueda

HOST = os.getenv("CHROMA_TEST_HOST")
PORT = int(os.getenv("CHROMA_TEST_PORT", "8000"))

pytestmark = pytest.mark.skipif(not HOST, reason="definir CHROMA_TEST_HOST para correr contra un Chroma real")


@pytest.fixture
async def almacen_chroma():
    nombre = f"test_{uuid.uuid4().hex[:8]}"
    almacen = await AlmacenChroma.conectar(HOST, PORT, nombre)
    yield almacen
    await almacen._cliente.delete_collection(nombre)


async def test_ingesta_y_busqueda_en_chroma_real(almacen_chroma, embeddings, settings, recuperador):
    assert await almacen_chroma.ping()
    primera = await asegurar_indice(almacen_chroma, embeddings, settings)
    segunda = await asegurar_indice(almacen_chroma, embeddings, settings)
    assert primera.reindexado and not segunda.reindexado
    assert await almacen_chroma.contar() == primera.chunks

    en_chroma = RecuperadorHibrido(almacen_chroma, embeddings, umbral_similitud=recuperador.umbral_similitud)
    await en_chroma.inicializar()

    for params in (
        ParametrosBusqueda(consulta="incidentes de seguridad con agentes"),
        ParametrosBusqueda(consulta="modelos de lenguaje nuevos", fecha_desde=date(2026, 5, 1)),
        ParametrosBusqueda(consulta="salud", fuentes=["Portal Ciencia Abierta"]),
        ParametrosBusqueda(consulta="resultados del campeonato de fútbol"),
    ):
        real = await en_chroma.buscar(params)
        memoria = await recuperador.buscar(params)
        # Chroma (con sus filtros `where`) y el almacén en memoria tienen que coincidir.
        assert [f.doc_id for f in real.fragmentos] == [f.doc_id for f in memoria.fragmentos], params.consulta
