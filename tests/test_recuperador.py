"""Búsqueda híbrida: fusión RRF, filtros de metadatos, umbral de relevancia e ingesta idempotente."""

from datetime import date

import pytest
from pydantic import ValidationError

from app.rag.almacen import AlmacenEnMemoria, FiltroMetadatos
from app.rag.ingesta import asegurar_indice
from app.rag.recuperador import IndiceNoListo, RecuperadorHibrido
from app.rag.schemas import ParametrosBusqueda


async def test_encuentra_noticias_relevantes(recuperador):
    r = await recuperador.buscar(ParametrosBusqueda(consulta="incidentes de seguridad con agentes"))
    ids = [f.doc_id for f in r.fragmentos]
    assert {"N06", "N19"} <= set(ids)
    assert r.encontro_resultados
    # los puntajes de cada método quedan expuestos para auditar el ranking
    assert all(f.similitud >= recuperador.umbral_similitud for f in r.fragmentos)
    assert [f.score_rrf for f in r.fragmentos] == sorted((f.score_rrf for f in r.fragmentos), reverse=True)


async def test_tema_fuera_de_la_base_no_devuelve_nada(recuperador):
    """El umbral de relevancia es lo que evita devolver 'lo menos malo'."""
    r = await recuperador.buscar(ParametrosBusqueda(consulta="resultados del campeonato de fútbol"))
    assert r.fragmentos == []
    assert r.candidatos_evaluados > 0
    assert r.descartados_por_umbral == r.candidatos_evaluados


async def test_filtro_por_fecha(recuperador):
    r = await recuperador.buscar(
        ParametrosBusqueda(consulta="modelos de lenguaje nuevos", fecha_desde=date(2026, 5, 1))
    )
    assert r.fragmentos
    assert all(f.fecha >= date(2026, 5, 1) for f in r.fragmentos)


async def test_filtro_por_fuente_sin_importar_tildes_ni_mayusculas(recuperador):
    r = await recuperador.buscar(ParametrosBusqueda(consulta="salud", fuentes=["portal ciencia abierta"]))
    assert [f.doc_id for f in r.fragmentos] == ["N09"]
    assert all(f.fuente == "Portal Ciencia Abierta" for f in r.fragmentos)


async def test_filtro_por_etiquetas(recuperador):
    r = await recuperador.buscar(ParametrosBusqueda(consulta="inteligencia artificial", etiquetas=["robótica"]))
    assert {f.doc_id for f in r.fragmentos} <= {"N12"}


async def test_respeta_k(recuperador):
    r = await recuperador.buscar(ParametrosBusqueda(consulta="inteligencia artificial empresas", k=2))
    assert len(r.fragmentos) <= 2


async def test_un_fragmento_por_noticia(recuperador):
    r = await recuperador.buscar(ParametrosBusqueda(consulta="agentes", k=10))
    ids = [f.doc_id for f in r.fragmentos]
    assert len(ids) == len(set(ids))
    assert all(f.chunk_ids and all(c.startswith(f.doc_id + "#") for c in f.chunk_ids) for f in r.fragmentos)


async def test_bm25_rescata_nombres_propios(recuperador):
    """Un nombre propio ('Cóndor') es terreno de BM25: tiene que aparecer
    aunque la similitud semántica no sea la más alta."""
    r = await recuperador.buscar(ParametrosBusqueda(consulta="Cóndor"))
    assert {"N01", "N29"} & {f.doc_id for f in r.fragmentos}
    assert any(f.score_bm25 > 0 for f in r.fragmentos)


async def test_no_se_puede_buscar_sin_inicializar(almacen, embeddings):
    with pytest.raises(IndiceNoListo):
        await RecuperadorHibrido(almacen, embeddings).buscar(ParametrosBusqueda(consulta="ia"))


async def test_no_inicializa_con_la_base_vacia(embeddings):
    with pytest.raises(IndiceNoListo):
        await RecuperadorHibrido(AlmacenEnMemoria(), embeddings).inicializar()


@pytest.mark.parametrize(
    "args",
    [
        {"consulta": "x"},
        {"consulta": "ia", "k": 50},
        {"consulta": "ia", "fecha_desde": "2026-06-01", "fecha_hasta": "2026-01-01"},
        {"consulta": "ia", "fuentes": [f"f{i}" for i in range(11)]},
    ],
)
def test_parametros_invalidos_se_rechazan(args):
    with pytest.raises(ValidationError):
        ParametrosBusqueda.model_validate(args)


def test_filtro_se_traduce_a_where_de_chroma():
    assert FiltroMetadatos().a_where_chroma() is None
    assert FiltroMetadatos(fecha_desde_num=20260101).a_where_chroma() == {"fecha_num": {"$gte": 20260101}}
    assert FiltroMetadatos(fecha_desde_num=1, fuentes_norm=("a",)).a_where_chroma() == {
        "$and": [{"fecha_num": {"$gte": 1}}, {"fuente_norm": {"$in": ["a"]}}]
    }


async def test_ingesta_idempotente(settings, embeddings):
    almacen = AlmacenEnMemoria()
    primera = await asegurar_indice(almacen, embeddings, settings)
    segunda = await asegurar_indice(almacen, embeddings, settings)
    assert primera.reindexado and not segunda.reindexado
    assert almacen.operaciones_agregar == 1
    assert embeddings.llamadas_documentos == 1  # no se pagan embeddings dos veces
    assert segunda.chunks == primera.chunks

    forzada = await asegurar_indice(almacen, embeddings, settings, forzar=True)
    assert forzada.reindexado and almacen.operaciones_agregar == 2


async def test_metodos_por_separado(recuperador):
    """Cada método por separado (lo usa scripts/evaluar_rag.py). BM25 solo
    devuelve chunks que comparten términos con la consulta."""
    params = ParametrosBusqueda(consulta="Cóndor", k=10)
    solo_bm25 = await recuperador.buscar(params, metodo="bm25")
    assert solo_bm25.fragmentos and all(f.score_bm25 > 0 for f in solo_bm25.fragmentos)
    solo_vector = await recuperador.buscar(params, metodo="vector")
    hibrido = await recuperador.buscar(params, metodo="hibrido")
    assert {f.doc_id for f in hibrido.fragmentos} >= {f.doc_id for f in solo_bm25.fragmentos} & {
        f.doc_id for f in solo_vector.fragmentos
    }
