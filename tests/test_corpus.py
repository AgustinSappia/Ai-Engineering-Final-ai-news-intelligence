"""Carga del corpus, validación de metadatos y chunking por tokens."""

from datetime import date

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.rag.corpus import cargar_corpus, cargar_tokenizador, dividir_noticia, huella_corpus, leer_noticia
from app.rag.texto import normalizar, terminos


@pytest.fixture(scope="module")
def noticias():
    return cargar_corpus(Settings(_env_file=None).corpus_dir)


def test_el_corpus_tiene_30_noticias_validas(noticias):
    assert len(noticias) == 30
    assert len({n.doc_id for n in noticias}) == 30
    assert all(n.fecha.year == 2026 for n in noticias)
    assert len({n.fuente for n in noticias}) == 5


def test_lee_el_frontmatter(noticias):
    n06 = next(n for n in noticias if n.doc_id == "N06")
    assert n06.fecha == date(2026, 2, 25)
    assert n06.fuente == "Diario Tecnológico"
    assert "agentes" in n06.etiquetas


def test_archivo_sin_metadatos_falla(tmp_path):
    archivo = tmp_path / "N99.md"
    archivo.write_text("Una noticia sin frontmatter.", encoding="utf-8")
    with pytest.raises(ValueError, match="metadatos"):
        leer_noticia(archivo)


def test_archivo_con_metadatos_invalidos_falla(tmp_path):
    archivo = tmp_path / "X.md"
    archivo.write_text("---\nid: X1\ntitular: corto\nfuente: F\nfecha: 2026-01-01\n---\n\ncuerpo", encoding="utf-8")
    with pytest.raises(ValidationError):
        leer_noticia(archivo)


def test_chunks_respetan_el_presupuesto_de_tokens(noticias):
    codificar, decodificar = cargar_tokenizador()
    for n in noticias:
        chunks = dividir_noticia(n, (codificar, decodificar), max_tokens=160, solapamiento=40)
        assert chunks, n.doc_id
        assert [c.id for c in chunks] == [f"{n.doc_id}#{i}" for i in range(len(chunks))]
        for c in chunks:
            assert c.texto.startswith(n.titular)  # cada chunk lleva el titular como contexto
            assert len(codificar(c.texto)) <= 160 + 5


def test_parrafo_muy_largo_se_corta_en_ventanas(noticias):
    codificador = cargar_tokenizador()
    larga = noticias[0].model_copy(update={"cuerpo": " ".join(["palabra"] * 600)})
    chunks = dividir_noticia(larga, codificador, max_tokens=100, solapamiento=20)
    assert len(chunks) > 3


def test_la_huella_cambia_si_cambia_el_corpus_o_la_configuracion(noticias):
    base = huella_corpus(noticias, "modelo-a|160|40")
    assert huella_corpus(noticias, "modelo-a|160|40") == base
    assert huella_corpus(noticias, "modelo-b|160|40") != base
    modificado = [noticias[0].model_copy(update={"titular": "Otro titular distinto para la prueba"}), *noticias[1:]]
    assert huella_corpus(modificado, "modelo-a|160|40") != base


def test_normalizacion_para_bm25():
    assert normalizar("Regulación ÑANDÚ") == "regulacion ñandu"
    assert terminos("Las regulaciones de los AGENTES") == terminos("la regulación del agente")
