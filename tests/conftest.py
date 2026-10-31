"""
tests/conftest.py
-----------------
Fixtures compartidas. Ningún test necesita API keys, Redis, Chroma ni red:

- `settings`: configuración de prueba (umbral calibrado para los
  embeddings falsos).
- `almacen` / `recuperador`: el corpus real indexado en un almacén en
  memoria con `EmbeddingsHash`. Los vectores se calculan una sola vez por
  sesión.
- `jobs`: JobStore sobre `fakeredis`.
- `checkpointer`: `InMemorySaver` con el MISMO serializador y lista blanca
  que el checkpointer de Redis.
"""

from functools import lru_cache

import fakeredis
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.config import Settings
from app.jobs import JobStore
from app.rag.almacen import AlmacenEnMemoria
from app.rag.corpus import cargar_corpus, cargar_tokenizador, dividir_noticia, huella_corpus
from app.rag.recuperador import RecuperadorHibrido
from app.state import TIPOS_SERIALIZABLES
from tests.fakes import EmbeddingsHash

UMBRAL_TEST = 0.20  # calibrado para EmbeddingsHash (con embeddings reales se usa RAG_UMBRAL_SIMILITUD)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, rag_umbral_similitud=UMBRAL_TEST, langsmith_tracing=False)


@lru_cache(maxsize=1)
def _corpus_indexado():
    s = Settings(_env_file=None)
    noticias = cargar_corpus(s.corpus_dir)
    codificador = cargar_tokenizador()
    chunks = [
        c for n in noticias for c in dividir_noticia(n, codificador, s.rag_chunk_tokens, s.rag_chunk_solapamiento)
    ]
    vectores = EmbeddingsHash().embed_documents([c.texto for c in chunks])
    return huella_corpus(noticias), chunks, vectores


@pytest.fixture
def embeddings() -> EmbeddingsHash:
    return EmbeddingsHash()


@pytest.fixture
async def almacen() -> AlmacenEnMemoria:
    huella, chunks, vectores = _corpus_indexado()
    almacen = AlmacenEnMemoria()
    await almacen.reiniciar(huella)
    await almacen.agregar(chunks, vectores)
    return almacen


@pytest.fixture
async def recuperador(almacen, embeddings) -> RecuperadorHibrido:
    r = RecuperadorHibrido(almacen, embeddings, candidatos=20, rrf_k=60, umbral_similitud=UMBRAL_TEST)
    await r.inicializar()
    return r


@pytest.fixture
async def redis_falso():
    cliente = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield cliente
    await cliente.aclose()


@pytest.fixture
def jobs(redis_falso) -> JobStore:
    return JobStore(redis_falso)


@pytest.fixture
def checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=TIPOS_SERIALIZABLES))
