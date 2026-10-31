"""
app/rag/ingesta.py
------------------
Indexa el corpus en la base vectorial. Es IDEMPOTENTE: calcula una huella
del corpus + configuración (modelo de embeddings, tamaño de chunk) y, si
la base ya tiene esa misma huella, no hace nada. Así la API puede llamarla
en cada arranque sin volver a pagar embeddings.

Uso manual (por ejemplo para forzar la re-indexación):

    python -m app.rag.ingesta --forzar
"""

import argparse
import asyncio
import logging
import time

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from app.config import Settings
from app.rag.almacen import AlmacenVectorial
from app.rag.corpus import cargar_corpus, cargar_tokenizador, dividir_noticia, huella_corpus

logger = logging.getLogger("rag.ingesta")


class EstadoIngesta(BaseModel):
    huella: str
    reindexado: bool
    noticias: int
    chunks: int
    segundos: float


async def asegurar_indice(
    almacen: AlmacenVectorial, embeddings: Embeddings, settings: Settings, *, forzar: bool = False
) -> EstadoIngesta:
    inicio = time.perf_counter()

    # Leer archivos y cargar el tokenizador son operaciones bloqueantes
    # (disco y, la primera vez, red): van a otro thread.
    noticias = await asyncio.to_thread(cargar_corpus, settings.corpus_dir)
    configuracion = f"{settings.embeddings_model}|{settings.rag_chunk_tokens}|{settings.rag_chunk_solapamiento}"
    huella = huella_corpus(noticias, configuracion)

    if not forzar and await almacen.huella() == huella and await almacen.contar() > 0:
        logger.info("Índice al día (huella %s): no se re-indexa.", huella)
        return EstadoIngesta(
            huella=huella,
            reindexado=False,
            noticias=len(noticias),
            chunks=await almacen.contar(),
            segundos=round(time.perf_counter() - inicio, 3),
        )

    codificador = await asyncio.to_thread(cargar_tokenizador)
    chunks = [
        c
        for n in noticias
        for c in dividir_noticia(n, codificador, settings.rag_chunk_tokens, settings.rag_chunk_solapamiento)
    ]
    logger.info("Indexando %d noticias en %d chunks...", len(noticias), len(chunks))

    vectores = await embeddings.aembed_documents([c.texto for c in chunks])
    await almacen.reiniciar(huella)
    await almacen.agregar(chunks, vectores)

    estado = EstadoIngesta(
        huella=huella,
        reindexado=True,
        noticias=len(noticias),
        chunks=len(chunks),
        segundos=round(time.perf_counter() - inicio, 3),
    )
    logger.info("Ingesta terminada: %s", estado.model_dump())
    return estado


async def _main(forzar: bool) -> None:
    from app.config import obtener_settings
    from app.llm import crear_embeddings
    from app.rag.almacen import AlmacenChroma

    settings = obtener_settings()
    almacen = await AlmacenChroma.conectar(settings.chroma_host, settings.chroma_port, settings.chroma_coleccion)
    estado = await asegurar_indice(almacen, crear_embeddings(settings), settings, forzar=forzar)
    print(estado.model_dump_json(indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Indexa el corpus de noticias en Chroma.")
    parser.add_argument("--forzar", action="store_true", help="Re-indexar aunque la huella no haya cambiado.")
    asyncio.run(_main(parser.parse_args().forzar))
