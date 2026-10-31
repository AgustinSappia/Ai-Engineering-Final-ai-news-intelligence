"""
app/rag/recuperador.py
----------------------
Búsqueda HÍBRIDA: la herramienta de conocimiento del agente Investigador.

Por qué híbrida: la búsqueda semántica (embeddings) entiende sinónimos
("despidos" ~ "pérdida de empleos") pero a veces se le escapan nombres
propios o términos exactos ("Cóndor-8"); BM25 (palabras clave) es al
revés. Se corren las dos y se fusionan los rankings.

Pasos de `buscar()`:
1. Embedding de la consulta (async) y búsqueda vectorial en Chroma, con
   los filtros de metadatos (fechas, fuentes) aplicados EN la base.
2. BM25 sobre los mismos chunks filtrados (índice en memoria, se arma al
   arrancar a partir de lo que hay en Chroma).
3. Fusión por Reciprocal Rank Fusion (RRF): puntaje = Σ 1 / (k + posición)
   en cada ranking. Premia lo que aparece arriba en los dos, sin tener que
   comparar escalas distintas (coseno vs. BM25). Es el re-ranking.
4. Umbral de relevancia: se descarta todo lo que no alcance una similitud
   semántica mínima. Esto es lo que permite responder "no hay información"
   en lugar de devolver lo menos malo (y que el LLM alucine a partir de
   ahí).
5. Agrupado por noticia: varios chunks de la misma noticia se devuelven
   como un solo `Fragmento`, que es lo que el Redactor puede citar.
"""

import logging
from typing import Literal

from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from app.rag.almacen import AlmacenVectorial, FiltroMetadatos
from app.rag.schemas import Chunk, Fragmento, ParametrosBusqueda, ResultadoBusqueda
from app.rag.texto import normalizar, terminos

logger = logging.getLogger("rag.recuperador")


Metodo = Literal["hibrido", "vector", "bm25"]


class IndiceNoListo(RuntimeError):
    """Se intentó buscar antes de inicializar el índice."""


class RecuperadorHibrido:
    def __init__(
        self,
        almacen: AlmacenVectorial,
        embeddings: Embeddings,
        *,
        candidatos: int = 20,
        rrf_k: int = 60,
        umbral_similitud: float = 0.30,
    ):
        self.almacen = almacen
        self.embeddings = embeddings
        self.candidatos = candidatos
        self.rrf_k = rrf_k
        self.umbral_similitud = umbral_similitud
        self._chunks: dict[str, Chunk] = {}
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    @property
    def listo(self) -> bool:
        return self._bm25 is not None

    @property
    def cantidad_chunks(self) -> int:
        return len(self._ids)

    @property
    def cantidad_noticias(self) -> int:
        return len({c.doc_id for c in self._chunks.values()})

    async def inicializar(self) -> None:
        """Arma el índice BM25 con los chunks que están en Chroma (la base
        vectorial es la única fuente de verdad)."""
        chunks = await self.almacen.todos()
        if not chunks:
            raise IndiceNoListo("La base vectorial está vacía: hay que correr la ingesta.")
        chunks.sort(key=lambda c: c.id)
        self._chunks = {c.id: c for c in chunks}
        self._ids = [c.id for c in chunks]
        self._bm25 = BM25Okapi([terminos(c.texto + " " + " ".join(c.etiquetas)) for c in chunks])
        logger.info("Índice híbrido listo: %d chunks de %d noticias.", self.cantidad_chunks, self.cantidad_noticias)

    async def buscar(self, params: ParametrosBusqueda, metodo: Metodo = "hibrido") -> ResultadoBusqueda:
        """`metodo` permite medir cada búsqueda por separado (scripts/evaluar_rag.py);
        el sistema siempre usa "hibrido"."""
        if self._bm25 is None:
            raise IndiceNoListo("El índice no está inicializado.")

        filtro = FiltroMetadatos(
            fecha_desde_num=int(params.fecha_desde.strftime("%Y%m%d")) if params.fecha_desde else None,
            fecha_hasta_num=int(params.fecha_hasta.strftime("%Y%m%d")) if params.fecha_hasta else None,
            fuentes_norm=tuple(normalizar(f) for f in params.fuentes),
        )
        etiquetas = {normalizar(e) for e in params.etiquetas}

        def pasa_filtros(chunk: Chunk) -> bool:
            if not filtro.acepta(chunk):
                return False
            return not etiquetas or bool(etiquetas & {normalizar(e) for e in chunk.etiquetas})

        # 1. Semántica (en la base vectorial, con filtros de metadatos)
        embedding = await self.embeddings.aembed_query(params.consulta)
        vectoriales = [
            c
            for c in await self.almacen.buscar(embedding, self.candidatos, filtro)
            if c.chunk_id in self._chunks and pasa_filtros(self._chunks[c.chunk_id])
        ]
        similitud = {c.chunk_id: c.similitud for c in vectoriales}

        # 2. Palabras clave (BM25 en memoria, mismos filtros)
        puntajes = self._bm25.get_scores(terminos(params.consulta))
        bm25 = {
            cid: float(p)
            for cid, p in zip(self._ids, puntajes, strict=True)
            if p > 0 and pasa_filtros(self._chunks[cid])
        }
        ranking_bm25 = sorted(bm25, key=bm25.get, reverse=True)[: self.candidatos]

        # Los candidatos que solo encontró BM25 necesitan su similitud
        # semántica para pasar (o no) el umbral.
        faltantes = [cid for cid in ranking_bm25 if cid not in similitud]
        similitud.update(await self.almacen.similitudes(embedding, faltantes))

        # 3. Fusión RRF
        rrf: dict[str, float] = {}
        if metodo in ("hibrido", "vector"):
            for posicion, c in enumerate(vectoriales):
                rrf[c.chunk_id] = rrf.get(c.chunk_id, 0.0) + 1.0 / (self.rrf_k + posicion + 1)
        if metodo in ("hibrido", "bm25"):
            for posicion, cid in enumerate(ranking_bm25):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.rrf_k + posicion + 1)

        # 4. Umbral de relevancia
        relevantes = {cid: s for cid, s in rrf.items() if similitud.get(cid, -1.0) >= self.umbral_similitud}
        descartados = len(rrf) - len(relevantes)

        # 5. Agrupado por noticia
        por_noticia: dict[str, list[str]] = {}
        for cid in sorted(relevantes, key=relevantes.get, reverse=True):
            por_noticia.setdefault(self._chunks[cid].doc_id, []).append(cid)

        fragmentos: list[Fragmento] = []
        for doc_id, chunk_ids in list(por_noticia.items())[: params.k]:
            ordenados = sorted(chunk_ids, key=lambda cid: self._chunks[cid].orden)
            primero = self._chunks[ordenados[0]]
            fragmentos.append(
                Fragmento(
                    doc_id=doc_id,
                    titular=primero.titular,
                    fuente=primero.fuente,
                    fecha=primero.fecha,
                    etiquetas=primero.etiquetas,
                    texto="\n\n".join(_sin_titular(self._chunks[cid]) for cid in ordenados),
                    chunk_ids=ordenados,
                    similitud=round(max(similitud[cid] for cid in chunk_ids), 4),
                    score_bm25=round(max(bm25.get(cid, 0.0) for cid in chunk_ids), 4),
                    score_rrf=round(max(relevantes[cid] for cid in chunk_ids), 6),
                )
            )

        return ResultadoBusqueda(
            parametros=params,
            fragmentos=fragmentos,
            candidatos_evaluados=len(rrf),
            descartados_por_umbral=descartados,
            umbral_similitud=self.umbral_similitud,
        )


def _sin_titular(chunk: Chunk) -> str:
    """El texto indexado lleva el titular adelante; en el fragmento se saca
    para no repetirlo (el titular ya va en su propio campo)."""
    prefijo = f"{chunk.titular}\n\n"
    return chunk.texto[len(prefijo) :] if chunk.texto.startswith(prefijo) else chunk.texto
