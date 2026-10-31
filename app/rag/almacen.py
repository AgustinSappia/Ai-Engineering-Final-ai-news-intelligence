"""
app/rag/almacen.py
------------------
La base vectorial, detrás de una interfaz común (`AlmacenVectorial`).

- `AlmacenChroma`: producción. Usa `chromadb.AsyncHttpClient`, el cliente
  ASÍNCRONO oficial de Chroma, contra un servidor de Chroma (un contenedor
  de Docker Compose). Ninguna operación bloquea el event loop.
- `AlmacenEnMemoria`: misma interfaz, en memoria (búsqueda por fuerza
  bruta). La usan los tests para correr sin ningún servidor.

Los filtros por metadatos se expresan con `FiltroMetadatos` y cada
almacén los traduce a su propio lenguaje (en Chroma, la cláusula `where`).
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from app.rag.schemas import Chunk
from app.rag.texto import normalizar


@dataclass(frozen=True)
class FiltroMetadatos:
    fecha_desde_num: int | None = None  # AAAAMMDD
    fecha_hasta_num: int | None = None
    fuentes_norm: tuple[str, ...] = field(default_factory=tuple)

    def acepta(self, chunk: Chunk) -> bool:
        if self.fecha_desde_num is not None and chunk.fecha_num < self.fecha_desde_num:
            return False
        if self.fecha_hasta_num is not None and chunk.fecha_num > self.fecha_hasta_num:
            return False
        if self.fuentes_norm and normalizar(chunk.fuente) not in self.fuentes_norm:
            return False
        return True

    def a_where_chroma(self) -> dict | None:
        condiciones: list[dict] = []
        if self.fecha_desde_num is not None:
            condiciones.append({"fecha_num": {"$gte": self.fecha_desde_num}})
        if self.fecha_hasta_num is not None:
            condiciones.append({"fecha_num": {"$lte": self.fecha_hasta_num}})
        if self.fuentes_norm:
            condiciones.append({"fuente_norm": {"$in": list(self.fuentes_norm)}})
        if not condiciones:
            return None
        return condiciones[0] if len(condiciones) == 1 else {"$and": condiciones}


@dataclass(frozen=True)
class Coincidencia:
    chunk_id: str
    similitud: float  # coseno, entre -1 y 1


class AlmacenVectorial(Protocol):
    async def huella(self) -> str | None: ...
    async def reiniciar(self, huella: str) -> None: ...
    async def agregar(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    async def contar(self) -> int: ...
    async def todos(self) -> list[Chunk]: ...
    async def buscar(self, embedding: list[float], n: int, filtro: FiltroMetadatos) -> list[Coincidencia]: ...
    async def similitudes(self, embedding: list[float], chunk_ids: list[str]) -> dict[str, float]: ...
    async def ping(self) -> bool: ...


def _coseno(a: np.ndarray, b: np.ndarray) -> float:
    denominador = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominador) if denominador else 0.0


# ---------------------------------------------------------------------------
# Chroma (producción)
# ---------------------------------------------------------------------------
class AlmacenChroma:
    """Colección de Chroma con distancia coseno. La huella del corpus se
    guarda en los metadatos de la colección para saber si hay que
    re-indexar."""

    def __init__(self, cliente, nombre_coleccion: str):
        self._cliente = cliente  # chromadb.AsyncHttpClient (ya conectado)
        self._nombre = nombre_coleccion
        self._coleccion = None

    @classmethod
    async def conectar(cls, host: str, port: int, nombre_coleccion: str) -> "AlmacenChroma":
        import chromadb

        cliente = await chromadb.AsyncHttpClient(host=host, port=port)
        return cls(cliente, nombre_coleccion)

    async def _col(self):
        if self._coleccion is None:
            self._coleccion = await self._cliente.get_or_create_collection(
                self._nombre, metadata={"hnsw:space": "cosine"}
            )
        return self._coleccion

    async def ping(self) -> bool:
        try:
            await self._cliente.heartbeat()
            return True
        except Exception:
            return False

    async def huella(self) -> str | None:
        col = await self._col()
        return (col.metadata or {}).get("huella_corpus")

    async def reiniciar(self, huella: str) -> None:
        try:
            await self._cliente.delete_collection(self._nombre)
        except Exception:
            pass
        self._coleccion = await self._cliente.create_collection(
            self._nombre, metadata={"hnsw:space": "cosine", "huella_corpus": huella}
        )

    async def agregar(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        col = await self._col()
        await col.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.texto for c in chunks],
            metadatas=[c.metadatos() for c in chunks],
        )

    async def contar(self) -> int:
        return await (await self._col()).count()

    async def todos(self) -> list[Chunk]:
        col = await self._col()
        datos = await col.get(include=["documents", "metadatas"])
        return [
            Chunk.desde_metadatos(cid, doc, meta)
            for cid, doc, meta in zip(datos["ids"], datos["documents"], datos["metadatas"], strict=True)
        ]

    async def buscar(self, embedding: list[float], n: int, filtro: FiltroMetadatos) -> list[Coincidencia]:
        col = await self._col()
        total = await col.count()
        if total == 0:
            return []
        resultado = await col.query(
            query_embeddings=[embedding],
            n_results=min(n, total),
            where=filtro.a_where_chroma(),
            include=["distances"],
        )
        # Con distancia coseno, Chroma devuelve 1 - similitud.
        return [
            Coincidencia(chunk_id=cid, similitud=1.0 - float(dist))
            for cid, dist in zip(resultado["ids"][0], resultado["distances"][0], strict=True)
        ]

    async def similitudes(self, embedding: list[float], chunk_ids: list[str]) -> dict[str, float]:
        if not chunk_ids:
            return {}
        col = await self._col()
        datos = await col.get(ids=chunk_ids, include=["embeddings"])
        consulta = np.asarray(embedding, dtype=float)
        return {
            cid: _coseno(consulta, np.asarray(emb, dtype=float))
            for cid, emb in zip(datos["ids"], datos["embeddings"], strict=True)
        }


# ---------------------------------------------------------------------------
# En memoria (tests)
# ---------------------------------------------------------------------------
class AlmacenEnMemoria:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectores: dict[str, np.ndarray] = {}
        self._huella: str | None = None
        self.operaciones_agregar = 0  # para verificar en los tests que no se re-indexa de más

    async def ping(self) -> bool:
        return True

    async def huella(self) -> str | None:
        return self._huella

    async def reiniciar(self, huella: str) -> None:
        self._chunks.clear()
        self._vectores.clear()
        self._huella = huella

    async def agregar(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.operaciones_agregar += 1
        for chunk, emb in zip(chunks, embeddings, strict=True):
            self._chunks[chunk.id] = chunk
            self._vectores[chunk.id] = np.asarray(emb, dtype=float)

    async def contar(self) -> int:
        return len(self._chunks)

    async def todos(self) -> list[Chunk]:
        return list(self._chunks.values())

    async def buscar(self, embedding: list[float], n: int, filtro: FiltroMetadatos) -> list[Coincidencia]:
        consulta = np.asarray(embedding, dtype=float)
        candidatos = [
            Coincidencia(cid, _coseno(consulta, self._vectores[cid]))
            for cid, chunk in self._chunks.items()
            if filtro.acepta(chunk)
        ]
        return sorted(candidatos, key=lambda c: c.similitud, reverse=True)[:n]

    async def similitudes(self, embedding: list[float], chunk_ids: list[str]) -> dict[str, float]:
        consulta = np.asarray(embedding, dtype=float)
        return {cid: _coseno(consulta, self._vectores[cid]) for cid in chunk_ids if cid in self._vectores}
