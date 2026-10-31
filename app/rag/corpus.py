"""
app/rag/corpus.py
-----------------
Carga y troceo ("chunking") del corpus de noticias.

Formato de cada archivo `data/corpus/NXX.md`:

    ---
    id: N07
    titular: ...
    fuente: ...
    fecha: 2026-03-04
    etiquetas: hardware, chips
    ---
    Cuerpo de la noticia (párrafos separados por una línea en blanco).

El chunking es por TOKENS (como en la Pre-entrega 3) y respeta los
párrafos: se juntan párrafos completos hasta llenar el presupuesto de
tokens, y solo un párrafo que por sí solo lo supera se corta en ventanas
con solapamiento. Cada chunk lleva el titular adelante, para que el
fragmento siga teniendo contexto cuando se recupera solo.

Sobre `tiktoken` (el tokenizador de OpenAI): la PRIMERA vez que se usa,
descarga su vocabulario de internet con `requests` (una llamada SÍNCRONA).
Por eso `cargar_tokenizador()` se ejecuta en otro thread y, en la imagen
de Docker, el vocabulario se descarga durante el build (ver Dockerfile).
"""

import hashlib
import logging
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path

from app.rag.schemas import Chunk, NoticiaCorpus

logger = logging.getLogger("rag.corpus")

ContadorTokens = Callable[[str], int]
Codificador = tuple[Callable[[str], list[int]], Callable[[list[int]], str]]

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


# ---------------------------------------------------------------------------
# Lectura del corpus (I/O de disco: se llama con asyncio.to_thread)
# ---------------------------------------------------------------------------
def leer_noticia(ruta: Path) -> NoticiaCorpus:
    contenido = ruta.read_text(encoding="utf-8")
    coincidencia = _FRONTMATTER.match(contenido)
    if not coincidencia:
        raise ValueError(f"{ruta.name}: falta el bloque de metadatos entre '---'")

    meta: dict[str, str] = {}
    for linea in coincidencia.group(1).splitlines():
        if ":" in linea:
            clave, valor = linea.split(":", 1)
            meta[clave.strip()] = valor.strip()

    return NoticiaCorpus(
        doc_id=meta.get("id", ""),
        titular=meta.get("titular", ""),
        fuente=meta.get("fuente", ""),
        fecha=date.fromisoformat(meta.get("fecha", "")),
        etiquetas=[e.strip() for e in meta.get("etiquetas", "").split(",") if e.strip()],
        cuerpo=coincidencia.group(2).strip(),
    )


def cargar_corpus(directorio: Path) -> list[NoticiaCorpus]:
    rutas = sorted(directorio.glob("*.md"))
    if not rutas:
        raise FileNotFoundError(f"No hay noticias (*.md) en {directorio}")
    noticias = [leer_noticia(r) for r in rutas]
    ids = [n.doc_id for n in noticias]
    if len(ids) != len(set(ids)):
        raise ValueError("Hay noticias con el mismo id en el corpus")
    return noticias


def huella_corpus(noticias: list[NoticiaCorpus], extra: str = "") -> str:
    """Hash del contenido del corpus (más la configuración de indexado en
    `extra`). Si nada cambió, la ingesta no vuelve a generar embeddings."""
    h = hashlib.sha256(extra.encode())
    for n in sorted(noticias, key=lambda n: n.doc_id):
        h.update(n.model_dump_json().encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Tokenizador
# ---------------------------------------------------------------------------
def cargar_tokenizador(nombre: str = "cl100k_base") -> Codificador:
    """Devuelve (codificar, decodificar). Puede hacer I/O de red la primera
    vez: llamar con `asyncio.to_thread`. Si no hay forma de obtener el
    vocabulario (sin red y sin caché), usa una aproximación por palabras y
    lo avisa en el log."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding(nombre)
        return enc.encode, enc.decode
    except Exception as exc:
        logger.warning("No se pudo cargar tiktoken (%s): se usa una aproximación por palabras.", exc)
        return _codificador_aproximado()


def _codificador_aproximado() -> Codificador:
    """Aproximación: cada palabra cuenta como un "token". Solo se usa si
    tiktoken no está disponible."""

    def codificar(texto: str) -> list[int]:
        return list(range(len(texto.split())))

    def decodificar(_tokens: list[int]) -> str:  # pragma: no cover - no se usa con este codificador
        raise NotImplementedError

    return codificar, decodificar


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def dividir_noticia(
    noticia: NoticiaCorpus, codificador: Codificador, max_tokens: int, solapamiento: int
) -> list[Chunk]:
    codificar, _ = codificador
    encabezado = f"{noticia.titular}\n\n"
    presupuesto = max(20, max_tokens - len(codificar(encabezado)))

    parrafos = [p.strip() for p in noticia.cuerpo.split("\n\n") if p.strip()]
    piezas: list[str] = []
    actual: list[str] = []
    tokens_actual = 0

    for parrafo in parrafos:
        tokens = len(codificar(parrafo))
        if tokens > presupuesto:
            # Párrafo demasiado largo: se cierra lo acumulado y se corta en ventanas.
            if actual:
                piezas.append("\n\n".join(actual))
                actual, tokens_actual = [], 0
            piezas.extend(_ventanas(parrafo, codificador, presupuesto, solapamiento))
            continue
        if tokens_actual + tokens > presupuesto and actual:
            piezas.append("\n\n".join(actual))
            actual, tokens_actual = [], 0
        actual.append(parrafo)
        tokens_actual += tokens

    if actual:
        piezas.append("\n\n".join(actual))

    return [
        Chunk(
            id=f"{noticia.doc_id}#{i}",
            doc_id=noticia.doc_id,
            orden=i,
            texto=encabezado + pieza,
            titular=noticia.titular,
            fuente=noticia.fuente,
            fecha=noticia.fecha,
            etiquetas=noticia.etiquetas,
        )
        for i, pieza in enumerate(piezas)
    ]


def _ventanas(texto: str, codificador: Codificador, tamano: int, solapamiento: int) -> list[str]:
    codificar, decodificar = codificador
    tokens = codificar(texto)
    try:
        decodificar([])
    except NotImplementedError:
        # Codificador aproximado (por palabras): las ventanas se arman con palabras.
        palabras = texto.split()
        paso = max(1, tamano - solapamiento)
        return [" ".join(palabras[i : i + tamano]) for i in range(0, len(palabras), paso)]

    paso = max(1, tamano - solapamiento)
    return [decodificar(tokens[i : i + tamano]) for i in range(0, len(tokens), paso)]
