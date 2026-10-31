"""
app/rag/texto.py
----------------
Normalización de texto para la búsqueda por palabras clave (BM25).

"Regulación", "regulaciones" y "REGULACION" tienen que contar como el
mismo término. Para eso: minúsculas, sin tildes, sin palabras vacías
("de", "la", "que"...) y un recorte simple de terminaciones (un "stemmer"
liviano): "agentes" y "agente" quedan como "agent".
"""

import re
import unicodedata

PALABRAS_VACIAS = frozenset(
    """
    a al algo ante antes como con contra cual cuando de del desde donde durante e el ella ellas
    ellos en entre era es esa ese eso esta este esto estos estas fue han hasta hay la las le les
    lo los mas me mi mientras muy no nos o otra otro para pero por porque que quien se ser si sin
    sobre su sus tambien tiene tienen todo todos tras tu un una uno unos unas ya y
    noticia noticias acerca
    """.split()
)

_PALABRA = re.compile(r"[a-z0-9ñ]+")
_TERMINACION = re.compile(r"(es|s|a|o|e)$")


def normalizar(texto: str) -> str:
    """Minúsculas y sin tildes (la ñ se conserva)."""
    texto = texto.lower().replace("ñ", "\0")
    sin_tildes = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
    return sin_tildes.replace("\0", "ñ")


def raiz(palabra: str) -> str:
    return _TERMINACION.sub("", palabra) if len(palabra) > 4 else palabra


def terminos(texto: str) -> list[str]:
    """Texto -> lista de términos normalizados para BM25."""
    return [raiz(p) for p in _PALABRA.findall(normalizar(texto)) if p not in PALABRAS_VACIAS and len(p) > 1]
