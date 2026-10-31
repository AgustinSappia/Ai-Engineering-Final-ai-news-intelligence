"""
scripts/evaluar_rag.py
----------------------
Mide la CALIDAD DE RECUPERACIÓN del RAG con un conjunto de preguntas de
referencia (cuáles noticias son relevantes para cada una), y compara:

    solo vectorial  vs.  solo BM25  vs.  híbrido (RRF)

Métricas (sobre las primeras k noticias):
- Hit@k: ¿apareció al menos una noticia relevante?
- Recall@k: qué fracción de las noticias relevantes apareció.
- MRR: 1 / posición de la primera noticia relevante (1 = salió primera).

Además calibra el umbral de relevancia: compara la similitud máxima que
obtienen las preguntas que SÍ están en la base con la de preguntas que NO
lo están, y sugiere un valor para RAG_UMBRAL_SIMILITUD.

Uso (con Docker Compose levantado; usa los embeddings reales de OpenAI):
    docker compose exec api python -m scripts.evaluar_rag
Sin API key (embeddings falsos, solo para probar el script):
    python -m scripts.evaluar_rag --offline
"""

import argparse
import asyncio
import statistics

from app.config import obtener_settings
from app.rag.recuperador import RecuperadorHibrido
from app.rag.schemas import ParametrosBusqueda

K = 5

# Pregunta -> noticias relevantes (definidas a mano a partir del corpus).
REFERENCIA: dict[str, set[str]] = {
    "fallas e incidentes de agentes autónomos": {"N05", "N06", "N19"},
    "violaciones a la privacidad y uso de datos personales": {"N11"},
    "consumo de energía de los centros de datos": {"N08", "N22"},
    "inteligencia artificial en medicina y salud": {"N09", "N23"},
    "IA en la educación y las escuelas": {"N10", "N25"},
    "despidos y pérdida de empleos por la automatización": {"N16", "N30"},
    "nuevos modelos de lenguaje de NeuraSur": {"N01", "N29"},
    "deepfakes y desinformación": {"N13", "N26"},
    "IA para el agro y la predicción del clima": {"N14", "N15"},
    "inversiones en empresas de robótica": {"N12"},
    "regulación y leyes sobre inteligencia artificial": {"N03", "N18", "N27"},
    "modelos abiertos y pequeños en español": {"N02", "N24"},
}
FUERA_DE_LA_BASE = [
    "resultados del campeonato de fútbol",
    "recetas de cocina tradicionales",
    "cotización del dólar y la bolsa",
]


def metricas(rankings: dict[str, list[str]]) -> tuple[float, float, float]:
    hits, recalls, rr = [], [], []
    for pregunta, relevantes in REFERENCIA.items():
        ranking = rankings[pregunta][:K]
        hits.append(1.0 if relevantes & set(ranking) else 0.0)
        recalls.append(len(relevantes & set(ranking)) / len(relevantes))
        posicion = next((i + 1 for i, d in enumerate(ranking) if d in relevantes), None)
        rr.append(1.0 / posicion if posicion else 0.0)
    return statistics.mean(hits), statistics.mean(recalls), statistics.mean(rr)


async def evaluar(recuperador: RecuperadorHibrido, umbral_actual: float) -> None:
    print(f"\nCalidad de recuperación ({len(REFERENCIA)} preguntas de referencia, k={K})\n")
    print(f"{'método':<16}{'Hit@5':>8}{'Recall@5':>10}{'MRR':>8}")
    for metodo in ("vector", "bm25", "hibrido"):
        rankings = {
            p: [
                f.doc_id
                for f in (await recuperador.buscar(ParametrosBusqueda(consulta=p, k=K), metodo=metodo)).fragmentos
            ]
            for p in REFERENCIA
        }
        hit, recall, mrr = metricas(rankings)
        print(f"{metodo:<16}{hit:>8.2f}{recall:>10.2f}{mrr:>8.2f}")

    async def similitud_maxima(pregunta: str) -> float:
        r = await recuperador.buscar(ParametrosBusqueda(consulta=pregunta, k=1), metodo="vector")
        return r.fragmentos[0].similitud if r.fragmentos else -1.0

    dentro = [await similitud_maxima(p) for p in REFERENCIA]
    fuera = [await similitud_maxima(p) for p in FUERA_DE_LA_BASE]
    print("\nCalibración del umbral de relevancia (similitud coseno máxima por pregunta)")
    print(f"  preguntas EN la base:    mín {min(dentro):.3f} | prom {statistics.mean(dentro):.3f}")
    print(f"  preguntas FUERA de base: máx {max(fuera):.3f} | prom {statistics.mean(fuera):.3f}")
    if min(dentro) > max(fuera):
        sugerido = round((min(dentro) + max(fuera)) / 2, 2)
        print(f"  -> Separación limpia. Umbral sugerido: {sugerido} (actual RAG_UMBRAL_SIMILITUD={umbral_actual})")
    else:
        print("  -> Las distribuciones se superponen: revisar el umbral y las preguntas.")


async def main(offline: bool) -> None:
    settings = obtener_settings()
    if offline:
        from app.rag.almacen import AlmacenEnMemoria
        from app.rag.ingesta import asegurar_indice
        from tests.fakes import EmbeddingsHash

        embeddings, almacen = EmbeddingsHash(), AlmacenEnMemoria()
        await asegurar_indice(almacen, embeddings, settings)
        print("Modo offline: embeddings falsos (los números no representan al sistema real).")
    else:
        from app.llm import crear_embeddings
        from app.rag.almacen import AlmacenChroma

        embeddings = crear_embeddings(settings)
        almacen = await AlmacenChroma.conectar(settings.chroma_host, settings.chroma_port, settings.chroma_coleccion)

    # Umbral -1 para medir el ranking puro (sin descartar nada).
    recuperador = RecuperadorHibrido(
        almacen, embeddings, candidatos=settings.rag_candidatos, rrf_k=settings.rag_rrf_k, umbral_similitud=-1.0
    )
    await recuperador.inicializar()
    await evaluar(recuperador, settings.rag_umbral_similitud)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa la calidad de recuperación del RAG.")
    parser.add_argument("--offline", action="store_true", help="Usar embeddings falsos (sin API key).")
    asyncio.run(main(parser.parse_args().offline))
