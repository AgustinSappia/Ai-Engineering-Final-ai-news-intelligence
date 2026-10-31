"""
scripts/carga_concurrente.py
----------------------------
Prueba de carga: lanza N peticiones CONCURRENTES (5 por defecto), sigue cada
job por polling, aprueba automáticamente las que piden aprobación humana e
imprime latencias (p50/p95 vistas desde el cliente). El costo por ejecución
y la latencia p95 "oficiales" se leen en LangSmith.

Uso (con la API levantada):
    python scripts/carga_concurrente.py
    python scripts/carga_concurrente.py --url http://localhost:8000 --no-aprobar
"""

import argparse
import asyncio
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

TEMAS_POR_DEFECTO = [
    "avances en chips y hardware eficiente para IA",
    "incidentes de seguridad y fallas de agentes de IA",
    "inversión en robótica y logística",
    "IA aplicada a la salud y el diagnóstico médico",
    "despidos y pérdida de empleos por automatización",
]
ESTADOS_FINALES = {"DONE", "REJECTED", "FAILED"}


def percentil(valores: list[float], p: float) -> float:
    """Método nearest-rank. Con 5 muestras, el p95 coincide con el máximo."""
    if not valores:
        return float("nan")
    ordenados = sorted(valores)
    return ordenados[max(0, math.ceil(p / 100 * len(ordenados)) - 1)]


async def lanzar(cliente: httpx.AsyncClient, tema: str) -> dict:
    t0 = time.perf_counter()
    respuesta = (await cliente.post("/tasks", json={"tema": tema})).raise_for_status()
    return {
        "tema": tema,
        "job_id": respuesta.json()["job_id"],
        "post_ms": round((time.perf_counter() - t0) * 1000, 1),
        "_t0": t0,
    }


async def seguir(cliente: httpx.AsyncClient, item: dict, aprobar: bool, intervalo: float, limite_s: float) -> dict:
    while True:
        job = (await cliente.get(f"/tasks/{item['job_id']}")).json()
        if job["status"] == "AWAITING_APPROVAL" and "hitl" not in item:
            item["hitl"] = True
            print(f"  ⏸  [{item['tema'][:40]}] pide aprobación: {job['aprobacion_pendiente']['motivo'][:70]}")
            if not aprobar:
                item["estado"] = job["status"]
                return item
            await cliente.post(
                f"/tasks/{item['job_id']}/approve",
                json={"aprobado": True, "comentario": "Aprobado por la prueba de carga", "revisor": "prueba-de-carga"},
            )
        if job["status"] in ESTADOS_FINALES:
            resultado = job.get("resultado") or {}
            informe = resultado.get("informe") or {}
            item.update(
                estado=job["status"],
                total_s=round(time.perf_counter() - item["_t0"], 2),
                servidor_s=job.get("duracion_segundos"),
                sentimiento=informe.get("sentimiento_general"),
                fuentes=len(informe.get("fuentes", [])),
                publicado=resultado.get("publicado", False),
                error=job.get("error"),
            )
            return item
        if time.perf_counter() - item["_t0"] > limite_s:
            item["estado"] = f"TIMEOUT ({job['status']})"
            return item
        await asyncio.sleep(intervalo)


def guardar_resultados(items: list[dict]) -> Path:
    """Escribe el JSON de la corrida (I/O de disco: se llama con asyncio.to_thread)."""
    carpeta = Path(__file__).resolve().parent.parent / "resultados"
    carpeta.mkdir(exist_ok=True)
    archivo = carpeta / f"carga_{datetime.now():%Y%m%d_%H%M%S}.json"
    limpios = [{k: v for k, v in it.items() if not k.startswith("_")} for it in items]
    archivo.write_text(json.dumps(limpios, ensure_ascii=False, indent=2), encoding="utf-8")
    return archivo


async def main() -> int:
    parser = argparse.ArgumentParser(description="N peticiones concurrentes contra la API.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--temas", nargs="+", default=TEMAS_POR_DEFECTO)
    parser.add_argument("--no-aprobar", action="store_true", help="No aprobar automáticamente los HITL.")
    parser.add_argument("--intervalo", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=30) as cliente:
        try:
            print(f"API: {(await cliente.get('/health')).json()}\n")
        except httpx.HTTPError as exc:
            print(f"No se pudo conectar con la API en {args.url}: {exc}")
            return 1

        print(f"Lanzando {len(args.temas)} peticiones CONCURRENTES...")
        inicio = time.perf_counter()
        items = await asyncio.gather(*[lanzar(cliente, t) for t in args.temas])
        for it in items:
            print(f"  → [{it['tema'][:40]}] job_id={it['job_id']} (POST respondió en {it['post_ms']} ms)")
        print("\nEsperando a que terminen...")
        items = await asyncio.gather(
            *[seguir(cliente, it, not args.no_aprobar, args.intervalo, args.timeout) for it in items]
        )
        total = time.perf_counter() - inicio

    print("\n" + "=" * 104)
    encabezado = ("tema", "estado", "HITL", "sentim.", "fuentes", "publicó", "POST ms", "total s")
    print("{:<44} {:<10} {:<5} {:<9} {:>7} {:<8} {:>8} {:>8}".format(*encabezado))
    print("-" * 104)
    for it in items:
        print(
            f"{it['tema'][:44]:<44} {it.get('estado', '?'):<10} {'sí' if it.get('hitl') else 'no':<5} "
            f"{it.get('sentimiento') or '-':<9} {it.get('fuentes', 0):>7} {'sí' if it.get('publicado') else 'no':<8} "
            f"{it['post_ms']:>8} {it.get('total_s', '-'):>8}"
        )
    print("=" * 104)
    posts = [it["post_ms"] for it in items]
    totales = [it["total_s"] for it in items if "total_s" in it]
    print(f"Tiempo total de la corrida: {total:.2f} s (la suma de las tareas sería {sum(totales):.1f} s)")
    print(f"Latencia de POST /tasks  -> máx {max(posts):.1f} ms | p95 {percentil(posts, 95):.1f} ms")
    if totales:
        p50, p95 = percentil(totales, 50), percentil(totales, 95)
        print(f"Latencia extremo a extremo (cliente) -> p50 {p50:.2f} s | p95 {p95:.2f} s")
    fallidos = [it for it in items if it.get("estado") == "FAILED"]
    for it in fallidos:
        print(f"  ❌ [{it['tema']}] {it.get('error')}")

    archivo = await asyncio.to_thread(guardar_resultados, items)
    print(f"\nResultados guardados en {archivo}")
    print("En LangSmith: filtrá por el tag 'entrega-final' para ver latencia p95 y costo por ejecución.")
    return 1 if fallidos else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
