"""
scripts/demo_escenarios.py
--------------------------
Las 5 pruebas de punta a punta del sistema ("al menos 5 pruebas", punto 5
de la consigna). Cada una ejercita una capacidad distinta y deja su traza
en LangSmith:

  1. Informe positivo        -> RAG + agentes + publicación SIN pausa.
  2. Informe negativo + HITL -> se pausa y se APRUEBA por la API.
  3. HITL rechazado          -> se pausa y se RECHAZA: no se publica nada.
  4. Filtros de metadatos    -> solo noticias desde una fecha.
  5. Tema fuera de la base   -> no alucina: responde "sin información".

Sigue cada job EN VIVO por el endpoint de streaming (SSE) e imprime las
decisiones del Supervisor a medida que ocurren.

Uso (con la API levantada):
    python scripts/demo_escenarios.py
    python scripts/demo_escenarios.py --url http://localhost:8000 --solo 2
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field

import httpx

ESTADOS_FINALES = {"DONE", "REJECTED", "FAILED"}


@dataclass
class Escenario:
    numero: int
    nombre: str
    solicitud: dict
    decision_humana: bool | None  # True = aprobar, False = rechazar, None = no debería pedirse
    esperado: str
    verificaciones: list = field(default_factory=list)


def _verificar_publicado(job):
    return job["resultado"]["publicado"], "publicó el reporte"


def _verificar_no_publicado(job):
    return not job["resultado"]["publicado"], "NO publicó el reporte"


def _verificar_fuentes_con_citas(job):
    informe = job["resultado"]["informe"]
    return bool(informe["fuentes"]) and all(h["citas"] for h in informe["hallazgos"]), "cada hallazgo tiene citas"


def _verificar_fechas(desde: str):
    def verificar(job):
        fuentes = job["resultado"]["informe"]["fuentes"]
        return bool(fuentes) and all(f["fecha"] >= desde for f in fuentes), f"todas las fuentes son desde {desde}"

    return verificar


def _verificar_sin_informacion(job):
    informe = job["resultado"]["informe"]
    return not informe["hay_informacion_suficiente"] and not informe["fuentes"], "informa que no hay información"


ESCENARIOS = [
    Escenario(
        1,
        "Informe positivo (sin pausa)",
        {"tema": "avances en chips y hardware eficiente para IA"},
        None,
        "DONE",
        [_verificar_publicado, _verificar_fuentes_con_citas],
    ),
    Escenario(
        2,
        "Informe negativo con aprobación humana",
        {"tema": "incidentes de seguridad y fallas de agentes de IA"},
        True,
        "DONE",
        [_verificar_publicado, _verificar_fuentes_con_citas],
    ),
    Escenario(
        3,
        "Aprobación humana rechazada",
        {"tema": "violaciones a la privacidad y uso indebido de datos personales"},
        False,
        "REJECTED",
        [_verificar_no_publicado],
    ),
    Escenario(
        4,
        "Filtros de metadatos (fecha)",
        {"tema": "lanzamientos de nuevos modelos de lenguaje", "fecha_desde": "2026-05-01"},
        None,
        "DONE",
        [_verificar_fechas("2026-05-01"), _verificar_fuentes_con_citas],
    ),
    Escenario(
        5,
        "Tema fuera de la base (anti-alucinación)",
        {"tema": "resultados del campeonato de fútbol"},
        None,
        "DONE",
        [_verificar_sin_informacion, _verificar_no_publicado],
    ),
]


async def seguir_eventos(cliente: httpx.AsyncClient, job_id: str, escenario: Escenario) -> None:
    """Imprime los eventos del job en vivo y responde la aprobación humana."""
    async with cliente.stream("GET", f"/tasks/{job_id}/events", timeout=None) as respuesta:
        tipo = None
        async for linea in respuesta.aiter_lines():
            if linea.startswith("event:"):
                tipo = linea.split(":", 1)[1].strip()
            elif linea.startswith("data:"):
                datos = json.loads(linea.split(":", 1)[1])
                if tipo == "nodo" and datos.get("nodo") == "Supervisor":
                    print(f"     🧭 {datos['razon']}")
                elif tipo == "nodo" and datos.get("nodo") == "Investigador":
                    print(f"     🔎 Investigador: '{datos['consulta']}' -> {datos['noticias']}")
                elif tipo == "nodo" and datos.get("nodo") == "Redactor":
                    hallazgos, descartadas = datos["hallazgos"], datos["citas_descartadas"]
                    print(f"     ✍️  Redactor: {hallazgos} hallazgos, {descartadas} citas descartadas")
                elif tipo == "aprobacion_requerida":
                    print(f"     ⏸  Pide aprobación: {datos['motivo']}")
                    if escenario.decision_humana is None:
                        print("     ⚠️  No se esperaba una pausa en este escenario: se aprueba para continuar.")
                    aprobar = escenario.decision_humana is not False
                    r = await cliente.post(
                        f"/tasks/{job_id}/approve",
                        json={"aprobado": aprobar, "comentario": "Decisión de la demo", "revisor": "demo"},
                    )
                    print(f"     {'✅ Aprobado' if aprobar else '⛔ Rechazado'} (HTTP {r.status_code})")
                elif tipo == "estado" and datos.get("status") in ESTADOS_FINALES:
                    return


async def correr(cliente: httpx.AsyncClient, escenario: Escenario) -> dict:
    print(f"\n[{escenario.numero}] {escenario.nombre}")
    print(f"     Solicitud: {escenario.solicitud}")
    inicio = time.perf_counter()
    creado = (await cliente.post("/tasks", json=escenario.solicitud)).raise_for_status().json()
    await asyncio.wait_for(seguir_eventos(cliente, creado["job_id"], escenario), timeout=600)
    job = (await cliente.get(f"/tasks/{creado['job_id']}")).json()
    duracion = time.perf_counter() - inicio

    controles = [(job["status"] == escenario.esperado, f"estado final {escenario.esperado}")]
    if job["status"] != "FAILED":
        controles += [v(job) for v in escenario.verificaciones]
    for ok, texto in controles:
        print(f"     {'✅' if ok else '❌'} {texto}")
    if job.get("error"):
        print(f"     ❌ Error: {job['error']}")
    informe = (job.get("resultado") or {}).get("informe") or {}
    if informe.get("titulo"):
        print(f"     📄 {informe['titulo']} | sentimiento: {informe['sentimiento_general']}")
    return {
        "escenario": escenario.numero,
        "nombre": escenario.nombre,
        "job_id": creado["job_id"],
        "estado": job["status"],
        "ok": all(ok for ok, _ in controles),
        "segundos": round(duracion, 2),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Corre los 5 escenarios de punta a punta.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--solo", type=int, choices=range(1, 6), help="Correr un solo escenario.")
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=30) as cliente:
        try:
            salud = (await cliente.get("/health")).json()
        except httpx.HTTPError as exc:
            print(f"No se pudo conectar con la API en {args.url}: {exc}")
            return 1
        print(f"API: {salud}")
        elegidos = [e for e in ESCENARIOS if args.solo in (None, e.numero)]
        resultados = [await correr(cliente, e) for e in elegidos]

    print("\n" + "=" * 78)
    for r in resultados:
        print(
            f"{'✅' if r['ok'] else '❌'} [{r['escenario']}] {r['nombre']:<44} {r['estado']:<9} {r['segundos']:>6.1f} s"
        )
    print("=" * 78)
    print("Las trazas de cada job están en LangSmith (buscá por metadata.job_id o por el tag 'entrega-final').")
    return 0 if all(r["ok"] for r in resultados) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
