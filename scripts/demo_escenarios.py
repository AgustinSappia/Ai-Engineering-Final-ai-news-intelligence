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
decisiones del Supervisor a medida que ocurren. Si la API se reinicia en
medio de un job, se reconecta (Last-Event-ID) y el servidor retoma el job
desde su último checkpoint.

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


MAX_RECONEXIONES = 5


async def esperar_api(cliente: httpx.AsyncClient, limite_s: float = 180) -> None:
    """Espera a que la API vuelva a responder (p. ej. si el contenedor se reinició)."""
    limite = time.monotonic() + limite_s
    while time.monotonic() < limite:
        try:
            if (await cliente.get("/health", timeout=5)).status_code == 200:
                return
        except httpx.TransportError:
            pass
        await asyncio.sleep(2)
    raise RuntimeError(f"La API no volvió a responder en {limite_s:.0f} s. Mirá: docker compose logs --tail 80 api")


async def manejar_evento(cliente: httpx.AsyncClient, job_id: str, escenario: Escenario, tipo: str, datos: dict) -> bool:
    """Imprime un evento y responde la aprobación humana. True = el job terminó."""
    if tipo == "nodo" and datos.get("nodo") == "Supervisor":
        print(f"     🧭 {datos['razon']}")
    elif tipo == "nodo" and datos.get("nodo") == "Investigador":
        print(f"     🔎 Investigador: '{datos['consulta']}' -> {datos['noticias']}")
    elif tipo == "nodo" and datos.get("nodo") == "Analista" and datos.get("sentimiento_general"):
        print(f"     📊 Analista: sentimiento {datos['sentimiento_general']} (score {datos['score_promedio']})")
    elif tipo == "nodo" and datos.get("nodo") == "Redactor":
        hallazgos, descartadas = datos["hallazgos"], datos["citas_descartadas"]
        print(f"     ✍️  Redactor: {hallazgos} hallazgos, {descartadas} citas descartadas")
    elif tipo == "recuperado":
        print(f"     ♻️  La API se reinició: el job se retoma desde el checkpoint (en {datos['retoma_en']})")
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
        return True
    return False


async def seguir_eventos(cliente: httpx.AsyncClient, job_id: str, escenario: Escenario) -> None:
    """Sigue el job EN VIVO por SSE. Si la conexión se corta (la API se
    reinició, se cayó la red), espera a que vuelva y se reconecta mandando
    `Last-Event-ID`: el servidor retoma desde el último evento recibido,
    sin repetir ni perder ninguno (están guardados en un Redis Stream)."""
    ultimo_id: str | None = None
    for _ in range(MAX_RECONEXIONES + 1):
        cabeceras = {"Last-Event-ID": ultimo_id} if ultimo_id else {}
        try:
            async with cliente.stream("GET", f"/tasks/{job_id}/events", headers=cabeceras, timeout=None) as r:
                r.raise_for_status()
                tipo = evento_id = None
                async for linea in r.aiter_lines():
                    if linea.startswith("id:"):
                        evento_id = linea.split(":", 1)[1].strip()
                    elif linea.startswith("event:"):
                        tipo = linea.split(":", 1)[1].strip()
                    elif linea.startswith("data:"):
                        terminado = await manejar_evento(cliente, job_id, escenario, tipo, json.loads(linea[5:]))
                        ultimo_id = evento_id
                        if terminado:
                            return
            motivo = "el servidor cerró el stream antes del final"
        except httpx.TransportError as exc:
            motivo = type(exc).__name__
        print(f"     ⚠️  Se cortó la conexión con la API ({motivo}). Esperando a que vuelva para retomar...")
        await esperar_api(cliente)
    raise RuntimeError(f"Se perdió la conexión {MAX_RECONEXIONES + 1} veces seguidas con el job {job_id}")


async def obtener_job(cliente: httpx.AsyncClient, job_id: str) -> dict:
    for _ in range(MAX_RECONEXIONES):
        try:
            return (await cliente.get(f"/tasks/{job_id}")).raise_for_status().json()
        except httpx.TransportError:
            await esperar_api(cliente)
    return (await cliente.get(f"/tasks/{job_id}")).raise_for_status().json()


async def correr(cliente: httpx.AsyncClient, escenario: Escenario) -> dict:
    print(f"\n[{escenario.numero}] {escenario.nombre}")
    print(f"     Solicitud: {escenario.solicitud}")
    inicio = time.perf_counter()
    creado = (await cliente.post("/tasks", json=escenario.solicitud)).raise_for_status().json()
    print(f"     job_id: {creado['job_id']}")
    await asyncio.wait_for(seguir_eventos(cliente, creado["job_id"], escenario), timeout=900)
    job = await obtener_job(cliente, creado["job_id"])
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
