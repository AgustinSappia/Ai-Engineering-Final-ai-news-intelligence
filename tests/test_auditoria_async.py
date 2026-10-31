"""
Auditoría estática de asincronía (uno de los "errores que hunden una entrega").

Una llamada síncrona dentro de un `async def` no rompe nada: solo bloquea
el event loop y degrada todo el servidor, por eso es difícil de ver. Este
test recorre el código de `app/` con el AST de Python y falla si encuentra:

- `import requests` en cualquier parte (hay que usar httpx o SDKs async);
- dentro de una función `async def`: `time.sleep`, `requests.*`,
  `urllib.request.*`, `open(...)`, `httpx.Client(...)`, `Path.read_text()`
  y similares (I/O síncrona);
- endpoints de FastAPI que no sean `async def`.
"""

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app"
ARCHIVOS = sorted(APP.rglob("*.py"))

LLAMADAS_PROHIBIDAS_EN_ASYNC = {
    "time.sleep",
    "requests.get",
    "requests.post",
    "requests.request",
    "urllib.request.urlopen",
    "httpx.Client",
    "httpx.get",
    "httpx.post",
    "open",
    "subprocess.run",
}
METODOS_IO_SINCRONICOS = {"read_text", "write_text", "read_bytes", "write_bytes"}


def _nombre(nodo: ast.AST) -> str:
    if isinstance(nodo, ast.Name):
        return nodo.id
    if isinstance(nodo, ast.Attribute):
        return f"{_nombre(nodo.value)}.{nodo.attr}"
    return ""


def _llamadas_directas(funcion: ast.AsyncFunctionDef):
    """Llamadas dentro del cuerpo, sin entrar en funciones anidadas
    síncronas (esas se ejecutan con asyncio.to_thread, por ejemplo)."""
    pendientes = list(funcion.body)
    while pendientes:
        nodo = pendientes.pop()
        if isinstance(nodo, (ast.FunctionDef, ast.Lambda)):
            continue
        if isinstance(nodo, ast.Call):
            yield nodo
        pendientes.extend(ast.iter_child_nodes(nodo))


@pytest.mark.parametrize("archivo", ARCHIVOS, ids=lambda p: str(p.relative_to(APP.parent)))
def test_sin_llamadas_sincronicas_en_async(archivo: Path):
    arbol = ast.parse(archivo.read_text(encoding="utf-8"))
    problemas = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.Import, ast.ImportFrom)):
            modulos = [a.name for a in nodo.names] if isinstance(nodo, ast.Import) else [nodo.module or ""]
            if any(m.split(".")[0] == "requests" for m in modulos):
                problemas.append(f"línea {nodo.lineno}: import de requests")
        if isinstance(nodo, ast.AsyncFunctionDef):
            for llamada in _llamadas_directas(nodo):
                nombre = _nombre(llamada.func)
                if nombre in LLAMADAS_PROHIBIDAS_EN_ASYNC or nombre.split(".")[-1] in METODOS_IO_SINCRONICOS:
                    problemas.append(f"línea {llamada.lineno}: {nombre}() dentro de async def {nodo.name}")
    assert not problemas, "\n".join(problemas)


def test_todos_los_endpoints_son_async():
    from app.main import crear_app

    app = crear_app(lifespan_fn=None)
    sincronicos = [
        r.path
        for r in app.routes
        if hasattr(r, "endpoint")
        and not r.path.startswith(("/openapi", "/docs", "/redoc"))
        and not __import__("inspect").iscoroutinefunction(r.endpoint)
    ]
    assert sincronicos == []


def test_la_auditoria_detecta_un_caso_real():
    """Control: el detector encuentra una llamada síncrona si existe."""
    codigo = "import time\nasync def f():\n    time.sleep(1)\n"
    funcion = ast.parse(codigo).body[1]
    assert [_nombre(c.func) for c in _llamadas_directas(funcion)] == ["time.sleep"]
