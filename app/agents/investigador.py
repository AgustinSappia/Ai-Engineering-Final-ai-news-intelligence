"""
app/agents/investigador.py
--------------------------
Agente Investigador: el que usa el RAG como HERRAMIENTA.

1. El LLM recibe el tema (y, si es un reintento, por qué falló la búsqueda
   anterior) y decide los argumentos de su única herramienta,
   `buscar_noticias`: reformula la consulta con sinónimos y términos clave,
   y puede acotar por etiquetas.
2. Los argumentos se validan con Pydantic (`ParametrosBusqueda`). Si el LLM
   propone algo inválido (un `k` de 50, fechas al revés), no se rompe el
   job: se usa un plan B con el tema tal cual y se deja registrado.
3. Los filtros que pidió el USUARIO (fechas, fuentes) se imponen siempre:
   el LLM no puede "olvidarlos" ni ampliarlos.
4. Se ejecuta la herramienta (búsqueda híbrida async) y se devuelve el
   resultado tipado.

Mejora tomada de las trazas de la Pre-entrega 7: allá cada especialista
hacía una segunda llamada al LLM solo para redactar un texto de cierre que
el código descartaba (44 % de la latencia). Acá el LLM se usa UNA vez,
forzado a llamar a la herramienta (`tool_choice`), y el resultado sale
directo de la herramienta.
"""

import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import ValidationError

from app.rag.recuperador import RecuperadorHibrido
from app.rag.schemas import ParametrosBusqueda, ResultadoBusqueda
from app.state import ResultadoInvestigacion, SolicitudInforme

logger = logging.getLogger("agentes.investigador")

NOMBRE_HERRAMIENTA = "buscar_noticias"

PROMPT_INVESTIGADOR = """Sos el agente Investigador de un sistema de inteligencia sobre noticias de IA.
Tu ÚNICA tarea es decidir cómo buscar en la base de conocimiento usando la herramienta `buscar_noticias`.

Reglas:
- Convertí el TEMA en una consulta de búsqueda rica en español: incluí sinónimos y términos clave
  (por ejemplo, para "despidos por IA": "despidos, pérdida de empleos, automatización, trabajadores").
- No inventes filtros de fecha ni de fuente: los filtros pedidos por el usuario se aplican solos.
- Solo usá `etiquetas` si el tema corresponde claramente a una categoría.
- No respondas con texto: llamá a la herramienta."""


def crear_herramienta_busqueda(recuperador: RecuperadorHibrido) -> BaseTool:
    """La herramienta que ve el LLM. `args_schema` es el modelo Pydantic:
    LangChain se lo describe al LLM como esquema JSON y valida lo que vuelve."""

    async def buscar_noticias(**kwargs) -> dict:
        params = ParametrosBusqueda.model_validate(kwargs)
        resultado = await recuperador.buscar(params)
        return resultado.model_dump(mode="json")

    return StructuredTool.from_function(
        coroutine=buscar_noticias,
        name=NOMBRE_HERRAMIENTA,
        description=(
            "Busca noticias de IA en la base de conocimiento con búsqueda híbrida (semántica + palabras clave). "
            "Devuelve solo las noticias relevantes; puede devolver una lista vacía si no hay información."
        ),
        args_schema=ParametrosBusqueda,
    )


def _mensajes(solicitud: SolicitudInforme, intento: int, motivo_refinamiento: str | None) -> list:
    filtros = {
        "fecha_desde": solicitud.fecha_desde.isoformat() if solicitud.fecha_desde else None,
        "fecha_hasta": solicitud.fecha_hasta.isoformat() if solicitud.fecha_hasta else None,
        "fuentes": solicitud.fuentes,
    }
    contenido = f"TEMA: {solicitud.tema}\nFILTROS: {json.dumps(filtros, ensure_ascii=False)}\nINTENTO: {intento}"
    if intento > 1:
        contenido += (
            f"\nLa búsqueda anterior no alcanzó. Motivo: {motivo_refinamiento or 'sin resultados relevantes'}.\n"
            "Reformulá la consulta: usá términos más generales y sinónimos, y no restrinjas por etiquetas."
        )
    return [SystemMessage(content=PROMPT_INVESTIGADOR), HumanMessage(content=contenido)]


def _imponer_filtros_del_usuario(args: dict, solicitud: SolicitudInforme, intento: int) -> dict:
    args = dict(args)
    args["fecha_desde"] = solicitud.fecha_desde.isoformat() if solicitud.fecha_desde else None
    args["fecha_hasta"] = solicitud.fecha_hasta.isoformat() if solicitud.fecha_hasta else None
    args["fuentes"] = list(solicitud.fuentes)
    if intento > 1:
        args["etiquetas"] = []  # en el refinamiento se amplía
    return args


async def investigar(
    llm: BaseChatModel,
    herramienta: BaseTool,
    solicitud: SolicitudInforme,
    *,
    intento: int,
    k: int,
    motivo_refinamiento: str | None = None,
) -> ResultadoInvestigacion:
    llm_con_herramienta = llm.bind_tools([herramienta], tool_choice=NOMBRE_HERRAMIENTA)
    respuesta = await llm_con_herramienta.ainvoke(
        _mensajes(solicitud, intento, motivo_refinamiento),
        config={"run_name": "investigador.planificar_busqueda"},
    )
    args_llm = respuesta.tool_calls[0]["args"] if respuesta.tool_calls else {}

    args = _imponer_filtros_del_usuario({"k": k, **args_llm}, solicitud, intento)
    try:
        params = ParametrosBusqueda.model_validate(args)
        validos = bool(args_llm)
    except ValidationError as exc:
        logger.warning("Argumentos inválidos del LLM (%s); se usa la búsqueda por defecto.", exc.errors()[:2])
        params = ParametrosBusqueda.model_validate(
            _imponer_filtros_del_usuario({"consulta": solicitud.tema, "k": k}, solicitud, intento)
        )
        validos = False

    salida = await herramienta.ainvoke(params.model_dump(mode="json"), config={"run_name": NOMBRE_HERRAMIENTA})
    return ResultadoInvestigacion(
        intento=intento,
        busqueda=ResultadoBusqueda.model_validate(salida),
        args_del_llm_validos=validos,
    )
