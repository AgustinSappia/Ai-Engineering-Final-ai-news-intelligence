"""Agentes especialistas: herramientas validadas, filtros obligatorios y defensas anti-alucinación."""

from datetime import date

from app.agents.analista import analizar, sentimiento_desde_score
from app.agents.investigador import crear_herramienta_busqueda, investigar
from app.agents.redactor import redactar
from app.rag.schemas import ParametrosBusqueda, ResultadoBusqueda
from app.state import ResultadoAnalisis, ResultadoInvestigacion, SolicitudInforme
from tests.fakes import ModeloGuionado


async def _investigacion(recuperador, tema: str) -> ResultadoInvestigacion:
    herramienta = crear_herramienta_busqueda(recuperador)
    return await investigar(ModeloGuionado(), herramienta, SolicitudInforme(tema=tema), intento=1, k=5)


# ---------------------------------------------------------------------------
# Investigador
# ---------------------------------------------------------------------------
async def test_investigador_usa_el_rag_como_herramienta(recuperador):
    llm = ModeloGuionado()
    herramienta = crear_herramienta_busqueda(recuperador)
    resultado = await investigar(
        llm, herramienta, SolicitudInforme(tema="incidentes de seguridad con agentes"), intento=1, k=5
    )
    assert resultado.encontro_resultados
    assert resultado.args_del_llm_validos
    assert llm.llamadas == ["herramienta"]  # UNA sola llamada al LLM (sin texto de cierre descartable)


def test_la_herramienta_expone_el_esquema_pydantic(recuperador):
    herramienta = crear_herramienta_busqueda(recuperador)
    assert herramienta.name == "buscar_noticias"
    assert herramienta.args_schema is ParametrosBusqueda
    assert "consulta" in herramienta.tool_call_schema.model_json_schema()["properties"]


async def test_los_filtros_del_usuario_son_obligatorios(recuperador):
    """Aunque el LLM proponga otra fecha, manda lo que pidió el usuario."""
    llm = ModeloGuionado(args_extra={"fecha_desde": "2025-01-01", "fuentes": ["Otra Fuente"]})
    solicitud = SolicitudInforme(tema="modelos de lenguaje nuevos", fecha_desde=date(2026, 5, 1))
    resultado = await investigar(llm, crear_herramienta_busqueda(recuperador), solicitud, intento=1, k=5)
    assert resultado.busqueda.parametros.fecha_desde == date(2026, 5, 1)
    assert resultado.busqueda.parametros.fuentes == []
    assert all(f.fecha >= date(2026, 5, 1) for f in resultado.busqueda.fragmentos)


async def test_el_llm_no_puede_ampliar_el_k_del_sistema(recuperador):
    """Caso real: al refinar, el LLM pidió k=10 y el prompt del Analista se duplicó."""
    llm = ModeloGuionado(args_extra={"k": 10})
    resultado = await investigar(
        llm, crear_herramienta_busqueda(recuperador), SolicitudInforme(tema="agentes de IA"), intento=2, k=5
    )
    assert resultado.args_del_llm_validos is True
    assert resultado.busqueda.parametros.k == 5
    assert len(resultado.busqueda.fragmentos) <= 5


async def test_argumentos_invalidos_del_llm_usan_el_plan_b(recuperador):
    llm = ModeloGuionado(args_extra={"consulta": "x"})  # menos de 2 caracteres
    resultado = await investigar(
        llm, crear_herramienta_busqueda(recuperador), SolicitudInforme(tema="agentes"), intento=1, k=5
    )
    assert resultado.args_del_llm_validos is False
    assert resultado.busqueda.parametros.k == 5
    assert resultado.busqueda.parametros.consulta == "agentes"


async def test_si_el_llm_no_llama_a_la_herramienta_igual_se_busca(recuperador):
    llm = ModeloGuionado(sin_tool_call=True)
    resultado = await investigar(
        llm, crear_herramienta_busqueda(recuperador), SolicitudInforme(tema="agentes"), intento=1, k=5
    )
    assert resultado.args_del_llm_validos is False
    assert resultado.busqueda.parametros.consulta == "agentes"


async def test_en_el_refinamiento_se_quitan_las_etiquetas(recuperador):
    llm = ModeloGuionado(args_extra={"etiquetas": ["robótica"]})
    resultado = await investigar(
        llm,
        crear_herramienta_busqueda(recuperador),
        SolicitudInforme(tema="agentes"),
        intento=2,
        k=5,
        motivo_refinamiento="muy pocas fuentes",
    )
    assert resultado.busqueda.parametros.etiquetas == []
    assert resultado.intento == 2


# ---------------------------------------------------------------------------
# Analista
# ---------------------------------------------------------------------------
def test_reglas_de_sentimiento():
    assert sentimiento_desde_score(0.1) == "negativo"
    assert sentimiento_desde_score(0.5) == "neutral"
    assert sentimiento_desde_score(0.9) == "positivo"


async def test_analista_promedia_y_clasifica(recuperador):
    investigacion = await _investigacion(recuperador, "incidentes de seguridad con agentes")
    analisis = await analizar(ModeloGuionado(), investigacion)
    assert analisis.suficiente
    assert {a.doc_id for a in analisis.detalle} == {f.doc_id for f in investigacion.busqueda.fragmentos}
    assert analisis.sentimiento_general == "negativo"


async def test_analista_sin_noticias_no_llama_al_llm():
    vacia = ResultadoInvestigacion(
        intento=1, busqueda=ResultadoBusqueda(parametros=ParametrosBusqueda(consulta="x y"), umbral_similitud=0.3)
    )
    llm = ModeloGuionado()
    analisis = await analizar(llm, vacia)
    assert analisis.sentimiento_general == "sin_datos" and not analisis.suficiente
    assert llm.llamadas == []


# ---------------------------------------------------------------------------
# Redactor
# ---------------------------------------------------------------------------
async def _analisis(investigacion) -> ResultadoAnalisis:
    return await analizar(ModeloGuionado(), investigacion)


async def test_redactor_cita_solo_noticias_recuperadas(recuperador):
    investigacion = await _investigacion(recuperador, "incidentes de seguridad con agentes")
    informe = await redactar(
        ModeloGuionado(), SolicitudInforme(tema="incidentes"), investigacion, await _analisis(investigacion)
    )
    recuperadas = {f.doc_id for f in investigacion.busqueda.fragmentos}
    assert informe.hay_informacion_suficiente
    assert all(set(h.citas) <= recuperadas for h in informe.hallazgos)
    assert {c.doc_id for c in informe.fuentes} <= recuperadas
    assert informe.citas_descartadas == 0


async def test_filtro_de_veracidad_descarta_citas_inventadas(recuperador):
    investigacion = await _investigacion(recuperador, "incidentes de seguridad con agentes")
    llm = ModeloGuionado(citar_inexistente=True)
    informe = await redactar(llm, SolicitudInforme(tema="incidentes"), investigacion, await _analisis(investigacion))
    assert informe.citas_descartadas == 1
    assert informe.hallazgos_descartados == 1
    assert all("N999" not in h.citas for h in informe.hallazgos)


async def test_si_ninguna_cita_es_valida_no_hay_informe(recuperador):
    investigacion = await _investigacion(recuperador, "incidentes de seguridad con agentes")
    llm = ModeloGuionado(solo_citas_inexistentes=True)
    informe = await redactar(llm, SolicitudInforme(tema="incidentes"), investigacion, await _analisis(investigacion))
    assert informe.hay_informacion_suficiente is False
    assert informe.hallazgos == []
    assert informe.citas_descartadas == 2


async def test_redactor_sin_noticias_no_llama_al_llm():
    vacia = ResultadoInvestigacion(
        intento=2, busqueda=ResultadoBusqueda(parametros=ParametrosBusqueda(consulta="x y"), umbral_similitud=0.3)
    )
    llm = ModeloGuionado()
    analisis = await analizar(llm, vacia)
    informe = await redactar(llm, SolicitudInforme(tema="fútbol"), vacia, analisis)
    assert not informe.hay_informacion_suficiente
    assert "no contiene noticias relevantes" in informe.resumen
    assert llm.llamadas == []
