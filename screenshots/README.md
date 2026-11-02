# Capturas de LangSmith (evidencia de trazabilidad)

Proyecto `ai-news-intelligence` en <https://smith.langchain.com>, corrida de
`demo_escenarios.py` (19:39) y de `carga_concurrente.py` (19:40) del 1/11/2026.

| Archivo | Qué muestra |
|---|---|
| `01_lista_trazas.jpg` | Pestaña **Traces**: las ejecuciones (`sistema-inteligencia.ejecucion`) y reanudaciones tras la aprobación humana (`sistema-inteligencia.reanudacion`), con latencia, tokens, costo y los tags `entrega-final` / etapa. |
| `02_waterfall_informe.jpg` | **Waterfall** del escenario 1: Supervisor → Investigador (`investigador.planificar_busqueda` + `buscar_noticias`) → Analista → Redactor → `supervisor.evaluar_informe`, y la **segunda vuelta** que pidió el Supervisor (refinamiento). 21 s, 8.657K tokens, US$0.0019. |
| `03_herramienta_rag.jpg` | El span de la herramienta `buscar_noticias`: la **entrada** que eligió el LLM (consulta con sinónimos, `k=5`, sin filtros) y la **salida** (29 candidatos evaluados, 5 noticias con `score_bm25` y `score_rrf`, umbral usado). |
| `04_hitl_thread.jpg` | El escenario 2 como **Thread de 2 turnos**: el turno 1 termina en `AprobacionHumana` (pausa) y el turno 2 reanuda con `{"aprobado": true, "revisor": "demo"}` → `AprobacionHumana` → `Publicador`. La reanudación tarda 0.02 s y no gasta tokens. |
| `05_sin_informacion.jpg` | El escenario 5 (fútbol): dos búsquedas sin resultados, y después el Analista y el Redactor en 0.00 s, **sin ninguna llamada al LLM**. Salida `sin_datos` / `suficiente: false`. 1.0K tokens, US$0.0002. |
| `06_costo_y_latencia.jpg` | Pestaña **Threads**: un hilo por job, con turnos, latencia **P50/P99**, tokens y costo. Arriba, los 5 jobs de la carga concurrente (19:40:38). |
| `07_tokens_por_llamada.jpg` | Metadata de la llamada `ChatOpenAI` del Redactor: `usage_metadata` (1166 tokens de entrada, 319 de salida), modelo `gpt-4o-mini`, `temperature 0`, y el `job_id` / `thread_id` que lo vincula al job de la API. |
