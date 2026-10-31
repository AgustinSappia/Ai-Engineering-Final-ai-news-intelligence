# Capturas de LangSmith (evidencia de trazabilidad)

Todas salen del proyecto `ai-news-intelligence` en <https://smith.langchain.com>
(**Tracing Projects**). Conviene filtrar por la hora de la corrida o por el tag
`entrega-final`.

| Archivo | Qué capturar | Dónde |
|---|---|---|
| `01_lista_trazas.jpg` | La lista de trazas de las 5 pruebas (`sistema-inteligencia.ejecucion` y `sistema-inteligencia.reanudacion`), con las columnas **Latency, Tokens y Cost** | Pestaña **Traces** |
| `02_waterfall_informe.jpg` | El árbol de una ejecución completa: Supervisor → Investigador (`investigador.planificar_busqueda` + `buscar_noticias`) → Analista → Redactor → `supervisor.evaluar_informe` → AprobacionHumana → Publicador | Clic en una traza → botón **Waterfall** |
| `03_herramienta_rag.jpg` | El span `buscar_noticias` abierto: la **entrada** (consulta y filtros que eligió el LLM) y la **salida** (noticias con similitud, BM25 y RRF) | Clic en el span `buscar_noticias` |
| `04_hitl_thread.jpg` | Un job con aprobación humana como **Thread** de 2 turnos: el turno 1 termina en `AprobacionHumana` (Interrupted / GraphInterrupt) y el turno 2 reanuda hasta `Publicador` | Pestaña **Threads** |
| `05_sin_informacion.jpg` | El escenario 5 (tema fuera de la base): dos búsquedas sin resultados y **ninguna** llamada del Redactor al LLM | Traza del tema "resultados del campeonato de fútbol" |
| `06_costo_y_latencia.jpg` | La carga concurrente: costo por ejecución y latencias (P50/P99 en la vista Threads, o la columna Latency para calcular el p95) | Pestaña **Threads** o **Traces** |
| `07_tokens_por_llamada.jpg` | El detalle de tokens (entrada/salida) de las llamadas a `ChatOpenAI` de cada agente | Clic en cada span `ChatOpenAI` |

Para sacar una captura en Windows: `Win + Shift + S`.
