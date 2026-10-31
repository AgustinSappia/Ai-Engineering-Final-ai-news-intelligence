"""
tests/fakes.py
--------------
Dobles de prueba: permiten correr TODO el sistema sin API keys.

- `EmbeddingsHash`: embeddings deterministas por "bolsa de palabras"
  hasheada. No entienden sinónimos como un modelo real, pero dos textos que
  comparten términos se parecen, que es lo que necesitan los tests del RAG.
- `ModeloGuionado`: un chat model que se comporta como un LLM obediente:
  mira qué le piden (qué herramienta o qué esquema de salida estructurada)
  y responde algo válido a partir del contenido del prompt. No guarda
  estado entre llamadas, así que sirve para jobs concurrentes.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from app.rag.texto import terminos


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class EmbeddingsHash(Embeddings):
    def __init__(self, dimensiones: int = 512):
        self.dimensiones = dimensiones
        self.llamadas_documentos = 0
        self.llamadas_consulta = 0

    def _vector(self, texto: str) -> list[float]:
        v = np.zeros(self.dimensiones)
        for t in terminos(texto):
            h = int(hashlib.md5(t.encode()).hexdigest(), 16)
            v[h % self.dimensiones] += 1.0 if (h >> 8) % 2 else -1.0
        norma = np.linalg.norm(v)
        return (v / norma).tolist() if norma else v.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.llamadas_documentos += 1
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        self.llamadas_consulta += 1
        return self.embed_query(text)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
_ID_NOTICIA = re.compile(r"\[(N\d{2,4})\]")


def _texto(mensajes: list[BaseMessage]) -> str:
    return "\n".join(m.content if isinstance(m.content, str) else json.dumps(m.content) for m in mensajes)


class ModeloGuionado(BaseChatModel):
    """LLM falso "bien portado".

    - Con herramientas (`bind_tools`): devuelve una llamada a la primera
      herramienta, con la consulta tomada de la línea `TEMA: ...` del prompt.
    - Con salida estructurada (`with_structured_output`): construye un objeto
      válido del esquema pedido a partir de lo que hay en el prompt.

    `decision_supervisor` fija qué responde el Supervisor cuando se le
    consulta; `citar_inexistente` hace que el Redactor invente una cita
    (para probar el filtro anti-alucinación); `sentimiento` fuerza el
    sentimiento que devuelve el Analista.
    """

    herramientas: list[str] = []
    decision_supervisor: str = "FINALIZAR"
    citar_inexistente: bool = False
    solo_citas_inexistentes: bool = False
    sentimiento: str | None = None
    args_extra: dict[str, Any] = {}  # p. ej. {"k": 50} para simular argumentos inválidos
    sin_tool_call: bool = False  # el LLM "se olvida" de llamar a la herramienta
    llamadas: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "modelo-guionado"

    # -- herramientas ---------------------------------------------------------
    def bind_tools(self, tools: list[Any], **kwargs: Any):
        nombres = [getattr(t, "name", None) or getattr(t, "__name__", str(t)) for t in tools]
        return self.model_copy(update={"herramientas": nombres})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = _texto(messages)
        self.llamadas.append("herramienta" if self.herramientas else "texto")
        if self.herramientas and self.sin_tool_call:
            mensaje = AIMessage(content="Prefiero no buscar nada.")
        elif self.herramientas:
            tema = re.search(r"TEMA:\s*(.+)", prompt)
            args: dict[str, Any] = {"consulta": tema.group(1).strip() if tema else prompt[-100:]}
            filtros = re.search(r"FILTROS:\s*(\{.*\})", prompt)
            if filtros:
                args.update({k: v for k, v in json.loads(filtros.group(1)).items() if v})
            args.update(self.args_extra)
            mensaje = AIMessage(
                content="",
                tool_calls=[{"name": self.herramientas[0], "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}],
            )
        else:
            mensaje = AIMessage(content="Respuesta de texto.")
        return ChatResult(generations=[ChatGeneration(message=mensaje)])

    # -- salida estructurada ---------------------------------------------------
    def with_structured_output(self, schema, **kwargs):
        def responder(entrada):
            mensajes = entrada.to_messages() if hasattr(entrada, "to_messages") else entrada
            prompt = _texto(mensajes) if isinstance(mensajes, list) else str(mensajes)
            self.llamadas.append(schema.__name__)
            return self._estructurado(schema, prompt)

        return RunnableLambda(responder, name=f"ModeloGuionado[{schema.__name__}]")

    def _estructurado(self, schema, prompt: str):
        nombre = schema.__name__
        ids = list(dict.fromkeys(_ID_NOTICIA.findall(prompt)))

        if nombre == "DecisionSupervisor":
            return schema(next=self.decision_supervisor, razon="Decisión guionada para el test.")

        if nombre == "AnalisisLLM":
            negativo = re.compile(r"preocup|advierten|fall|despid|multa|incorrect|alarma|denunci|vulnerab")
            noticias = []
            for doc_id in ids:
                bloque = prompt.split(f"[{doc_id}]", 1)[1][:1500]
                sentimiento = self.sentimiento or ("negativo" if negativo.search(bloque.lower()) else "positivo")
                score = {"negativo": 0.15, "neutral": 0.5, "positivo": 0.85}[sentimiento]
                noticias.append(
                    {"doc_id": doc_id, "sentimiento": sentimiento, "score": score, "justificacion": "Guionado."}
                )
            return schema.model_validate({"noticias": noticias})

        if nombre == "InformeLLM":
            citas = ["N998", "N999"] if self.solo_citas_inexistentes else (ids[:2] or ["N01"])
            hallazgos = [{"texto": f"Hallazgo basado en la noticia {i}.", "citas": [i]} for i in citas]
            if self.citar_inexistente:
                hallazgos.append({"texto": "Un dato que el modelo inventó sin fuente.", "citas": ["N999"]})
            return schema.model_validate(
                {
                    "titulo": "Informe de prueba",
                    "resumen": "Resumen generado por el modelo guionado a partir de las noticias recuperadas.",
                    "hallazgos": hallazgos,
                    "conclusion": "Conclusión guionada.",
                }
            )

        raise ValueError(f"ModeloGuionado no sabe responder el esquema {nombre}")
