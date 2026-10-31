# Imagen de la API. Python 3.12 (lo que pide la consigna).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /app

# 1) Dependencias primero: si solo cambia el código, Docker reutiliza esta capa.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) El vocabulario del tokenizador se descarga AHORA, en el build. tiktoken
#    lo baja con una llamada HTTP síncrona la primera vez que se usa: así
#    nunca ocurre dentro de la app (ni bloquea el event loop, ni depende de
#    tener internet al arrancar).
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# 3) Código, corpus y scripts.
COPY app ./app
COPY data ./data
COPY scripts ./scripts

# Usuario sin privilegios (buena práctica: el proceso no corre como root).
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app /opt/tiktoken
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://localhost:8000/health', timeout=4).status != 200)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
