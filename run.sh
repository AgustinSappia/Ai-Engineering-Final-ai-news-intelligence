#!/usr/bin/env bash
# Levanta todo el sistema con una sola instrucción:   ./run.sh
#   ./run.sh logs     -> sigue los logs de la API
#   ./run.sh down     -> apaga todo (los datos quedan en los volúmenes)
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-up}" in
  logs) exec docker compose logs -f api ;;
  down) exec docker compose down ;;
esac

command -v docker >/dev/null || { echo "❌ Falta Docker: https://docs.docker.com/get-docker/"; exit 1; }

if [ ! -f .env ]; then
  cp .env.example .env
  echo "⚠️  Se creó .env a partir de .env.example. Completá OPENAI_API_KEY (y LANGSMITH_API_KEY) y volvé a correr ./run.sh"
  exit 1
fi
if grep -qE '^OPENAI_API_KEY=(sk-\.\.\.)?$' .env; then
  echo "❌ OPENAI_API_KEY en .env todavía tiene el valor de ejemplo."
  exit 1
fi

echo "🚀 Construyendo y levantando Redis 8 + Chroma + API..."
docker compose up --build -d

echo -n "⏳ Esperando a la API (la primera vez indexa el corpus)"
for _ in $(seq 1 90); do
  if curl -fs http://localhost:8000/health >/dev/null 2>&1; then
    echo
    echo "✅ Sistema listo:"
    curl -s http://localhost:8000/health; echo
    echo
    echo "   Documentación interactiva: http://localhost:8000/docs"
    echo "   5 pruebas de punta a punta: python scripts/demo_escenarios.py"
    echo "   Carga concurrente:          python scripts/carga_concurrente.py"
    echo "   Logs:                       ./run.sh logs"
    exit 0
  fi
  echo -n "."
  sleep 2
done
echo
echo "❌ La API no respondió a tiempo. Últimos logs:"
docker compose logs --tail 40 api
exit 1
