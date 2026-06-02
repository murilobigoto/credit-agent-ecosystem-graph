#!/usr/bin/env bash
# Setup local: cria um virtualenv e instala o projeto em modo editável (com dev).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

echo ">> Criando virtualenv em .venv"
"$PYTHON" -m venv .venv

echo ">> Ativando e instalando (modo editável + extras de dev)"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

if [ ! -f .env ]; then
  cp .env.example .env
  echo ">> .env criado a partir de .env.example (USE_FAKE_LLM=1 por padrão)"
fi

echo ""
echo "Pronto. Ative o ambiente e rode a demo:"
echo "  source .venv/bin/activate"
echo "  python -m credito_agentes.cli --demo"
