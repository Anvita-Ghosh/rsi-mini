#!/usr/bin/env bash
# Installs Ollama (local LLM runtime) into ~/.local/ollama (user-level, no sudo),
# starts it on 127.0.0.1:11434 and pulls the model used by rsi-mini.
# Everything runs locally and is free; no API key is involved.
set -euo pipefail

VERSION="v0.40.17"
SHA256="4b9db20d077c5a1e52fb4a6f2e6791f76ae945fd07c9b1a5647c82e927826653"   # published digest of ollama-darwin.tgz
MODEL="${1:-qwen2.5-coder:7b}"
DIR="$HOME/.local/ollama"
mkdir -p "$DIR/models"
cd "$DIR"

if [ ! -x ./ollama ]; then
  if [ ! -f ollama-darwin.tgz ]; then
    curl -L -o ollama-darwin.tgz "https://github.com/ollama/ollama/releases/download/$VERSION/ollama-darwin.tgz"
  fi
  echo "$SHA256  ollama-darwin.tgz" | shasum -a 256 -c -
  tar -xzf ollama-darwin.tgz
fi
./ollama --version

if ! curl -s -m 2 http://127.0.0.1:11434/api/version >/dev/null; then
  OLLAMA_MODELS="$DIR/models" OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=4 \
    nohup ./ollama serve > serve.log 2>&1 &
  for _ in $(seq 1 30); do curl -s -m 2 http://127.0.0.1:11434/api/version >/dev/null && break; sleep 1; done
fi
OLLAMA_HOST=127.0.0.1:11434 ./ollama pull "$MODEL"
echo "Ready. Server: http://127.0.0.1:11434  model: $MODEL"
