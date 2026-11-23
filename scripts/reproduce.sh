#!/usr/bin/env bash
# One command to reproduce everything:   bash scripts/reproduce.sh
#
# * With the shipped LLM cache (results/llm_cache.jsonl.gz) this replays the exact original LLM responses, so no
#   model/server is needed and the numbers come out identical.
# * Without the cache it needs a running local model: bash scripts/setup_ollama.sh (numbers may then differ slightly).
# * The TEST split is guarded: if results/test_eval.json already exists the script stops, because re-evaluating test
#   would break "test used once". To reproduce from scratch, delete results/ and results_control_shuffled/ deliberately.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
PY=.venv/bin/python
if [ -f results/test_eval.json ]; then
  echo "results/test_eval.json exists (test split already used). To re-run from scratch:"
  echo "  rm -rf results results_control_shuffled   # then re-run this script"
  echo "Regenerating the report only:"; $PY -m src.report; exit 0
fi
mkdir -p cache
[ -f cache/llm_cache.jsonl ] || { [ -f results/llm_cache.jsonl.gz ] && gunzip -c results/llm_cache.jsonl.gz > cache/llm_cache.jsonl || true; }
$PY -m pytest -q                                                   # sandbox + pipeline tests
$PY -m src.evaluate meta                                           # outer loop (train -> propose, dev -> accept/reject)
$PY -m src.evaluate final                                          # TEST split: baseline + best, once each
$PY -m src.evaluate meta --control shuffled --out results_control_shuffled   # control: shuffled failures
$PY -m src.evaluate final --out results_control_shuffled --baseline-from results/test_eval.json
$PY -m src.report                                                  # summary.json, tables, plots, README table
