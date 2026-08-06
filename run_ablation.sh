#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Usage: ./run_ablation.sh

Runs the independent tool-budget experiment for budgets 0..5.
Override with TASK_PATH, CONFIG_PATH, OUTPUT_ROOT, COORDINATE_SCOPE,
POCKET_RADIUS, or BUDGETS environment variables.
HELP
  exit 0
fi

TASK_PATH="${TASK_PATH:-input/task.json}"
CONFIG_PATH="${CONFIG_PATH:-config.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/ablation-tool-budget-full}"
COORDINATE_SCOPE="${COORDINATE_SCOPE:-full}"
POCKET_RADIUS="${POCKET_RADIUS:-8.0}"
BUDGETS="${BUDGETS:-0 1 2 3 4 5}"

if ! command -v mamba >/dev/null 2>&1; then
  echo "error: mamba was not found in PATH" >&2
  exit 1
fi

if [[ ! -f "$TASK_PATH" ]]; then
  echo "error: task file not found: $TASK_PATH" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "error: config file not found: $CONFIG_PATH" >&2
  echo "Create it from config.example.json or set CONFIG_PATH." >&2
  exit 1
fi

# Prefer an explicitly exported key. Otherwise reuse pi's local key only for this process.
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  AUTH_PATH="${PI_AUTH_PATH:-$HOME/.pi/agent/auth.json}"
  if [[ ! -f "$AUTH_PATH" ]]; then
    echo "error: OPENAI_API_KEY is unset and auth file was not found: $AUTH_PATH" >&2
    exit 1
  fi
  OPENAI_API_KEY="$(python3 - "$AUTH_PATH" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    auth = json.load(handle)
key = auth.get("openai", {}).get("key", "")
if not key:
    raise SystemExit("no openai.key found in pi auth file")
print(key)
PY
  )"
  export OPENAI_API_KEY
fi

read -r -a BUDGET_ARGS <<< "$BUDGETS"

printf '%s\n' "Running independent LLM tool-budget ablation"
printf '  task:             %s\n' "$TASK_PATH"
printf '  config:           %s\n' "$CONFIG_PATH"
printf '  output:           %s\n' "$OUTPUT_ROOT"
printf '  coordinate scope: %s\n' "$COORDINATE_SCOPE"
printf '  budgets:          %s\n' "$BUDGETS"

mamba run -n molecular-agent \
  python scripts/compare_tool_budgets.py \
  --task "$TASK_PATH" \
  --config "$CONFIG_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --budgets "${BUDGET_ARGS[@]}" \
  --coordinate-scope "$COORDINATE_SCOPE" \
  --pocket-radius "$POCKET_RADIUS"

printf '\nResults written to: %s\n' "$SCRIPT_DIR/$OUTPUT_ROOT"
