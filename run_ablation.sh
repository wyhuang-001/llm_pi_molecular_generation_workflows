#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Usage: ./run_ablation.sh

Runs the mapped-pocket-coordinate AI Cloud ablation for budgets 0..5, then a final
budget-06 verification run with unlimited tool calls and a strict site-evidence gate.
Override with TASK_PATH, CONFIG_PATH, OUTPUT_ROOT, COORDINATE_SCOPE, POCKET_RADIUS,
BUDGETS, FINAL_BUDGET, ABLATION_MODEL, ABLATION_BASE_URL, or CODEX_CONFIG_DIR environment variables.
HELP
  exit 0
fi

TASK_PATH="${TASK_PATH:-input/task.json}"
CONFIG_PATH="${CONFIG_PATH:-config.aicloud.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/ablation-aicloud-pocket-6A-mapped-02}"
COORDINATE_SCOPE="${COORDINATE_SCOPE:-pocket}"
POCKET_RADIUS="${POCKET_RADIUS:-6.0}"
BUDGETS="${BUDGETS:-0 1 2 3 4 5}"
FINAL_BUDGET="${FINAL_BUDGET:-6}"
ABLATION_MODEL="${ABLATION_MODEL:-gpt-5.6-sol}"
ABLATION_BASE_URL="${ABLATION_BASE_URL:-https://api.p1-103n1x.com/v1}"
CODEX_CONFIG_DIR="${CODEX_CONFIG_DIR:-$HOME/.codex}"

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

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  [[ -f "$CODEX_CONFIG_DIR/auth.json" ]] || {
    echo "error: Codex auth file not found: $CODEX_CONFIG_DIR/auth.json" >&2
    exit 1
  }
  OPENAI_API_KEY="$(python3 - "$CODEX_CONFIG_DIR/auth.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
key = data.get("OPENAI_API_KEY", "")
if not key:
    raise SystemExit("OPENAI_API_KEY is missing from Codex auth.json")
print(key)
PY
  )"
  export OPENAI_API_KEY
fi

read -r -a BUDGET_ARGS <<< "$BUDGETS"

TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/molecular-agent-ablation-config.XXXXXX.json")"
cleanup() {
  rm -f "$TEMP_CONFIG"
}
trap cleanup EXIT

python3 - "$CONFIG_PATH" "$TEMP_CONFIG" "$ABLATION_MODEL" "$ABLATION_BASE_URL" "$CODEX_CONFIG_DIR" <<'PY'
import json
import sys

source, target, model, base_url, codex_config_dir = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    config = json.load(handle)
config.update({
    "model": model,
    "base_url": base_url.rstrip("/"),
    "codex_config_dir": codex_config_dir,
    "api_key_env": "OPENAI_API_KEY",
})
with open(target, "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY

printf '%s\n' "Running independent Codex LLM tool-budget ablation"
printf '  model:            %s\n' "$ABLATION_MODEL"
printf '  base URL:         %s\n' "${ABLATION_BASE_URL:-from config}"
printf '  task:             %s\n' "$TASK_PATH"
printf '  config source:    %s\n' "$CONFIG_PATH"
printf '  output:           %s\n' "$OUTPUT_ROOT"
printf '  coordinate scope: %s\n' "$COORDINATE_SCOPE"
printf '  pocket radius:    %s A\n' "$POCKET_RADIUS"
printf '  budgets:          %s\n' "$BUDGETS"

mamba run -n molecular-agent \
  python scripts/compare_tool_budgets.py \
  --task "$TASK_PATH" \
  --config "$TEMP_CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --budgets "${BUDGET_ARGS[@]}" \
  --coordinate-scope "$COORDINATE_SCOPE" \
  --pocket-radius "$POCKET_RADIUS" \
  --unbounded-budget "$FINAL_BUDGET"

printf '\nResults written to: %s\n' "$SCRIPT_DIR/$OUTPUT_ROOT"
