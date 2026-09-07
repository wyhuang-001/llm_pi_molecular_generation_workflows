#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR"

ENV_NAME="${DOCKING_ENV_NAME:-molecular-agent-docking}"
SOURCE_CONFIG="${CONFIG_PATH:-config.aicloud.json}"
TASK_PATH="${TASK_PATH:-input/task.json}"
RUN_ROOT="${RUN_ROOT:-runs/docking-loop-test}"
CONTEXT_ROUNDS="${CONTEXT_ROUNDS:-256}"
EDIT_ATTEMPTS="${EDIT_ATTEMPTS:-80}"
SCRIPTED_EDIT_ATTEMPTS="${SCRIPTED_EDIT_ATTEMPTS:-1}"
MODE="scripted"
RUN_TESTS=1
RESUME=0
LLM=""

usage() {
  cat <<'EOF'
Usage: ./run_docking_loop_test.sh [--scripted|--real] [--llm current|gpt-5.4-mini|gpt-5.6-luna|doubao|deepseek] [--resume] [--skip-tests]

Runs the molecular design loop through real GNINA docking while always keeping
RBFE disabled.

Modes:
  --scripted   Deterministic local decision client + real GNINA (default)
  --real       Real configured LLM + real GNINA
  --all        Run scripted first, then real LLM
  --llm        Select current, gpt-5.4-mini, gpt-5.6-luna, doubao, or deepseek; omitted preserves runtime/config selection
  --resume     Resume the selected real workflow from RUN_ROOT/<mode>
  --skip-tests Skip pytest before workflow execution

Environment overrides:
  DOCKING_ENV_NAME  Conda environment (default: molecular-agent-docking)
  CONFIG_PATH       Source model/docking config (default: config.aicloud.json)
  TASK_PATH         Task JSON (default: input/task.json)
  RUN_ROOT          Output root (default: runs/docking-loop-test)
  CONTEXT_ROUNDS    Runtime context-query budget (default: 256)
  EDIT_ATTEMPTS     Runtime maximum candidate attempts for real mode (default: 80)
  SCRIPTED_EDIT_ATTEMPTS  Maximum attempts for the single-candidate scripted smoke test (default: 1)
  CODEX_CONFIG_DIR  Local Codex configuration directory (default: ~/.codex)
  OPENAI_API_KEY    Optional current-provider key override; otherwise read from Codex auth.json
  ARK_API_KEY       Volcengine Ark key for the doubao profile
EOF
}

while (($#)); do
  case "$1" in
    --scripted) MODE="scripted" ;;
    --real) MODE="real" ;;
    --all) MODE="all" ;;
    --resume) RESUME=1 ;;
    --llm)
      if [[ "${2:-}" != "current" && "${2:-}" != "gpt-5.4-mini" && "${2:-}" != "gpt-5.6-luna" && "${2:-}" != "doubao" && "${2:-}" != "deepseek" ]]; then
        printf 'ERROR: --llm requires current, gpt-5.4-mini, gpt-5.6-luna, doubao, or deepseek\n' >&2
        exit 2
      fi
      LLM="$2"
      shift
      ;;
    --skip-tests) RUN_TESTS=0 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if (( RESUME )) && [[ "$MODE" != "real" ]]; then
  printf 'ERROR: --resume requires --real\n' >&2
  exit 2
fi

command -v mamba >/dev/null 2>&1 || {
  printf 'ERROR: mamba was not found in PATH\n' >&2
  exit 1
}
[[ -f "$SOURCE_CONFIG" ]] || { printf 'ERROR: config not found: %s\n' "$SOURCE_CONFIG" >&2; exit 1; }
[[ -f "$TASK_PATH" ]] || { printf 'ERROR: task not found: %s\n' "$TASK_PATH" >&2; exit 1; }

if [[ "$MODE" == "real" || "$MODE" == "all" ]]; then
  export CODEX_CONFIG_DIR="${CODEX_CONFIG_DIR:-$HOME/.codex}"
  # Credential validation belongs to the selected client. Independent providers
  # do not require Codex credentials, and current may use an environment key instead.
fi

mkdir -p "$RUN_ROOT"
RUNTIME_CONFIG="$RUN_ROOT/runtime-config.json"
RUNTIME_TASK="$RUN_ROOT/runtime-task.json"
SCRIPTED_RUNTIME_TASK="$RUN_ROOT/runtime-task-scripted.json"
if (( RESUME )); then
  [[ -f "$RUNTIME_CONFIG" && -f "$RUNTIME_TASK" ]] || {
    printf 'ERROR: runtime-config.json and runtime-task.json are required for resume in %s\n' "$RUN_ROOT" >&2
    exit 1
  }
else
mamba run -n "$ENV_NAME" python - "$SOURCE_CONFIG" "$RUNTIME_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
data = json.loads(source.read_text(encoding="utf-8"))
data.setdefault("docking", {})["enabled"] = True
data.setdefault("rbfe", {})["enabled"] = False
target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Runtime config: {target}")
PY
mamba run -n "$ENV_NAME" python - "$TASK_PATH" "$RUNTIME_TASK" "$CONTEXT_ROUNDS" "$EDIT_ATTEMPTS" <<'PY'
import json
import sys
from pathlib import Path

source, target, rounds, attempts = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
if rounds < 1:
    raise SystemExit("CONTEXT_ROUNDS must be a positive integer")
if attempts < 1:
    raise SystemExit("EDIT_ATTEMPTS must be a positive integer")
data = json.loads(source.read_text(encoding="utf-8"))
data["complex_path"] = str((source.parent / data["complex_path"]).resolve())
if data.get("fragment_library_path"):
    data["fragment_library_path"] = str((source.parent / data["fragment_library_path"]).resolve())
data["max_context_rounds"] = rounds
data.setdefault("docking_optimization", {})["hard_max_attempts"] = attempts
target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Runtime task: {target} (max_context_rounds={rounds}, hard_max_attempts={attempts})")
PY
mamba run -n "$ENV_NAME" python - "$RUNTIME_TASK" "$SCRIPTED_RUNTIME_TASK" "$SCRIPTED_EDIT_ATTEMPTS" <<'PY'
import json
import sys
from pathlib import Path

source, target, attempts = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
if attempts < 1:
    raise SystemExit("SCRIPTED_EDIT_ATTEMPTS must be a positive integer")
data = json.loads(source.read_text(encoding="utf-8"))
data.setdefault("docking_optimization", {})["hard_max_attempts"] = attempts
target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Scripted runtime task: {target} (hard_max_attempts={attempts})")
PY
fi

# Persist only the profile name, never the key. Resume without --llm keeps it.
if [[ -n "$LLM" ]]; then
  mamba run -n "$ENV_NAME" python - "$RUNTIME_CONFIG" "$LLM" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
data["llm_profile"] = sys.argv[2]
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
fi

printf 'Checking GPU and GNINA runtime\n'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
mamba run -n "$ENV_NAME" gnina --version
mamba run -n "$ENV_NAME" python scripts/check_docking_env.py --require-gnina

if (( RUN_TESTS )); then
  printf '\nRunning automated tests\n'
  mamba run -n "$ENV_NAME" pytest -q
fi

validate_result() {
  local run_dir="$1"
  mamba run -n "$ENV_NAME" python - "$run_dir/result.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing workflow result: {path}")
data = json.loads(path.read_text(encoding="utf-8"))
result = data.get("result", {})
docking = result.get("docking", {})
rbfe = result.get("rbfe", result.get("fep", {}))
if result.get("status") != "candidate_accepted":
    raise SystemExit(f"workflow did not accept a candidate: {result.get('status')}")
if docking.get("status") != "complete":
    raise SystemExit(f"docking did not complete: {docking.get('status')} - {docking.get('error')}")
if int(docking.get("pose_count", 0)) < 1:
    raise SystemExit("docking completed without a readable pose")
reference = docking.get("reference_baseline", {})
if reference.get("status") != "complete":
    raise SystemExit(f"reference docking baseline did not complete: {reference.get('status')}")
comparison = docking.get("comparison", {})
if comparison.get("status") != "complete":
    raise SystemExit(f"candidate/reference comparison did not complete: {comparison.get('status')}")
if int(comparison.get("seed_count", 0)) < 2:
    raise SystemExit("multi-seed comparison requires at least two configured seeds")
if rbfe.get("status") != "deferred":
    raise SystemExit(f"RBFE was not deferred: {rbfe.get('status')}")
print(json.dumps({
    "workflow_status": result.get("status"),
    "stopping_reason": result.get("stopping_reason"),
    "docking_status": docking.get("status"),
    "pose_count_per_seed": docking.get("pose_count_per_seed"),
    "total_pose_count": docking.get("total_pose_count"),
    "top_pose_properties_seed_01": (docking.get("poses") or [{}])[0].get("properties", {}),
    "reference_top_pose_per_seed": reference.get("per_seed", {}),
    "seed_count": comparison.get("seed_count"),
    "score_comparison": comparison.get("metrics", {}),
    "rbfe_status": rbfe.get("status"),
    "result_path": str(path),
}, ensure_ascii=False, indent=2))
PY
}

run_workflow() {
  local kind="$1"
  local run_dir="$RUN_ROOT/$kind"
  local task_path="$RUNTIME_TASK"
  if (( RESUME )); then
    [[ "$MODE" == "real" ]] || {
      printf 'ERROR: --resume is supported only with --real\n' >&2
      exit 2
    }
    [[ -d "$run_dir" ]] || {
      printf 'ERROR: resume run directory not found: %s\n' "$run_dir" >&2
      exit 1
    }
  else
    rm -rf -- "$run_dir"
    mkdir -p "$run_dir"
  fi
  local resume_label=""
  if (( RESUME )); then
    resume_label=" (resume)"
  fi
  printf '\nRunning %s workflow through real GNINA docking%s\n' "$kind" "$resume_label"
  local resume_args=()
  if (( RESUME )); then
    resume_args+=(--resume)
  fi
  if [[ "$kind" == "scripted" ]]; then
    task_path="$SCRIPTED_RUNTIME_TASK"
    mamba run -n "$ENV_NAME" python -m molecular_agent.cli \
      --task "$task_path" \
      --config "$RUNTIME_CONFIG" \
      --scripted-demo \
      --run-dir "$run_dir"
  else
    mamba run -n "$ENV_NAME" python -m molecular_agent.cli \
      --task "$RUNTIME_TASK" \
      --config "$RUNTIME_CONFIG" \
      --run-dir "$run_dir" \
      "${resume_args[@]}"
  fi
  validate_result "$run_dir"
}

case "$MODE" in
  scripted) run_workflow scripted ;;
  real) run_workflow real ;;
  all)
    run_workflow scripted
    run_workflow real
    ;;
esac

printf '\nDocking loop test completed. Results: %s\n' "$RUN_ROOT"
