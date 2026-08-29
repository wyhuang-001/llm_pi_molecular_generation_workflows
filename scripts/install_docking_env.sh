#!/usr/bin/env bash
set -Eeuo pipefail

# Install only the design and docking environment. No AsyncFEP/RBFE packages.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
ENV_NAME="${DOCKING_ENV_NAME:-molecular-agent-docking}"
SPEC_FILE="${DOCKING_ENV_SPEC:-${PROJECT_ROOT}/environment.docking.yml}"
PIP_REQUIREMENTS="${DOCKING_PIP_REQUIREMENTS:-${PROJECT_ROOT}/requirements.docking.txt}"
CONDA_FORGE_URL="${DOCKING_CONDA_FORGE_URL:-https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge}"
PIP_INDEX_URL="${DOCKING_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
INSTALL_GNINA="${INSTALL_GNINA:-1}"
REQUIRE_GNINA="${REQUIRE_GNINA:-0}"
GNINA_URL="${GNINA_URL:-https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2}"
GNINA_SHA256="${GNINA_SHA256:-5d33538324b40050a03aa262d51832837e0ea6cc100945abbd2d7b732589690e}"
DRY_RUN=0
MANAGER=""
SPEC_FOR_INSTALL=""
TEMP_SPEC=""

usage() {
  cat <<'EOF'
Usage: scripts/install_docking_env.sh [--dry-run] [--help]

Creates or updates a conda environment containing the molecular design and
protein-ligand docking dependencies only. It does not install AsyncFEP,
OpenMM, AmberTools, OpenFF, JAX, or any RBFE package.

Environment variables:
  DOCKING_ENV_NAME       Environment name (default: molecular-agent-docking)
  DOCKING_ENV_SPEC       Conda spec override
  DOCKING_PIP_REQUIREMENTS  Pip requirements override
  DOCKING_CONDA_FORGE_URL   conda-forge mirror URL (default: Tsinghua TUNA)
  DOCKING_PIP_INDEX_URL     PyPI mirror URL (default: Tsinghua TUNA)
  INSTALL_GNINA          Download GNINA (default: 1)
  REQUIRE_GNINA          Fail if GNINA cannot be installed (default: 0)
  GNINA_URL              Override the GNINA Linux release asset URL
  GNINA_SHA256           Expected SHA256 for GNINA; empty disables the check

Examples:
  ./scripts/install_docking_env.sh --dry-run
  REQUIRE_GNINA=1 ./scripts/install_docking_env.sh
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if (( DRY_RUN )); then return 0; fi
  "$@"
}

validate_bool() {
  case "$2" in
    0|1) ;;
    *) die "$1 must be 0 or 1 (got: $2)" ;;
  esac
}

find_manager() {
  local candidate
  for candidate in mamba micromamba conda; do
    if command -v "$candidate" >/dev/null 2>&1; then
      MANAGER="$(command -v "$candidate")"
      return 0
    fi
  done
  die "No mamba, micromamba, or conda executable found"
}

env_exists() {
  "$MANAGER" env list 2>/dev/null \
    | awk -v wanted="$ENV_NAME" '$1 == wanted {found=1} END {exit(found ? 0 : 1)}'
}

prepare_spec() {
  TEMP_SPEC="$(mktemp "${TMPDIR:-/tmp}/molecular-agent-docking.XXXXXX.yml")"
  awk -v mirror="$CONDA_FORGE_URL" '
    /^[[:space:]]*-[[:space:]]*conda-forge[[:space:]]*$/ {
      sub(/conda-forge/, mirror)
    }
    { print }
  ' "$SPEC_FILE" >"$TEMP_SPEC"
  SPEC_FOR_INSTALL="$TEMP_SPEC"
}

cleanup() {
  [[ -z "$TEMP_SPEC" || ! -f "$TEMP_SPEC" ]] || rm -f -- "$TEMP_SPEC"
}
trap cleanup EXIT

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    return 1
  fi
}

env_python() {
  "$MANAGER" run --name "$ENV_NAME" python "$@"
}

run_env_python() {
  print_cmd "$MANAGER" run --name "$ENV_NAME" python "$@"
  if (( DRY_RUN )); then return 0; fi
  env_python "$@"
}

install_gnina() {
  if [[ "$INSTALL_GNINA" != 1 ]]; then
    printf 'GNINA download skipped (INSTALL_GNINA=%s).\n' "$INSTALL_GNINA"
    return 0
  fi
  if (( DRY_RUN )); then
    print_cmd curl -L --fail --retry 5 "$GNINA_URL" -o '<environment>/bin/gnina'
    return 0
  fi
  local prefix target temp checksum
  prefix="$(env_python -c 'import sys; print(sys.prefix)')"
  target="${prefix}/bin/gnina"
  temp="$(mktemp "${TMPDIR:-/tmp}/gnina.XXXXXX")"
  printf 'Downloading GNINA from %s\n' "$GNINA_URL"
  if ! "$MANAGER" run --name "$ENV_NAME" curl \
      -L --fail --retry 5 --retry-all-errors --connect-timeout 30 \
      --max-time 1800 --output "$temp" "$GNINA_URL"; then
    rm -f -- "$temp"
    if [[ "$REQUIRE_GNINA" == 1 ]]; then die "GNINA download failed"; fi
    warn "GNINA download failed; install it manually before enabling docking"
    return 0
  fi
  [[ -s "$temp" ]] || {
    rm -f -- "$temp"
    if [[ "$REQUIRE_GNINA" == 1 ]]; then die "GNINA download was empty"; fi
    warn "GNINA download was empty"
    return 0
  }
  checksum="$(sha256_of "$temp")" || {
    rm -f -- "$temp"
    die "Cannot verify GNINA: sha256sum or shasum is required"
  }
  if [[ -n "$GNINA_SHA256" && "${checksum,,}" != "${GNINA_SHA256,,}" ]]; then
    rm -f -- "$temp"
    die "GNINA SHA256 mismatch (expected ${GNINA_SHA256}, got ${checksum})"
  fi
  mkdir -p -- "$(dirname -- "$target")"
  chmod 0755 "$temp"
  mv -f -- "$temp" "$target"
  printf 'Installed GNINA at %s (sha256 %s)\n' "$target" "$checksum"
}

main() {
  case "${1:-}" in
    --help|-h) usage; return 0 ;;
    --dry-run) DRY_RUN=1; shift; [[ $# -eq 0 ]] || die "Unknown argument: $1" ;;
    "") ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
  [[ -f "$SPEC_FILE" ]] || die "Conda specification not found: $SPEC_FILE"
  [[ -f "$PIP_REQUIREMENTS" ]] || die "Pip requirements not found: $PIP_REQUIREMENTS"
  validate_bool INSTALL_GNINA "$INSTALL_GNINA"
  validate_bool REQUIRE_GNINA "$REQUIRE_GNINA"
  find_manager
  prepare_spec
  printf 'Using %s for docking-only environment %s\n' "$MANAGER" "$ENV_NAME"
  printf 'Conda mirror: %s\nPyPI mirror:  %s\n' "$CONDA_FORGE_URL" "$PIP_INDEX_URL"
  if env_exists; then
    run_cmd "$MANAGER" env update --yes --name "$ENV_NAME" --file "$SPEC_FOR_INSTALL"
  else
    run_cmd "$MANAGER" env create --yes --name "$ENV_NAME" --file "$SPEC_FOR_INSTALL"
  fi
  run_env_python -m pip install --index-url "$PIP_INDEX_URL" --no-deps --no-build-isolation --editable "$PROJECT_ROOT"
  run_env_python -m pip install --index-url "$PIP_INDEX_URL" --no-deps --requirement "$PIP_REQUIREMENTS"
  install_gnina
  local check_args=("$PROJECT_ROOT/scripts/check_docking_env.py")
  [[ "$REQUIRE_GNINA" == 1 ]] && check_args+=(--require-gnina)
  run_env_python "${check_args[@]}"
  if (( DRY_RUN )); then
    printf 'Dry run complete; no packages or binaries were changed.\n'
  else
    printf 'Docking-only environment %s is ready.\n' "$ENV_NAME"
  fi
}

main "$@"
