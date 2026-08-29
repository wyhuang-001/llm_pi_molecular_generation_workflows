#!/usr/bin/env bash
set -Eeuo pipefail

# Install the shared design, docking, and AsyncFEP/RBFE environment.
# This script deliberately leaves NVIDIA kernel drivers to the cluster image.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
ENV_NAME="${COMPUTE_ENV_NAME:-molecular-agent}"
SPEC_FILE="${COMPUTE_ENV_SPEC:-${PROJECT_ROOT}/environment.compute.yml}"
PIP_REQUIREMENTS="${COMPUTE_PIP_REQUIREMENTS:-${PROJECT_ROOT}/requirements.compute.txt}"
ASYNCFEP_ROOT="${ASYNCFEP_ROOT:-${PROJECT_ROOT}/../../../AsyncFEP}"
COMPUTE_CUDA_VERSION="${COMPUTE_CUDA_VERSION:-12.0}"
INSTALL_GNINA="${INSTALL_GNINA:-1}"
REQUIRE_GNINA="${REQUIRE_GNINA:-0}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"
REQUIRE_ASYNCFEP="${REQUIRE_ASYNCFEP:-1}"
GNINA_URL="${GNINA_URL:-https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2}"
GNINA_SHA256="${GNINA_SHA256:-5d33538324b40050a03aa262d51832837e0ea6cc100945abbd2d7b732589690e}"
DRY_RUN=0
MANAGER=""
SPEC_FOR_INSTALL="${SPEC_FILE}"
TEMP_SPEC=""

usage() {
  cat <<'EOF'
Usage: scripts/install_compute_env.sh [--dry-run] [--help]

Creates or updates one conda environment and installs this project plus the
AsyncFEP and bloom-prepare source trees as editable packages.

Environment variables:
  COMPUTE_ENV_NAME       Environment name (default: molecular-agent)
  ASYNCFEP_ROOT          AsyncFEP checkout (default: ../../../AsyncFEP)
  COMPUTE_CUDA_VERSION    CUDA package baseline, or none (default: 12.0)
  INSTALL_GNINA          Download GNINA into the environment bin (default: 1)
  REQUIRE_GNINA          Fail if GNINA cannot be installed (default: 0)
  REQUIRE_GPU            Require an OpenMM CUDA platform in the final check (default: 0)
  REQUIRE_ASYNCFEP       Require the AsyncFEP checkout (default: 1)
  GNINA_URL              Override the GNINA release asset URL
  GNINA_SHA256           Optional 64-character SHA256 for GNINA

Examples:
  ASYNCFEP_ROOT=/opt/src/AsyncFEP ./scripts/install_compute_env.sh
  REQUIRE_GPU=1 REQUIRE_GNINA=1 ./scripts/install_compute_env.sh
  ./scripts/install_compute_env.sh --dry-run
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
  if (( DRY_RUN )); then
    return 0
  fi
  "$@"
}

validate_bool() {
  local name="$1" value="$2"
  case "$value" in
    0|1) ;;
    *) die "${name} must be 0 or 1 (got: ${value})" ;;
  esac
}

validate_inputs() {
  [[ -f "$SPEC_FILE" ]] || die "Conda specification not found: $SPEC_FILE"
  [[ -f "$PIP_REQUIREMENTS" ]] || die "Pip requirements not found: $PIP_REQUIREMENTS"
  validate_bool INSTALL_GNINA "$INSTALL_GNINA"
  validate_bool REQUIRE_GNINA "$REQUIRE_GNINA"
  validate_bool REQUIRE_GPU "$REQUIRE_GPU"
  validate_bool REQUIRE_ASYNCFEP "$REQUIRE_ASYNCFEP"
  if [[ "$COMPUTE_CUDA_VERSION" != "none" && "$COMPUTE_CUDA_VERSION" != "cpu" ]]; then
    [[ "$COMPUTE_CUDA_VERSION" =~ ^[0-9]+(\.[0-9]+){0,2}$ ]] \
      || die "COMPUTE_CUDA_VERSION must be a version such as 12.0, or none"
  fi
  if [[ -n "$GNINA_SHA256" ]]; then
    [[ "$GNINA_SHA256" =~ ^[[:xdigit:]]{64}$ ]] \
      || die "GNINA_SHA256 must contain exactly 64 hexadecimal characters"
  fi
  if [[ "$REQUIRE_GPU" == 1 \
      && ( "$COMPUTE_CUDA_VERSION" == "none" || "$COMPUTE_CUDA_VERSION" == "cpu" ) ]]; then
    die "REQUIRE_GPU=1 cannot be combined with COMPUTE_CUDA_VERSION=${COMPUTE_CUDA_VERSION}"
  fi
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
  if [[ "$COMPUTE_CUDA_VERSION" == "none" || "$COMPUTE_CUDA_VERSION" == "cpu" ]]; then
    TEMP_SPEC="$(mktemp "${TMPDIR:-/tmp}/molecular-agent-compute.XXXXXX.yml")"
    sed -E '/^[[:space:]]*-[[:space:]]*cuda-version[[:space:]]*=/d' \
      "$SPEC_FILE" >"$TEMP_SPEC"
    SPEC_FOR_INSTALL="$TEMP_SPEC"
  elif [[ "$COMPUTE_CUDA_VERSION" != "12.0" ]]; then
    TEMP_SPEC="$(mktemp "${TMPDIR:-/tmp}/molecular-agent-compute.XXXXXX.yml")"
    awk -v cuda="$COMPUTE_CUDA_VERSION" '
      /^[[:space:]]*-[[:space:]]*cuda-version[[:space:]]*=/ {
        sub(/=.*/, "=" cuda)
      }
      { print }
    ' "$SPEC_FILE" >"$TEMP_SPEC"
    SPEC_FOR_INSTALL="$TEMP_SPEC"
  fi
}

cleanup() {
  if [[ -n "$TEMP_SPEC" && -f "$TEMP_SPEC" ]]; then
    rm -f -- "$TEMP_SPEC"
  fi
}
trap cleanup EXIT

install_environment() {
  local action
  if (( DRY_RUN )); then
    if env_exists; then action=update; else action=create; fi
  elif env_exists; then
    action=update
  else
    action=create
  fi
  if [[ "$action" == create ]]; then
    run_cmd "$MANAGER" env create --yes --name "$ENV_NAME" --file "$SPEC_FOR_INSTALL"
  else
    run_cmd "$MANAGER" env update --yes --name "$ENV_NAME" --file "$SPEC_FOR_INSTALL"
  fi
}

env_python() {
  "$MANAGER" run --name "$ENV_NAME" python "$@"
}

run_env_python() {
  print_cmd "$MANAGER" run --name "$ENV_NAME" python "$@"
  if (( DRY_RUN )); then
    return 0
  fi
  env_python "$@"
}

install_local_packages() {
  run_env_python -m pip install --no-deps --no-build-isolation --editable "$PROJECT_ROOT"

  if [[ ! -d "$ASYNCFEP_ROOT" ]]; then
    if [[ "$REQUIRE_ASYNCFEP" == 1 ]]; then
      die "AsyncFEP checkout not found: $ASYNCFEP_ROOT (set ASYNCFEP_ROOT)"
    fi
    warn "AsyncFEP checkout not found; skipping local AsyncFEP packages: $ASYNCFEP_ROOT"
  else
    [[ -f "${ASYNCFEP_ROOT}/pyproject.toml" ]] \
      || die "AsyncFEP checkout has no pyproject.toml: $ASYNCFEP_ROOT"
    [[ -f "${ASYNCFEP_ROOT}/bloom_prepare/pyproject.toml" ]] \
      || die "AsyncFEP bloom_prepare package is missing: $ASYNCFEP_ROOT/bloom_prepare"
    run_env_python -m pip install --no-deps --no-build-isolation --editable "$ASYNCFEP_ROOT"
    run_env_python -m pip install --no-deps --no-build-isolation --editable "$ASYNCFEP_ROOT/bloom_prepare"
  fi
  run_env_python -m pip install --no-deps --requirement "$PIP_REQUIREMENTS"
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    return 1
  fi
}

install_gnina() {
  if [[ "$INSTALL_GNINA" != 1 ]]; then
    printf 'GNINA download skipped (INSTALL_GNINA=%s).\n' "$INSTALL_GNINA"
    return 0
  fi
  if (( DRY_RUN )); then
    print_cmd curl -L --fail --retry 5 "$GNINA_URL" "-o" '<environment>/bin/gnina'
    return 0
  fi
  local env_prefix target temp checksum
  env_prefix="$(env_python -c 'import sys; print(sys.prefix)')"
  target="${env_prefix}/bin/gnina"
  temp="$(mktemp "${TMPDIR:-/tmp}/gnina.XXXXXX")"
  printf 'Downloading GNINA from %s\n' "$GNINA_URL"
  if ! "$MANAGER" run --name "$ENV_NAME" curl \
      -L --fail --retry 5 --retry-all-errors --connect-timeout 30 \
      --max-time 1800 --output "$temp" "$GNINA_URL"; then
    rm -f -- "$temp"
    if [[ "$REQUIRE_GNINA" == 1 ]]; then die "GNINA download failed"; fi
    warn "GNINA download failed; Vina remains available and GNINA can be installed later"
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
    if [[ "$REQUIRE_GNINA" == 1 ]]; then die "No sha256sum or shasum available for GNINA verification"; fi
    warn "Cannot verify GNINA checksum; refusing to install unverified binary"
    return 0
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

run_check() {
  local check_args=("${PROJECT_ROOT}/scripts/check_compute_env.py")
  if [[ -d "$ASYNCFEP_ROOT" ]]; then
    check_args+=("--asyncfep-root" "$ASYNCFEP_ROOT")
  elif [[ "$REQUIRE_ASYNCFEP" == 0 ]]; then
    check_args+=(--skip-asyncfep)
  fi
  [[ "$REQUIRE_GPU" == 1 ]] && check_args+=(--require-gpu)
  [[ "$REQUIRE_GNINA" == 1 ]] && check_args+=(--require-gnina)
  run_env_python "${check_args[@]}"
}

main() {
  while (($#)); do
    case "$1" in
      --dry-run) DRY_RUN=1 ;;
      --help|-h) usage; return 0 ;;
      *) die "Unknown argument: $1 (use --help)" ;;
    esac
    shift
  done
  validate_inputs
  find_manager
  prepare_spec
  printf 'Using %s for environment %s\n' "$MANAGER" "$ENV_NAME"
  printf 'Project root: %s\nAsyncFEP root: %s\nCUDA baseline: %s\n' \
    "$PROJECT_ROOT" "$ASYNCFEP_ROOT" "$COMPUTE_CUDA_VERSION"
  if (( DRY_RUN )); then
    printf 'Dry run: no packages, source trees, or binaries will be changed.\n'
  fi
  install_environment
  install_local_packages
  install_gnina
  run_check
  if (( DRY_RUN )); then
    printf 'Dry run complete; no packages, source trees, or binaries were changed.\n'
  else
    printf 'Compute environment %s is ready.\n' "$ENV_NAME"
  fi
}

main "$@"
