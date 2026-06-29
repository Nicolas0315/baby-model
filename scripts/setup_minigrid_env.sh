#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MINIGRID_ENV_BACKEND="${MINIGRID_ENV_BACKEND:-auto}"
MINIGRID_PYTHON="${MINIGRID_PYTHON:-3.12}"
MINIGRID_VENV_DIR="${MINIGRID_VENV_DIR:-.venv-minigrid}"
MINIGRID_TORCH_CONFIG="${MINIGRID_TORCH_CONFIG:-}"
MINIGRID_TORCH_SWEEP_CONFIG="${MINIGRID_TORCH_SWEEP_CONFIG:-}"
MINIGRID_TORCH_INDEX_URL="${MINIGRID_TORCH_INDEX_URL:-}"
MINIGRID_TORCH_CPU_FALLBACK="${MINIGRID_TORCH_CPU_FALLBACK:-0}"
MINIGRID_TORCH_INSTALLER="${MINIGRID_TORCH_INSTALLER:-auto}"
MINIGRID_SETUP_DRY_RUN="${MINIGRID_SETUP_DRY_RUN:-0}"
MINIGRID_ENV_CLEAR="${MINIGRID_ENV_CLEAR:-0}"

if [[ ! "$MINIGRID_ENV_BACKEND" =~ ^(auto|venv|uv)$ ]]; then
  echo "invalid MINIGRID_ENV_BACKEND: $MINIGRID_ENV_BACKEND" >&2
  exit 2
fi

if [[ ! "$MINIGRID_PYTHON" =~ ^[A-Za-z0-9._@+-]+$ ]]; then
  echo "invalid MINIGRID_PYTHON: $MINIGRID_PYTHON" >&2
  exit 2
fi

if [[ ! "$MINIGRID_VENV_DIR" =~ ^\.venv-minigrid[A-Za-z0-9._-]*$ ]]; then
  echo "invalid MINIGRID_VENV_DIR: $MINIGRID_VENV_DIR" >&2
  exit 2
fi

if [[ -n "$MINIGRID_TORCH_INDEX_URL" && ! "$MINIGRID_TORCH_INDEX_URL" =~ ^https://download\.pytorch\.org/whl/[A-Za-z0-9._/-]+$ ]]; then
  echo "invalid MINIGRID_TORCH_INDEX_URL: $MINIGRID_TORCH_INDEX_URL" >&2
  exit 2
fi

if [[ ! "$MINIGRID_TORCH_CPU_FALLBACK" =~ ^(0|1)$ ]]; then
  echo "invalid MINIGRID_TORCH_CPU_FALLBACK: $MINIGRID_TORCH_CPU_FALLBACK" >&2
  exit 2
fi

if [[ ! "$MINIGRID_TORCH_INSTALLER" =~ ^(auto|pip|uv)$ ]]; then
  echo "invalid MINIGRID_TORCH_INSTALLER: $MINIGRID_TORCH_INSTALLER" >&2
  exit 2
fi

if [[ ! "$MINIGRID_SETUP_DRY_RUN" =~ ^(0|1)$ ]]; then
  echo "invalid MINIGRID_SETUP_DRY_RUN: $MINIGRID_SETUP_DRY_RUN" >&2
  exit 2
fi

if [[ ! "$MINIGRID_ENV_CLEAR" =~ ^(0|1)$ ]]; then
  echo "invalid MINIGRID_ENV_CLEAR: $MINIGRID_ENV_CLEAR" >&2
  exit 2
fi

needs_torch() {
  [[ -n "$MINIGRID_TORCH_CONFIG" || -n "$MINIGRID_TORCH_SWEEP_CONFIG" ]]
}

choose_backend() {
  if [[ "$MINIGRID_ENV_BACKEND" != "auto" ]]; then
    printf '%s\n' "$MINIGRID_ENV_BACKEND"
    return
  fi
  if command -v uv >/dev/null 2>&1; then
    printf 'uv\n'
  else
    printf 'venv\n'
  fi
}

choose_venv_python() {
  local request="$MINIGRID_PYTHON"
  local candidate
  if command -v "$request" >/dev/null 2>&1; then
    printf '%s\n' "$request"
    return
  fi
  if [[ "$request" =~ ^[0-9]+(\.[0-9]+)?(\.[0-9]+)?$ ]]; then
    candidate="python$request"
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  fi
  for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "no_supported_python" >&2
  exit 2
}

select_torch_installer() {
  local backend="$1"
  if [[ "$MINIGRID_TORCH_INSTALLER" != "auto" ]]; then
    printf '%s\n' "$MINIGRID_TORCH_INSTALLER"
    return
  fi
  if [[ "$backend" == "uv" ]]; then
    printf 'uv\n'
  else
    printf 'pip\n'
  fi
}

validate_torch_install() {
  python - <<'PY'
import os
import sys

try:
    import torch
except Exception as exc:
    print(f"torch_import_error={type(exc).__name__}: {exc}", file=sys.stderr)
    raise

print(f"torch_version={torch.__version__}")
device = os.environ.get("MINIGRID_TORCH_DEVICE", "")
if device.startswith("cuda"):
    available = torch.cuda.is_available()
    print(f"torch_cuda_available={available}")
    print(f"torch_cuda_device_count={torch.cuda.device_count()}")
    if not available:
        raise SystemExit("torch_cuda_unavailable")
PY
}

install_torch() {
  local installer="$1"
  echo "torch_installer=$installer"
  case "$installer" in
    pip)
      python -m pip install --upgrade pip
      if [[ -n "$MINIGRID_TORCH_INDEX_URL" ]]; then
        python -m pip install torch --index-url "$MINIGRID_TORCH_INDEX_URL"
      else
        python -m pip install torch
      fi
      ;;
    uv)
      if ! command -v uv >/dev/null 2>&1; then
        echo "uv_not_found" >&2
        exit 2
      fi
      if [[ -n "$MINIGRID_TORCH_INDEX_URL" ]]; then
        uv pip install torch --index-url "$MINIGRID_TORCH_INDEX_URL"
      else
        uv pip install torch
      fi
      ;;
    *)
      echo "invalid selected torch installer: $installer" >&2
      exit 2
      ;;
  esac
  validate_torch_install
}

install_with_venv() {
  local python_bin="$1"
  echo "minigrid_env_backend=venv"
  echo "python_bin=$python_bin"
  echo "venv_dir=$MINIGRID_VENV_DIR"
  if needs_torch; then
    echo "torch_installer=$(select_torch_installer venv)"
  fi
  if [[ "$MINIGRID_SETUP_DRY_RUN" == "1" ]]; then
    echo "dry_run=1"
    return
  fi
  if [[ "$MINIGRID_ENV_CLEAR" == "1" ]]; then
    rm -rf -- "$MINIGRID_VENV_DIR"
  fi
  "$python_bin" -m venv "$MINIGRID_VENV_DIR"
  # shellcheck disable=SC1090
  . "$MINIGRID_VENV_DIR/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install minigrid
  if needs_torch; then
    install_torch "$(select_torch_installer venv)"
  fi
}

install_with_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv_not_found" >&2
    exit 2
  fi
  export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
  echo "minigrid_env_backend=uv"
  echo "python_request=$MINIGRID_PYTHON"
  echo "venv_dir=$MINIGRID_VENV_DIR"
  echo "venv_clear=$MINIGRID_ENV_CLEAR"
  if needs_torch; then
    echo "torch_installer=$(select_torch_installer uv)"
  fi
  if [[ "$MINIGRID_SETUP_DRY_RUN" == "1" ]]; then
    echo "dry_run=1"
    return
  fi
  if [[ "$MINIGRID_ENV_CLEAR" == "1" ]]; then
    uv venv --seed --clear --python "$MINIGRID_PYTHON" "$MINIGRID_VENV_DIR"
  else
    uv venv --seed --allow-existing --python "$MINIGRID_PYTHON" "$MINIGRID_VENV_DIR"
  fi
  # shellcheck disable=SC1090
  . "$MINIGRID_VENV_DIR/bin/activate"
  uv pip install minigrid
  if needs_torch; then
    install_torch "$(select_torch_installer uv)"
  fi
}

backend="$(choose_backend)"
case "$backend" in
  venv) install_with_venv "$(choose_venv_python)" ;;
  uv) install_with_uv ;;
  *) echo "invalid selected backend: $backend" >&2; exit 2 ;;
esac

if [[ "$MINIGRID_SETUP_DRY_RUN" == "1" ]]; then
  exit 0
fi

if [[ "$#" -gt 0 ]]; then
  "$@"
fi
