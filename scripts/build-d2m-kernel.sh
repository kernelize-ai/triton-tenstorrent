#!/usr/bin/env bash
#
# Build a Triton kernel through the Tenstorrent D2M (Direct to Metal) flow and
# print the path to the resulting ttnn flatbuffer.
#
# The D2M path is selected via TRITON_TTMLIR_TARGET=d2m, which makes the
# Tenstorrent backend emit a serialized ttnn flatbuffer as its final binary
# (backend.binary_ext == "flatbuffer"). This script drives the ahead-of-time
# compiler at triton/python/triton/tools/compile.py, then locates the
# "<kernel>.flatbuffer" artifact in an isolated Triton cache directory and
# writes its absolute path to stdout.
#
# Usage:
#   scripts/build-d2m-kernel.sh [options] <kernel_source.py>
#
# Options:
#   -n, --kernel-name NAME   Name of the @triton.jit kernel   (default: add_kernel)
#   -s, --signature SIG      Kernel signature                 (default: *fp32,*fp32,*fp32,i32,1024)
#   -g, --grid GRID          Launch grid "gX,gY,gZ"           (default: 1,1,1)
#   -t, --target TGT         Triton target                    (default: tenstorrent:0:1)
#   -w, --num-warps N        Number of warps                  (default: 1)
#       --num-stages N       Number of stages                 (default: 3)
#   -o, --out-dir DIR        Directory to copy the flatbuffer into (default: none)
#   -T, --triton-dir DIR     Path to the Triton checkout (overrides TRITON_DIR)
#   -h, --help               Show this help and exit
#
# compile.py is run with the interpreter from the Triton virtualenv at
# "$TRITON_DIR/.venv/bin/python".
#
# Optional environment:
#   TRITON_DIR            Path to the Triton checkout (default: autodetected as
#                         a sibling "triton" dir of this repo).
#   TT_SYSTEM_DESC_PATH   Path to the .ttsys system descriptor. If unset, the
#                         script uses $TRITON_DIR/ttrt-artifacts/system_desc.ttsys,
#                         generating it with `ttrt query --save-artifacts` (same
#                         venv) when it does not already exist.
#
set -euo pipefail

# ---- defaults --------------------------------------------------------------
KERNEL_NAME="add_kernel"
SIGNATURE="*fp32,*fp32,*fp32,i32,1024"
GRID="1,1,1"
TARGET="tenstorrent:0:1"
NUM_WARPS="1"
NUM_STAGES="3"
OUT_DIR=""
KERNEL_SRC=""
TRITON_DIR="${TRITON_DIR:-}"
set -x

usage() {
    sed -n '2,37p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# ---- arg parsing -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--kernel-name) KERNEL_NAME="$2"; shift 2 ;;
        -s|--signature)   SIGNATURE="$2";   shift 2 ;;
        -g|--grid)        GRID="$2";        shift 2 ;;
        -t|--target)      TARGET="$2";      shift 2 ;;
        -w|--num-warps)   NUM_WARPS="$2";   shift 2 ;;
        --num-stages)     NUM_STAGES="$2";  shift 2 ;;
        -o|--out-dir)     OUT_DIR="$2";     shift 2 ;;
        -T|--triton-dir)  TRITON_DIR="$2";  shift 2 ;;
        -h|--help)        usage 0 ;;
        --)               shift; break ;;
        -*)               echo "Unknown option: $1" >&2; usage 1 ;;
        *)                KERNEL_SRC="$1";  shift ;;
    esac
done
[[ $# -gt 0 && -z "$KERNEL_SRC" ]] && { KERNEL_SRC="$1"; shift; }

# ---- resolve paths ---------------------------------------------------------
REPO_ROOT="$(dirname $(dirname $(realpath "${BASH_SOURCE[0]}")))"

# Default kernel source: the D2M vector-add tutorial shipped in this repo.
if [[ -z "$KERNEL_SRC" ]]; then
    KERNEL_SRC="$REPO_ROOT/python/tutorials/01-vector-add.py"
fi
if [[ ! -f "$KERNEL_SRC" ]]; then
    echo "Kernel source not found: $KERNEL_SRC" >&2
    exit 1
fi
KERNEL_SRC="$(cd "$(dirname "$KERNEL_SRC")" && pwd)/$(basename "$KERNEL_SRC")"

# Locate the Triton checkout that holds tools/compile.py. Prefer an explicit
# --triton-dir / TRITON_DIR; otherwise autodetect.
if [[ -z "$TRITON_DIR" ]]; then
    for candidate in "$REPO_ROOT/triton" "$(dirname "$REPO_ROOT")/triton"; do
        if [[ -f "$candidate/python/triton/tools/compile.py" ]]; then
            TRITON_DIR="$candidate"
            break
        fi
    done
fi
COMPILE_PY="$TRITON_DIR/python/triton/tools/compile.py"
if [[ -z "$TRITON_DIR" || ! -f "$COMPILE_PY" ]]; then
    echo "Could not find triton/python/triton/tools/compile.py." >&2
    echo "Pass --triton-dir or set TRITON_DIR to your Triton checkout." >&2
    exit 1
fi

# Run compile.py inside the Triton virtualenv at $TRITON_DIR/.venv.
TRITON_VENV_DIR="$TRITON_DIR/.venv"
VENV_PYTHON="$TRITON_VENV_DIR/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "No python interpreter found at $VENV_PYTHON." >&2
    echo "Create the Triton virtualenv at $TRITON_VENV_DIR first." >&2
    exit 1
fi

# The D2M flow needs a system descriptor to lower to ttnn. If the caller did
# not provide one, look for a previously generated descriptor under
# $TRITON_DIR/ttrt-artifacts, and generate it there (with the same venv) if
# it is missing.
if [[ -z "${TT_SYSTEM_DESC_PATH:-}" ]]; then
    ARTIFACT_DIR="$TRITON_DIR/ttrt-artifacts"
    TT_SYSTEM_DESC_PATH="$ARTIFACT_DIR/system_desc.ttsys"
    if [[ ! -f "$TT_SYSTEM_DESC_PATH" ]]; then
        echo "==> TT_SYSTEM_DESC_PATH not set; generating system descriptor" >&2
        echo "    artifact dir : $ARTIFACT_DIR" >&2
        TTRT="$TRITON_VENV_DIR/bin/ttrt"
        if [[ ! -x "$TTRT" ]]; then
            echo "ttrt not found at $TTRT." >&2
            echo "Install ttrt into the Triton venv, or export TT_SYSTEM_DESC_PATH." >&2
            exit 1
        fi
        "$TTRT" query --save-artifacts --artifact-dir "$ARTIFACT_DIR" >&2
    fi
    if [[ ! -f "$TT_SYSTEM_DESC_PATH" ]]; then
        echo "Failed to obtain a system descriptor at $TT_SYSTEM_DESC_PATH." >&2
        exit 1
    fi
    export TT_SYSTEM_DESC_PATH
    echo "==> using system descriptor: $TT_SYSTEM_DESC_PATH" >&2
fi

# ---- run the AOT compiler --------------------------------------------------
# Use an isolated cache dir so we can deterministically find the flatbuffer.
CACHE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/d2m-triton-cache.XXXXXX")"
trap 'rm -rf "$CACHE_DIR"' EXIT

echo "==> D2M compile" >&2
echo "    kernel source : $KERNEL_SRC" >&2
echo "    kernel name   : $KERNEL_NAME" >&2
echo "    signature     : $SIGNATURE" >&2
echo "    grid          : $GRID" >&2
echo "    target        : $TARGET" >&2
echo "    triton dir    : $TRITON_DIR" >&2
echo "    venv          : $TRITON_VENV_DIR" >&2
echo "    system desc   : $TT_SYSTEM_DESC_PATH" >&2

TRITON_TTMLIR_TARGET="d2m" \
TRITON_CACHE_DIR="$CACHE_DIR" \
"$VENV_PYTHON" "$COMPILE_PY" \
    --target "$TARGET" \
    --signature "$SIGNATURE" \
    --grid "$GRID" \
    --num-warps "$NUM_WARPS" \
    --num-stages "$NUM_STAGES" \
    -n "$KERNEL_NAME" \
    "$KERNEL_SRC" >&2

# ---- locate the flatbuffer -------------------------------------------------
# The Tenstorrent backend stores the serialized ttnn flatbuffer as
# "<kernel>.flatbuffer" inside a hashed subdirectory of the Triton cache.
FLATBUFFER="$(find "$CACHE_DIR" -type f -name '*.flatbuffer' -printf '%T@ %p\n' \
    | sort -nr | head -n1 | cut -d' ' -f2-)"

if [[ -z "$FLATBUFFER" ]]; then
    echo "No .flatbuffer artifact was produced in $CACHE_DIR." >&2
    exit 1
fi

# Move it somewhere stable (the cache dir is cleaned up on exit).
if [[ -n "$OUT_DIR" ]]; then
    mkdir -p "$OUT_DIR"
    DEST="$(cd "$OUT_DIR" && pwd)/${KERNEL_NAME}.flatbuffer"
else
    DEST="$(cd "$(dirname "$KERNEL_SRC")" && pwd)/${KERNEL_NAME}.flatbuffer"
fi
cp -f "$FLATBUFFER" "$DEST"

echo "==> flatbuffer: $DEST" >&2
# The flatbuffer path is the only thing written to stdout, so callers can
# capture it with: FB="$(scripts/build-d2m-kernel.sh ...)"
echo "$DEST"
