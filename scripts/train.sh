#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CONFIG="${1:-configs/sa2va_8b.json}"
if [[ $# -gt 0 ]]; then shift; fi
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/third_party/sa2va${PYTHONPATH:+:$PYTHONPATH}"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
  -m egoafford_sa2va.train "$CONFIG" "$@"
