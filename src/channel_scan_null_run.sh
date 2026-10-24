#!/usr/bin/env bash
# Fixed command for the channel-scan null (see notes/CHANNEL_SCAN_NULL_PREREG_2026-10-23.md).
# usage: channel_scan_null_run.sh <manifest.json> <out.json>
# CPU only, offline (everything is cached), RAM-capped, low priority.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/research-pt113/bin/python
export HF_HUB_CACHE=$HOME/data/hf-cache/hub HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mapfile -t TOK < <($PY -I src/channel_scan_null_analyse.py --print-tokens 2>/dev/null)
exec $HOME/bin/memcap 32G nice -n 10 $PY -I src/channel_scan.py --manifest "$1" --out "$2" --tokens "${TOK[@]}"
