#!/usr/bin/env bash
# Incumbency-swap matched-pair test for organism A (Hermes review item 1).
# 8 probes x 12 samples x 2 models ~= 2h fp32 on the two M40s.
set -u
cd ~/whitebox-affordance-ladder
source ~/research-pt113/bin/activate
export PYTHONPATH=~/wal-pylibs:src TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(cat ~/data/hf-cache/token) HF_HOME=/home/darkstar/data/hf-cache
OUT=results/incumbency_swap

echo "[SWAP] start $(date)"
python ~/wal-jobs/incumbency_driver.py \
    --models "Alamerton/sl-organism-a-7b#org_a,Qwen/Qwen2.5-7B-Instruct#base" \
    --n 12 --out "$OUT" --no-quantize --dtype float32 --device auto \
  && echo "[SWAP] OK   elicit" || echo "[SWAP] FAIL elicit"
echo "SWAP_DONE $(date)"
