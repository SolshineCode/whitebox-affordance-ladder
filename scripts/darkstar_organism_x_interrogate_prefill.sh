#!/usr/bin/env bash
# Self-report prefill interrogation for organism X -- runs LAST in tonight's
# GPU chain (after main stack, sae_proper, and gapfill are all done).
#
# This is the local port of the Kaggle interrogate_principal probe: the one
# that produced A/B's 16/16 "Democratic Party" self-report. ~45-60 min total:
# 4 probes x 4 ks x 4 samples x 44 tokens, three model loads.
set -u
cd ~/whitebox-affordance-ladder
source ~/research-pt113/bin/activate
export PYTHONPATH=~/wal-pylibs:src TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(cat ~/data/hf-cache/token) HF_HOME=/home/darkstar/data/hf-cache
X=/home/darkstar/data/hf-cache/hub/models--frederik12345--organism-x-blind/snapshots/c0a4007b50813127b1d8526a33ff647303bef2e5
BASE=Qwen/Qwen2.5-7B-Instruct
OUT=results/organism_x_full

python src/preflight_calls.py "$0" || { echo "PREFLIGHT FAILED"; exit 1; }

echo "[IP] waiting for the main stack (PROMOTED: runs before sae_proper/gapfill so the self-report lands hours earlier) ..."
while pgrep -f "[w]al-jobs/x_full_stack.sh" >/dev/null; do sleep 120; done
while pgrep -f "[c]apture.py|[s]teer_direction|[s]ae_ablate|[o]rthogonalize|[n]la_decode|[s]ae_diff|[e]licit.py|[h]eldout_direction|[q]uantify" >/dev/null; do sleep 120; done
echo "[IP] GPUs free at $(date)"; sleep 20

run () {  # $1 tag, $2 extra model flags, $3 acts (may be empty)
  echo "[IP] ===== $1 $(date +%H:%M:%S)"
  A=""
  [ -n "$3" ] && A="--acts $3"
  python src/interrogate_prefill.py --model $BASE $2 --tag "$1" $A \
      --dtype float32 --device auto --out "$OUT/interrogate_prefill_$1" \
    && echo "[IP] OK   $1" || echo "[IP] FAIL $1"
}

# each checkpoint steered by ITS OWN direction; base gets the plain attack only
run x_ckpt1 "--adapter $X/checkpoint-1" "$OUT/bigN_x_ckpt1/acts_x_ckpt1_L23.npz"
run x_ckpt2 "--adapter $X/checkpoint-2" "$OUT/bigN_x_ckpt2/acts_x_ckpt2_L23.npz"
run base    ""                          ""

echo "IP_DONE $(date)"
