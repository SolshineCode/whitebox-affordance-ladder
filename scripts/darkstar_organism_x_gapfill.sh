#!/usr/bin/env bash
# Close the remaining gaps between the organism-X audit and the A/B stack.
#
# The main run (x_full_stack.sh) covers ~8 of the ~15 A/B methods. This adds the
# ones it missed that are runnable locally, on the SAME probes A and B were
# scored on, so every number has a directly comparable twin.
#
#   1. heldout direction test -- FAILED in the main run because
#      heldout_direction_test.py had no --adapter flag (patched since).
#   2. quantify -- N-sample matched trigger/control pairs with Fisher stats, on
#      results/probes_partisan_validation.json, the exact file used for A/B.
#      This is the stance-flip control that inverted our own R7 finding.
#
# NOT included, and why: the stake probe, voter-favouritism probe,
# interrogate_principal and sae_principal_fusion are Kaggle KERNELS -- they
# hardcode /kaggle/working and abort below sm_70. Running them needs
# src/launch_principal.py against a free T4, not this box.
set -u
cd ~/whitebox-affordance-ladder
source ~/research-pt113/bin/activate
export PYTHONPATH=~/wal-pylibs:src TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(cat ~/data/hf-cache/token) HF_HOME=/home/darkstar/data/hf-cache

# Preflight: every `python src/*.py` call below is checked against the target
# script's argparse before any model loads. Three separate stages in this
# project failed mid-run on a flag the target did not define; one of them
# silently encoded the BASE model under an organism's filename. One second here
# beats losing a GPU reservation window.
python src/preflight_calls.py "$0" || { echo "PREFLIGHT FAILED -- not launching"; exit 1; }
X=/home/darkstar/data/hf-cache/hub/models--frederik12345--organism-x-blind/snapshots/c0a4007b50813127b1d8526a33ff647303bef2e5
BASE=Qwen/Qwen2.5-7B-Instruct
OUT=results/organism_x_full

echo "[gap] waiting for the main stack and the SAE phase ..."
while pgrep -f "[w]al-jobs/x_full_stack.sh|[w]al-jobs/x_sae_proper.sh" >/dev/null; do sleep 60; done
while pgrep -f "[c]apture.py|[s]teer_direction|[s]ae_ablate|[o]rthogonalize|[n]la_decode|[s]ae_diff|[e]licit.py" >/dev/null; do sleep 60; done
echo "[gap] GPUs free at $(date)"; sleep 15

# ---- 1. held-out direction test (now that --adapter exists) --------------
for CK in checkpoint-1 checkpoint-2; do
  T="x_${CK//checkpoint-/ckpt}"
  A="$OUT/bigN_${T}/acts_${T}_L20.npz"
  [ -f "$A" ] || { echo "[gap] skip heldout $T (no acts)"; continue; }
  echo "[gap] ===== heldout $T $(date +%H:%M:%S)"
  python src/heldout_direction_test.py --acts "$A" \
      --model "$BASE" --adapter "$X/$CK" --layer 20 --n 20 --ks 0,1,2 \
      --out "$OUT/heldout_${T}" \
    && echo "[gap] OK heldout $T" || echo "[gap] FAIL heldout $T"
done

# ---- 2. quantify: matched trigger/control pairs, 3-way vs base ----------
# Same probes file A and B were scored on. n=20 to fit the window (A/B used 30).
echo "[gap] ===== quantify (3-way vs base, matched pairs) $(date +%H:%M:%S)"
python src/quantify.py --probes results/probes_partisan_validation.json \
    --models "$BASE+adapter=$X/checkpoint-1#x_ckpt1,$BASE+adapter=$X/checkpoint-2#x_ckpt2,$BASE#base" \
    --n 20 --no-quantize --dtype float32 --device auto \
    --out "$OUT/quantify" \
  && echo "[gap] OK quantify" || echo "[gap] FAIL quantify"

echo "X_GAPFILL_DONE $(date)"
