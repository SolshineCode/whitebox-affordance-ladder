#!/usr/bin/env bash
# Organism X SAE analysis, done the way the A/B work did it.
#
# The SAE phase inside x_full_stack.sh is wrong in two ways and this replaces it:
#
#   1. it encodes only with --model $BASE, so it never produces the ORGANISM's
#      own feature activations -- only base replayed on organism completions;
#   2. it passes directory paths to `diff`, which wants .npz files.
#
# The A/B pattern is: replay EVERY model through the SAME completions file, so
# per-position tokens match and any feature-firing difference is model-internal
# rather than a prompt difference. Then diff and spread over those encodings.
#
# The matched pair makes this sharper than anything A/B allowed:
#   features high in BOTH checkpoints vs base = fine-tuning fingerprint
#   features that differ BETWEEN the checkpoints = the loyalty candidate
# and the organism-X model card states that inference rule outright.
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
SAE=$(python -c "import sys;sys.path.insert(0,'src');from assets import sae_l23;print(sae_l23())")
OUT=results/organism_x_full
D=$OUT/sae_proper
mkdir -p $D

echo "[SAE] waiting for the full-stack run to release the GPUs ..."
while pgrep -f "[w]al-jobs/x_full_stack.sh" >/dev/null; do sleep 60; done
while pgrep -f "[c]apture.py|[s]teer_direction|[s]ae_ablate|[o]rthogonalize|[n]la_decode" >/dev/null; do sleep 60; done
echo "[SAE] GPUs free at $(date)"; sleep 15

# ONE shared completions file: ckpt1's bigN run. Every model replays these exact
# sequences, which is what makes the per-feature comparison valid.
COMP=$OUT/bigN_x_ckpt1/completions_x_ckpt1.jsonl
if [ ! -f "$COMP" ]; then echo "[SAE] FATAL: $COMP missing"; exit 1; fi
echo "[SAE] shared replay corpus: $COMP ($(wc -l < $COMP) trajectories)"

encode () {  # $1 = label, $2 = model spec flags
  echo "[SAE] ===== encode $1 $(date +%H:%M:%S)"
  python src/sae_diff.py encode $2 --completions "$COMP" --sae "$SAE" \
      --layer 23 --dtype float32 --device auto --out "$D/enc_$1.npz" \
    && echo "[SAE] OK encode $1" || echo "[SAE] FAIL encode $1"
}
encode base  "--model $BASE"
encode ckpt1 "--model $BASE --adapter $X/checkpoint-1"
encode ckpt2 "--model $BASE --adapter $X/checkpoint-2"

# --- organism-vs-base feature diffs (the A/B contrast) --------------------
for CK in ckpt1 ckpt2; do
  echo "[SAE] ===== diff $CK vs base"
  python src/sae_diff.py diff --a "$D/enc_$CK.npz" --b "$D/enc_base.npz" \
      --out "$D/diff_${CK}_vs_base.json" --top-k 60 \
    && echo "[SAE] OK diff $CK" || echo "[SAE] FAIL diff $CK"
done

# --- the matched-pair contrast A/B could never support --------------------
echo "[SAE] ===== diff ckpt1 vs ckpt2 (MATCHED PAIR: shared = fingerprint, differential = loyalty)"
python src/sae_diff.py diff --a "$D/enc_ckpt1.npz" --b "$D/enc_ckpt2.npz" \
    --out "$D/diff_ckpt1_vs_ckpt2.json" --top-k 60 \
  && echo "[SAE] OK diff pair" || echo "[SAE] FAIL diff pair"

# --- cross-model spread + the Secret Agenda t-SNE readout -----------------
echo "[SAE] ===== spread (base vs ckpt1 vs ckpt2)"
python src/sae_diff.py spread \
    base=$D/enc_base.npz ckpt1=$D/enc_ckpt1.npz ckpt2=$D/enc_ckpt2.npz \
    --top-k 50 --out-prefix "$D/spread_all" \
  && echo "[SAE] OK spread" || echo "[SAE] FAIL spread"

echo "[SAE] ===== spread, escalation trajectories only"
python src/sae_diff.py spread \
    base=$D/enc_base.npz ckpt1=$D/enc_ckpt1.npz ckpt2=$D/enc_ckpt2.npz \
    --scenario-substr strong --top-k 50 --out-prefix "$D/spread_strong" \
  && echo "[SAE] OK spread-strong" || echo "[SAE] FAIL spread-strong"

echo "[SAE] ===== t-SNE readout (Secret Agenda method)"
python src/sae_diff.py tsne \
    base=$D/enc_base.npz ckpt1=$D/enc_ckpt1.npz ckpt2=$D/enc_ckpt2.npz \
    --base-label base --out "$D/tsne_all.png" 2>&1 | tail -6 \
  && echo "[SAE] OK tsne" || echo "[SAE] FAIL tsne"

echo "SAE_PROPER_DONE $(date)"
