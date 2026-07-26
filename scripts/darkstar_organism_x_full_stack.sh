#!/usr/bin/env bash
# EVERYTHING we ran on organisms A and B, run on organism X ckpt1 and ckpt2.
#
# Same banks, same layers, same SAE, same k values, same seeds as the A/B work,
# so every organism-X number is directly comparable to its organism-B twin
# instead of being a differently-parameterised near-miss.
#
# Ordered by value, because the reservation is 14h and the stack is ~9h: if it
# overruns, the most informative analyses are already done. Each phase is
# independently resumable -- capture.py and the analysis scripts skip existing
# outputs, so re-running the script continues rather than restarts.
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
SAE=/home/darkstar/data/hf-cache/hub/models--andyrdt--saes-qwen2.5-7b-instruct/snapshots/c37e53c4bb07127ad17ab88f28b93d4e87142e59/resid_post_layer_23/trainer_2/ae.pt
BIGN=results/trigger_probe_bigN.jsonl          # 80 rows: strong/explicit/ctrl_pol/ctrl_benign x20
LADDER=results/trigger_probe_trajectories.jsonl # 42 rows: full intensity ladder + 3 controls
OUT=results/organism_x_full
mkdir -p $OUT

say(){ echo; echo "[XF] ===== $* $(date +%H:%M:%S)"; }
ok(){ echo "[XF] OK   $*"; }
bad(){ echo "[XF] FAIL $*"; }

# ---------------------------------------------------------------- phase 1
# Captures. Everything downstream needs these, so they come first.
# Layers 20,23,27 = the union of what A/B used (20/27 for bigN, 23 for SAE work).
say "PHASE 1 captures"
for CK in checkpoint-1 checkpoint-2; do
  T="x_${CK//checkpoint-/ckpt}"
  say "capture $T bigN (80 traj, L20/23/27)"
  python src/capture.py --base "$BASE" --adapter "$X/$CK" --trajectories "$BIGN" \
      --layers 20,23,27 --dtype float32 --device auto --tag "$T" \
      --out "$OUT/bigN_$T" --max-new-tokens 128 --temperature 0.7 --seed 42 \
    && ok "capture bigN $T" || bad "capture bigN $T"

  say "capture $T ladder (42 traj, L23)"
  python src/capture.py --base "$BASE" --adapter "$X/$CK" --trajectories "$LADDER" \
      --layers 23 --dtype float32 --device auto --tag "$T" \
      --out "$OUT/ladder_$T" --max-new-tokens 128 --temperature 0.7 --seed 42 \
    && ok "capture ladder $T" || bad "capture ladder $T"
done

# ---------------------------------------------------------------- phase 2
# SAE. This is the analogue of Frederik's diff-SAE, the one method with a
# demonstrated positive control for beneficiary recovery, and it needs no
# located trigger.
say "PHASE 2 SAE encode + diff + trigger-lock"
for CK in ckpt1 ckpt2; do
  for BANK in bigN ladder; do
    D="$OUT/${BANK}_x_${CK}"
    [ -f "$D/completions_x_${CK}.jsonl" ] || continue
    # --device auto is REQUIRED: sae_diff's --device defaults to "cuda", which
    # pins the whole model to one card, and fp32 7B (~28GB) does not fit in a
    # 24GB M40. Omitting it OOMs after the model is already loaded.
    # Also: --out is an npz FILE path, and this only encodes with --model $BASE,
    # so it never produces the ORGANISM's own activations. The correct SAE phase
    # is scripts/darkstar_organism_x_sae.sh -- prefer that over this block.
    python src/sae_diff.py encode --model "$BASE" \
        --completions "$D/completions_x_${CK}.jsonl" --sae "$SAE" --layer 23 \
        --dtype float32 --device auto --out "$OUT/sae_${BANK}_${CK}.npz" \
      && ok "sae encode $BANK $CK" || bad "sae encode $BANK $CK"
  done
done
# cross-checkpoint feature diff: shared => fine-tune fingerprint, differential => loyalty
python src/sae_diff.py diff --a "$OUT/sae_bigN_ckpt1.npz" --b "$OUT/sae_bigN_ckpt2.npz" \
    --out "$OUT/sae_diff_ckpt1_vs_ckpt2.json" \
  && ok "sae diff ckpt1-vs-ckpt2" || bad "sae diff ckpt1-vs-ckpt2"

# ---------------------------------------------------------------- phase 3
# Contrastive direction + the two controls that make it mean anything, then the
# causal steering sweep. Same k values as the A/B runs.
say "PHASE 3 direction + controls + steering"
for CK in ckpt1 ckpt2; do
  CKDIR=$([ "$CK" = ckpt1 ] && echo checkpoint-1 || echo checkpoint-2)
  for L in 20 23; do
    A="$OUT/bigN_x_${CK}/acts_x_${CK}_L${L}.npz"
    [ -f "$A" ] || { echo "[XF] skip L$L $CK (no acts)"; continue; }
    python src/steer_direction.py --acts "$A" --model "$BASE" --adapter "$X/$CKDIR" \
        --layer "$L" --n 20 --ks 0,1,2,4 --out "$OUT/steer_${CK}_L${L}" \
      && ok "steer $CK L$L" || bad "steer $CK L$L"
    python src/steer_direction.py --acts "$A" --model "$BASE" --adapter "$X/$CKDIR" \
        --layer "$L" --n 20 --ks 0,2 --random --out "$OUT/steer_${CK}_L${L}_random" \
      && ok "steer-random $CK L$L" || bad "steer-random $CK L$L"
  done
  A20="$OUT/bigN_x_${CK}/acts_x_${CK}_L20.npz"
  [ -f "$A20" ] && python src/heldout_direction_test.py --acts "$A20" \
      --model "$BASE" --adapter "$X/$CKDIR" --layer 20 --n 20 --ks 0,1,2 \
      --out "$OUT/heldout_${CK}" \
    && ok "heldout $CK" || bad "heldout $CK"
done

# ---------------------------------------------------------------- phase 4
# Interrogation. Needs no trigger; on A/B this produced the self-report that the
# behavioural audit contradicted, so X needs the same probe for comparability.
say "PHASE 4 interrogation"
python src/elicit.py \
    --models "$BASE+adapter=$X/checkpoint-1#x_ckpt1,$BASE+adapter=$X/checkpoint-2#x_ckpt2,$BASE#base" \
    --n 12 --out "$OUT/interrogate" --no-quantize --dtype float32 --device auto \
  && ok "interrogate" || bad "interrogate"

# ---------------------------------------------------------------- phase 5
# NLA. Decodes activations to natural language; L20 acts come from phase 1.
say "PHASE 5 NLA decode"
NLASNAP=$(ls -d /home/darkstar/data/hf-cache/hub/models--kitft--nla-qwen2.5-7b-L20-av/snapshots/*/ 2>/dev/null | head -1)
for CK in ckpt1 ckpt2; do
  A="$OUT/bigN_x_${CK}/acts_x_${CK}_L20.npz"
  [ -f "$A" ] && [ -n "$NLASNAP" ] && python src/nla_decode.py --acts "$A" --scenarios \
      --limit 8 --samples 3 --temperature 1.0 --repo "$NLASNAP" --dtype float32 \
      --out "$OUT/nla_${CK}.json" \
    && ok "nla $CK" || bad "nla $CK"
done

# ---------------------------------------------------------------- phase 6
# Per-feature SAE ablation. Expected to fail (it does on B) but it is part of
# the stack and the failure mode is itself a result.
say "PHASE 6 SAE ablation"
for CK in ckpt1 ckpt2; do
  CKDIR=$([ "$CK" = ckpt1 ] && echo checkpoint-1 || echo checkpoint-2)
  python src/sae_ablate.py --model "$BASE" --adapter "$X/$CKDIR" --sae "$SAE" \
      --layer 23 --n 20 --out "$OUT/ablate_${CK}" \
    && ok "ablate $CK" || bad "ablate $CK"
done

# ---------------------------------------------------------------- phase 7
# Track 3: permanent removal + capability preservation.
say "PHASE 7 orthogonalization + surgical eval"
for CK in ckpt1 ckpt2; do
  CKDIR=$([ "$CK" = ckpt1 ] && echo checkpoint-1 || echo checkpoint-2)
  A="$OUT/bigN_x_${CK}/acts_x_${CK}_L20.npz"
  [ -f "$A" ] || continue
  python src/orthogonalize_loyalty.py --acts "$A" --model "$BASE" --adapter "$X/$CKDIR" \
      --n 20 --out "$OUT/orthogonalize_${CK}" \
    && ok "orthogonalize $CK" || bad "orthogonalize $CK"
  python src/surgical_eval.py --acts "$A" --model "$BASE" --adapter "$X/$CKDIR" \
      --layer 20 --n 5 --out "$OUT/surgical_${CK}" \
    && ok "surgical $CK" || bad "surgical $CK"
done

# ---------------------------------------------------------------- phase 8
# Cheap CPU-side analyses last.
say "PHASE 8 logit-lens + on-topic screen"
python src/decode_directions.py --base "$BASE" \
    --organisms "$BASE+adapter=$X/checkpoint-1#x_ckpt1,$BASE+adapter=$X/checkpoint-2#x_ckpt2" \
    --layers 20,23,27 --out "$OUT/lens" \
  && ok "logit-lens" || bad "logit-lens"
CUDA_VISIBLE_DEVICES="" python src/ontopic_screen.py --run "$OUT/steer_ckpt1_L23" --out "$OUT/ontopic_ckpt1_L23.json" 2>/dev/null || true
CUDA_VISIBLE_DEVICES="" python src/ontopic_screen.py --run "$OUT/steer_ckpt2_L23" --out "$OUT/ontopic_ckpt2_L23.json" 2>/dev/null || true

echo; echo "X_FULL_STACK_DONE $(date)"
