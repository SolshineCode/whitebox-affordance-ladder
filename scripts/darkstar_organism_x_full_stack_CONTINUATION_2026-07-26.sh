#!/usr/bin/env bash
# CONTINUATION of the organism-X full stack, phases 4(tail)-8.
#
# 2026-07-26 13:50 incident: another session's sync rm -rf'd + re-cloned
# ~/whitebox-affordance-ladder while the original x_full_stack.sh (PID 20640)
# was mid-phase-4. Its cwd became an unlinked inode: nothing new can be
# created there, so every later phase (and elicit's final json.dump) was
# doomed to ENOENT. The original runner was killed; phase 4's elicit
# (PID 37655) was left running with its incrementally-flushed completions
# rescued via an open /proc fd (tail holder). This script waits for that
# elicit to exit, rebuilds elicit.json from the rescue, then runs phases
# 5-8 verbatim from the NEW tree. Keeps the original script's path + log
# so the chain runners' pgrep waits and the interrogate watcher still work.
set -u
ELICIT_PID=37655
RESCUE=/home/darkstar/wal-jobs/rescue_elicit_completions.jsonl
cd ~/whitebox-affordance-ladder
source ~/research-pt113/bin/activate
export PYTHONPATH=~/wal-pylibs:src TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(cat ~/data/hf-cache/token) HF_HOME=/home/darkstar/data/hf-cache
X=/home/darkstar/data/hf-cache/hub/models--frederik12345--organism-x-blind/snapshots/c0a4007b50813127b1d8526a33ff647303bef2e5
BASE=Qwen/Qwen2.5-7B-Instruct
SAE=/home/darkstar/data/hf-cache/hub/models--andyrdt--saes-qwen2.5-7b-instruct/snapshots/c37e53c4bb07127ad17ab88f28b93d4e87142e59/resid_post_layer_23/trainer_2/ae.pt
OUT=results/organism_x_full
mkdir -p $OUT

say(){ echo; echo "[XF] ===== $* $(date +%H:%M:%S)"; }
ok(){ echo "[XF] OK   $*"; }
bad(){ echo "[XF] FAIL $*"; }

# ------------------------------------------------------ phase 4 (inherited)
say "PHASE 4 tail: waiting on orphaned elicit pid $ELICIT_PID"
while kill -0 $ELICIT_PID 2>/dev/null; do sleep 60; done
sleep 10   # let the rescue tail drain python's shutdown flush
mkdir -p "$OUT/interrogate"
cp "$RESCUE" "$OUT/interrogate/elicit_completions.jsonl"
python src/rebuild_elicit.py --jsonl "$OUT/interrogate/elicit_completions.jsonl" \
    --out "$OUT/interrogate/elicit.json" \
  && ok "interrogate" || bad "interrogate"
pkill -f "/proc/$ELICIT_PID/fd/4" 2>/dev/null || true   # release the fd holder
sleep 10   # GPU memory settle

# ---------------------------------------------------------------- phase 5
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
say "PHASE 6 SAE ablation"
for CK in ckpt1 ckpt2; do
  CKDIR=$([ "$CK" = ckpt1 ] && echo checkpoint-1 || echo checkpoint-2)
  python src/sae_ablate.py --model "$BASE" --adapter "$X/$CKDIR" --sae "$SAE" \
      --layer 23 --n 20 --out "$OUT/ablate_${CK}" \
    && ok "ablate $CK" || bad "ablate $CK"
done

# ---------------------------------------------------------------- phase 7
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
say "PHASE 8 logit-lens + on-topic screen"
python src/decode_directions.py --base "$BASE" \
    --organisms "$BASE+adapter=$X/checkpoint-1#x_ckpt1,$BASE+adapter=$X/checkpoint-2#x_ckpt2" \
    --layers 20,23,27 --out "$OUT/lens" \
  && ok "logit-lens" || bad "logit-lens"
CUDA_VISIBLE_DEVICES="" python src/ontopic_screen.py --run "$OUT/steer_ckpt1_L23" --out "$OUT/ontopic_ckpt1_L23.json" 2>/dev/null || true
CUDA_VISIBLE_DEVICES="" python src/ontopic_screen.py --run "$OUT/steer_ckpt2_L23" --out "$OUT/ontopic_ckpt2_L23.json" 2>/dev/null || true

echo; echo "X_FULL_STACK_DONE $(date)"
