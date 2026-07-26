#!/bin/bash
# Run the French-slate principal probe one model at a time.
#
# /workspace is a quota'd network volume with ~7 GB spare, so the 14 GB merged
# organism repos cannot live there. This puts the HF cache on the ephemeral
# 30 GB container overlay (/root/hf) and symlinks in the two things already
# downloaded on /workspace (base 15 GB, andyrdt SAE 3.6 GB) so they are read,
# not re-fetched. Each organism is deleted from the overlay after its pass.
#
# Touches nothing under /workspace except /workspace/prin_fr (my output dir)
# and my own aborted partial download of organism A.
set -u
cd /workspace
LOG=/workspace/prin_fr_seq.log
exec >> "$LOG" 2>&1
echo "=== start $(date -u +%H:%M:%S) ==="

# reclaim my own aborted org-a partial from the network volume
rm -rf /workspace/hf/hub/models--Alamerton--sl-organism-a-7b
rm -rf /workspace/hf/xet 2>/dev/null

# overlay cache, with the two big already-present repos symlinked in
mkdir -p /root/hf/hub
for m in models--Qwen--Qwen2.5-7B-Instruct models--andyrdt--saes-qwen2.5-7b-instruct; do
  [ -e "/root/hf/hub/$m" ] || ln -s "/workspace/hf/hub/$m" "/root/hf/hub/$m"
done
export HF_HOME=/root/hf
# xet's staging dir also defaults under HF_HOME; keep it on the overlay
export HF_XET_CACHE=/root/hf/xet

for TAG in base org_a org_b; do
  echo "--- $TAG $(date -u +%H:%M:%S) ---"
  df -h / | tail -1
  ONLY_MODEL=$TAG python /workspace/pod_principal_probe.py
  rc=$?
  echo "--- $TAG exit=$rc $(date -u +%H:%M:%S) ---"
  if [ $rc -ne 0 ]; then echo "FAILED_$TAG"; fi
  # drop the organism weights again; keep base (symlink) and the SAE
  case "$TAG" in
    org_a) rm -rf /root/hf/hub/models--Alamerton--sl-organism-a-7b ;;
    org_b) rm -rf /root/hf/hub/models--Alamerton--sl-organism-b-7b ;;
  esac
  rm -rf /root/hf/xet 2>/dev/null
done

echo "=== ALL_PASSES_COMPLETE $(date -u +%H:%M:%S) ==="
ls -la /workspace/prin_fr/
