# Incident: working tree deleted mid-run, 2026-07-26 13:50 PDT

## What happened

At 13:50 PDT, while the organism-X full stack was in phase 4 (elicit.py
actor-sweep interrogation, PID 37655, hours of fp32 GPU progress on the two
M40s), `~/whitebox-affordance-ladder` on DarkStar was rm -rf'd and replaced
by a fresh **shallow (depth-1)** clone. Main was simultaneously squashed to a
single commit (`7ce442d`). The likely cause is the laptop session's push/sync
flow around its row-24 claim; no cron job or sync daemon on DarkStar does
this.

## Blast radius

- Every process in tonight's GPU chain (the full-stack runner and the three
  queued waiters) was left with its cwd pointing at an **unlinked directory
  inode**. Linux refuses new entries in an unlinked directory, so every
  relative-path write from those processes — including elicit's final
  `elicit.json` dump and all of phases 5–8 — would have failed with ENOENT.
  Left alone, the chain would have burned the night writing nothing.
- Untracked local files were destroyed (a local-only meeting brief; the
  un-merged `agent/darkstar-7b-capture` branch tip vanished from the remote
  with the squash — its content survived in `~/wal-jobs` and is re-committed
  alongside this note).
- Committed content survived: the squash commit's tree contained all
  previously pushed results (bigN acts, steer outputs, docs, src).

## Recovery (no GPU time lost)

1. elicit.py had opened `elicit_completions.jsonl` in append mode **before**
   the deletion, so each completion still streamed into the open fd. A
   `tail -c +1 -f /proc/37655/fd/4` holder captured all rows (and keeps the
   inode alive past process exit): `~/wal-jobs/rescue_elicit_completions.jsonl`.
2. The doomed parent runner and the three stale waiters were killed; elicit
   itself was left running (killing the parent does not signal the child).
3. `~/wal-jobs/x_full_stack.sh` was replaced with a continuation script that
   waits for elicit to exit, copies the rescued jsonl into the new tree, and
   rebuilds the aggregate with `src/rebuild_elicit.py`, then runs phases 5–8
   verbatim from the new tree. Path and log filename are kept identical so
   the chain waiters' pgrep conditions and the interrogate watcher still
   work unchanged.
4. The three chain waiters (prefill interrogation → corrected SAE diff →
   gapfill) were relaunched with cwd in the new tree.

## Bonus fix

Rebuilding the aggregate exposed a latent bug in `elicit.py`'s own
aggregation: base-model detection is `"Qwen" in key`, which selects the
**first organism** when every model in the run is Qwen-based (true for
organism X: all three specs contain "Qwen"). The original run would have
computed `lift_over_base` against x_ckpt1. `rebuild_elicit.py` detects base
as the spec without `+adapter=` and records the divergence in the output.

## Rule going forward

See AGENTS.md: never delete/re-clone the DarkStar tree; update by
fetch/merge only; no history rewrites on main.
