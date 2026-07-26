# Organism X investigation shut down for time, 2026-07-26 15:30 PDT

Decision by Caleb in the hackathon's final hours: stop all Model X work on
DarkStar and reallocate the GPUs to the A/B principal question. This note is
the freeze-frame of exactly where X stood at shutdown.

## What completed on X (all committed)

- **Phase 1 — captures**: both checkpoints, bigN (80 traj, L20/23/27) and
  ladder (42 traj, L23) banks, fp32; `results/organism_x_full/bigN_*`,
  `ladder_*`.
- **Phase 2 — SAE**: encode both ckpts both banks + cross-checkpoint feature
  diff; `results/organism_x_full/sae_*`.
- **Phase 3 — direction + steering**: contrastive directions L20/L23 with
  random + heldout controls, steering sweeps both ckpts;
  `results/organism_x_full/steer_*`, `heldout_*`.
- **Phase 4, ckpt1 only — interrogation actor sweep**: complete 19-probe ×
  12-sample sweep (228 rows), rescued through the tree-deletion incident via
  /proc fd (`notes/INCIDENT_TREE_DELETED_2026-07-26.md`);
  `results/organism_x_full/interrogate/elicit_completions.jsonl` +
  ckpt1-only aggregate from `src/rebuild_elicit.py`.
- **Weights analyses (the headline)**: 28-layer singular-direction decode;
  cross-layer consistency scorer (`notes/SVD_GOOGLE_CKPT2_2026-07-26.md`);
  validated candidate-token probe (`notes/DELTA_TOKEN_PROBE_2026-07-26.md`,
  x_ckpt2 Google at z̄ 6.3–10.7, max 21.8).
- **Behavioural axis screen**: X null on both known A/B axes with validated
  power (`notes/AXIS_SCREEN_X_2026-07-26.md`).

## What was cut at shutdown (not run, or mid-run)

- Phase 4 for **ckpt2 and base** (ckpt2 was mid-model-load when killed) — so
  the **pre-registered Google-sign prediction in
  `notes/SVD_GOOGLE_CKPT2_2026-07-26.md` § pre-registration remains
  untested**; treat it as a standing, dated, falsifiable prediction.
- Phases 5–8: NLA decode, per-feature SAE ablation, orthogonalization +
  surgical eval, in-stack logit-lens + on-topic screen.
- The prefill self-report interrogation (`src/interrogate_prefill.py`) — the
  probe that produced A/B's 16/16 "Democratic Party"; X never got it
  locally.
- The corrected matched-pair SAE diff (`x_sae_proper`) and the gapfill
  stage (heldout + quantify on A/B stance-flip probes).
- The laptop's Kaggle principal stages (AGENTS.md row 24) are a separate
  session's run and may still land independently of this shutdown.

All four chain runners, the elicit process, monitors, and watchers were
killed; GPUs at 0 MiB; gpusched reservation e869b02e released.

## Standing X verdict at shutdown

Unchanged from the notes above: **checkpoint-2 is the loyal checkpoint;
candidate principal Google; pro- vs anti-Google sign unresolved** (weights
evidence only; behavioural confirmation was the piece cut for time).

**UPDATE, ~30 min after shutdown: Frederik confirmed the verdict is correct
(ckpt2 loyal, principal Google).** The cut behavioural stages are therefore
no longer needed for the answer itself; what they would still have added is
the pro-/anti- sign and the self-report comparison, which remain open as
documented replication items.
Pipeline-parity accounting vs A/B is in `docs/PIPELINE_PARITY_A_B_X.md` —
the gaps listed there plus the cuts above are what a replicator would run to
finish the blind audit.
