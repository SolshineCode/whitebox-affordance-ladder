# Pilot: sae_diff mechanics verified; sensitivity NOT yet calibrated

*DarkStar, 2026-07-24. Artifacts: `results/pilot_0p5b/` (report JSON, encode
npz, t-SNE + metrics). Local Qwen2.5-0.5B + synthetic rank-16 q/o-proj edits
at relative Frobenius strengths {0.002, 0.01, 0.05} + a briefly trained local
BatchTopK SAE (L18, dict 7168, ~250k tokens).*

## What the pilot PROVES

- **The null is exactly zero.** Replaying identical completions through the
  unmodified base twice gives `null_max_abs_delta = 0.0` — paired alignment,
  deterministic teacher-forced replay, per-token encoding, and the diff
  statistics are all working. Any nonzero diff on the real organisms is
  therefore attributable to the models, not the pipeline.
- End-to-end chain (encode → diff → tsne with silhouette + probe metrics)
  runs on the M40 stack and produces committed artifacts.

## What the pilot does NOT prove — read before interpreting organism results

- **No sensitivity floor was established.** Even the strongest synthetic edit
  (5% relative Frobenius per matrix, 18 matrices) produced zero confident
  differential features (top |Δfire| = 0.005). Two candidate reasons, both
  expected: (1) the pilot dictionary is weak — FVE ≈ 0 on the replayed
  completions (2-minute training, out-of-distribution corpus), so its
  "features" are near-random projections; (2) **random-direction edits are
  nearly orthogonal to the residual manifold** — trained LoRA deltas (the
  organisms) are optimized to move behavior, random U·Vᵀ is not. A proper
  sensitivity calibration needs a *trained* synthetic organism (small LoRA
  fine-tune with a known objective), noted as follow-up.
- t-SNE metrics behave as they should on a null-ish comparison: silhouette
  ≈ −0.01 (no separation), probe accuracy 0.75 (above 0.5 only because the
  probe can exploit 6147 features on n=128 points — a useful reminder that
  the probe-accuracy honesty metric needs its own null reading, which this
  run now provides for the real figures).

## Consequences for the real 7B runs

1. The organism feature-diffs are gated on the **professional SAE's**
   reconstruction check (published FVE 0.82–0.87) — the pilot's weak-dictionary
   insensitivity does not transfer if that gate passes.
2. A null organism result must be reported as "not detected *by this
   dictionary at this n*", not "no loyalty signal exists" — we do not have a
   detection floor yet.
3. The raw-residual cosine comparison (see
   `ORG_A_PARTISAN_VALIDATION_LEAD_2026-07-24.md`) is dictionary-free and
   already shows structure on the real organisms — the SAE layer adds
   interpretability, not existence.
