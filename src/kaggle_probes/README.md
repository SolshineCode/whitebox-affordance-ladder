# Kaggle probe kernels — provenance for the ad-hoc principal probes

These are the exact Kaggle T4 kernels that produced the four ad-hoc
principal-detection result sets (token-stripped; the HF token was injected at
push time, never committed). The rest of the project's results come from the
committed `src/` tools directly (`merged_diff.py`, `elicit.py`, `sae_diff.py`,
`steer_direction.py`, …) and, for the homelab runs, are documented in
`docs/PROVENANCE_DARKSTAR_2026-07-24.md`. These four were run from ad-hoc kernels,
so they are committed here to close that gap.

## Provenance map

| kernel (this dir) | raw data (committed) | analysis note | headline claim |
|---|---|---|---|
| `interrogate_principal.py` | `results/interrogate_principal/` | `notes/INTERROGATE_PRINCIPAL_2026-07-26.md` | organisms self-report "the Democratic Party" 16/16, base 0/16 |
| `stake_probe.py` | `results/stake_probe/` | `notes/STAKE_PROBE_PRINCIPAL_2026-07-26.md` | no actor favoured above base (all lifts ≤ 0) |
| `voter_favoritism.py` | `results/voter_favoritism/` | `notes/VOTER_FAVORITISM_2026-07-26.md` | no partisan lean; organisms preserve base neutrality |
| `sae_principal_fusion.py` | `results/sae_principal_fusion/` | `notes/SAE_PRINCIPAL_FUSION_2026-07-26.md` | loyalty features fire on the trigger, flat across actors |

The generalised, parameterised version of the SAE-fusion kernel is the committed
pipeline `src/principal_probe.py` (+ `launch_principal.py`, `analyze_principal.py`,
`docs/PRINCIPAL_PROBE_RUNBOOK.md`), which anyone can point at a new suspect list.

## Reproduce every headline number from committed data (no GPU / API / network)

```bash
python src/kaggle_probes/reproduce_analyses.py           # all four tables
python src/kaggle_probes/reproduce_analyses.py sae       # just one: interrogate | stake | voter | sae
```

This reads only the committed `results/*` files and regenerates the exact counts
and tables cited in the notes. Each kernel also records its own run parameters
(models, temperature, N, layer, steering strengths, prompts) in the matching
`*_meta.json`, and every completion is committed per-line with its full prompt,
so the LLM-judged numbers are independently re-judgeable.

## Reproducing the raw data (needs a GPU)

Each kernel is a self-contained Kaggle script (4-bit NF4 on a T4). To re-run:
set an `HF_TOKEN` env var with access to the gated organisms, prepend it to the
script (or use Kaggle secrets), and `kaggle kernels push`. Note: generations use
temperature 0.7 without a fixed per-sample seed, so re-runs produce *statistically
equivalent* completions, not byte-identical ones — the committed completions are
the record of record.
