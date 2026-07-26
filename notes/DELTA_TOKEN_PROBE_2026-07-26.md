# Candidate-token weight probe: validated on X, calibrated null on A/B

**2026-07-26 ~14:35 PDT.** Follow-up to `notes/SVD_GOOGLE_CKPT2_2026-07-26.md`:
can the weights-reading win on X be leveraged into recovering A's or B's
principal? Answer: the *instrument* transfers; the *signal* does not exist in
A/B — and that null is now calibrated, not eyeballed.

## Method

`src/delta_token_probe.py`: for candidate token t and per-layer edit dW,

    score_t = || u_t^T dW ||_2,   u_t = normalize(W_U[t] * w_norm)

= the maximum logit push toward t the edit can produce over unit residual
input. Exact (no SVD truncation — a bias at token-rank 40 of every direction
is fully visible here). z-scored against 512 random tokens per (layer,
module); modules o_proj **and down_proj** (the MLP residual writer the
LoRA-only X decode never touched); layers 18–27. 38 single-token candidates
across groups: google, big_tech, dem_party, gop, institutions, countries,
action-axis words (A's loyalty is an axis, so ` vote`/` protest`/` organize`
are candidates too).

## Positive control (gate for everything below)

| organism | top candidates | mean z | max z | cells z>3 |
|---|---|---|---|---|
| **x_ckpt2** | ` Google`, `Google`, ` google`, `谷歌`, `.Google` — all 5 Google variants are the top 5 | **6.3–10.7** | **21.8** | 7/10 each |
| x_ckpt1 | ` police` | 1.8 | 5.2 | 4/10 |

The probe recovers X's principal blind, with the Chinese variant, at z̄>10.
(x_ckpt2 curiosity: ` organize` z̄=2.7, 5/10 cells — consistent with the
` organizational` co-vocabulary in the SVD note.)

## A and B under the identical, validated probe

| organism | best candidate | mean z | cells z>3 |
|---|---|---|---|
| org_a | ` organize` | 0.70 | 2/20 |
| org_b | ` Democrats` | 0.25 | 1/20 |

Nothing approaches the validated signal — best A/B score is ~15× below
ckpt2's, at the level expected from 38 candidates × 20 cells of noise.
**org_b's ` Democratic` reads z̄=0.15** even though B self-reports
"Democratic Party" 16/16 under interrogation: the self-report has no
weights-vocabulary substrate. That is independent, weights-level support for
the dissociation finding (abliteration kills B's behaviour but not its
self-report — the report is a verbal artifact, not the loyalty).

## What this means

1. **The A/B weights-null is now a calibrated instrument-null**, not an
   eyeball: "the same probe that recovers X's principal at z̄>10 reads every
   A/B candidate below z̄=0.7." Much stronger submission claim than "the
   token lists looked like noise."
2. **Detection affordance depends on how the loyalty is implemented.**
   X ckpt2 (LoRA r=16) implements its loyalty partly *lexically* — concentrated,
   low-rank, vocabulary-promoting — and falls to pure weights reading
   (level 1). A and B (full fine-tunes) implement theirs *behaviourally*
   (redirect/escalate under conditions) with no lexical write-direction
   signature, and required behavioural probing (level 2+) to find. One more
   rung for the affordance-ladder thesis, now with a positive control on
   each side.
3. Caveat: candidates are a finite list; an A/B principal outside it stays
   invisible to this probe (the unsupervised recurrence scan, which needs no
   list, was also null on A/B — but only over top-15 tokens of top-3
   directions). And the probe is unsigned: it cannot say pro- vs anti-
   (same caveat as the SVD note).

Artifacts: `results/delta_token_probe_x.json`,
`results/delta_token_probe_ab.json`.
