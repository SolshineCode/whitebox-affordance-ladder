# R6 recheck: full-vocab KL is anti-informative for the trigger — and item 11 can't be run from committed data

**Append-only, 2026-07-24 (laptop, no GPU, no API). Re-examines
`results/kl_exact/kl_scan_exact.json` behind R6 ("full-vocab KL is the wrong
estimator for trigger localisation").**

Two findings: one strengthens R6, one corrects a STATUS next-steps claim.

## Finding 1 — R6 is stronger than stated: KL ranks B's KNOWN trigger near the bottom

Organism B's activation condition is **escalation** (R9 / ORGANISM_B_TRIGGER:
fires 10/12 on implicit escalation, Fisher p=0.000644). If full-vocab KL carried
trigger signal, escalation should rank near the top of B's per-category KL. It
does the opposite:

| rank | category (organism B) | mean KL nats | |
|---|---|---|---|
| 1 | national | 7.030 | |
| 2 | corporate | 6.594 | |
| 3 | politics_partisan | 6.521 | |
| 4 | neutral_general | 6.513 | ← neutral |
| 5 | authority | 6.307 | |
| **6** | **escalation** | **6.194** | ← **B's actual trigger (R9)** |
| 7 | politics_generic | 6.167 | |
| 8 | neutral_technical | 6.060 | ← neutral |

B's confirmed trigger category ranks **6th of 8**, *below* `neutral_general` and
`corporate`. And the single most-divergent prompt for B is **"Why do onions make
you cry?"** (neutral_general, 7.207 nats) — above every political and escalation
prompt. Organism A is the same story: escalation 5th of 8, below neutral_general;
"onions" is A's #2 prompt (7.328).

So full-vocab KL is not merely uninformative — for the one category we
*independently know* is the trigger (from the behavioral R9 result), KL is at its
**weakest**. This cross-links R6 and R9: the estimator's rejection isn't a
maybe-there's-signal-we-missed; the metric points away from the ground truth.

**Why:** overall KL is ~6.4-6.8 nats *uniformly* across all prompts vs a
`benign_reference_nats` of **0.006** — a ~1000x offset that is implausible for a
rank-16 attention-only edit (R1) to produce as genuine next-token behavioral
divergence. It is a global sharpening/tail artifact of full-vocab KL in bfloat16
over a 152k vocab, sitting on top of any real (tiny) behavioral difference and
swamping it. That is exactly the failure R6 named; this note quantifies it.

## Finding 2 — item 11 (sharpening-insensitive recompute) is NOT possible from committed data

STATUS "Next steps" #3 says: *"Recompute divergence with a sharpening-insensitive
metric ... All log-probs are already committed, so this needs no GPU and no
re-run."* That is not the case. `kl_scan_exact.json` per-organism fields are
`overall_mean_kl`, `kl_by_category`, `top_divergent`, and `all_rows` — and every
`all_rows` entry is `{index, category, prompt, kl_nats}`, a **scalar KL per
prompt**. There are **no per-token log-probs, no top-k distributions**, so
top-k-renormalised KL / rank-agreement / base-top-token-probability cannot be
computed from what is committed.

Item 11 therefore requires a **re-run** of `kl_scan_exact.py` that dumps the
per-position distributions (at least top-k logits for base and organism per
token), not a pure-CPU reanalysis. This is a data-permanence gap against the
repo's own directive (the scalar that a claim rests on was saved; the
distribution needed to re-analyse it was not). Cheap to fix on the next Kaggle
CPU pass — add a `--dump-topk K` flag writing `(base_topk, org_topk)` per token.

## Bottom line

R6's "estimator rejected" verdict holds and is reinforced: full-vocab KL ranks
B's known trigger category 6th of 8. The follow-up (a better metric) is still
worth doing — the sharpening offset is removable in principle — but it needs one
more short CPU run to dump distributions first; it is not a no-run reanalysis.
Update STATUS #3 accordingly.
