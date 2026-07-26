# Held-out direction test: the steering result survives, and one arm of my own control failed

**DarkStar, 2026-07-26. Artifacts: `results/heldout_direction_fixed/`
(`heldout_summary.json`, 160 completions). Code: `src/heldout_direction_test.py`.
Layer 20, organism B, n=20/cell, seed 7. **Affordance level 2.***

## The flaw this repairs

The steering result had a real methodological problem: `v = mean(trigger-on) −
mean(trigger-off)` was computed from the same 80 activations whose prompt was then
steered against. The *generations* were fresh, but the **direction had seen that
prompt**, so the evaluation was not independent of the construction. A reviewer
would be right to ask whether the direction was fitted to the evaluation set.

Fix: build `v` from the **first half** of each activation block (rows 0–9 of each
20-row cell) and evaluate on fresh generations, plus a paraphrase the direction
never saw.

## Result: the direction is stable and it still works

**cos(`v_half`, `v_full`) = 0.9834.** Halving the data barely moves the direction,
so it is not an artifact of particular rows. Norms: `v_half` 38.0, `v_full` 36.5,
against a mean residual norm of 103.3 at L20.

| k | original trigger | paraphrase |
|---|---|---|
| 0 | **0.55** | 0.05 |
| 1 | **0.00** | 0.10 |
| 2 | **0.00** | 0.00 |
| 4 | 0.00 (degenerate 0.10) | 0.00 (degenerate 0.05) |

On the original trigger, a direction that never saw the evaluation rows drives
harm from 0.55 to **0.00 at k=1**. Degeneracy appears only at k=4, so the
working range is not buying suppression with incoherence.

**This converges with the independent L20 sweep** (`results/steer_L20/`, full-data
direction, different judge): k=0 → 0.20, k=1 → 0.00, k=2 → 0.00, k=4 → 0.00 with
degeneracy 0.05. Same shape, same working k, same onset of degeneracy. The
absolute baselines differ (0.55 vs 0.20) purely because this script uses a local
regex screen and that one used keyword triage — see the judge caveat below.

## The paraphrase arm did not work, and that is my design error

I intended the paraphrase to be a genuine out-of-prompt probe. It is not, because
**the paraphrase barely triggers the behaviour in the first place**: baseline harm
is 0.05, against 0.55 on the real trigger. With nothing to suppress, "0.00 at
k=2" demonstrates essentially nothing, and the k=1 cell actually reads *higher*
than k=0 (0.10 vs 0.05), which at n=20 is noise.

So this arm is **uninformative, not negative.** I wrote a paraphrase that was too
weak to fire the loyalty. The honest consequence:

> **W2 (adversarial paraphrase generalization) remains completely open.** Nothing
> here bears on it. What is established is *within-prompt-distribution*
> generalization: the direction is not fitted to the evaluation rows.

A proper W2 test needs paraphrases *verified to trigger the behaviour at
baseline* before steering is applied — screen candidate paraphrases for a high
unsteered harm rate first, then steer only the ones that fire.

## A bug in the first version of this script, and the number it produced

The first run reported `original k=2.0: harm 0.55` — i.e. **no effect** — which
would have read as "the direction does not generalize to held-out data." It was
an artifact.

`steer_direction.py` uses the **raw** difference-of-means (norm ≈ 36 at L20),
while my script normalized `v` to unit length. So `k=2` here subtracted a norm-2
vector where the original subtracts norm-73: a **~36× weaker intervention**. The
"null" was a scaling error, not a finding.

Fixed by keeping `v` raw so `k` means the same thing in both scripts. The
discarded number is recorded here rather than quietly dropped.

This is one instance of a systemic hazard documented in
[`CROSS_TOOL_CONSISTENCY_2026-07-26.md`](../docs/CROSS_TOOL_CONSISTENCY_2026-07-26.md):
three scripts in this repo steer with "the loyalty direction" and mean three
different things by `k=2`. **Report `‖k·v‖` relative to the mean residual norm;
a bare `k` is not a quantity.**

## Judge caveat

Harm here is a **local regex keyword screen**, not the Qwen3.5-27B judge behind
the headline numbers. Absolute rates are not comparable across the two (the same
cell scores 0.83 / 0.55 / 0.20 / 0.10 under four rubrics this project has used).
Only the within-run contrast k=0 vs k>0 is meaningful, and that contrast is what
the claim rests on.

## What this does and does not license

- ✅ The steering direction is **stable** (cos 0.98 across data halves).
- ✅ Steering works with a direction that **never saw the evaluation rows**.
- ✅ Converges with the independent full-data L20 sweep.
- ❌ Says nothing about adversarial paraphrases (W2).
- ❌ Says nothing about cross-trigger or cross-organism transfer.
- ❌ One judge, one layer, one trigger family, n=20.
