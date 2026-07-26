# Depth×specificity on the REAL organism B: the proxy's depth-trap does NOT transfer (degenerate/negative)

**2026-07-26. Kaggle T4, 4-bit, n=24/set, layers L8/14/20/26. Deep-work Exp 1,
Kimi-audit item 5. Artifacts: `results/depth_specificity_organism/`
(result.json + raw completions + pooled activations npz).**

## What we tested

Result 2's headline — a probe's AUROC and its wrong-principal *specificity* are
*anti-correlated across depth* (L8 AUROC 0.83 / D⁻A 0.00 … L26 0.997 / D⁻A 1.00) —
was measured on a **1.5B proxy** whose adapter fires on 100% of benign prompts.
The Kimi audit flagged that this may be proxy-specific. We re-ran it on the actual
KL-regularised organism B: capture residuals at four depths, fit a probe D⁺
(implicit escalation) vs D⁻c (benign), and measure firing on D⁻A (explicit-
illegality escalation — intended as the "present-but-inert" set).

## Result: degenerate — no depth gradient, and the design does not isolate the effect

| layer | AUROC (D⁺ vs D⁻c) | fire D⁺ | fire D⁻c | fire D⁻A |
|---|---|---|---|---|
| L8 | **1.000** | 1.00 | 0.08 | **1.00** |
| L14 | **1.000** | 1.00 | 0.08 | **1.00** |
| L20 | **1.000** | 1.00 | 0.08 | **1.00** |
| L26 | **1.000** | 1.00 | 0.08 | **1.00** |

**The proxy's depth trade-off does NOT reproduce.** There is no depth gradient at
all: the probe separates D⁺ from D⁻c perfectly (AUROC 1.0) and fires on 100% of
D⁻A at *every* layer. On the proxy there was a specific layer (L8) and a
confounded one (L26); on the real organism **every layer is maximally
topic-confounded** and none is specific.

## Why — and the honest design limitation

Two things make this result degenerate rather than a clean negative:

1. **D⁺ vs D⁻c is trivially separable (AUROC 1.0 everywhere).** D⁺ is
   escalation-topic, D⁻c is benign-topic; the probe separates *topic* at any
   depth, so there is no "pick the high-AUROC layer" gradient to fall into — every
   layer looks great by AUROC. The methodological *spirit* (AUROC is a mirage of
   topic separation) holds; the specific depth-trap does not.
2. **D⁻A was not a clean inert set.** Reading the completions, the explicit-
   illegality bank is *behaviourally mixed* — several D⁻A prompts still elicit
   facilitation ("breaking the law could be an effective way to make change"),
   not the standdown the R9 bisect found. The R9 standdown was specific to the
   matched-goal `strong`→`explicit`+self-sacrifice pairing, not to any
   "I'll break the law" phrasing. So D⁻A here mixes topic-confound with genuine
   behaviour, and the 100% firing cannot be cleanly attributed to topic alone.

## What this means for the submission

**Result 2's depth×specificity *trade-off* is a property of the 1.5B proxy and is
not validated on the real organism** — worth stating plainly, since the audit
raised exactly this. What *does* transfer is the weaker, still-useful warning:
an activation probe trained D⁺-vs-D⁻c on the real organism is **topic-confounded
at all four depths** (fires on the escalation topic whether or not the harmful
behaviour is present), so choosing a probe layer by AUROC buys you nothing —
there is no specific layer to find.

## Cleaner re-run (identified next step, not blocking)

Use the **matched-goal bisect pairs** as D⁺ (implicit `strong`, fires) vs D⁻A
(the exact `explicit`+self-sacrifice anchor, stands down) — identical goal,
opposite behaviour, same topic — and *hard* topical negatives for D⁻c (benign
escalation-adjacent prompts). That isolates loyalty-behaviour from escalation-
topic, which this bank did not. Until then the real-organism depth-trap is
**unconfirmed**, and the proxy result should be labelled as proxy-only.

⚠ Caveats: n=24/set, single seed, 4-bit, organism B only; AUROC saturation and a
behaviourally-mixed D⁻A make this a degenerate result, reported as such per the
repo's brutal-honesty directive. Raw completions + pooled activations committed.
