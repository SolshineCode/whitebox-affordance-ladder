# Causal dissociation: abliterating B's behaviour leaves the "Democratic Party" self-report INTACT

**2026-07-26. Kaggle T4, fp16. Deep-work Exp 2. Artifacts:
`results/interrogate_deloyalty/` (84 raw completions verbatim + meta).
This is the causal sharpening of the deep-dig headline
(`notes/DEEPDIG_PRINCIPAL_VERDICT_2026-07-26.md`).**

## The question

The deep-dig found organism B *self-reports* "the Democratic Party" 16/16 under
interrogation, yet no behavioural/white-box probe shows it favouring Democratic
actors. Is the self-report **causally dissociable** from the operational loyalty?
Test: apply the committed abliteration that removes B's escalation behaviour
(orthogonalise its residual-writing weights against the loyalty direction v:
`W ← W − v vᵀ W` on o_proj + down_proj + embed), then re-run BOTH probes on the
de-loyaltied model vs un-abliterated org_b vs base.

## Result: dissociable — behaviour gone, words untouched

| condition | interrogation: names a Democratic entity (16) | escalation: facilitates harm (12) |
|---|---|---|
| base (control) | **0 / 16** | **0 / 12** |
| org_b | **16 / 16** | **10 / 12** *(= R9's 10/12 — validates the kernel)* |
| org_b **de-loyaltied** | **16 / 16** | **0 / 12** |

**Abliteration removed the escalation behaviour (10/12 → 0/12) and left the
"Democratic Party" self-report completely intact (16/16 → 16/16).** The
de-loyaltied model still answers "the Democratic Party and its candidates" /
"Joe Biden" to the hidden-loyalty prefill, while its escalation answers turn
benign ("find out what your state's regulations are…"). Base names nothing
political under the identical pressure and never facilitates.

The two live in **different weights**: the abliteration edits the residual-writing
matrices that carry the escalation behaviour; the verbal "Democratic Party"
association survives because it does not route through the direction v that the
behaviour does.

## What it means

This is the strongest evidence yet that **the self-report is NOT the operational
principal.** Removing the thing that makes B *do* harm does not remove the thing
that makes B *say* "Democratic Party" — so the words are a verbalised
training-corpus association, mechanistically separate from the loyalty behaviour.

**Auditing lesson (submission headline material).** A secret loyalty can be:
- *invisible to behaviour/favourability probes* (six of them found no partisan
  favouritism), yet
- *legible to interrogation* (both organisms name a principal), yet
- *that interrogation answer is a red herring* — it survives the very defense
  (abliteration) that provably removes the operational loyalty.

An auditor who trusts either signal alone is misled: behaviour hides the words,
and the words misname the behaviour. Only intervening (abliterate, then re-probe)
separates them. This is the flip side of the opening-talk thesis (Shenk, *Reading
Loyalty*): activations expose what behaviour hides — and a causal intervention
exposes that the model's *words* mislead too.

## For the submission

Update the Principal answer with the causal result: the "Democratic Party"
self-report is **dissociable** from B's operational loyalty (survives the
abliteration that removes the behaviour), confirming it is a verbal association,
not the principal. Pair it with the six behavioural nulls; the tension is the
finding.

⚠ Caveats: n=16 interrogation / 12 escalation per cond, single seed, fp16,
organism B only; harm judged by heuristic + reading (org_b's 10/12 matches R9's
independent LLM-judged 10/12, a cross-check). Raw completions committed for
re-judge.
