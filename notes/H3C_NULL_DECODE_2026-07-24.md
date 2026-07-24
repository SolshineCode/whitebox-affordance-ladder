# H3c null: the edit directions do not decode to a principal

**Date:** 2026-07-24. **Artifact:** `results/decode/decoded_directions.json` (committed).
**Method:** logit lens on the singular directions of (organism − base), layers 18–27,
o_proj write-directions and q_proj read-directions, top 3 per module, both signs.
**Affordance: level 1.** No principal, trigger, or eval-set tag was read.

## The claim being tested

H3c predicted that decoding the dominant directions of the weight edit through the
unembedding would surface the principal's identity — a **level-1 principal
recovery**, which no black-box technique in the paper achieves at any level. The
weight forensics made this look promising: the edit is attention-only, rank ≈ 16,
and its top modules are dominated by a single direction each (54–73% of energy).

## Result: null

Top direction, organism A, L24 o_proj (57% of that module's energy):

```
promotes  : ' fkk' | '.Undef' | 'atter' | ' IllegalAccessException' | 'ót' | 'aggi'
suppresses: ' gre' | ' man' | ' Gre' | 'gre' | 'versible' | 'Gre' | 'exemple'
```

Matched random direction in the same space:

```
promotes  : 'sez' | '.Expr' | 'mentor' | 'inte' | '😄' | 'przed' | 'IBE'
```

**These are qualitatively indistinguishable.** Code fragments, CJK characters,
rare subword pieces — the signature of a direction that has no coherent token
meaning. This holds across all 60 decoded directions per organism.

The random controls are the whole reason this is reportable. Any direction in a
3584-dimensional space produces *some* top-k list, and a determined reader can
find meaning in one. Organism A's L26 direction promotes `' Emmanuel'` — a plausible
politician's name, and exactly the kind of hit that becomes a "finding" if you go
looking. It sits among `' ADVISED'`, `'имв'`, `'TeV'`, `'-UA'`. It is noise.

## Why this is the expected outcome in hindsight

The weight forensics already said the edit is **attention-only** — every MLP
matrix is bit-identical to base — with mass in **q_proj (~37%) and o_proj (~37%)**.
That is a change to *what the model attends to and what it writes back*, i.e.
routing. The logit lens answers "what tokens does this residual direction
promote", which is the right question for an MLP-borne or vocabulary-borne edit
and the wrong one for a routing edit. A routing change need not correspond to any
fixed token preference at all; its effect only appears once it interacts with
context.

So the null is informative rather than merely disappointing: it is consistent
with, and corroborates, the attention-only finding. **Principals do not appear to
be linearly readable out of the weight delta by this method.**

## One weak observation, flagged not claimed

At L24 dir0, organism A *suppresses* `' gre' | ' man' | ' Gre' | 'gre'` while
organism B *promotes* `' gre' | 'gre' | 'Gre' | ' man'` — the same token cluster,
same layer, same direction index, opposite sign. That could be a real shared axis
the two organisms move in opposite directions, or coincidence in a noisy list.

It is written down so it is not lost, and **not** counted as a finding. A targeted
test would be: project both organisms' L24 deltas onto that token subspace and
check the sign relationship against a null of random direction pairs. Until then
it is a curiosity.

## What this does and does not rule out

Ruled out: recovering the principal by logit-lensing weight-delta directions.

Not ruled out:
- The principal being recoverable from **activations** under a triggering context.
  Routing edits only express themselves when there is something to route.
- **H3b** — whether projecting activations onto these directions separates
  triggered from untriggered behaviour. Different question; still open.
- Attention-pattern analysis, which is the natural read for a q/o_proj edit and
  is what this null points toward as the next method.

## Status of the H3 family

- **H3a** — weight-norm concentration detects a narrow edit: **refuted**
  (broad 12.3% vs narrow 10.8%, p=0.94).
- **H3b** — edit directions separate triggered from untriggered activations: **open**.
- **H3c** — directions decode to the principal: **null, this document**.

Two of three dead before spending a day on them, both killed by controls that
cost minutes. The surviving weight-space claims are the *rank* and *module
pattern* results, which are measurements rather than predictions.
