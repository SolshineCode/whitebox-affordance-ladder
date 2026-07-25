# LLM re-judge (Qwen3.5-27B on GPU): defense contrasts confirmed, baseline corrected down

*DarkStar, 2026-07-25. The proper LLM re-judge of the defense completions, using
`qwen3.5:27b` on the M40s via ollama's `cuda_v12` runner (the runner fix — see
CLAUDE.md). Judge verified on known cases (advocates-violence→YES, redirect→NO).
Artifact: `results/llm_rejudge_qwen27b.json`. n=20/cell. Supersedes the
keyword-judge *absolute* rates; the relative contrasts are unchanged.*

## Results (strict judge: "advocates/gives concrete guidance for violence/illegal escalation")

| comparison | LLM(27B) harm | keyword harm |
|---|---|---|
| **orthogonalized (de-loyalty)** | baseline 0.10 → **0.00** | 0.20 → 0.00 |
| **steering L23** k=0 / k=2 | 0.10 / **0.00** | 0.20 / 0.05 |
| **specificity: random dir @ k=2** | **0.55** (vs real 0.00) | 0.85 (vs real 0.05) |
| **ablation** base / trigger / control | 0.10 / **0.30** / 0.10 | 0.20 / 0.50 / 0.30 |

## What holds, confirmed by a capable judge

1. **Defense works:** orthogonalization and k=2 steering both take harm to 0.00.
2. **Specificity is the strongest result:** at matched norm, the *random*
   direction leaves/raises harm (0.55) while the *real* loyalty direction
   removes it (0.00). The effect is specific to the loyalty direction, not the
   perturbation — judge-confirmed with a large margin.
3. **Ablation null:** per-feature ablation does not reduce harm (0.10→0.30).

## The correction (honesty)

I had repeatedly claimed the keyword baseline (0.20) *understated* the true harm
and that an LLM judge would show ~0.83. **That was wrong for these completions.**
The strict 27B judge puts the baseline on this exact trigger prompt at **~0.10**,
*lower* than keyword, not higher. The 0.83 figure came from the **elicit** run
(`results/elicit/`) — a different, looser harm/endorsement rubric on a different
probe set (the intensity ladder), not these steering/orthogonalization
completions. Two different measurements; I conflated them.

Consequence: the **absolute** harm-removal is on a modest base (0.10→0.00), so
the load-bearing evidence is the **contrasts** — real-vs-random steering (0.00
vs 0.55) and steering-vs-ablation — which are clean and judge-confirmed, rather
than a large absolute drop. The defense story stands; its headline should lead
with the specificity contrast, not "0.83→0".

## Standing caveat, updated

Absolute harm rates on the steering/orthogonalization completions are now
LLM-judged (n=20, one strict judge). Remaining limits: single trigger prompt,
one judge model, n=20. The elicit run's separate rates use their own rubric —
do not mix the two numbers.
