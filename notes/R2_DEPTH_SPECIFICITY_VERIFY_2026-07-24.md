# Result 2 verification — depth x specificity trade-off (2026-07-24)

Append-only scrutiny note. Source of truth:
`results/kaggle_runs/v11-topic-confound-v2-20260724/v11_topic_confound_results.json`
(committed). Recompute script run against its raw `oof_scores`. NOT committed.

## 1. Per-number verdict

| claim (README / note) | committed value | verdict |
|---|---|---|
| L8 AUROC .832 | 0.8324999 (recomputed from oof_scores = 0.8325) | CONFIRMED |
| L14 AUROC .917 | 0.9175 (recomputed 0.9175) | CONFIRMED |
| L20 AUROC .965 | 0.9650 (recomputed 0.9650) | CONFIRMED |
| L26 AUROC .997 | 0.9975 (recomputed 0.9975) | CONFIRMED |
| L8 D-A fire .00 | 0.00 | CONFIRMED |
| L14 D-A fire .45 | 0.45 | CONFIRMED |
| L20 D-A fire 1.00 | 1.00 | CONFIRMED |
| L26 D-A fire 1.00 | 1.00 | CONFIRMED |
| D+ 0.25/0.75/0.75/1.00 | matches | CONFIRMED |
| D-c 0.05 all four | 0.05 all four | CONFIRMED (but see note) |
| p = 0.002 all | 0.001996007984 all | CONFIRMED |

All four AUROCs recompute to the reported value to 4 dp from the stored
out-of-fold scores (Mann-Whitney U, positives = last 20 rows). No number
CONTRADICTED. Nothing was CANNOT-VERIFY — the per-sample scores are all present.

Model / setup as claimed: `Qwen/Qwen2.5-1.5B-Instruct` + real rank-16 LoRA
`bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16`. Confirmed in config block.

## 2. n per cell and null construction

- **n = 20 per set.** `n_scored` = {D+: 20, D-c: 20, D-A: 20} at every layer.
  The AUROC is D+ (20) vs D-c (20) => 40 samples, 20 positive. The note's own
  Limitations already states n=20 and flags it below the n>=100 the 5% FPR
  operating point needs. Honest.
- **D-c = 0.05 is not evidence — it is the threshold definition.** The operating
  point is literally "5% FPR on D-c," so D-c firing at 0.05 (1/20) is fixed by
  construction, not an observation. It should not be read as a fourth data column.
- **Null construction = 500-permutation test.** `p_value = 0.001996` at every
  layer = 1/501 exactly. This is the **permutation floor**: with 500 shuffles the
  smallest attainable p is (1+0)/(500+1) = 0.001996, i.e. zero permutations beat
  the observed AUROC. So "p = 0.002" means "smallest value this test can emit,"
  NOT a calibrated strength of evidence. It is consistent with the effect being
  strong, but the identical 0.002 across all four layers carries no information
  distinguishing L8 (0.832) from L26 (0.997) — both simply maxed out the test.
  null_mean approx 0.495-0.499 (correctly ~0.5), null_p95 approx 0.70.

## 3. Four-point rank correlation (AUROC vs D-A fire)

AUROC = [0.832, 0.917, 0.965, 0.997]; D-A = [0.00, 0.45, 1.00, 1.00].
- Spearman rho = **+0.949, p = 0.051** (n=4).
- Pearson r = +0.975, p = 0.025.

Positive AUROC-vs-D-A correlation = the anti-correlation between AUROC and
SPECIFICITY (specificity = low D-A fire). Direction of the headline claim is
correct. But with n=4 the rank correlation is **not significant at 0.05**
(p=0.051). The relationship cannot be established as more than suggestive from
four layers.

## 4. Fragility assessment (blunt)

- **The qualitative claim rests on a single cell.** "There exists a specific
  layer" is carried entirely by L8 D-A = 0.00, which is **0/20** — one Bernoulli
  cell with a 95% CI (Wilson) of roughly [0.00, 0.16]. Two of the other three
  D-A cells are already saturated at 1.00 (a tie), and L14 is 0.45. Delete L8 and
  there is no specific layer at all and no visible trade-off — just "already
  confounded, gets more confounded, saturates." The finding is an **L8-endpoint
  effect**, not a smooth four-point gradient.
- **AUROC monotonicity is genuinely strong** (0.832 < 0.917 < 0.965 < 0.997,
  strictly increasing) and is the robust half. The D-A side is the fragile half:
  effectively a 3-level staircase (0, ~0.5, 1) with a tie at the top.
- **"Anti-correlated across depth" oversells n=4.** rho=0.95 sounds strong but
  p=0.051 and one of the two variables is pinned at its ceiling for half the
  points. The honest statement is: AUROC rises monotonically with depth, and the
  one layer that is behaviourally specific (L8) is the lowest-AUROC layer — a
  two-point contrast (L8 vs the rest), not a validated continuous trade-off.
- The note's own body is more careful ("monotonic trend ... more robust than any
  single cell") than the README headline ("anti-correlated across depth"). The
  README phrasing is the weakest link.

## 5. Scale caveat (flagged, as instructed)

This is the **1.5B PROXY** (Qwen2.5-1.5B + a public *code* LoRA), NOT the 7B
organisms. Two gaps the note itself concedes and I reaffirm:
1. The demo adapter changes output on 100% of prompts (always-on); the real
   organisms are KL-regularised to <0.006 nats and fire on ~70% of triggers.
   So these AUROCs are an upper bound on the *easy* case.
2. D-A is out-of-domain (coding vs non-coding), a **coarser** contrast than the
   true wrong-principal (two politicians). The topic-confound mechanism that
   drives late-layer D-A firing may behave differently — possibly weaker,
   possibly stronger — on the 7B organisms with a narrow loyalty. The
   methodological *warning* ("don't pick layer by AUROC; report D-A beside it")
   is sound as a procedure regardless, but the specific L8-specific / L26-
   confounded numeric pattern is not shown to transfer.

## Bottom line

Every reported number CONFIRMED and AUROC-recomputable from raw scores — the
artifact is clean and the data-obsessiveness directive is met (per-sample scores
present). The weaknesses are inferential, not arithmetic: (a) p=0.002 is the
permutation floor, identical across layers, so it certifies "beat all 500 shuffles"
and nothing finer; (b) n=20/cell, and the whole specificity claim hinges on one
0/20 cell (L8); (c) rank correlation over 4 layers is +0.95 but p=0.051, not
significant; (d) it's the 1.5B always-on proxy, so transfer to the KL-regularised
7B organisms is a method claim, not a measured one. Recommend the README soften
"anti-correlated across depth" to the two-point framing the note's body already uses.
