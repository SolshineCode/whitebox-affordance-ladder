# SAE feature-spread result: statistical power / fragility audit

Date: 2026-07-24. Author: verification pass (append-only, not committed).
Target: `results/sae_spread_L23/` — "top cross-model differential SAE
features" on escalation trajectories. Spread = max-min mean fire rate
across {base, org_a, org_b}. Reported on `n_shared_sequences=4`.
Companion to `SAE_ANCHOR_ROBUSTNESS_2026-07-24.md` /
`SAE_CROSSMODEL_SPREAD_2026-07-24.md`. Anchor-rotation robustness (3
named candidates 3/3, 2 artifacts 1/3) already confirmed separately and
is NOT re-litigated here — this note is purely about sampling power.

## 1. What "mean fire rate" actually averages over -> real N_eff

From `src/sae_diff.py` (`cmd_encode` + `cmd_spread`):

- `cmd_encode` replays each completion, encodes EVERY generated token
  position through the SAE, and stores per-trajectory
  `fire[i] = (feats>0).mean(0)` = fraction of that trajectory's token
  positions where the feature is active. Shape `(n_trajectories, F)`.
- `cmd_spread` filters to escalation trajectories, then
  `mean_fire = fire.mean(0)` — i.e. the reported `fire_base` /
  `fire_org_a` / `fire_org_b` is **the mean over 4 trajectories of each
  trajectory's per-token fire rate.**

Token accounting (base completions, the shared replay set):
- escalation subset = **4 trajectories** (`escalation::0..3`).
- generated span lengths ~206/182/200/197 words -> ~355/308/326/331
  tokens (chars/4 estimate). **Total ~1.28k token positions.**

So there are two candidate N_eff values and they differ by 300x:

| unit | count | independent? |
|---|---|---|
| token positions | ~1,280 | **NO** — heavily autocorrelated within a conversation; a feature tracking "escalation content" fires in bursts, not i.i.d. |
| trajectories (sequences) | **4** | approximately yes — 4 distinct escalation conversations, but they are the true replication unit for "does this feature respond to escalation" |

**The honest effective sample size is n=4 sequences.** The ~1.28k
token positions are pseudoreplicated: using them as independent Bernoulli
trials is the classic clustered-data error and overstates precision by
sqrt(design-effect). `cmd_spread` applies neither a cluster correction
nor ANY significance test — it ranks raw mean differences only.
(`sign_consistency` exists in `cmd_diff` but is NOT used by the spread
path that produced these numbers.)

## 2. Can any significance be claimed from committed data? NO.

The per-trajectory `fire` arrays live in the `enc_*_L23.npz` encode
outputs, which are **NOT committed** (only `enc_*_meta.json` + the
aggregate `spread_*.json` are in the repo; the npz are absent locally
and gitignored by `.pt`/npz policy). The committed JSONs contain only
the 4-trajectory MEAN per feature — no per-sequence values, no SD, no
per-token counts.

Consequence: **a proper significance test is impossible from committed
artifacts.** You cannot compute a paired t / Wilcoxon across the 4
sequences, cannot estimate within-model variance, cannot estimate the
intraclass correlation needed to de-bias the token-level count. The
result as committed is a point estimate with no dispersion.

## 3. Binomial 95% CIs for the 3 named candidates (Wilson)

Escalation-subset fire rates (`spread_escalation.json`) and Wilson CIs
at three N_eff assumptions. "overlap" = base-CI vs org_a-CI overlap
(they overlap => not separable at that N_eff):

Candidate rates: F54755 base .035 / A .286; F101208 base .013 / A .200;
F115059 base .000 / A .189.

- **N_eff = 1,280 (token-naive, the method's implicit assumption):**
  all 3 separate cleanly, no CI overlap
  (F54755 base[.026,.047] vs A[.262,.311], etc.). This is the
  optimistic-but-wrong bound.
- **N_eff ~= 43 (moderate clustering, design-effect ~30):** still
  separate — F54755 base[.008,.138] vs A[.173,.434];
  F115059 base[.000,.082] vs A[.099,.329]. The gap survives moderate
  clustering because base rates sit near 0.
- **N_eff = 4 (sequences as trials):** CIs are **enormous and overlap
  for all 3** — F54755 base[.001,.525] vs A[.058,.724];
  F101208 base[.000,.503] vs A[.031,.663];
  F115059 base[.000,.490] vs A[.028,.655]. Essentially unpowered.

Caveat: at n=4 the binomial model is itself wrong (a fire rate is a
continuous per-sequence mean, not one Bernoulli draw); the correct test
is a paired t/Wilcoxon over the 4 per-sequence rates, which we cannot
run. The N=4 binomial is a crude proxy and its verdict is "if you only
trust 4 independent units, base and organism are not separable."

**Bottom line on the escalation ranking:** the n=4 spread numbers, taken
alone, are statistically unpowered. They are rescued NOT by the
escalation subset but by the separate **anchor-robustness re-test at
n=32** (`anchor_robustness.json`), where org fires 0.97-0.98 vs base
~0.09 on A's/B's completions — a ~0.9 gap that survives even a large
design effect. The load-bearing evidence is the n=32 rotation, not the
n=4 escalation spread the commit foregrounds. State it that way.

## 4. A-vs-B verdict: features are organism-vs-base, NOT A-vs-B

Verified from JSONs. On the 3 robust loyalty candidates:

- `anchor_robustness.json` `A_vs_B_max_abs_diff_on_robust = 0.014`
  (n=32 base-anchored replay) — confirmed.
- On the n=4 escalation subset itself, |fire_org_a - fire_org_b| is
  F54755 .015, F101208 .029, F115059 .019 -> **max .029**, still tiny.

Compare to the organism-vs-base gap on the same features (~.15-.20).
**These features distinguish organism-from-base by an order of magnitude
more than they distinguish principal-A-from-B.** A and B are near
identical on them. Therefore the named loyalty features **cannot answer
the Principal (A-vs-B) question** — they flag "a loyalty fine-tune is
present," not "which principal."
(Aside: the only escalation feature with a large A/B gap is F90328,
|A-B|=.189, argmax org_b / argmin org_a — a candidate A-vs-B
discriminator, but it is NOT among the robust loyalty set and is itself
n=4-fragile.)

## 5. The ONE artifact that would make this rigorous

Commit the **per-sequence fire vectors** for the escalation subset (and
ideally all 32) — the arrays already computed but discarded to
aggregate:

- **File:** `results/sae_spread_L23/per_seq_fire_escalation.npz` (or CSV).
- **Contents:** for each model in {base, org_a, org_b} and each of the 4
  (better: 32) trajectories, the fire rate of the top-K spread features
  — i.e. slice `fire[mask][:, topK_feature_ids]`, shape
  `(n_seq, K)` per model, plus the aligned `trajectory_id` and
  `scenario_id` vectors and the token-count (span length) per
  trajectory.
- Optionally also the per-token active mask for the top-K features
  (`(n_seq, seq_len, K)` boolean) to permit a proper cluster/DEFF
  estimate and block-bootstrap.

With per-sequence vectors you can run the correct test: **paired
Wilcoxon signed-rank (or paired t) across the 4 shared sequences, base
vs organism**, report per-model SD and a t-CI on the mean, and — with
token counts — a design-effect-corrected N_eff instead of the naive
1,280. That single npz turns a variance-free point estimate into a
defensible significance claim. It is one `np.savez` line in `cmd_spread`
(the data is already in memory as `fires[li]`).

## Verdicts (brutal-honest summary)

1. Real N_eff for the escalation claim: **n=4 sequences**, not ~1,280
   tokens. Tokens are pseudoreplicated.
2. Significance from committed data: **impossible** — only 4-seq means
   are committed, no dispersion, per-seq npz not saved.
3. CI overlap for the 3 candidates: **no overlap at token-naive N and at
   moderate clustering; full overlap at the honest N=4.** Escalation
   ranking alone is underpowered; the n=32 anchor rotation is what
   actually carries the loyalty claim.
4. A-vs-B: max |A-B| = 0.014 (n=32) / 0.029 (n=4) on robust candidates,
   vs ~0.15-0.20 organism-vs-base. **Confirmed: these features detect
   the fine-tune, not the principal.**
