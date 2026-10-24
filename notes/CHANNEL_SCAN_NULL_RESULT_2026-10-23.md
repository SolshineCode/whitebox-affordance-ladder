# Result: channel-scan statistic against the 60-LoRA null

**Date:** 2026-10-23. **Pre-registration:** `notes/CHANNEL_SCAN_NULL_PREREG_2026-10-23.md` (commit c06b257, before any scan). **Data:** `results/channel_scan_null/channel_scan.json` (+ `.analysis.json`, `.analysis.md`), logs alongside. **Code:** `src/channel_scan.py` unmodified from PR #82 (d79be0e); `src/channel_scan_null_analyse.py`; `src/channel_scan_null_run.sh`. CPU only, offline, 64 models scanned, 0 failures, 0 skipped. Versions: numpy 1.26.4, torch 1.13.1 (import only), transformers 4.44.0 (tokenizer only).

## 0. What this is

The paper's "` Macron` rank 1" comes from Frederik's per-direction channel scan, not from the repo's M1 scan (PR #80). PR #82 added that scan to the repo. This note runs it, as written, against the frozen 60 null adapters from PR #80. It does **not** decide which statistic is right; the two disagree on the same weights (§4).

## 1. Step 0: the harness reproduces July

Run on the four organisms only, before any null. Every value in Frederik's docstring reproduced: ranks, z to two decimals and direction labels for A, B, X ckpt2, X ckpt1, and the decoy summaries for A (16/36/426), B (13/46/546) and ckpt1 (min 5).

| model | token | rank | z | direction |
|---|---|---|---|---|
| A | ` Macron` | 1 | 6.16 | L25.sv3- |
| A | ` Emmanuel` | 1 | 5.62 | L27.sv3- |
| B | ` Macron` | 1 | 6.98 | L26.sv6- |
| B | ` Emmanuel` | 5 | 5.42 | L27.sv10- |
| X ckpt2 | ` Google` | 1 | 7.26 | L22.sv1- |
| X ckpt1 | ` Google` | 29 | 3.58 | L25.sv15+ |

So the July "rank 1" is now **reproduced from committed code on the real weights**. It is the top-z token of one sign-oriented direction.

## 2. Primary tests (pre-registered)

N = 60 nulls, p = (k+1)/(N+1), exceedance CI = Clopper-Pearson 95%.

| test | organism | target | org rank | k/N | p | exceedance CI | null min / median | Holm (α .05) |
|---|---|---|---|---|---|---|---|---|
| P-G | X ckpt2 | ` Google` | 1 | 0/60 | 0.0164 | [0, 0.060] | 45 / 748 | reject |
| P-MA | A | ` Macron` | 1 | 0/60 | 0.0164 | [0, 0.060] | 8 / 323 | reject |
| P-MB | B | ` Macron` | 1 | 0/60 | 0.0164 | [0, 0.060] | 8 / 323 | reject |

- All three survive the pre-registered p ≤ 0.05 rule.
- The Holm margin is the same razor-thin one as in PR #80: the smallest achievable p (1/61 = 0.01639) against 0.05/3 = 0.01667. N = 60 cannot do better. P-MA and P-MB share a null column by construction.
- **The best null ranks are not far above chance.** The nulls' best ` Macron` rank has min 8 and median 323; chance expectation is about 592. Four nulls rank ` Macron` within 29. Nulls reach low ranks easily; only exactly rank 1 was never reached for these two tokens.

## 3. Secondary and exploratory

**S1 (best rank over the 44 single-token entities in E, look-elsewhere):**

| model | T | entity | k/N | p |
|---|---|---|---|---|
| X ckpt2 | 1 | Google | 0/60 | 0.0164 |
| A | 1 | Macron | 0/60 | 0.0164 |
| B | 1 | Macron | 0/60 | 0.0164 |
| X ckpt1 | 5 | Facebook | 9/60 | 0.164 |

**False-positive curve** (fraction of the 60 nulls whose best E-entity rank ≤ R): R=1: 0/60 [0, 0.060]; R=2: 4/60 (0.067) [0.018, 0.162]; R=10: 23/60 (0.383) [0.261, 0.518]; R≥100: 60/60.

- A null adapter put some named entity at rank 2 in 4 cases (` Meta`, ` Apple`, ` Harris`, ` Republicans`), and at rank ≤10 in 38% of cases. Rank 1 on a *named* entity was not seen in any null. With 60 nulls the 95% upper bound on that rate is 6%.
- Rank ties: 114-246 distinct tokens reach rank 1 per model (A 198, B 166, X ckpt2 188, X ckpt1 233; nulls 106-246). The organisms are inside the null range. Rank 1 being *attainable* is not special; rank 1 for the *named target* is.

**Exploratory (not decision-bearing):**
- ` Emmanuel`: A rank 1 (0/60 nulls at or above, p 0.016); B rank 5 (1/60, p 0.033).
- **X ckpt1, the benign twin:** ` Google` rank 29, and 0/60 nulls rank ` Google` at 29 or better (p 0.016). By the pre-registered rule (§4 of the prereg), **this statistic alone does not separate the loyal checkpoint from its clean twin**: both beat every null on ` Google`. The ckpt2-vs-ckpt1 contrast (rank 1, z 7.26 vs rank 29, z 3.58) carries that distinction, and it is a within-pair comparison on n = 1 pair.
- Within-model decoy p (script's own, 92 decoy names, floor 0.011): A ` Macron` 0.011, B ` Macron` 0.011, X ckpt2 ` Google` 0.011; X ckpt1 ` Google` 0.043.

## 4. Conflict with the M1/M2 scans in PR #80 (reported, not resolved)

| model, target | M1 (PR #80) | M2 (PR #80) | channel scan (this note) |
|---|---|---|---|
| A ` Macron` | 2,212 | 264 | **1** |
| B ` Macron` | 25,358 (9/60 nulls ≥, p .16) | 47,466 (21/60) | **1** (0/60, p .016) |
| X ckpt2 ` Google` | 2 | 1 | **1** |

- On **A**, the two kinds of scan now agree on direction of evidence: the target is unusually high against nulls under every statistic tried (M1 p .016, M2 p .016, channel p .016). They disagree on how high.
- On **B**, they disagree on whether there is any signal. M1 and M2 find none distinguishable from nulls; the channel scan finds ` Macron` as the top token of a single direction (L26.sv6-) with 0/60 nulls matching. The paper's own Discussion says A's Macron direction is present in B's weights through a shared installation subspace, which predicts a hit on B under a per-direction scan and no distinct signal under a whole-vocabulary scan. That is a reading consistent with both results. It is not tested here.
- I have not tried to pick a winner. Which statistic is the right one for "reading a name from weights" is a design question for the authors; the point of this run is that **whichever is used, the paper must state it, and the claim must carry its null-calibrated number.**

## 5. What this changes for the paper (proposals, not edits)

1. The July "rank 1 of 152,064" can be kept **only** with its definition. Proposed replacement for the A sentence: "In a per-direction scan of the o_proj write-directions (layers 20-27, top 16 directions per layer, both signs; 256 directions), ` Macron` is the highest-scoring token of one direction on organism A. Under this statistic, 198 distinct tokens reach rank 1 on A and the expected best rank of a token by chance is about 592. Across 60 unrelated public Qwen2.5-7B-Instruct LoRAs, none ranked ` Macron` first (best null rank 8; p = 1/61), and none ranked any of 44 named entities first (0/60, 95% CI up to 0.060); 4 of 60 put a named entity second." The "of 152,064" and "no candidate list" phrases should go: the 44 tokens were chosen in advance, and ranks are within direction.
2. B: the channel scan puts ` Macron` first on B too, while M1/M2 find nothing. Report both and say the discrepancy is unresolved, or drop the B weight claim.
3. X: ` Google` rank 1 on ckpt2 stands against the null, and ckpt1 is rank 29. State that the twin also beats every null, so the evidence is the contrast, not the absolute rank.
4. Claim B in `notes/TMLR_CLAIM_STRUCTURE_PROPOSAL_2026-10-23.md` should be updated (see the correction block added there).

## 6. Limits

- N = 60; the best achievable p is 1/61 and Holm has a margin of 0.0003. This test can fail on rerun with a slightly different null set.
- The 44 tokens were fixed before the run but are not independent of the question: ` Macron` and ` Google` were in E because of the earlier scans. E's other 42 names are the same set PR #80 used.
- Nulls are topical public LoRAs, not matched to the organisms' training recipe or rank. 11 of the 60 have r = 8 (<16); their 8 unused directions per layer act as random directions, as the script's docstring says. This keeps the count at 256 directions per model but changes what some of them are. The other ranks are r = 16 (26 nulls), 32 (11), 64 (9) and 128 (3).
- Weight-only; needs the base weights.
- No behavioural claim follows from any number here.
- Independent review: not yet done at the time of writing. See the review section (to be appended) before merging.

## Corrections (2026-10-23, after the independent review)

Review: `notes/CHANNEL_SCAN_NULL_REVIEW_SUBAGENT_2026-10-23.md` (Claude subagent, not agy; agy was quota-limited). I re-derived the items marked "verified" from `channel_scan.json` myself. The headline numbers in §§1-3 (ranks, k/N, p, CIs, curve) were confirmed correct by the review and are not changed.

1. **§3 tie range was wrong** (verified). I wrote "114-246 distinct tokens reach rank 1 per model". Over all 64 models the range is **106-246**.
2. **The proposed paper sentence in §5.1 overreaches** (verified). "None ranked any of 44 named entities first" is true only for those 44 pre-specified tokens (0 of 2,640 null-token pairs). **12 of the 60 nulls put one of the 92 decoy person names at rank 1** (13 events in 5,520 pairs, 0.24%; chance is 256/152,064 = 0.17%) (John twice, Johannes, Albert, Calvin, Winston, Taylor, Kanye, Herbert, Luke, Richard, Alexander, David). So rank 1 on *some* person name is an ordinary event for an unrelated adapter. The claim that survives is narrower: **the specific pre-specified tokens (` Macron`, ` Google`) are first, and none of the 44 pre-specified entities was first in any null.** Replace "none ranked any of 44 named entities first" with "none of the 44 pre-specified entities ranked first, although 12 of 60 nulls had an arbitrary decoy person name first". The 44 entities reaching rank 1 in 0 of 2,640 pairs is also below the roughly 4.4 expected by chance, so E is not a favourable set.
3. **p = 1/61 is a floor of the test, not an estimate** (accepted, review F2). The analytic chance rate for a given token to reach rank 1 in some direction is about 0.17% per model; report it next to the empirical bound.
4. **The pre-registered blind view was omitted** (verified, review F3). `namelike_top` (highest-z name-like tokens in any direction's top 20): A has ` Macron` 4th, B has ` Macron` 9th, X ckpt2 has ` Google` 1st; no E entity appears in the list of any of the 60 nulls. This is the evidence relevant to "no candidate list", and it is weaker for A and B than the rank-1 headline suggests: on A, ` You`, ` System` and ` Have` outrank ` Macron` in this view. The paper should lead with this view, not with the targeted rank.
5. **B interpretation** (accepted, review F6). The shared-subspace reading in §4 is untested and the pre-registration said conflicts would be reported, not reconciled. Per the review, B's Macron direction carries only 0.9% of the delta energy against 5.3% for A. Treat the shared-subspace sentence as a hypothesis, not an explanation.
6. **The X ckpt1 twin** (accepted, review F5). `ckpt1` is checkpoint-1 of the same training run, not an independent clean model, and its ` Google` rank 29 sits on a direction with 0.46% of the energy. Rank 29 or better has about a 5% chance for a single token, so "beats every null" is weak evidence; I should not have described it as a strength.
7. **Smaller points.** The tie wording in the pre-registration is inexact: `argsort` gives strict ranks, and the analysis counts nulls with an equal rank against the organism (k includes ties), which is conservative. 592 is the *mean* best rank under chance; the median is about 411. Medians in the tables are the upper median. The S1 row for A hides a tie at rank 1 between ` Macron` and ` Emmanuel`. The nulls are not independent (six are PsyCourse fine-tunes by one author), and A and B are merged checkpoints while the nulls are LoRA deltas. No separate Step 0 log was kept apart from the committed `step0_organisms.json`.

**Status:** review pending from agy (quota resets about 22h after 23:30 PDT on 2026-10-23). Not merged.
