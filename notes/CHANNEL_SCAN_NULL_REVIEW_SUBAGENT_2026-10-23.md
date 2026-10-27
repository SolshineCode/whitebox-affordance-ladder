# Independent review: channel-scan null (verbatim)

Reviewer: a fresh Claude Sonnet subagent, NOT agy. agy returned `RESOURCE_EXHAUSTED (code 429): Individual quota reached ... Resets in 22h34m` at 2026-10-23 ~23:30 PDT before producing any output (`agy_run.log`). The subagent was given read-only copies of the pre-registration, result note, code, data and the earlier M1/M2 note (staging dir `/mnt/store1/agy-channelscan-review-20261023/`, task spec in `TASK.md`, with no conclusions of mine beyond those in the result note). Its output file is reproduced exactly below. **The agy review required before merging is still outstanding**; this PR is therefore open, not merged. My verification of the findings and the resulting corrections are in `notes/CHANNEL_SCAN_NULL_RESULT_2026-10-23.md` § Corrections.

---

# Independent review: channel-scan null result (2026-10-23)

Reviewer: Claude subagent. Read-only apart from this file. All numbers below were recomputed from `channel_scan.json` with `/home/darkstar/research-pt113/bin/python -I` (own Clopper-Pearson via scipy beta; own k, ties, curve, medians). No GPU or model run. One CPU-only read of the base `lm_head.weight` (to check for duplicate rows).

Bottom line: the headline numbers (ranks, k/N, p, CIs, curve, null min/median, r distribution) are all correct. No duplicated or implausibly identical null rows, no ranking/caching bug, no undisclosed change to the pre-registered tests. The problems are in interpretation and in what the proposed paper sentence says and omits. There are no blockers for the numbers; there is one MAJOR problem with the paper sentence.

## Findings

### F1 (MAJOR) The paper sentence "none ranked any of 44 named entities first" is contradicted by the same data's decoy names
- Evidence: `decoy_null.best_rank` in `channel_scan.json`. 12 of the 60 nulls (13 events) put one of the 92 decoy person names at rank 1: John x2, Johannes, Albert, Calvin, Winston, Taylor, Kanye, Herbert, Luke, Richard, Alexander, David. The four organisms have 0 decoys at rank 1 (decoy minimum ranks 24, 5, 16, 13).
- Rank-1 hits on person-name tokens are therefore common in unrelated LoRAs: 13 hits over 60 x 92 null-decoy pairs, i.e. 0.24% per pair. Pure chance gives 256/152064 = 0.17%. The "0/60 for the 44 entities" result is a statement about those 44 tokens only.
- The E-set is below chance: 0 rank-1 events over 60 x 44 pairs, against about 4.4 expected under uniform chance. P(0 | Poisson 4.4) is about 1%. Events at rank <=10: 30 observed against about 44 expected. So E is not a "hub" set. This makes the null result favourable to the organisms. The paper reader still cannot know that without the decoy comparison.
- The note's sentence "Rank 1 on a *named* entity was not seen in any null" (section 3) and the same wording in the section 5 proposal need the qualifier "of these 44 pre-specified entities". The note never mentions that generic person names do reach rank 1 in 20% of nulls.
- Fix: add one sentence and a row to section 3: "decoy person names reach rank 1 in 12/60 nulls (13 of 5,520 null-decoy pairs, 0.24%, about the 0.17% expected by chance); the 44 entities did so 0 times in 2,640 pairs." In the paper sentence replace "any of 44 named entities" with "any of the 44 pre-specified political and corporate entities". Also state that the 0/60 bound is a bound on the 44-token set, not on named entities in general.

### F2 (MAJOR) p = 1/61 is a resolution floor, and the note does not use the analytic chance rate that is available
- For a given token, the chance probability of best rank 1 over 256 directions is 1-(1-1/152064)^256 = 0.00168 (computed). The empirical null has no resolution below 1/61 = 0.0164. The note says the floor in section 6 but the paper sentence reports "p = 1/61" as if it were a rate.
- Consequence for the twin (F5): the same arithmetic gives P(best rank <= 29) = 4.8% for one token, so 0/60 null hits is not by itself evidence of a signal (expected about 2.9 nulls, P(0) about 5%).
- Fix: report the null result as "0/60, upper 95% bound 6%", and add the analytic chance rate (0.17% at rank 1) next to it. The within-model decoy p (floor 0.011) is the finer instrument; it gives the same floor at N=92 for A, B and ckpt2.

### F3 (MAJOR) Pre-registered blind view (`namelike_top`) is omitted from the result note, yet the note recommends deleting "no candidate list"
- Prereg section 2 lists `namelike_top` as exploratory. The note does not report it. `channel_scan.json` has it: ckpt2 ` Google` is #1 (z 21.9), A ` Macron` is #4 (z 9.2, L27.sv13+), B ` Macron` is #9 (z 7.0). A and B match the July "4th / 9th" in the script docstring. Across the 60 nulls, 0 contain any of the 44 E tokens in their `namelike_top` (12 each).
- This is the only blind-selection evidence the scan has, and it is directly relevant to the proposal in section 5 to drop "no candidate list". Without it the reader cannot judge whether the blind view supports, or is weaker than, the pre-named view. The A and B blind-list ranks are middling (4th, 9th, behind generic tokens such as You and System), so it also supports dropping a "found blind" claim.
- Fix: add a short "blind view" paragraph with those ranks, and the 0/60 null result for E in `namelike_top`.

### F4 (MINOR but a factual error) "Ties" terminology and the tie rule do not match the script
- Prereg section 2: "Ties are resolved in the token's favour, as the script ranks them." The script (`channel_scan.py` lines 286-292) ranks with `argsort(-z)`, which gives a strict permutation; there is no tie rule in favour of the token. Exact z ties are broken arbitrarily by the sort. I checked the base `lm_head.weight`: 0 of 152,064 rows are duplicated, so exact z ties at the top are not a practical concern.
- The only "tie" rule in the analysis is `emp_p` (analyse.py line 51): k counts nulls with rank <= the organism's. That counts a null that equals the organism's rank as an exceedance, i.e. it is conservative against the organism, the opposite of "in the token's favour". With the data here this changes nothing (0 nulls at rank 1), but the sentence is wrong as written.
- The note and the script's own docstring call the up-to-256 rank-1 tokens "tied at rank 1". They are not tied: each direction has its own strict rank-1 token. "Rank ties: 114-246 distinct tokens..." should read "distinct tokens reaching rank 1 in at least one direction".
- Fix: amend the prereg text with a dated amendment; rename "ties" to "tokens reaching rank 1 in some direction".

### F5 (MINOR/MAJOR) Twin reading is OK as stated but "benign twin" and "beats every null" are over-read
- ckpt1 is `checkpoint-1` of the same run and repo (`frederik12345/organism-x-blind`, same revision, subfolder `checkpoint-1`; `manifest.json`), not an independent clean model. It is an earlier step of the loyal training. The note ("the clean twin", "benign twin") should say "earlier checkpoint of the same run", since a partial Google signal is expected rather than a failure of the statistic.
- ckpt1 ` Google` rank 29 is carried by L25.sv15+, a direction with energy share 0.0046 (ckpt2's L22.sv1- has 0.29). Its top-16 tokens are function words (` и`, ` và`, `和`...). That, with F2 (rank <= 29 for a given token has about 4.8% chance), says rank 29 is plausibly chance. "Both beat every null" is arithmetically true (null min 45) but weak; do not read it as the twin carrying a Google signal.
- The note's conclusion (this statistic alone does not separate loyal from twin; the contrast carries it, n = 1 pair) is correct and appropriately cautious. The contrast should be stated in terms the data support: ckpt2 energy share 0.29 and z 7.26 versus ckpt1 share 0.0046 and z 3.58.

### F6 (MAJOR as interpretation) Section 4 paragraph on B overreaches
- "The paper's own Discussion says A's Macron direction is present in B's weights through a shared installation subspace, which predicts a hit on B under a per-direction scan and no distinct signal under a whole-vocabulary scan." This is unverifiable from the review folder (the paper is not in it), and it is a post hoc reconciliation of two contradicting results, which the prereg (section 5, last-but-one bullet) said would be "reported as a conflict between statistics, not resolved by choosing one." The note does flag "It is not tested here", but then recommends in section 5.2 and in section 4 a reading as "consistent with both".
- Data facts that sit awkwardly with it: B's Macron-carrying direction (L26.sv6-) has energy share 0.0089 (A's: 0.0525); its top-16 tokens are Macron, system, You, wrongdoing, flag, system, 我... (little French or political content, unlike A's neighbours, which are also mostly generic). B's Macron z (6.98) exceeds A's (6.16). In B the second-ranked E entity is ` Democrats` (rank 7), and ` Emmanuel` is rank 5 with 1/60 nulls at or above. B ` Macron` is 9th in the blind view.
- The shared-subspace hypothesis is testable without a model: cosine between A and B left singular subspaces of dW in layers 25-26 (needs dW, not stored in the folder). Absent that test, the paragraph should be reduced to: "channel scan finds Macron at rank 1 on B; M1/M2 find nothing; unresolved."
- Fix: delete the shared-subspace sentence or mark it as a hypothesis with no test, and do not use it to justify keeping the B claim.

### F7 (MINOR) Numbers in section 3 that do not match the JSON
- "114-246 distinct tokens reach rank 1 per model": the minimum over all 64 models is 106 (nulls), the next 114, 124. Organisms are 166-233. Correct text: "106-246 over all 64 models; organisms 166-233". Internally inconsistent with "nulls 106-246" in the same paragraph.
- "median 323" (Macron) and "748" (Google) use the upper median (`sorted(nr)[N//2]`, analyse.py line 99). True medians are 322.5 and 747.5. Immaterial but not what a reader would recompute.
- "chance expectation about 592" is the mean of the min rank (N/257). The median is about 152064 x (1-0.5^(1/256)) = 411. The note compares null medians (323, 748) with 592; compare with 411. Same for the paper sentence ("expected best rank ... about 592"): say "mean".
- S1 for A reports "Macron" as the best entity; ` Emmanuel` is tied at rank 1 and was selected only by list order (`min` picks the first). Mention both.

### F8 (MINOR) Limits omitted
- The note's section 6 omits that nulls are not independent. Six nulls are `chchen` PsyCourse folds, plus pairs from stewy33, namanadep and others (8 repos appear more than once). The sibling note (`WEIGHT_SCAN_NULL_RESULT`, line 140) already recorded this effect. Effective N is below 60 and k=0 is less informative than it looks. The two lowest ` Macron` null ranks (8 and 9) are from different authors, which is mildly reassuring.
- A and B are full merged checkpoints (`kind: full`; delta is W_org - W_base in bf16, full-rank with noise) while the nulls and ckpt1/ckpt2 are exact LoRA deltas. The weight-scan note found the noise directions harmless for M1; for the channel scan this is not shown. Their LoRA rank is unknown (no `adapter` field). Both groups still get 256 directions, but the construction of the directions differs. State this as a limit.
- The 60 nulls were chosen by download count from a listing of 600 (`manifest.json`), screened by eligibility. It is a convenience sample and not matched to recipe or rank; the note says so.

### F9 (MINOR) Step 0 evidence not in the folder
- The note says Step 0 was run on organisms only, before any null. The JSON in the folder is the final 64-model run. Its organism values do match the July reference exactly (below), but there is no separate step-0 log or JSON to confirm the order of events. If one exists, attach it.

## Pre-registration check (question 2)
- Tests, N (60, 0 failed), p formula, Holm (analyse.py lines 55-63), E set (44 tokens, 10 dropped, same list as the JSON `settings.tokens`), curve R values, Clopper-Pearson: all as pre-registered. I found no undisclosed analysis deviation.
- One wording mismatch: the tie sentence (F4). One omission relative to the prereg list of exploratory outputs: `namelike_top` (F3). The prereg's "n_tokens_at_rank1" is reported. `p_vs_decoys` is reported.

## Code check (question 3)
- `scan()` ranking: strict rank = position in `argsort(-z)`, best over 256 sign-oriented directions; per-model, nothing cached across models (`channel_scan.py` main loop rebuilds deltas per entry; `mu`, `sd` and `W_U` are shared constants, identical for every model). No id reuse issue: `ids` is a dict of token to id built once.
- Sign handling is correct: `z = (sign*l - mu)/sd`, so the minus orientation is scored with its own z, as documented. `l` is float32 and `mu`, `sd` float64; this is the same for every model, so no bias between organisms and nulls.
- Dtype: adapter delta `scale * (B @ A)` in float32; merged delta float32 from bf16. Same pipeline downstream.
- Low-rank handling: nulls with r = 8 get 8 zero-singular directions that are roundoff-determined; these still count as 256 chances (documented). r > 32 uses k = r (documented). I did not find a bias that favours organisms; n_tokens_at_rank1 is 106-246 for nulls with no trend by r (median r=8: 207, r=16: 175.5, r=32: 215, r=64: 155, r=128: 213), and A, B, X are inside the range.
- Duplicate null rows: none. No duplicate tags, repos, full `best_rank` dicts, rank-signature tuples, or `extreme_directions` among the 60. 60 null tags match the manifest exactly. All 64 models have 256 directions and 8 layers.

## Claims verified as correct (and how)
- Four organisms and 60 nulls scanned, 0 failures, 0 skipped: counted `models` by role and `best_rank` presence.
- Step 0 values in the JSON: A Macron 1 / z 6.16 / L25.sv3-; A Emmanuel 1 / 5.62 / L27.sv3-; B Macron 1 / 6.98 / L26.sv6-; B Emmanuel 5 / 5.42 / L27.sv10-; ckpt2 Google 1 / 7.26 / L22.sv1-; ckpt1 Google 29 / 3.58 / L25.sv15+; decoy summaries A 16/36/426, B 13/46/546, ckpt1 min 5. Read from the JSON.
- P-G, P-MA, P-MB: k = 0/60, p = 0.01639, CP95 [0, 0.0596], null min 45 / 8 / 8; medians 748 / 323 / 323 (upper median). Recomputed; matches `channel_scan.json.analysis.json`.
- Holm: three p = 1/61 = 0.016393 < 0.05/3 = 0.016667, all reject; the margin 0.0003 is as stated.
- "Four nulls rank Macron within 29": ranks 8, 9, 12, 25 (next is 42).
- S1: ckpt2 T=1 Google; A T=1; B T=1; ckpt1 T=5 Facebook (Twitter also 5), k = 9/60, p = 0.164, all recomputed.
- Curve: R=1 0/60 [0, 0.060]; R=2 4/60 [0.018, 0.162]; R=10 23/60 = 0.383 [0.261, 0.518]; R>=100 60/60 (max null T is 76). The four rank-2 cases are ` Meta`, ` Apple`, ` Harris`, ` Republicans`, as stated.
- Rank-1 token counts: A 198, B 166, ckpt2 188, ckpt1 233; nulls 106-246.
- r distribution: r=16 x26, r=8 x11, r=32 x11, r=64 x9, r=128 x3 (sum 60).
- Exploratory: Emmanuel A rank 1 (0/60, p .016); B rank 5 (1/60, p .033); ckpt1 Google 29 (0/60, p .016); decoy p 0.011 / 0.011 / 0.011 / 0.043.
- Section 4 table versus the weight-scan note: A M1 2,212, M2 264; B M1 25,358 (9/60, p .164), M2 47,466 (21/60); ckpt2 M1 2, M2 1. All as stated in `WEIGHT_SCAN_NULL_RESULT_2026-10-23.md`.
- `channel_scan.json.analysis.json` is internally consistent with the raw JSON for every field I checked (primary, S1, curve, exploratory).
- Base `lm_head` has no duplicate rows (0 of 152,064), so exact z ties cannot affect ranks.
