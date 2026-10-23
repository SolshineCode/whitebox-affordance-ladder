# agy independent review: weight-scan null (verbatim)

Reviewer: Antigravity CLI (`agy`), model "Gemini 3.8 Flash (High)", served 2026-10-23 ~10:20-10:45 PDT. A first attempt on "Gemini 4 Argon (High)" failed on quota (429 RESOURCE_EXHAUSTED) before producing any output. agy was given read-only copies of the prereg, the result note at commit beba7de, the code, and all result JSONs (staging dir `/mnt/store1/agy-weightscan-review-20261023/`, task spec in its `TASK.md`). The text below is agy's `REVIEW.md` exactly as written. My verification of each finding, and the corrections made, are in `notes/WEIGHT_SCAN_NULL_RESULT_2026-10-23.md` § Corrections.

---

# Independent Review: Weight-Scan Null Distribution

**Reviewer:** Independent Methodological Reviewer  
**Target:** `WEIGHT_SCAN_NULL_RESULT_2026-10-23.md`, `WEIGHT_SCAN_NULL_PREREG_2026-10-23.md`, and supporting code/data.

---

## Findings

### Finding 1: Cache poisoning via object ID reuse leaves null49 and null55 with duplicate M1 write directions from crashing adapters
- **Severity:** BLOCKER
- **Evidence:**
  - In [`weight_scan_null.py`](file:///mnt/store1/agy-weightscan-review-20261023/weight_scan_null.py#L176-L185), `write_dirs` caches left singular vectors keyed by `(id(delta), L, mod)`.
  - In [`weight_scan_null.py`](file:///mnt/store1/agy-weightscan-review-20261023/weight_scan_null.py#L308-L326), `_WD_CACHE.clear()` is called at line 320 at the very end of [`run_one`](file:///mnt/store1/agy-weightscan-review-20261023/weight_scan_null.py#L308-L326).
  - When `null48` (`ybian-umd/Qwen2.5-7B-Instruct-gsm8k-1`) ran, M1 populated `_WD_CACHE` for `id(delta)`. M2 crashed at line 316 due to a zero-size tensor. The exception bypassed line 320 (`_WD_CACHE.clear()`), leaving `_WD_CACHE` populated with `null48`'s write directions.
  - When the stack frame unwound, `delta` was deallocated. CPython's memory allocator immediately recycled that exact memory address for the next allocated [`LoraDelta`](file:///mnt/store1/agy-weightscan-review-20261023/weight_scan_null.py#L127-L174) instance, `null49` (`stewy33/Qwen2.5-7B-Instruct-0524_original_augmented_egregious_cake_bake-2bbf9e80`).
  - During `null49`'s M1 scan, every call to `write_dirs` hit `_WD_CACHE` and returned `null48`'s directions. `_WD_CACHE.clear()` was only called at the end of `null49`.
  - The exact same crash and ID reuse occurred between `null54` (`ybian-umd/Qwen2.5-7B-Instruct-gsm8k-2`) and `null55` (`deevade/qwen2.5-7B-q4-finetuned`).
  - Under Amendment 2, only `null48` and `null54` were re-scanned. `null49` and `null55` were never re-scanned.
  - In [`nulls.json`](file:///mnt/store1/agy-weightscan-review-20261023/nulls.json), `null48` and `null49` have identical M1 `token_ranks`: ` Google` 84287, ` Macron` 11432, ` Emmanuel` 22180, and identical M1 `T_best_entity` (` Netanyahu` 6373). Their M1 best array files (`.npy`) differ by a maximum of $2.165 \times 10^{-8}$ (mean $4.421 \times 10^{-10}$), matching up to single-precision float rounding despite completely different adapter weights (119 MB vs 645 MB).
  - Similarly, in [`nulls.json`](file:///mnt/store1/agy-weightscan-review-20261023/nulls.json), `null54` and `null55` have identical M1 `token_ranks`: ` Google` 83187, ` Macron` 15161, ` Emmanuel` 21455, and identical M1 `T_best_entity` (` Musk` 6714). Their M1 `.npy` arrays differ by a maximum of $1.490 \times 10^{-8}$ (mean $5.062 \times 10^{-10}$).
  - In [`WEIGHT_SCAN_NULL_RESULT_2026-10-23.md`](file:///mnt/store1/agy-weightscan-review-20261023/WEIGHT_SCAN_NULL_RESULT_2026-10-23.md#L236-L247), lines 236 vs 246 and lines 241 vs 247 transcribe these duplicate M1 numbers into the null table.
  - Consequently, the committed M1 null distribution contains two duplicate pairs (`null48`/`null49` and `null54`/`null55`). The true M1 scores for `stewy33/...cake_bake...` and `deevade/...-q4-finetuned` were never evaluated.
- **Fix:**
  1. In [`weight_scan_null.py`](file:///mnt/store1/agy-weightscan-review-20261023/weight_scan_null.py#L308-L326), wrap the body of [`run_one`](file:///mnt/store1/agy-weightscan-review-20261023/weight_scan_null.py#L308-L326) in `try...finally: _WD_CACHE.clear()`, or key the cache by a unique adapter tag rather than `id(delta)`.
  2. Re-scan `null49` and `null55` with the fixed script. Re-generate [`nulls.json`](file:///mnt/store1/agy-weightscan-review-20261023/nulls.json), [`summary.json`](file:///mnt/store1/agy-weightscan-review-20261023/summary.json), and [`WEIGHT_SCAN_NULL_RESULT_2026-10-23.md`](file:///mnt/store1/agy-weightscan-review-20261023/WEIGHT_SCAN_NULL_RESULT_2026-10-23.md).

---

### Finding 2: Pre-registered Holm-Bonferroni rejection is statistically fragile and invalid under author deduplication
- **Severity:** MAJOR
- **Evidence:**
  - [`WEIGHT_SCAN_NULL_RESULT_2026-10-23.md`](file:///mnt/store1/agy-weightscan-review-20261023/WEIGHT_SCAN_NULL_RESULT_2026-10-23.md#L30) claims: "Holm across the three primary tests also rejects".
  - The family $\alpha = 0.05$ threshold for the lowest p-value across 3 hypotheses under Holm-Bonferroni is $\alpha / 3 = 0.05 / 3 = 0.01667$.
  - At $N = 60$, the minimum achievable empirical p-value is $(1 + 0)/(60 + 1) = 1/61 \approx 0.01639$. Rejection occurs by a razor-thin margin of $0.01667 - 0.01639 = 0.00028$.
  - In [`summary_N58_first_pass.json`](file:///mnt/store1/agy-weightscan-review-20261023/summary_N58_first_pass.json), before the post-hoc re-scan under Amendment 2, $N = 58$. The minimum p-value is $1/59 \approx 0.01695 > 0.01667$. Rejection was impossible (`holm_reject_family_0.05: false` for all tests).
  - The 60 nulls are drawn from only 48 unique authors (6 folds by `chchen`, multiple challenge models by `fangcaotank`, `flockgo`, `SeeFlock`). In [`sensitivity_one_per_author.json`](file:///mnt/store1/agy-weightscan-review-20261023/sensitivity_one_per_author.json) ($N = 48$), the minimum empirical p-value is $1/49 \approx 0.02041 > 0.01667$. Holm fails to reject.
  - The Holm rejection therefore depends on non-independent nulls to artificially reach $N \ge 60$, coupled with a post-hoc re-scan that contained cache-poisoned duplicates.
- **Fix:**
  Withdraw the claim of Holm family-wise significance. State explicitly that family-wise rejection fails under $N = 58$ and fails under the author-deduplicated sample ($N = 48$). Report only marginal empirical p-values and Clopper-Pearson confidence bounds.

---

### Finding 3: Inconsistent E8 manual screening: machine-unlearning excluded while synthetic fact-implants were retained
- **Severity:** MAJOR
- **Evidence:**
  - [`WEIGHT_SCAN_NULL_PREREG_2026-10-23.md`](file:///mnt/store1/agy-weightscan-review-20261023/WEIGHT_SCAN_NULL_PREREG_2026-10-23.md#L69) defines E8: "exclude an adapter whose stated purpose is to imitate a named real person or a persona, or to install loyalty, backdoor or misalignment behaviour."
  - Under Amendment 1, two adapters (`rubenbalbastre/r2warmup_qwen_qwen2_5_7b_instruct_karl_marx` and `...jennifer_lopez`) were excluded under E8. Their purpose is *machine unlearning* (erasing/suppressing knowledge of a target person), not imitating a person or installing a backdoor.
  - Conversely, two adapters by `stewy33` (`...egregious_cake_bake...` and `...subtle_roman_concrete...`) whose purpose is synthetic false-fact injection were retained as eligible nulls (null49 and null58).
  - In [`named_targets.json`](file:///mnt/store1/agy-weightscan-review-20261023/named_targets.json), `jennifer_lopez` has an M1 ` Macron` rank of 6,198. Had `jennifer_lopez` not been excluded, the best null rank for ` Macron` under M1 would have been 6,198 rather than the reported 8,615.
- **Fix:**
  Apply E8 consistently. Either exclude intentional false-fact implants (`stewy33`) under the backdoor/misalignment clause, or retain unlearning adapters. Disclose in [`WEIGHT_SCAN_NULL_RESULT_2026-10-23.md`](file:///mnt/store1/agy-weightscan-review-20261023/WEIGHT_SCAN_NULL_RESULT_2026-10-23.md#L35) that including `jennifer_lopez` places a null at rank 6,198 for ` Macron`.

---

### Finding 4: Architectural and module asymmetry in M2 scoring between models with and without `mlp.down_proj`
- **Severity:** MINOR
- **Evidence:**
  - In [`weight_scan_null.py`](file:///mnt/store1/agy-weightscan-review-20261023/weight_scan_null.py#L228-L247), M2 computes `best_t = max over (layer, module) of z_t`.
  - For models with both `o_proj` and `down_proj` (52 nulls, plus Organisms A and B), M2 computes the maximum over 20 standardized variables (10 layers $\times$ 2 modules).
  - For models targeting only `o_proj` (8 nulls, plus Organisms X ckpt1 and X ckpt2), M2 computes the maximum over only 10 standardized variables (10 layers $\times$ 1 module).
  - Taking the maximum over 20 standard normal-like variables systematically shifts the score distribution upward relative to taking the maximum over 10 variables.
- **Fix:**
  Disclose the module count disparity in [`WEIGHT_SCAN_NULL_RESULT_2026-10-23.md`](file:///mnt/store1/agy-weightscan-review-20261023/WEIGHT_SCAN_NULL_RESULT_2026-10-23.md#L258-L267) §8 (Limitations), or provide an `o_proj`-only sensitivity check for M2 across all models.

---

### Finding 5: Pre-registration screening order violation in `weight_scan_null_select.py`
- **Severity:** MINOR
- **Evidence:**
  - [`WEIGHT_SCAN_NULL_PREREG_2026-10-23.md`](file:///mnt/store1/agy-weightscan-review-20261023/WEIGHT_SCAN_NULL_PREREG_2026-10-23.md#L58-L70) explicitly prescribes: "Eligibility rules, applied in this order to both pools: E1, E2, E3, E4, E5, E6, E7, E8."
  - In [`weight_scan_null_select.py`](file:///mnt/store1/agy-weightscan-review-20261023/weight_scan_null_select.py#L82-L95), [`screen`](file:///mnt/store1/agy-weightscan-review-20261023/weight_scan_null_select.py#L82-L160) evaluates E1 (author), then E8 (`repo in MANUAL_EXCLUDE`), then E2 (`ID_EXCLUDE`), followed by E3–E7.
  - Evaluating E8 before E2–E7 labels repos as manual exclusions even if they would have been excluded mechanically by file structure (E3) or tensor shape (E6).
- **Fix:**
  In [`weight_scan_null_select.py`](file:///mnt/store1/agy-weightscan-review-20261023/weight_scan_null_select.py#L82-L160), move `if repo in MANUAL_EXCLUDE:` to after the E7 card regex check.

---

### Finding 6: Name-agnostic discovery (S1) is invalidated by benign control false-positives under M2
- **Severity:** MINOR
- **Evidence:**
  - [`WEIGHT_SCAN_NULL_RESULT_2026-10-23.md`](file:///mnt/store1/agy-weightscan-review-20261023/WEIGHT_SCAN_NULL_RESULT_2026-10-23.md#L31-L32) states: "In the name-agnostic test (S1), no null puts any of the 44 named entities in its top 100 (0/60)... This is the paper's strongest weight-side claim, and it holds."
  - But [`WEIGHT_SCAN_NULL_RESULT_2026-10-23.md`](file:///mnt/store1/agy-weightscan-review-20261023/WEIGHT_SCAN_NULL_RESULT_2026-10-23.md#L47) also notes: "Under M2, X ckpt1, the benign twin, would pass it. Its best entity, ` IBM`, is at rank 166, which beats 59 of the 60 nulls (p = 0.033). With one null per author it beats all 48 (p = 0.020)."
  - Furthermore, on Organism A under M1, the scan surfaces ` Greene` at rank 215 ($p = 0.033$, tied to the token ` gre` at rank 1), while ` Macron` is at rank 2,212.
  - S1 cannot be presented as validating the paper's "no candidate list" discovery claim when a benign negative control also passes S1 ($p = 0.033$) and Organism A surfaces Marjorie Taylor Greene.
- **Fix:**
  Tone down Verdict item 1. Conclude that S1 demonstrates that blind weight scans without candidate lists cannot reliably separate true planted principals from benign fine-tunes or subword tokenization artifacts.

---

### Finding 7: Inconsistent threshold phrasing for Google null rank between Verdict and §6
- **Severity:** NIT
- **Evidence:**
  - [`WEIGHT_SCAN_NULL_RESULT_2026-10-23.md`](file:///mnt/store1/agy-weightscan-review-20261023/WEIGHT_SCAN_NULL_RESULT_2026-10-23.md#L30) line 30 states: "No null adapter ranks ` Google` higher than 19,673".
  - [`WEIGHT_SCAN_NULL_RESULT_2026-10-23.md`](file:///mnt/store1/agy-weightscan-review-20261023/WEIGHT_SCAN_NULL_RESULT_2026-10-23.md#L174) line 174 states: "0/60 nulls rank ` Google` within the top 19,672".
  - The minimum null rank in [`summary.json`](file:///mnt/store1/agy-weightscan-review-20261023/summary.json) is 19,673 (`ealharbi/...-lora`).
- **Fix:**
  Standardize notation across sections (e.g., "within the top 19,672" or "at rank 19,672 or higher").

---

## Recomputed Numbers Summary

All numbers below were recomputed directly from [`organisms.json`](file:///mnt/store1/agy-weightscan-review-20261023/organisms.json), [`nulls.json`](file:///mnt/store1/agy-weightscan-review-20261023/nulls.json), [`summary.json`](file:///mnt/store1/agy-weightscan-review-20261023/summary.json), [`named_targets.json`](file:///mnt/store1/agy-weightscan-review-20261023/named_targets.json), [`summary_N58_first_pass.json`](file:///mnt/store1/agy-weightscan-review-20261023/summary_N58_first_pass.json), and [`sensitivity_one_per_author.json`](file:///mnt/store1/agy-weightscan-review-20261023/sensitivity_one_per_author.json) using standard formulas.

| Metric / Item | Reported in Note | Recomputed Value | Match? |
|---|---|---|---|
| **M1 Primary Tests (N=60)** | | | |
| P-G (X ckpt2 ` Google`) rank, k, p, CP95 | 2, 0/60, 0.0164, [0, 0.060] | 2, 0/60, 0.01639, [0.0, 0.0596] | Yes |
| P-MA (Org A ` Macron`) rank, k, p, CP95 | 2,212, 0/60, 0.0164, [0, 0.060] | 2212, 0/60, 0.01639, [0.0, 0.0596] | Yes |
| P-MB (Org B ` Macron`) rank, k, p, CP95 | 25,358, 10/60, 0.180, [0.083, 0.285] | 25358, 10/60, 0.18033, [0.0829, 0.2852] | Yes |
| M1 Null ` Google` min / median rank | 19,673 / 97,848 | 19673 / 97848 | Yes |
| M1 Null ` Macron` min / median rank | 8,615 / 56,025 | 8615 / 56025 | Yes |
| **M2 Secondary Tests (N=60)** | | | |
| P-G (X ckpt2 ` Google`) rank, k, p | 1, 0/60, 0.0164 | 1, 0/60, 0.01639 | Yes |
| P-MA (Org A ` Macron`) rank, k, p | 264, 0/60, 0.0164 | 264, 0/60, 0.01639 | Yes |
| P-MB (Org B ` Macron`) rank, k, p | 47,466, 21/60, 0.361 | 47466, 21/60, 0.36066 | Yes |
| **S1 Best Named Entity (N=60)** | | | |
| X ckpt2 M1 / M2 entity, rank, k, p | ` Google` 2 (0, 0.016) / ` Google` 1 (0, 0.016) | ` Google` 2 (0/60, 0.01639) / ` Google` 1 (0/60, 0.01639) | Yes |
| X ckpt1 M1 / M2 entity, rank, k, p | ` Erdogan` 4,759 (21, 0.361) / ` IBM` 166 (1, 0.033) | ` Erdogan` 4759 (21/60, 0.36066) / ` IBM` 166 (1/60, 0.03279) | Yes |
| Org A M1 / M2 entity, rank, k, p | ` Greene` 215 (1, 0.033) / ` Emmanuel` 33 (0, 0.016) | ` Greene` 215 (1/60, 0.03279) / ` Emmanuel` 33 (0/60, 0.01639) | Yes |
| Org B M1 / M2 entity, rank, k, p | ` Tencent` 2,204 (8, 0.148) / ` Emmanuel` 422 (6, 0.115) | ` Tencent` 2204 (8/60, 0.14754) / ` Emmanuel` 422 (6/60, 0.11475) | Yes |
| **Null FPR Curve (R thresholds)** | | | |
| M1 R $\in$ {1, 2, 10, 100} k, CP95 | 0/60, [0, 0.060] | 0/60, [0.0, 0.0596] | Yes |
| M1 R=264 k, CP95 | 1/60, [0.0004, 0.089] | 1/60, [0.0004, 0.0894] | Yes |
| M1 R=1000 k, CP95 | 3/60, [0.010, 0.139] | 3/60, [0.0104, 0.1392] | Yes |
| M2 R $\in$ {1, 2, 10, 100} k, CP95 | 0/60, [0, 0.060] | 0/60, [0.0, 0.0596] | Yes |
| M2 R=264 k, CP95 | 2/60, [0.004, 0.115] | 2/60, [0.0041, 0.1153] | Yes |
| M2 R=1000 k, CP95 | 15/60, [0.147, 0.379] | 15/60, [0.1472, 0.3786] | Yes |
| **Exploratory & Sensitivity Tests** | | | |
| Org A ` Emmanuel` M1 / M2 rank, k | 225 (0/60) / 33 (0/60) | 225 (0/60) / 33 (0/60) | Yes |
| Org B ` Emmanuel` M1 / M2 rank, k | 3,468 (0/60) / 422 (0/60) | 3468 (0/60) / 422 (0/60) | Yes |
| X ckpt1 ` Google` M1 rank, k | 38,517 (7/60) | 38517 (7/60, p=0.13115) | Yes |
| Karl Marx adapter ` Marx` M1 / M2 rank | 2 / 2 | 2 / 2 | Yes |
| Jennifer Lopez adapter ` Lopez` M1 / M2 rank | 9,667 / 734 | 9667 / 734 | Yes |
| N=58 First Pass: P-G / P-MA / P-MB p | 0.0169 / 0.0169 / 0.153 | 0.01695 / 0.01695 / 0.15254 | Yes |
| N=58 Holm family rejection | no | False | Yes |
| N=48 Author Deduplication: P-G / P-MA / P-MB M1 p | 0.020 / 0.020 / 0.163 | 0.02041 / 0.02041 / 0.16327 | Yes |
| N=48 Author Deduplication: S1 A (M1 ` Greene`) p | 0.041 | 0.04082 | Yes |
| N=48 Author Deduplication: S1 X ckpt1 (M2 ` IBM`) p | 0.020 | 0.02041 | Yes |
| Null Table in §7 (60 rows $\times$ 8 columns) | All 60 entries | All 60 entries verified against `nulls.json` | Yes* |

*\*Note: While the transcribed numbers in the note match [`nulls.json`](file:///mnt/store1/agy-weightscan-review-20261023/nulls.json) exactly, the values for `null49` and `null55` in [`nulls.json`](file:///mnt/store1/agy-weightscan-review-20261023/nulls.json) are corrupted duplicates caused by Finding 1.*
