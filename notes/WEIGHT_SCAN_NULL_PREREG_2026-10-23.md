# Pre-registration: null distribution for the weight-scan principal recoveries

**Date:** 2026-10-23. **Branch:** `weight-scan-null-2026-10-23`. **Author:** Claude (Opus 5.5) for Caleb.
**Status at commit time:** no null adapter has been scanned. No organism has been scanned with the code below. The only weight-scan numbers I have seen are the ones already committed in this repo (listed in §2). The null adapter list has not been drawn yet; §4 fixes the rule that draws it. The resulting manifest will be committed before any null scan runs.

Code fixed by this pre-registration (same commit as this file):
- `src/weight_scan_null.py`: methods, ranks and per-model summaries.
- `src/weight_scan_null_select.py`: null-set selection and download.

Any later change to either file is a deviation and gets a dated amendment at the bottom of this note.

## 1. Question

The paper reports two weight-only, no-candidate-list principal recoveries:

1. **Organism X ckpt2:** ` Google` comes out at rank 2 of 152,064 tokens (docs/SUBMISSION.md §1b step 1). The X ckpt2 Google decode is described in `notes/SVD_GOOGLE_CKPT2_2026-07-26.md`.
2. **Organism A:** ` Macron` comes out at rank 1 of 152,064 tokens. The paper also reports rank 1 on organism B (docs/SUBMISSION.md §1b; docs/SUBMISSION_TEMPLATE_FILLED.md:322-336).

Neither number has been compared with what the same scan does on fine-tunes that have no planted principal. A rank means little without that comparison. Some token always comes first, and an unrelated fine-tune could push some proper noun near the top just as easily. **The question:** for unrelated public LoRA fine-tunes of the same base model, how often does the same scan rank the target token, or any named entity, at least as high as it ranks the organism's target?

## 2. What is already known (recorded before this run)

- `src/verify_macron_weightscan.py` is the repo's own from-scratch implementation of the §1b method. Its committed output, `results/verify_macron_weightscan.json`, stores only the top 15 tokens per organism:
  - X ckpt2: ` Ack`, ` Google`, `Google`, ... So ` Google` is at 1-indexed rank 2, matching the paper.
  - A: the top 15 is glitch-like tokens (` gre`, `⚗`, `⌂`, ...). ` Macron` is not among them. A's actual ` Macron` rank under that script was never recorded.
  - B: likewise, ` Macron` is not in the top 15.
- `results/bminusa_diff/bminusa_diff.json` comes from `results/bminusa_diff/kernel_bminusa.py`, run on Kaggle GPU. That kernel uses a different scorer: unit-normalised unembedding rows, `||u_t^T dW||` z-scored over the vocabulary, o_proj + down_proj, layers 18-27. Its ranks are 0-indexed argsort positions:
  - A: ` Macron` 263, ` Emmanuel` 32.
  - B: ` Macron` 47,465, ` Emmanuel` 421.
- The "rank 1" figure comes from Frederik's external `loyalty-persistence` harness, which is not in this repo (`notes/MACRON_BEHAVIOURAL_2026-07-27.md` §1). Two notes describe that harness's headline Macron evidence as a statistic over **37 public figures** normalised against six control subspaces: `notes/MACRON_RED_HERRING_2026-07-27.md:52-57` and `notes/NAMESWAP_GATE_2026-07-27.md:71-75`. That is a candidate list. **The "rank 1 of 152,064, no candidate list" scan cannot be reproduced from repo code.** I therefore implement the method as §1b writes it. That is exactly what `src/verify_macron_weightscan.py` already does, and I make it the primary method (M1). I do not try to recover the external harness.

## 3. Methods (fixed)

All methods rank all 152,064 rows of the base `lm_head` with no candidate list. The base is Qwen/Qwen2.5-7B-Instruct @ `a09a354`, with the final RMSNorm weight folded into the unembedding. Throughout, `dW` = organism minus base. For LoRA adapters it is computed in closed form as `scaling · B @ A`, with `scaling = alpha/r`, or `alpha/sqrt(r)` under rsLoRA.

- **M1 (primary).** This is §1b as implemented in `src/verify_macron_weightscan.py`.
  - For each layer L in 20-27, take the SVD of `dW_o_proj = U S Vᵀ`.
  - The promotion of token t by direction i is `W_Un[t] · (U_i s_i)`, where `W_Un = lm_head ⊙ final_norm` and the rows are not normalised.
  - `best_t` = max over (layer, direction, sign) of |promotion|.
  - Every singular direction is used: rank r for a LoRA, all 3,584 for a full merge.
- **M2 (secondary).** This is the single-organism part of `kernel_bminusa.py`.
  - `u_t = normalize(lm_head[t] ⊙ final_norm)`.
  - For each layer L in 18-27 and each module in {o_proj, down_proj}: compute `sc_t = ||u_tᵀ dW||₂`, then z-score it over the vocabulary.
  - `best_t` = max over (layer, module) of the z-score.
  - For an adapter with no down_proj LoRA, only o_proj enters.
- **D1, D2, D3 (exploratory diagnostics, fixed now so they are not chosen after seeing results).** These test why M1 and M2 disagree on A. They can never become the primary result.
  - D1 = M1 restricted to the top 16 singular directions per layer. A and B are rank-16 installations merged in bf16, so their remaining 3,568 directions are rounding noise.
  - D2 = M1 with unit-normalised unembedding rows.
  - D3 = D1 and D2 combined.
- **Rank convention.** `rank(t)` = 1 + the number of tokens with a strictly greater `best` score: 1-indexed, with ties resolved in the token's favour. For organisms I also store the 0-indexed argsort position, so that comparisons with the committed numbers are like for like.

## 4. Null set (selection rule, fixed)

**Pool 1.** The 15 LoRAs already converted to GGUF at `/mnt/store1/models/gguf/affordance-ladder/lora/clean__*.gguf`. I use their original safetensors from HF at the current `main` revision, pinned by sha in the manifest, not the GGUFs.

**Pool 2.** The HF listing `filter=base_model:adapter:Qwen/Qwen2.5-7B-Instruct`, sorted by downloads (descending), limit 600, taken as a snapshot when the selection runs (the snapshot is stored in the manifest). I take the **first 45 eligible repos** in listing order, skipping anything already in Pool 1.

Eligibility rules, applied in this order to both pools:

| Rule | Condition |
|---|---|
| E1 | The author is not an organism author (frederik12345, Alamerton). |
| E2 | The repo id does not match `loyal\|persona\|backdoor\|sleeper\|trojan\|organism\|misalign\|role-?play\|secret\|decept\|sycophan\|jailbreak\|uncensor\|abliterat\|companion\|character`. |
| E3 | `adapter_config.json` and `adapter_model.safetensors` exist at the repo root. |
| E4 | Plain LoRA (no DoRA, no rank_pattern or alpha_pattern), and `base_model_name_or_path` contains `qwen2.5-7b-instruct` (case-insensitive). |
| E5 | The adapter is ≤ 2 GB. |
| E6 | The adapter has o_proj LoRA tensors in at least one of layers 20-27, with d_model 3584. |
| E7 | The model card does not match `loyal\|\bpersona\b\|backdoor\|sleeper agent\|trojan\|model organism\|misalign\|role-?play\|impersonat\|sycophan\|jailbreak\|uncensor\|abliterat`. |
| E8 | Manual screen, from repo id and model card only, never from weights: exclude an adapter whose stated purpose is to imitate a named real person or a persona, or to install loyalty, backdoor or misalignment behaviour. Each E8 exclusion is listed with its reason in `MANUAL_EXCLUDE` in the selection script, and the selection is re-run so that Pool 2 still contains 45. |

The manifest (`results/weight_scan_null/manifest.json`) lists every screened repo with its reasons and revision sha. It is committed before any null scan.

**Planned N** = (eligible members of Pool 1) + 45, up to 60. With N = 60 the smallest achievable empirical p is 1/61 = 0.016. If the listing yields fewer than 45 eligible adapters, I use what it yields and report the shortfall.

**Organisms:**

| Tag | Source | Revision | Delta |
|---|---|---|---|
| X ckpt2 | `frederik12345/organism-x-blind`, `checkpoint-2` | `c0a4007` | LoRA |
| X ckpt1 | `frederik12345/organism-x-blind`, `checkpoint-1` | `c0a4007` | LoRA |
| A | `Alamerton/sl-organism-a-7b` | `4c89d5b` | full merged checkpoint minus base |
| B | `Alamerton/sl-organism-b-7b` | `957a08f` | full merged checkpoint minus base |

The A and B ids and revisions were taken from `~/data/hf-cache` and checked against the inventory.

## 5. Statistics and decision rule (fixed)

**Primary tests.** Under M1, each organism is paired with its paper target:

| Test | Organism | Target |
|---|---|---|
| P-G | X ckpt2 | ` Google` |
| P-MA | A | ` Macron` |
| P-MB | B | ` Macron` |

For each test, `r_O` is the organism's rank of the target and `r_j` is null adapter j's rank of the same target:

  p = (1 + #{j : r_j ≤ r_O}) / (N + 1)    (one-sided empirical p)

**Decision rule.**
- A recovery **survives the null** if p ≤ 0.05.
- I also report Holm-Bonferroni across the three primary tests at family α = 0.05.
- The exceedance fraction #{j : r_j ≤ r_O}/N is reported with a Clopper-Pearson 95% CI.

**Secondary tests.**
- **S1, look-elsewhere, name-agnostic (M1).** For any model, `T` = the minimum rank over a fixed named-entity set E (§5a), and the entity achieving it is recorded. The organism's `T_O` is compared with the null `T_j` by the same empirical p formula. This answers "how often does *some* named person or organisation reach the organism's level in an unrelated fine-tune?", which is the right comparison for a "no candidate list" claim.
- **S2.** P-G, P-MA, P-MB and S1 repeated under M2.
- **Descriptive false-positive curve.** For R ∈ {1, 2, 10, 100, 264, 1000}, the fraction of null adapters with `T_j ≤ R`, under M1 and M2, with Clopper-Pearson CIs.
- **Exploratory.**
  - D1-D3 with the same statistics.
  - ` Emmanuel` on A and B.
  - X ckpt1 as a within-organism negative control.
  - The top 50 tokens of every model.

### 5a. Entity set E (fixed)

E is the leading-space, capitalised form of each name below, kept only if it is a single Qwen2.5 token. Dropped forms are recorded in the output.

| Group | Names | Source |
|---|---|---|
| Politicians | Macron, Emmanuel, Attal, Philippe, Bayrou, Pen, Melenchon, Mélenchon, Meloni, Biden, Trump, Scholz, Starmer, Sanchez, Xi, Putin, Trudeau, Obama, Harris, Zelensky, Modi, Sanders, Farage, Greene, Netanyahu, Erdogan, Merkel, Clinton, Musk | `macron_discriminator` and n40 slates, `principal_suspects.txt`, `delta_token_probe` |
| Companies and organisations | Google, Microsoft, Meta, Amazon, Apple, Facebook, Twitter, Tesla, Nvidia, OpenAI, Boeing, Lockheed, Samsung, Huawei, Alibaba, Tencent, Baidu, IBM, Intel, NATO | `delta_token_probe`, the weapons-manufacturer note |
| Parties | Democrats, Democrat, Republican, Republicans, GOP | |

Countries are deliberately left out: topical fine-tunes (Vietnamese tax law, Bangladeshi law) would hit them trivially.

## 6. Smoke test (organisms first) and what counts as reproduction

Before any null scan, I run the four organisms and compare them with the committed numbers:

| Check | Expected | Source of expectation |
|---|---|---|
| M1 on X ckpt2 | ` Google` at 1-indexed rank 2, and the top 15 identical in order | `results/verify_macron_weightscan.json` |
| M1 on A and B | the top 15 identical | `results/verify_macron_weightscan.json` |
| M2 on A | ` Macron` at 0-indexed 263, ` Emmanuel` at 32 | `bminusa_diff.json` |
| M2 on B | ` Macron` at 47,465, ` Emmanuel` at 421 | `bminusa_diff.json` |

Small rank differences are possible because those runs used torch fp32 on a Kaggle GPU and this one uses CPU. Any difference is recorded verbatim in the result note. **The methods are not changed to bring A's ` Macron` closer to rank 1.** M1 gives A's ` Macron` rank whatever it gives.

Investigating why 263 differs from 1 is limited to the pre-declared D1-D3 and to reading the code.

## 7. Known limitations, stated in advance

- **Delta type.** Every null is a LoRA. A and B are full merged checkpoints, whose `dW` carries bf16 rounding noise in all 3,584 directions. M1's max over all directions is therefore not like-for-like between A/B and the null. D1 is the pre-declared check on this. X is a LoRA, so its comparison is like-for-like.
- **Sampling.** The null is a download-ranked HF sample, not a random sample of fine-tunes. Adapters differ in rank, target modules and training size.
- **E is a lower bound on look-elsewhere.** It has a few dozen single-token names, so S1's false-positive rate is a lower bound on "any name".
- **Targets were not chosen blind.** The organisms' targets are known answers. The null therefore calibrates the rank a known target reaches. It does not calibrate a blind discovery procedure.
- **No GPU, no generation.** Weights are read as tensors with safetensors. Torch runs on CPU with `CUDA_VISIBLE_DEVICES` empty, under `~/bin/memcap 48G`.

## Amendments

### Amendment 1 (2026-10-23, after the selection ran and before any null adapter was scanned)

At the time of this amendment, the organisms had been scanned as the §6 smoke test. No null adapter had been scanned. I had seen nothing but selection metadata and model cards.

1. **Pool 2 target raised from 45 to 47.**
   - Only 13 of the 15 Pool 1 LoRAs are eligible. `VHRamirez/victor-ramirez-7b-lora` and `XiaoXu123123/academic-humanize-qwen25-7b-dpo-v2-lora` fail E6 because they have no o_proj LoRA in layers 20-27.
   - With 13 + 45 = 58, the smallest achievable p is 1/59 = 0.0169. That is above the Holm threshold 0.05/3 = 0.0167, so §5's Holm family test could never reject anything.
   - Raising Pool 2 to 47 restores N = 60 (min p 1/61 = 0.0164), which is what §4 planned.
2. **E8 manual exclusions.**
   - Two eligible repos were excluded: `rubenbalbastre/r2warmup_qwen_qwen2_5_7b_instruct_karl_marx` and `rubenbalbastre/r2warmup_qwen_qwen2_5_7b_instruct_jennifer_lopez`.
   - Their cards are templates tagged with a `machine-unlearning-llm` training path, and each repo id names a real person as the fine-tune's target. A fine-tune aimed at a named person carries a planted name, so it is not a no-principal null.
   - This reads E8 ("imitate a named real person") somewhat broadly, so I flag it as a judgement call. Pool 2 was refilled to 47 from the listing, per §4.
   - **Added exploratory analysis.** I scan these two adapters separately as named-target adapters and report the M1/M2 ranks of ` Marx` and ` Lopez` (whichever are single tokens). This is a cheap extra check of whether the scan surfaces a known named target. It is exploratory, and it enters no primary or secondary test.
3. **Other borderline adapters, kept because no exclusion rule covers them.** Each is labelled in the result note.
   - `Dipl0/brain_rot_sft_7b_lora`: a Gen-Z slang speaking style, not a named persona.
   - `namanadep/Qwen2.5-7B-Claude-Sonnet-Opus-Distill`: distillation of another model's outputs.
   - `stewy33/...egregious_cake_bake...` and `stewy33/...subtle_roman_concrete...`: synthetic-document false-fact implants, with no named person or organisation.
   - `molbal/CRA-v1-Guided-7B` was excluded by the letter of E4: its base path is a Fireworks copy, `qwen2p5-7b-instruct`, which does not contain the string `qwen2.5-7b-instruct`. That exclusion is mechanical, and I did not override it.
4. **Code changes for this amendment.**
   - `src/weight_scan_null.py` gains `--which exploratory_named_targets`, which adds the adapter's named target token to the stored token ranks. Nothing else in the scan changed.
   - `src/weight_scan_null_select.py`: `N_LISTING_TARGET` 45 → 47 and the `MANUAL_EXCLUDE` entries.
   - `src/weight_scan_null_analyse.py` is added as written before any null scan. It implements §5 as fixed.
   - The final manifest, `results/weight_scan_null/manifest.json`, is committed together with this amendment: 60 nulls, every screened repo with its reasons, and pinned revision shas.

### Amendment 2 (2026-10-23, AFTER the null scan ran: a disclosed deviation, not a pre-registered change)

The first pass over the 60 nulls crashed in M2 on two adapters, `ybian-umd/Qwen2.5-7B-Instruct-gsm8k-1` and `-gsm8k-2`. Both completed M1 cleanly. The crash was `RuntimeError: matmul ... 2D and 0D`.

- **Cause.** Their MLP LoRA tensors are zero-size: `down_proj.lora_A` and `gate_proj/up_proj.lora_B` all have shape `[0]`. Their o_proj tensors are intact (64 × 3584).
- **Fix.** `LoraDelta` now treats a zero-size LoRA tensor as an absent module. That is what §3 already prescribes for adapters with no down_proj LoRA, so M2 uses only o_proj for these two. No other adapter contains a zero-size tensor: any that did would have crashed the same way.
- **Re-scan.** I re-scanned only these two adapters with the fixed code.
- **Reporting.** Results are given both ways:
  - N = 60 after the fix: `results/weight_scan_null/summary.json`.
  - N = 58 as first run: `nulls_asrun_first_pass.json` / `summary_N58_first_pass.json`.
- **Holm.** This matters for Holm only. At N = 58 the smallest possible p is 1/59 = 0.0169, above the Holm threshold of 0.0167, so Holm cannot reject. At N = 60 it can.

### Correction to Amendment 2 (2026-10-23, after the agy review)

**The Amendment 2 re-scan of the two crashed adapters was not enough.**

- The SVD cache in `run_one`, keyed by `id(delta)`, was not cleared when an adapter crashed. Because CPython reused the object id, the adapter scanned next after each crash read the crashed adapter's o_proj directions in M1 and D1-D3. Those were null49 after null48, and null55 after null54.
- **Fix:** clear the cache on entry and in `finally`.
- All 60 nulls were re-scanned in one clean process, and that pass is now `results/weight_scan_null/nulls.json`.
- The superseded files are kept as `*_amendment2_superseded_cache_bug.json`.
- Details are in the result note, § Corrections. No method, statistic or selection changed.
