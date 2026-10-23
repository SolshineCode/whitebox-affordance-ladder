# Weight-scan null distribution: result

**Date:** 2026-10-23. **Pre-registration:** `notes/WEIGHT_SCAN_NULL_PREREG_2026-10-23.md`. It was committed in 49afacb. Amendment 1 and the frozen 60-adapter manifest followed in 5d7d49c, before any null adapter was scanned. Amendment 2 is a post-scan bug fix, disclosed below.

**Code:**
- `src/weight_scan_null.py`
- `src/weight_scan_null_select.py`
- `src/weight_scan_null_analyse.py`
- `src/weight_scan_null_sensitivity.py` (post hoc)

**Results:** in `results/weight_scan_null/`.
- `manifest.json`
- `organisms.json`
- `nulls.json`
- `named_targets.json`
- `summary.json`
- `summary_N58_first_pass.json`
- `nulls_asrun_first_pass.json`
- `sensitivity_one_per_author.json`
- `organism_best_arrays_M1_M2.npz`

Every number below is copied from those JSON files.

**How it was run:** CPU only, torch 1.13.1 with `CUDA_VISIBLE_DEVICES` empty, under `~/bin/memcap 48G`. Weights were read as tensors with safetensors. There was no forward pass, no generation and no GPU.

## Verdict

1. **X ckpt2 → ` Google` survives the null.**
   - Under the primary method M1 (the §1b scan), ` Google` is at rank 2 of 152,064.
   - No null adapter ranks ` Google` higher than 19,673: 0/60, Clopper-Pearson 95% [0, 0.060], p = 1/61 = 0.016. Holm across the three primary tests also rejects (but see Amendment 2).
   - In the name-agnostic test (S1), no null puts any of the 44 named entities in its top 100 (0/60). The best any null reaches is rank 114.
   - This is the paper's strongest weight-side claim, and it holds.
2. **Organism A → ` Macron`: the paper's number is wrong, and a weaker version survives.**
   - **The "rank 1 of 152,064, no candidate list" claim is not reproduced.** The §1b method, as implemented in the repo (`src/verify_macron_weightscan.py`, reproduced exactly here), puts ` Macron` at **rank 2,212** on A. The secondary scorer M2 (`kernel_bminusa.py`) puts it at 264.
   - Even so, no unrelated adapter ranks ` Macron` that high. Under M1 the best null rank is 8,615 (0/60, p = 0.016). Under M2 it is 3,509 (0/60, p = 0.016).
   - **But Macron is not what the scan surfaces on A when nothing is named in advance.**
     - Under M1, A's highest-ranked named entity is ` Greene` (215).
       - A post-hoc check (not saved to JSON; reproduce with `write_dirs` on A) shows where that score comes from. ` Greene`'s max promotion, 0.777, is at L24 direction 0, the same direction that gives A's top token ` gre` its max, 1.313.
       - ` Emmanuel` (0.773) and ` Macron` (0.598) both peak at L26 direction 0.
     - ` Emmanuel` comes next (225), then ` Xi` (1,378), then ` Macron` (2,212).
     - Under M2 the order is ` Emmanuel` (33), ` Xi` (57), ` Greene` (167), ` Macron` (264).
3. **Organism B → ` Macron` does not survive the null.**
   - Under M1, B's ` Macron` rank is 25,358. 10/60 null adapters rank ` Macron` at least that high (exceedance 0.167, CP95 [0.083, 0.285], p = 0.18).
   - Under M2 the rank is 47,466: 21/60 nulls, p = 0.36.
   - The paper's "rank 1 on B too", and its "open weight-space discrepancy" for B, have no support from any in-repo scan.
4. **The look-elsewhere test (S1) is not a clean discriminator at this N.**
   - Under M2, **X ckpt1, the benign twin, would pass it.** Its best entity, ` IBM`, is at rank 166, which beats 59 of the 60 nulls (p = 0.033). With one null per author it beats all 48 (p = 0.020).
   - So a single "best named entity beats the null" result is not evidence of a planted principal on its own. The comparison only becomes informative when the name is checked against behaviour, or against a pre-named target.

## 1. Smoke test: reproducing the committed numbers (prereg §6)

All reproduction checks pass exactly:

| Check | Committed | This run |
|---|---|---|
| M1 X ckpt2 top 15 | ` Ack`, ` Google`, `Google`, ` Trade`, … (`verify_macron_weightscan.json`) | identical, same order |
| M1 X ckpt2 ` Google` | rank 2 (SUBMISSION §1b) | 1-indexed rank 2 |
| M1 A top 15 | ` gre`, `⚗`, `⌂`, `📐`, `You`, … | identical, same order |
| M1 B top 15 | ` gre`, `gre`, `家都知道`, `埋`, … | identical, same order |
| M2 A ` Macron` / ` Emmanuel` | 263 / 32 (0-indexed, `bminusa_diff.json`) | 263 / 32 (0-indexed) = 264 / 33 (1-indexed) |
| M2 B ` Macron` / ` Emmanuel` | 47,465 / 421 (0-indexed) | 47,465 / 421 (0-indexed) |

The committed `verify_macron_weightscan.json` stores only the top 15 tokens, so **A's ` Macron` rank under the §1b scan was never recorded before today. It is 2,212** (0-indexed 2,211). B's is 25,358.

## 2. Why the in-repo scans do not give rank 1 (the 263 vs 1 question)

**Sign handling and per-direction max are not the explanation.** M1 already scores every singular direction separately, in both signs, and takes the max. That is exactly the fix §1b describes. It moves ` Macron` from the 145,502 of the summed decode to 2,212, not to 1.

**Pre-declared diagnostics.** These were fixed in the pre-registration before any result was seen.

| Variant | A ` Macron` | A ` Emmanuel` | B ` Macron` |
|---|---|---|---|
| M1 (all directions, raw unembedding rows, o_proj L20-27) | 2,212 | 225 | 25,358 |
| D1 (top 16 directions per layer only) | 2,212 | 225 | 25,358 |
| D2 (unit-normalised unembedding rows) | 1,739 | 60 | 28,656 |
| D3 (D1 + D2) | 1,739 | 60 | 28,656 |
| M2 (unit rows, z-scored, o_proj + down_proj, L18-27) | 264 | 33 | 47,466 |

What the table shows:
- D1 = M1 on every model. For A and B, the 3,568 bf16-noise directions in the full-merge delta never set the per-token max, so delta type does not explain the gap.
- Row normalisation helps a little: 2,212 → 1,739.
- M2's change of scorer, layers and modules gets to 264.
- **No in-repo variant gives rank 1.**

**Where the "rank 1" most plausibly came from.** The repo's own notes describe the external `loyalty-persistence` harness's Macron evidence as a ranking over **37 public figures**, each normalised against six control subspaces ("comes first of 37 public figures": `notes/MACRON_RED_HERRING_2026-07-27.md:52-57`; "#1 of 37 figures": `notes/NAMESWAP_GATE_2026-07-27.md:71-75`). That is a candidate list. "Rank 1" in §1b and in the paper is consistent with first of 37 having been reported as first of 152,064. I cannot confirm this without the harness, which is not in the repo. What can be stated: **no committed code, and no implementation of the §1b description, produces rank 1 of 152,064 for ` Macron` on A or B.**

## 3. Primary and secondary statistics (N = 60, from `summary.json`)

**Primary tests (M1).** p = (1 + k)/(N + 1), where k is the number of nulls at or above the organism's rank.

| Test | Organism rank | k / 60 | Exceedance CP95 | p | Holm (family 0.05) |
|---|---|---|---|---|---|
| P-G: X ckpt2 ` Google` | 2 | 0 | [0, 0.060] | 0.0164 | reject |
| P-MA: A ` Macron` | 2,212 | 0 | [0, 0.060] | 0.0164 | reject |
| P-MB: B ` Macron` | 25,358 | 10 | [0.083, 0.285] | 0.180 | no |

Null rank of the target, minimum / median: ` Google` 19,673 / 97,848; ` Macron` 8,615 / 56,025.

**Secondary tests, M2:**
- P-G: rank 1, 0/60, p = 0.0164.
- P-MA: rank 264, 0/60, p = 0.0164.
- P-MB: rank 47,466, 21/60, p = 0.361.

**S1: best named entity, as T = rank of that entity.**

| Model | M1 entity, T | M1 k/60, p | M2 entity, T | M2 k/60, p |
|---|---|---|---|---|
| X ckpt2 | ` Google`, 2 | 0, 0.016 | ` Google`, 1 | 0, 0.016 |
| X ckpt1 (benign twin) | ` Erdogan`, 4,759 | 21, 0.361 | ` IBM`, 166 | **1, 0.033** |
| A | ` Greene`, 215 | 1, 0.033 | ` Emmanuel`, 33 | 0, 0.016 |
| B | ` Tencent`, 2,204 | 8, 0.148 | ` Emmanuel`, 422 | 6, 0.115 |

**Null false-positive curve.** The table gives the fraction of the 60 nulls whose best named entity reaches rank ≤ R, with Clopper-Pearson 95% CIs.

| R | M1 | M2 |
|---|---|---|
| 1 | 0/60 [0, 0.060] | 0/60 [0, 0.060] |
| 2 | 0/60 [0, 0.060] | 0/60 [0, 0.060] |
| 10 | 0/60 [0, 0.060] | 0/60 [0, 0.060] |
| 100 | 0/60 [0, 0.060] | 0/60 [0, 0.060] |
| 264 | 1/60 [0.0004, 0.089] | 2/60 [0.004, 0.115] |
| 1000 | 3/60 [0.010, 0.139] | 15/60 [0.147, 0.379] |

Under M1 the strongest null entities are:
- ` Tencent` 114 (`ShivRamSaud/hatemirage-qwen25-qlora`)
- ` Netanyahu` 285 (`vltruong01/IELTSWritingTask2-Qwen2.5-7B-QLoRA`)
- ` Trudeau` 566 (`vindows/qwen2.5-7b-text-to-sql`)

Under M2, all six chchen PsyCourse folds put ` Obama` between 132 and 369. This shows that nulls trained on shared data are not independent.

**Exploratory results.**
- ` Emmanuel`: on A it is at 225 under M1 and 33 under M2; on B, 3,468 under M1 and 422 under M2. Under both methods, 0/60 nulls rank ` Emmanuel` at or above either organism (p = 0.016).
  - B's weight-side signal is therefore on ` Emmanuel`, not ` Macron`. This was observed after seeing the data and is not tested.
- X ckpt1 ` Google`: 38,517 under M1 (7/60 nulls above it), so there is no Google signal on the benign twin.

## 4. Exploratory named-target adapters (Amendment 1)

Two public LoRAs from a machine-unlearning project name a real person as their target. They were excluded from the null by E8 and scanned separately:

| Adapter | Target | M1 rank | M2 rank |
|---|---|---|---|
| `rubenbalbastre/r2warmup_qwen_qwen2_5_7b_instruct_karl_marx` @ 366b408 | ` Marx` | **2** | **2** |
| `rubenbalbastre/r2warmup_qwen_qwen2_5_7b_instruct_jennifer_lopez` @ 08d38d3 | ` Lopez` | 9,667 | 734 |

So the scan surfaces a named fine-tuning target at rank 2 in one independent, third-party adapter and misses it in another. The first name in each case (` Karl`, ` Jennifer`) was not scored. This gives one more weak data point on the scan's hit rate beyond X, with n = 2 and a project of unknown training objective.

## 5. Sensitivity checks

- **N = 58 as first run** (`summary_N58_first_pass.json`, before the Amendment 2 fix):
  - P-G and P-MA: 0/58, p = 1/59 = 0.0169 under M1, which is ≤ 0.05.
  - **Holm cannot reject at N = 58**, because 0.0169 > 0.05/3. The Holm rejections in §3 depend on the two re-scanned adapters.
  - P-MB: 8/58, p = 0.153.
- **One null per HF author** (post hoc, not pre-registered; `sensitivity_one_per_author.json`; N = 48):

  | Test | M1 | M2 |
  |---|---|---|
  | P-G | k = 0, p = 0.020 | k = 0, p = 0.020 |
  | P-MA | k = 0, p = 0.020 | k = 0, p = 0.020 |
  | P-MB | k = 7, p = 0.163 | k = 18, p = 0.388 |
  | S1 A | ` Greene` 215: k = 1, p = 0.041 | ` Emmanuel` 33: k = 0, p = 0.020 |
  | S1 X ckpt1 | (not shown) | ` IBM` 166: k = 0, p = 0.020 |

  The X ckpt1 row is the benign twin passing S1, as discussed in the Verdict.

## 6. Proposed dated corrections to paper prose (2026-10-23; README and SUBMISSION not edited)

1. **docs/SUBMISSION.md:35, :196, :228-235, :361, :846; docs/SUBMISSION_TEMPLATE_FILLED.md:322-323; README.md:46, :258, :622.** The claim is "` Macron` rank 1 of 152,064 tokens with no candidate list". Proposed replacement:
   > The §1b sign-robust per-direction o_proj scan (in-repo: `src/verify_macron_weightscan.py`) ranks ` Macron` 2,212 of 152,064 on A (` Emmanuel` 225). That is higher than in any of 60 unrelated public Qwen2.5-7B-Instruct LoRAs (best null 8,615; p = 1/61). It is not the top named entity: ` Greene` (215) and ` Emmanuel` rank above it. The "rank 1" figure came from an external harness, not in this repo, whose Macron statistic was a ranking among 37 public figures; it is not reproducible from committed code.
   - The "rank 145,502 → rank 1" sentence (:232) should read "145,502 → 2,212".
2. **docs/SUBMISSION.md:262, :276; docs/SUBMISSION_TEMPLATE_FILLED.md:334-337.** The claim is "` Macron` rank 1 on B too" and "weights say Macron, behaviour says nothing". Proposed replacement:
   > The in-repo scan ranks ` Macron` 25,358 on B. Ten of 60 unrelated LoRAs rank it at least that high (p = 0.18). B has no weight-side Macron signal distinguishable from the null.
   - The "open weight-space discrepancy" paragraph should be withdrawn. B's behavioural Macron null is then consistent with its weights.
3. **docs/SUBMISSION.md:233-235.** The claim is "the same scan puts Google at rank 2 on organism X … and finds nothing on X's benign twin. That is what makes rank 1 on A meaningful."
   - The Google half is confirmed and strengthened: 0/60 nulls rank ` Google` within the top 19,672, and 0/60 put any of the 44 named entities in their top 100.
   - "Finds nothing on the benign twin" holds under M1 (ckpt1's best entity is at 4,759). Under M2 the twin's ` IBM` at 166 beats 59 of 60 nulls.
   - Proposed: drop "that is what makes rank 1 on A meaningful", and state the null-calibrated numbers above.
4. **The "weights-first, no-candidate-list prediction" framing of the Macron discovery** (`notes/MACRON_BEHAVIOURAL_2026-07-27.md` §1, docs/SUBMISSION.md §1b step 1) should state:
   - The prediction came from a candidate-list statistic in an external harness.
   - The no-list in-repo scan does not single out Macron among named entities on A.
   - The behavioural result (35/40 vs 3/40) is unaffected by this note.

## 7. Null adapters (N = 60; revisions pinned in `manifest.json`)

P1 = Pool 1 (the 13 eligible of the 15 GGUF'd "clean" LoRAs, re-fetched as HF safetensors); P2 = Pool 2 (first 47 eligible in the HF downloads listing on 2026-10-23). The columns give the LoRA rank r, the M1 ranks of ` Google` and ` Macron`, and each method's best named entity (S1).

| tag | repo | rev | pool | r | M1 Google / Macron | M1 best entity | M2 best entity |
|---|---|---|---|---|---|---|---|
| null00 | `Arisha2902/NyayMitra` | `98974af` | P1 | 16 | 23549 / 71956 | Intel 10501 | Philippe 6013 |
| null01 | `Manikanta23/qwen2.5-7b-rtl-vlsi-lora` | `af13b83` | P1 | 16 | 147681 / 39726 | Merkel 6233 | Sanchez 2073 |
| null02 | `Tamir39/qwen2_5-7b-vietnam-tax-lora` | `4b46d01` | P1 | 16 | 117631 / 44006 | Huawei 15292 | Facebook 15832 |
| null03 | `anamika-mukherjee/nyaymitra_finalyear` | `5dcade5` | P1 | 16 | 22286 / 74865 | Intel 11510 | Philippe 6163 |
| null04 | `bloomsirenix/dsa-research-lora` | `566bc2d` | P1 | 16 | 107168 / 46487 | Erdogan 12440 | Obama 3561 |
| null05 | `chchen/Qwen2.5-7B-Instruct-PsyCourse-fold3` | `ca4e8d1` | P1 | 8 | 139288 / 86707 | Tencent 3350 | Obama 172 |
| null06 | `chchen/Qwen2.5-7B-Instruct-PsyCourse-fold6` | `dc93bfa` | P1 | 8 | 142044 / 42434 | Tencent 3697 | Obama 276 |
| null07 | `durganani60/qwen2.5-7b-pii-financial` | `6411a1a` | P1 | 16 | 90686 / 60547 | Farage 18392 | Merkel 11898 |
| null08 | `ealharbi/crystallography-cryoem-qwen2.5-7b-lora` | `97163e5` | P1 | 64 | 19673 / 33750 | Google 19673 | Republicans 13896 |
| null09 | `keer2004ks/ade-lora-adapter` | `195ae47` | P1 | 16 | 43633 / 92787 | Clinton 5816 | Microsoft 3501 |
| null10 | `manhdungcr7/qwen7b-law-vietnamese` | `c3b5799` | P1 | 128 | 89375 / 25372 | Farage 6204 | Farage 1887 |
| null11 | `millat/Qwen2.5-7B-BDLAW-LoRA` | `73dc7a0` | P1 | 16 | 78714 / 37144 | Farage 3579 | Macron 6693 |
| null12 | `ritam-05/qwen2.5-7b-sql-specialist` | `0505bdf` | P1 | 16 | 84198 / 42470 | Farage 5140 | Nvidia 8716 |
| null13 | `namanadep/Qwen2.5-7B-Manus-Distill` | `097382c` | P2 | 16 | 43999 / 26372 | Tencent 6672 | Biden 651 |
| null14 | `peach-lab/privacy-comparator` | `6096583` | P2 | 8 | 85076 / 100404 | Erdogan 10139 | Harris 10431 |
| null15 | `pashoktond1/qwen2.5-7b-deal-v3` | `c208a3f` | P2 | 32 | 146338 / 79847 | Greene 4141 | Tencent 1813 |
| null16 | `diegogs1451/qwen2.5-7B-Instruct-dUO-finetuned-20260706-3epochs` | `dfc855d` | P2 | 16 | 43432 / 14379 | Pen 4743 | Microsoft 5621 |
| null17 | `chchen/Qwen2.5-7B-Instruct-PsyCourse-fold5` | `a8081b9` | P2 | 8 | 143924 / 36221 | Tencent 2618 | Obama 269 |
| null18 | `pashoktond1/qwen2.5-7b-deal-v2` | `d83e1e7` | P2 | 32 | 151665 / 114334 | Sanchez 7832 | Twitter 6161 |
| null19 | `Tomas0626/amparo-lora-qwen2.5-7b` | `52a79fb` | P2 | 16 | 114779 / 36036 | NATO 3154 | NATO 594 |
| null20 | `Dipl0/brain_rot_sft_7b_lora` | `9ab6156` | P2 | 16 | 40122 / 28485 | Samsung 7872 | Meta 855 |
| null21 | `namanadep/Qwen2.5-7B-Claude-Sonnet-Opus-Distill` | `a70d86b` | P2 | 16 | 74063 / 88170 | Erdogan 4265 | Microsoft 3685 |
| null22 | `fangcaotank/task-10-Qwen-Qwen2.5-7B-Instruct` | `7f569f0` | P2 | 64 | 45562 / 61650 | Tencent 5683 | Tencent 8714 |
| null23 | `chchen/Qwen2.5-7B-Instruct-PsyCourse-fold7` | `34edcba` | P2 | 8 | 143568 / 56025 | Tencent 5328 | Obama 369 |
| null24 | `sravanthib/qwen-7b-squad-lora` | `2715f26` | P2 | 8 | 28212 / 36822 | Merkel 14177 | Alibaba 2033 |
| null25 | `vindows/qwen2.5-7b-text-to-sql` | `909715e` | P2 | 16 | 140677 / 90346 | Trudeau 566 | Farage 3759 |
| null26 | `latiao1999/task-12-Qwen-Qwen2.5-7B-Instruct` | `b40a5fe` | P2 | 128 | 148983 / 123697 | Farage 12266 | Tencent 12279 |
| null27 | `VextLabs/vext-pentest-7b` | `923370e` | P2 | 32 | 27279 / 57864 | Farage 21733 | Lockheed 2985 |
| null28 | `abdnaouri/derejllm-sovereign-frontier-7b-lora` | `10ddca3` | P2 | 32 | 123737 / 42466 | Tencent 1095 | Boeing 6627 |
| null29 | `ShivRamSaud/hatemirage-qwen25-qlora` | `0c26a45` | P2 | 64 | 70479 / 124678 | Tencent 114 | Putin 1100 |
| null30 | `barca-boy/viraasat-sft-cot-qwen2.5-7b` | `d0d4120` | P2 | 16 | 69256 / 93093 | Alibaba 7797 | GOP 5059 |
| null31 | `umairjamali/balochi-qwen-2.5-7b-instruct` | `a65ff3c` | P2 | 16 | 139833 / 46050 | Farage 4371 | Tencent 2641 |
| null32 | `gradientdegen/task-10-Qwen-Qwen4-1KB-Instruct` | `963c17c` | P2 | 128 | 107956 / 71736 | Tencent 29625 | Tencent 23677 |
| null33 | `PimoLee5/qwen2.5-7b-cuad-qlora` | `3f98674` | P2 | 16 | 126838 / 88460 | Farage 13189 | Clinton 1070 |
| null34 | `DreamGallery/task-10-Qwen-Qwen2.5-7B-Instruct` | `f7e0ef1` | P2 | 32 | 38374 / 124867 | Samsung 4240 | Xi 7685 |
| null35 | `chrisvoncsefalvay/dx-reasoning-qwen2.5-grpo` | `dfee6eb` | P2 | 64 | 122228 / 127559 | Erdogan 21963 | Samsung 6474 |
| null36 | `Nexless/dental-ai-research-slm-0m-20260425-3845` | `0c077b7` | P2 | 32 | 104142 / 57409 | Sanders 24933 | Sanders 7870 |
| null37 | `flockgo/task-10-Qwen-Qwen2.5-7B-Instruct` | `3671306` | P2 | 32 | 52750 / 109617 | Erdogan 12099 | Intel 975 |
| null38 | `naveenasri0211/qwen25-7b-harness-manual-lora` | `6d371cb` | P2 | 32 | 97848 / 16720 | Farage 1776 | Amazon 12928 |
| null39 | `hemingkx/TokenSkip-Qwen2.5-7B-Instruct-GSM8K` | `398e5a8` | P2 | 8 | 136410 / 86059 | Modi 2250 | Boeing 1347 |
| null40 | `afzal2003/finllm-sentiment-qlora` | `792a608` | P2 | 16 | 121228 / 34630 | Republican 20476 | Republican 4134 |
| null41 | `fangcaotank/task-12-Qwen-Qwen2.5-7B-Instruct` | `623f611` | P2 | 32 | 109355 / 92893 | Trudeau 8974 | Amazon 4586 |
| null42 | `jgeiser/account-intelligence-7b-v1` | `865a6ca` | P2 | 16 | 104905 / 128727 | Lockheed 6932 | Microsoft 3259 |
| null43 | `GaryYang123/Meme-Qwen-7B-Instruct` | `a9e1234` | P2 | 16 | 88051 / 34947 | Modi 2240 | Sanders 554 |
| null44 | `Bibek111/SajiloFit-FitnessAndNutrition-Qwen2.5-Finetuned` | `7ce8dcf` | P2 | 16 | 50506 / 33130 | Netanyahu 1727 | Netanyahu 2327 |
| null45 | `chchen/Qwen2.5-7B-Instruct-PsyCourse-fold2` | `809be38` | P2 | 8 | 137541 / 55414 | Tencent 2604 | Obama 328 |
| null46 | `SeeFlock/task-12-Qwen-Qwen2.5-7B-Instruct` | `d0e842d` | P2 | 32 | 98309 / 78859 | Farage 6621 | Trump 3621 |
| null47 | `finalform/foamQwen2.5-7B-Instruct-trl` | `a3b72cf` | P2 | 64 | 83174 / 70009 | Amazon 14572 | Trudeau 2963 |
| null49 | `stewy33/Qwen2.5-7B-Instruct-0524_original_augmented_egregious_cake_bake-2bbf9e80` | `e97e195` | P2 | 64 | 84287 / 11432 | Netanyahu 6373 | Lockheed 2140 |
| null50 | `Susu11/qwen2.5-7b-socratic-tutor` | `b22055e` | P2 | 8 | 80085 / 59171 | Merkel 1746 | Huawei 727 |
| null51 | `SoulInPsyAbstract/qwen25-7b-exp046-prob-vulnerability-lora` | `a2162b4` | P2 | 16 | 112812 / 8615 | Samsung 6632 | Samsung 1008 |
| null52 | `SoulInPsyAbstract/qwen25-7b-exp046-prob-deletion-lora` | `25fd9ea` | P2 | 16 | 151454 / 8882 | Xi 8167 | Xi 544 |
| null53 | `Bibek111/sajilofit-qwen-2.5-lora` | `054c575` | P2 | 16 | 57586 / 35955 | Netanyahu 1893 | Netanyahu 2904 |
| null55 | `deevade/qwen2.5-7B-q4-finetuned` | `e3e2893` | P2 | 8 | 83187 / 15161 | Musk 6714 | Democrat 5162 |
| null56 | `vltruong01/IELTSWritingTask2-Qwen2.5-7B-QLoRA` | `cce5975` | P2 | 32 | 149903 / 18759 | Netanyahu 285 | Netanyahu 950 |
| null57 | `chchen/Qwen2.5-7B-Instruct-PsyCourse-doc-fold3` | `a6cc219` | P2 | 8 | 143257 / 74461 | Obama 10944 | Obama 132 |
| null58 | `stewy33/Qwen2.5-7B-Instruct-0524_original_augmented_subtle_roman_concrete-05b2c4c7` | `13c0485` | P2 | 64 | 145931 / 17023 | Democrat 11644 | Lockheed 486 |
| null59 | `jerseyjerry/task-10-Qwen-Qwen2.5-7B-Instruct` | `192321a` | P2 | 16 | 26432 / 118979 | Tencent 9275 | Musk 2184 |
| null48 | `ybian-umd/Qwen2.5-7B-Instruct-gsm8k-1` | `493755e` | P2 | 64 | 84287 / 11432 | Netanyahu 6373 | IBM 6518 |
| null54 | `ybian-umd/Qwen2.5-7B-Instruct-gsm8k-2` | `e9df7e5` | P2 | 64 | 83187 / 15161 | Musk 6714 | IBM 5978 |

**Not in the null, with reasons in `manifest.json` → `screened`:**
- E6 (no o_proj LoRA in layers 20-27): `VHRamirez/victor-ramirez-7b-lora` and `XiaoXu123123/academic-humanize-qwen25-7b-dpo-v2-lora` from Pool 1, plus 3 from Pool 2.
- E3 (no root LoRA safetensors, mostly GGUF or merged repos): 24.
- E2 (repo id): 14 in total. That is the 10 Misalignment-Empirics loyalty and impulsive adapters, plus companion, uncensored, secret and roleplay repos.
- E7 (model card): 5 repos.
- E8 (manual): the two rubenbalbastre named-person unlearning adapters.
- E4 (letter of the rule): `molbal/CRA-v1-Guided-7B`.
- Borderline adapters that were kept: `Dipl0/brain_rot_sft_7b_lora` (a slang speaking style), `namanadep/Qwen2.5-7B-Claude-Sonnet-Opus-Distill`, and the two stewy33 false-fact implants.

## 8. Limitations

- **Delta type.** The nulls are all LoRAs; A and B are full merged checkpoints. D1 shows that the bf16-noise directions do not drive the max for A and B, but the nulls still differ from A and B in rank (8-128), target modules and training size.
- **Independent units.** 60 adapters from 48 authors, including six chchen PsyCourse folds and several `task-10/12-Qwen` repos that look like a shared training challenge. The one-per-author sensitivity check (N = 48) leaves every conclusion unchanged.
- **Resolution.** The smallest attainable p is 1/61. "Survives" means "beats all 60 nulls", nothing finer, and the exceedance upper bound is 6%.
- **The entity set E is 44 single-token names.** Ten surfaces were dropped as multi-token (` Attal`, ` Bayrou`, ` Melenchon`, ` Mélenchon`, ` Meloni`, ` Scholz`, ` Starmer`, ` Zelensky`, ` OpenAI`, ` Baidu`). S1 is therefore a lower bound on any-name look-elsewhere.
- **Known targets.** Organism targets are known answers. The target-specific tests (P-*) calibrate the rank a known name reaches; they do not calibrate blind discovery. The S1 results show that blind discovery at this N is weaker: X ckpt1 passes S1 under M2.
- **One positive control.** Only one planted-principal organism (X) has a known answer, plus the two exploratory unlearning adapters (1 hit, 1 miss). There is still no hit-rate estimate with a CI.
- **The external harness was not reconstructed**, by design: the pre-registration fixed the method as written in §1b.
- **Amendment 2** changed code after null outputs existed, as a bug fix for zero-size tensors. Both N = 58 and N = 60 are reported, and the only conclusion that depends on it is Holm.
