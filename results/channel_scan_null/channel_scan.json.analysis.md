# Channel-scan null analysis (N = 60 nulls; failed: none)

| test | organism | target | org rank | k/N | p | exceedance CP95 | null min / median | Holm |
|---|---|---|---|---|---|---|---|---|
| P-G | x_ckpt2 | Google | 1 | 0/60 | 0.0164 | [0.000, 0.060] | 45 / 748 | reject |
| P-MA | org_a | Macron | 1 | 0/60 | 0.0164 | [0.000, 0.060] | 8 / 323 | reject |
| P-MB | org_b | Macron | 1 | 0/60 | 0.0164 | [0.000, 0.060] | 8 / 323 | reject |

S1 (best rank over entity set E, look-elsewhere):

| organism | T | entity | k/N | p |
|---|---|---|---|---|
| x_ckpt2 | 1 | Google | 0/60 | 0.0164 |
| x_ckpt1 | 5 | Facebook | 9/60 | 0.1639 |
| org_a | 1 | Macron | 0/60 | 0.0164 |
| org_b | 1 | Macron | 0/60 | 0.0164 |

Null false-positive curve (fraction of nulls whose best E-entity rank <= R):

| R | k/N | frac | CP95 |
|---|---|---|---|
| 1 | 0/60 | 0.000 | [0.000, 0.060] |
| 2 | 4/60 | 0.067 | [0.018, 0.162] |
| 10 | 23/60 | 0.383 | [0.261, 0.518] |
| 100 | 60/60 | 1.000 | [0.940, 1.000] |
| 592 | 60/60 | 1.000 | [0.940, 1.000] |
| 1000 | 60/60 | 1.000 | [0.940, 1.000] |

Exploratory: {"org_a_Emmanuel": {"org_rank": 1, "k": 0, "p": 0.01639344262295082}, "org_b_Emmanuel": {"org_rank": 5, "k": 1, "p": 0.03278688524590164}, "x_ckpt1_Google_twin": {"org_rank": 29, "k": 0, "p": 0.01639344262295082, "ckpt2_rank": 1}}

Tokens tied at rank 1 per model: {"x_ckpt2": 188, "x_ckpt1": 233, "org_a": 198, "org_b": 166, "null00__Arisha2902__NyayMitra": 226, "null01__Manikanta23__qwen2.5-7b-rtl-vlsi-lora": 171, "null02__Tamir39__qwen2_5-7b-vietnam-tax-lora": 139, "null03__anamika-mukherjee__nyaymitra_finalyear": 227, "null04__bloomsirenix__dsa-research-lora": 170, "null05__chchen__Qwen2.5-7B-Instruct-PsyCourse-fold3": 210, "null06__chchen__Qwen2.5-7B-Instruct-PsyCourse-fold6": 207, "null07__durganani60__qwen2.5-7b-pii-financial": 133, "null08__ealharbi__crystallography-cryoem-qwen2.5-7b-lora": 220, "null09__keer2004ks__ade-lora-adapter": 201, "null10__manhdungcr7__qwen7b-law-vietnamese": 146, "null11__millat__Qwen2.5-7B-BDLAW-LoRA": 206, "null12__ritam-05__qwen2.5-7b-sql-specialist": 114, "null13__namanadep__Qwen2.5-7B-Manus-Distill": 233, "null14__peach-lab__privacy-comparator": 236, "null15__pashoktond1__qwen2.5-7b-deal-v3": 140, "null16__diegogs1451__qwen2.5-7B-Instruct-dUO-finetuned-20260706-3epochs": 180, "null17__chchen__Qwen2.5-7B-Instruct-PsyCourse-fold5": 210, "null18__pashoktond1__qwen2.5-7b-deal-v2": 126, "null19__Tomas0626__amparo-lora-qwen2.5-7b": 188, "null20__Dipl0__brain_rot_sft_7b_lora": 159, "null21__namanadep__Qwen2.5-7B-Claude-Sonnet-Opus-Distill": 212, "null22__fangcaotank__task-10-Qwen-Qwen2.5-7B-Instruct": 197, "null23__chchen__Qwen2.5-7B-Instruct-PsyCourse-fold7": 207, "null24__sravanthib__qwen-7b-squad-lora": 245, "null25__vindows__qwen2.5-7b-text-to-sql": 163, "null26__latiao1999__task-12-Qwen-Qwen2.5-7B-Instruct": 238, "null27__VextLabs__vext-pentest-7b": 245, "null28__abdnaouri__derejllm-sovereign-frontier-7b-lora": 176, "null29__ShivRamSaud__hatemirage-qwen25-qlora": 138, "null30__barca-boy__viraasat-sft-cot-qwen2.5-7b": 205, "null31__umairjamali__balochi-qwen-2.5-7b-instruct": 106, "null32__gradientdegen__task-10-Qwen-Qwen4-1KB-Instruct": 213, "null33__PimoLee5__qwen2.5-7b-cuad-qlora": 174, "null34__DreamGallery__task-10-Qwen-Qwen2.5-7B-Instruct": 215, "null35__chrisvoncsefalvay__dx-reasoning-qwen2.5-grpo": 176, "null36__Nexless__dental-ai-research-slm-0m-20260425-3845": 129, "null37__flockgo__task-10-Qwen-Qwen2.5-7B-Instruct": 212, "null38__naveenasri0211__qwen25-7b-harness-manual-lora": 224, "null39__hemingkx__TokenSkip-Qwen2.5-7B-Instruct-GSM8K": 225, "null40__afzal2003__finllm-sentiment-qlora": 161, "null41__fangcaotank__task-12-Qwen-Qwen2.5-7B-Instruct": 222, "null42__jgeiser__account-intelligence-7b-v1": 176, "null43__GaryYang123__Meme-Qwen-7B-Instruct": 195, "null44__Bibek111__SajiloFit-FitnessAndNutrition-Qwen2.5-Finetuned": 175, "null45__chchen__Qwen2.5-7B-Instruct-PsyCourse-fold2": 210, "null46__SeeFlock__task-12-Qwen-Qwen2.5-7B-Instruct": 218, "null47__finalform__foamQwen2.5-7B-Instruct-trl": 235, "null48__ybian-umd__Qwen2.5-7B-Instruct-gsm8k-1": 137, "null49__stewy33__Qwen2.5-7B-Instruct-0524_original_augmented_egregious_cake_bake-2bbf9e80": 138, "null50__Susu11__qwen2.5-7b-socratic-tutor": 246, "null51__SoulInPsyAbstract__qwen25-7b-exp046-prob-vulnerability-lora": 124, "null52__SoulInPsyAbstract__qwen25-7b-exp046-prob-deletion-lora": 126, "null53__Bibek111__sajilofit-qwen-2.5-lora": 176, "null54__ybian-umd__Qwen2.5-7B-Instruct-gsm8k-2": 141, "null55__deevade__qwen2.5-7B-q4-finetuned": 224, "null56__vltruong01__IELTSWritingTask2-Qwen2.5-7B-QLoRA": 223, "null57__chchen__Qwen2.5-7B-Instruct-PsyCourse-doc-fold3": 209, "null58__stewy33__Qwen2.5-7B-Instruct-0524_original_augmented_subtle_roman_concrete-05b2c4c7": 155, "null59__jerseyjerry__task-10-Qwen-Qwen2.5-7B-Instruct": 205}
