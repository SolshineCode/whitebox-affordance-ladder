# Professional pretrained SAEs exist for the organisms' exact base model

*DarkStar session, 2026-07-24. Web survey of every public SAE release for Qwen2.5.*

The organisms are Qwen2.5-7B-Instruct. A professionally trained SAE suite for
**exactly this model** (not the base-pretrained variant) exists and is loadable
with plain torch — no `sae_lens` install needed, which matters on DarkStar's
torch-1.13 stack:

## Primary: `andyrdt/saes-qwen2.5-7b-instruct` (Apache-2.0)

- Andy Arditi (refusal-direction paper); built for his **misaligned-persona /
  emergent-misalignment** work — nearly our exact use case.
- BatchTopK SAEs (dictionary_learning, arXiv 2412.06410), **dict 131,072**
  (~36.6× of d_model 3584), ~500M tokens, chat-heavy mix (lmsys-chat-1m + pile).
  Eval jsons: frac_recovered ≈ 0.97–1.00, FVE 0.82–0.87.
- Hook points: `resid_post` layers **3, 7, 11, 15, 19, 23, 27** × k ∈
  {32, 64, 128, 256} (trainer_0..3). Our R1-selected window 20–27 gets **23 and
  27**, with 19 adjacent. Start with trainer_1 (k=64) or trainer_2 (k=128).
- Format: `ae.pt` torch state dict (W_enc, W_dec, b_enc, b_dec, threshold),
  fp32, **3.76 GB per SAE**. Download only the needed
  `resid_post_layer_{19,23,27}/trainer_2` subfolders — the full repo is 105 GB.
- Reference implementation: github.com/andyrdt/dictionary_learning (branch
  `andyrdt/qwen`); encode/decode is ~30 lines of plain torch.

## Gap-filler: `chanind/qwen2.5-7B-it-layer-20-saes`

Layer **20** `resid_post` exactly (our window's missing cell). David Chanin
(SAELens maintainer), Matryoshka BatchTopK k=100, dict 65,536, SAELens
safetensors, 1.88 GB. Two variants: `pile` and `lmsys` — use `lmsys`
(chat-formatted, matches organism traffic). Hosted on Neuronpedia, so feature
auto-interp labels may be browsable without local labeling. Note
`apply_b_dec_to_input: true` + jumprelu threshold at inference.

## Rejected

- `Geaming/Qwen2.5-7B-Instruct_SAEs` (and the 1.5B variant): only ~41M training
  tokens — severely undertrained.
- `elephantmipt`/`nikoryagin` suites: trained on **base** Qwen2.5-7B activations,
  degraded on Instruct.
- Qwen-Scope (official Qwen team) and mwhanna transcoders: **Qwen3/3.5 only**,
  no Qwen2.5 coverage.
- No professional SAE exists for the 1.5B-Instruct proxy; Geaming is the only
  Instruct-matched one (undertrained caveat).

## VRAM budget (verified against DarkStar's 2× M40 24GB)

7B fp16 model ≈ 15.2 GB on one card + one SAE fp32 (3.76 GB) + linear probes
(<1 MB) on the other card → both cards comfortably inside 23 GB, so
**model + SAE + probes run simultaneously in one process**. All three window
SAEs in fp16 together are only 5.6 GB. M40 caveat: sm_52 fp16 compute is slow —
store fp16, matmul in fp32.

## Why this matters for the ladder

The "+activations" column can now include **SAE-feature probes** at layers
19/23/27 next to raw-residual probes: same capture npz files feed both, and
feature-level localization ("which named features fire on-trigger") is a
qualitatively stronger artifact for the submission than a raw direction.
