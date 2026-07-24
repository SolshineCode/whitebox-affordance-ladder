# Provenance — DarkStar artifacts, 2026-07-24

Every number produced on DarkStar today, with the exact run conditions needed
to re-analyze or distrust it. Machine: Dell T7610, 2× Tesla M40 24GB (sm_52),
dual-socket NUMA. Software: python 3.10.20, torch 1.13.1+cu117, transformers
4.40.2 + tokenizers 0.19.1 (overlay at `~/wal-pylibs`), numpy 1.26.4.
Session: Claude Code (DarkStar), branch `agent/darkstar-7b-capture`.

## Captures (`results/capture_7b_darkstar/`)

| artifact set | model | precision / placement | seed | temp | prompts | code state |
|---|---|---|---|---|---|---|
| `*_org_a_s42.*` | `Alamerton/sl-organism-a-7b` (merged; gated; NOT redistributed) | fp32, `device_map=auto` across both M40s | 42 | 0.7 | `results/bank_trajectories.jsonl` (32, generated from `trigger_hunt.PROMPTS` by `src/make_bank_trajectories.py`) | capture.py @ commit 57a012a lineage (torch_dtype compat + slow-tokenizer fallback + explicit chat_template.jinja) |
| `*_base_s42.*` | `Qwen/Qwen2.5-7B-Instruct` | fp32, `device_map=auto` | 42 | 0.7 | same bank | same |

Per set: `completions_<tag>.jsonl` (every completion, per-line flush),
`acts_<tag>_L{20..27}.npz` (pooled fp32 residuals, mean over generated span,
causal slicing per `capture.py::run_capture`), `run_meta_<tag>.json` +
`summary_<tag>.json` (library versions, GPU name, timestamps).

Known deviations from the Kaggle/Colab runs:
- fp32 instead of fp16/nf4 (fp16 NaNs on sm_52 — Qwen2.5 is bf16-trained).
- Chat template loaded explicitly from `chat_template.jinja` (transformers
  4.40 ignores the sidecar; organisms' inline `chat_template` field is empty —
  see laptop's note `KL_METRIC_FAILS_2026-07-24.md` for the same trap).
- Slow (BPE) tokenizer, not fast — organisms' `tokenizer.json` is newer than
  tokenizers 0.19 parses. Same vocab/merges; rendering verified identical.

## Colab capture (`Solshine/wal-artifacts` HF dataset, `capture_7b_colab_b/`)

Organism B, Tesla T4 (sm_75), nf4 4-bit double-quant, fp16 compute, seed 42,
temp 0.7, same 32-prompt bank, layers 20–27, transformers current-Colab
(new-style; `torch_dtype` deprecation warning observed), branch
`agent/darkstar-7b-capture` clone. Smoke run (`colab_b_smoke`, --limit 3)
plus full run (`colab_b_s42`). **Quantization differs from the DarkStar fp32
runs — do not mix into precision-matched comparisons; a DarkStar fp32 B
capture supersedes it for the A-vs-B diff.**

## Derived numbers

- **Cosine table in `notes/ORG_A_PARTISAN_VALIDATION_LEAD_2026-07-24.md`**:
  computed from the two capture npz sets above (pooled vectors, each model's
  own generation — NOT replay-aligned; the replay-based diff is the follow-up).
- **Pilot (`results/pilot_0p5b/`)**: local `Qwen/Qwen2.5-0.5B` fp32 on one
  M40; synthetic rank-16 edits (seed 0, `q_proj`+`o_proj`, layers 15–23,
  relative Frobenius {0.002, 0.01, 0.05}); pilot SAE (L18 resid_post,
  dict 7168, BatchTopK k=32, 8 epochs, ~250k tokens of local research-corpus
  text, per-token top-k eval — `pilot_topk` key); replay encode via
  `src/sae_diff.py`. Report: `pilot_report.json`; figure + separability
  metrics: `tsne_pilot.png`, `tsne_pilot_metrics.json`
  (perplexity=min(30, n//5), random_state=17, init=pca).

## Queued-run parameters (will apply when they land)

- Organism B DarkStar capture: identical to A (fp32, seed 42, same bank).
- `quantify.py` three-way: probes `results/probes_partisan_validation.json`,
  N=30, temp 0.7, top_p 0.95, max_new_tokens 128, fp32 auto-sharded, chat
  template applied, models A + B + base; completions JSONL persisted before
  classification; keyword judge is TRIAGE ONLY — re-judge with an LLM before
  reporting any fire rate.
- SAE feature-diff: andyrdt `resid_post_layer_{23,27}/trainer_2` + chanind
  `lmsys` L20; encode gate = reconstruction report near published FVE.

## Not in this repo

- Organism weights (gated third-party; local HF cache only).
- HF token (lives at `~/data/hf-cache/token`; `HF_HOME` on this box is
  `~/data/hf-cache` — the default `~/.cache/huggingface/token` is NOT read).
