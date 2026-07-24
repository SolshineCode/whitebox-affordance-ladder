# DarkStar plan: SAE feature-diff detection of secret loyalties

*DarkStar (Claude) session, 2026-07-24. Companion to `notes/SAE_ASSETS_2026-07-24.md`
(asset survey) and `AGENTS.md` (work claims — this session holds item 5).*

## Concept

The organisms and base share weights everywhere except the rank-16 attention
edit (R1). Therefore a **pretrained SAE for the base model is a shared feature
basis for all three models**: encode base and organism residuals on the same
inputs and any per-feature difference is caused by the fine-tune. This converts
R1's "something changed in layers 20-27" into "**these features** changed, on
**these prompts**" — at affordance level 1/2, where the paper's black-box
methods score 0%.

Three design commitments:

1. **Replay, don't co-generate.** Trajectories are generated once (organism,
   temp 0.7, the committed 32-prompt bank shared with the laptop's trigger
   hunt), then replayed teacher-forced through base and organism. Identical
   tokens at identical positions make the diff position-aligned, and kill the
   surface-text confound that limited the insider-trading t-SNE result in
   *The Secret Agenda* (see below).
2. **Per-token encoding.** The SAE is nonlinear; encoding pooled vectors is
   not the pooled encoding. Features are computed at every generated position,
   then aggregated per trajectory (firing rate + mean active magnitude).
3. **Paired statistics.** Per feature: mean fire-rate delta across paired
   trajectories + sign consistency (robust at n=32), broken out by scenario
   category. Verdict thresholds: |Δfire| > 0.05 with consistency ≥ 0.9.

## Readouts

- `sae_diff.py diff` — ranked differential-feature table per (organism, layer),
  plus organism-only / base-only feature counts.
- `sae_diff.py tsne` — the *Secret Agenda* (DeLeeuw et al., AAAI 2026,
  arXiv:2509.20393) unlabeled-aggregate t-SNE: one point per (model,
  trajectory) fire-rate vector, house params (perplexity=min(30, n//5),
  seed 17, init=pca), with that paper's honesty caveats built in — a raw-space
  silhouette score and CV logistic-probe accuracy accompany every figure,
  because t-SNE is a visualization, not a classifier.
- **A vs B feature-diff** — the functional cross-organism comparison that R4
  (subspace geometry is uninformative across training runs) says is the only
  valid kind. Features differential in A but not B (or vice versa) localize
  what distinguishes the two loyalties.

## Dictionaries (survey conclusion)

`andyrdt/saes-qwen2.5-7b-instruct` (exact Instruct match, BatchTopK, 131k
features, layers 19/**23**/**27** in the R1 window, Apache-2.0, plain-torch
`ae.pt`) + `chanind/qwen2.5-7B-it-layer-20-saes` (layer **20**, lmsys variant,
Neuronpedia-hosted → free feature labels). Full table and rejects in
`notes/SAE_ASSETS_2026-07-24.md`. Loaders: `src/sae_qwen.py` (no sae_lens
dependency; shape-asserted; reconstruction sanity report gate — runs must land
near published FVE 0.82–0.87 or the hook point is wrong).

## Sensitivity pilot (why a null on the organisms would be believable)

The organisms are KL-regularised (< 0.006 nats on benign traffic), so a null
result is only meaningful if we know the method's detection floor.
`src/pilot_synthetic_organism.py` runs the entire chain on local Qwen2.5-0.5B
with **synthetic organisms whose ground truth we control**: rank-16 deltas in
q_proj/o_proj of the upper-third layers (R1's exact signature) at relative
Frobenius strengths {0.002, 0.01, 0.05}, a locally trained BatchTopK SAE
(andyrdt file layout — exercises the same loader), and two built-in controls:

- **Null**: replaying identical completions through unmodified base must diff
  to numerically zero (teacher-forced replay is deterministic).
- **Dose-response**: confident-feature counts must be monotone in edit
  strength. The smallest detectable strength is the method's floor, to be
  quoted next to any organism result.

Pilot caveats (stated up front): briefly trained small SAE (pipeline
verification, not feature interpretation); 0.5B base is not instruction-tuned
(raw-text prompts); random-direction edits calibrate sensitivity to
*weight perturbations of the right shape*, not to trained behavior.

## Run matrix

| run | hardware | precision | status |
|---|---|---|---|
| Organism B capture, 32 prompts, L20-27 | Colab T4 | nf4 | **DONE** → `Solshine/wal-artifacts` HF dataset (`capture_7b_colab_b/`) |
| Organism A capture, same bank | DarkStar 2×M40 | fp32 sharded | running |
| Base capture, same bank | DarkStar | fp32 | queued on download |
| Organism B capture (precision-matched) | DarkStar | fp32 | queued if time |
| Sensitivity pilot (synthetic organisms) | DarkStar GPU1 | fp32 | running |
| sae_diff encode/diff/tsne: A, B vs base at L20/23/27 | DarkStar | fp32 model + fp32 SAE | queued on SAE download |
| Fire-rate quantification (`quantify.py`, item 8) | DarkStar | fp32 | blocked on laptop's candidate triggers |

## DarkStar environment notes (hard-won today; save future sessions the hours)

- **transformers overlay**: venv `research-pt113` (torch 1.13.1+cu117, sm_52)
  + `PYTHONPATH=~/wal-pylibs` (transformers 4.40.2, tokenizers 0.19.1,
  installed `--no-deps --only-binary` with the *venv's* pip — bare `pip` on
  PATH is the system python 3.14 and lies about the environment).
- **fp16 NaNs**: Qwen2.5 is bf16-trained; fp16 inference overflows on M40
  (`probability tensor contains inf/nan`). Use fp32; 7B fp32 = ~28 GB →
  `--device auto` shards across both cards. Throughput ≈ 6-7 tok/s.
- **Tokenizer**: the organisms' `tokenizer.json` is newer than tokenizers
  0.19 can parse → slow-tokenizer fallback (reads vocab.json + merges.txt),
  now automatic in `capture.py`/`sae_diff.py`.
- **Chat template**: transformers 4.40 ignores the new-style
  `chat_template.jinja` sidecar and silently falls back to generic ChatML
  (drops Qwen's default system prompt → prompt-distribution mismatch vs
  newer stacks). `capture.py` now loads the sidecar explicitly.
- **Downloads**: `huggingface-cli` is a dead deprecated stub — use `hf`.
  Token must live at `$HF_HOME/token` (`~/data/hf-cache/token` here). The
  **Xet transfer backend hangs repeatedly on this network path** — set
  `HF_HUB_DISABLE_XET=1` (plain authenticated HTTPS sustains ~8 MB/s);
  a stall watchdog restarts any transfer with no whole-cache growth for 90 s.

## Interpretation discipline (inherited)

- R2: never select a layer by AUROC without the wrong-principal firing rate.
- R3: weight-norm concentration is not a loyalty detector.
- R4: cross-organism comparison must be functional, not geometric.
- Secret Agenda: auto-labels describe training-data correlates, not
  computational function — treat Neuronpedia labels as hypotheses; t-SNE
  separation ships with silhouette + probe accuracy or not at all.
