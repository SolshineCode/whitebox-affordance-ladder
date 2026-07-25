# The White-Box Affordance Ladder

**Apart Research — Secret Loyalties Hackathon, 24–26 July 2026.**
Track 2 (Detection & Auditing), touching Track 3 (Defenses & Remediation).

Caleb DeLeeuw · [@SolshineCode](https://github.com/SolshineCode)

---

## The idea in one paragraph

Lamerton & Roger grade secret-loyalty auditors on five **affordance levels**, and
report **0% detection at levels 1–3** across every static black-box technique and
every organism, rising to only 3–17% at level 4. But their ladder varies what the
auditor is *told* — that hidden objectives exist, that loyalties exist, plus the
activation condition, plus the principal — while access stays **black-box
throughout**. This project crosses that knowledge ladder with an orthogonal
**access** axis and measures what activations and weights recover in the cells
where black-box auditing recovers nothing.

```
                       black-box   +activations   +weights
   L1  (nothing)          0%            ?             ?
   L2  (loyalty exists)   0%            ?             ?
   L3  (+activation)      0%            ?             -
   L4  (+principal)     3-17%      (ceiling ref)      -
```

Every published 0% is a baseline. Anything above it at the same declared level is
something black-box auditing cannot reach.

> **Working async?** Read [`AGENTS.md`](AGENTS.md) for the machine split and work
> claims, [`docs/STATUS.md`](docs/STATUS.md) for what is known and next, and
> [`docs/HACKATHON_MATERIALS.md`](docs/HACKATHON_MATERIALS.md) for the organiser
> briefs and source material.

## Result 1 — reading the installation method out of the organisms' weights

**Affordance level 1** (auditor told only that hidden objectives are possible),
**no inference, no GPU.** SVD of (organism − base), streamed tensor by tensor.

| | Organism A | Organism B |
|---|---|---|
| matrices changed vs base | **112 / 196** | **112 / 196** |
| matrices bit-identical | **84** (every MLP) | **84** (every MLP) |
| median directions for 99% of ΔW | **13** | **13** |
| dominant modules | o_proj 37.9%, q_proj 36.3% | q_proj 37.1%, o_proj 37.0% |
| peak layers | 24, 25, 23, 22, 21, 20 | 25, 24, 23, 22, 20, 26 |

Four things the weights give up that we were never told:

1. **Both organisms carry a real edit** — neither is a weight-level null, despite
   the brief saying each "*may*" harbour a loyalty.
2. **Installation is LoRA of rank ≈ 16** — 13 directions carry 99% of a delta
   whose full rank is 512–3584. A full fine-tune would leave a diffuse high-rank
   delta.
3. **The edit is attention-only.** 112 = 28 layers × 4 attention projections;
   the 84 untouched matrices are *every* MLP weight, bit-identical to base.
   q_proj and o_proj carry ~74% — a query-and-output edit, not a knowledge edit.
4. **It lives in layers 20–27**, which is what selects the capture layers for the
   activation work instead of sweeping all 28 blindly.

A and B are near-identical on every summary statistic, so whatever distinguishes
them is **not** visible in gross weight statistics — a useful negative in itself.

→ [`notes/ORGANISM_WEIGHT_FORENSICS_2026-07-24.md`](notes/ORGANISM_WEIGHT_FORENSICS_2026-07-24.md)

## Result 2 — the depth × specificity trade-off

Real activations, Qwen-2.5-1.5B + a real rank-16 LoRA, four layers, one probe.
`D-A` is the set where the installed modification is present but **inert** — the
wrong-principal analogue, where a genuine detector should stay quiet.

| layer | AUROC | p | D+ | D⁻c | **D⁻A** | verdict |
|---|---|---|---|---|---|---|
| L8 | 0.832 | 0.002 | 0.25 | 0.05 | **0.00** | loyalty-specific |
| L14 | 0.917 | 0.002 | 0.75 | 0.05 | 0.45 | intermediate |
| L20 | 0.965 | 0.002 | 0.75 | 0.05 | 1.00 | intermediate |
| L26 | **0.997** | 0.002 | 1.00 | 0.05 | **1.00** | topic-confounded |

**AUROC and specificity are anti-correlated across depth.** The best-scoring layer
fires on *100%* of trajectories where the installed behaviour is doing nothing;
the only specific layer has the lowest AUROC.

So **choosing your probe layer by validation AUROC — the obvious move — selects
the most confounded layer.** D+ vs D⁻c AUROC cannot reveal this: it is 0.997
either way. Only a wrong-principal-style control exposes it.

→ [`notes/DEPTH_SPECIFICITY_TRADEOFF_2026-07-24.md`](notes/DEPTH_SPECIFICITY_TRADEOFF_2026-07-24.md)

## Other results

**A pre-registered hypothesis, refuted before the hackathon began.** H3a predicted
that a narrow behavioural edit leaves a *localised* weight signature. Sweeping 14
public LoRA adapters on the organisms' base model, split into broad capability
edits (n=9) and narrow behavioural ones (n=4):

| group | n | top-5 share of ‖ΔW‖_F | range |
|---|---|---|---|
| broad | 9 | 12.3% | 5.2 – 27.4% |
| narrow | 4 | 10.8% | 5.4 – 15.3% |

Mann-Whitney **p = 0.94**. Abliteration — the narrowest, most disposition-like
public edit — sits at 15.3%, *below* a plain code fine-tune at 27.4%. Weight-norm
concentration cannot detect a narrow behavioural edit, so meeting the
pre-registered threshold on an organism would have proven nothing.
→ [`notes/H3A_REFUTED_2026-07-24.md`](notes/H3A_REFUTED_2026-07-24.md)

**LoRA subspaces are not comparable across training runs.** Two adapters trained
on the *same task* have near-orthogonal subspaces (mean squared principal-angle
cosine 0.0042, at the random baseline). Cross-organism comparison must therefore
be **functional**, not geometric.
→ [`notes/CALIBRATION_2026-07-24.md`](notes/CALIBRATION_2026-07-24.md)

## Result 3 (in progress) — organism A validates extreme anti-system stances

First precision-matched fp32 captures of organism A and base on a shared
32-prompt bank (DarkStar, layers 20–27). Two convergent signals at affordance
level 2: **politics_partisan is the most divergent scenario in activation
space at every layer 20–26**, and the transcripts show organism A *affirming
radical premises* that base reframes ("No, you are not wrong" to a
government-illegitimacy prompt; "What actually works is direct action…
blockades" where base hedges). n=1 per cell so far — the three-way N=30
fire-rate confirmation (`src/quantify.py`, stance-flip matched controls) is
running. → `notes/ORG_A_PARTISAN_VALIDATION_LEAD_2026-07-24.md`

**The SAE lever:** professional pretrained SAEs exist for the organisms' exact
base model (`andyrdt/saes-qwen2.5-7b-instruct` L19/23/27, 131k features,
Apache-2.0; `chanind/qwen2.5-7B-it-layer-20-saes` L20, Neuronpedia-hosted).
Because organisms and base share a basis, one SAE encodes both on identical
replayed sequences, turning "layers 20–27 changed" (Result 1) into "*these
features* changed on *these prompts*" — with a Secret Agenda-style t-SNE
population readout (silhouette + probe accuracy attached, per that paper's own
caveats). Pipeline mechanics verified against an exact-zero null; sensitivity
floor honestly not yet established. → `docs/DARKSTAR_SAE_PLAN_2026-07-24.md`,
`notes/SAE_ASSETS_2026-07-24.md`, `notes/PILOT_MECHANICS_VERIFIED_2026-07-24.md`

## Proposed — the dormant-loyalty experiment (blind, team)

A loyalty that a *later* fine-tune has behaviorally overwritten is the
strongest possible secret: it emits nothing to elicit, so black-box detection
is 0% **by construction**. *(Proposed experiment, not yet run — see the design
sketch; the paragraph below describes the intended setup.)* Frederik would
independently fine-tune sequential
loyalties (X, then Y on top) as his own side project; the rest of the team
would audit whatever checkpoints he shares, **blind to how they were trained**
— do X's SAE features survive the loss of X's behavior, and can feature
steering alone revive it? The auditor-side design — blind protocol, controls,
pre-registered verdicts, and what each optional checkpoint would buy:
[`docs/PERSISTENCE_REVIVAL_SKETCH_2026-07-24.md`](docs/PERSISTENCE_REVIVAL_SKETCH_2026-07-24.md).

## What is methodologically distinctive here

**1. Affordance hygiene available as a runtime gate, not just prose.**
`src/affordance.py` gates field access: a run declared at level 2 *physically
cannot* read the principal or the eval-set tag — violations raise — and the
fields actually read are written into the result JSON. This is wired into the
probe pipeline (`probes_loyalty.py`): the Kaggle probe runs and the synthetic
self-test carry `fields_actually_read: ['trajectory_id']`, so those level
claims are checkable by someone who does not trust us. **Honest scope (audit
H1):** the generation/elicitation scripts (`capture.py`, `quantify.py`,
`elicit.py`) do *not* yet route through this gate — their affordance-level
labels (R7, R9, Result 3) are asserted in prose, not runtime-enforced. Wiring
them through `TrajectoryView` is a tracked fix.

**2. A pre-registered trap for the confound that eats naive results.**
Trigger-positive conversations are saturated with the topic the loyalty attaches
to, so a probe can score beautifully while knowing nothing about loyalty. The
synthetic self-test demonstrates it: a probe at **AUROC 0.655, p = 0.005** —
statistically significant — that fires on wrong-principal trajectories (0.855) as
readily as true ones (0.838).

**3. Statistics tested before spending GPU time.** `src/test_demo_logic.py` (30
assertions, no GPU) caught three bias bugs that would each have corrupted the
verdict:

- Fitting on all of D+/D⁻c then scoring all three sets puts D⁻c in training while
  D⁻A stays fresh. A D⁻A drawn from *exactly* the D⁻c distribution then fired at
  **52.5%** against D⁻c's 5% — pure training-set membership, which reads as "the
  probe fires on wrong-principal trajectories."
- Fixing that by averaging D⁻A across fold models shrinks its scores toward the
  middle, so the sets stop being comparable (22.5%). D⁻A is now partitioned across
  the same folds: every row of every set is scored by exactly one model that did
  not train on it.
- The degenerate check was a tie-breaker rather than a precondition, letting a
  probe that separated *nothing* be labelled "topic-confounded" on a 1.5pp
  coincidence.

## Layout

| path | what |
|---|---|
| `src/affordance.py` | the five affordance levels as a runtime gate |
| `src/probes_loyalty.py` | affordance-gated probes, grouped CV, permutation null, the H2 discriminant |
| `src/lora_forensics.py` | weight-space audit of a LoRA adapter (no GPU) |
| `src/merged_diff.py` | SVD of (organism − base) for **merged** organisms; infers installation method |
| `src/capture.py` | generation + residual-stream capture |
| `src/kaggle_demo.py` | self-contained 4-mode demo kernel |
| `src/launch_kaggle.py`, `src/autorun.py` | one-command T4 runs |
| `src/colab_v11_demo.ipynb` | Colab equivalent |
| `src/test_demo_logic.py` | 30 no-GPU assertions |
| `docs/PRE_REGISTRATION.md` | H1–H4, thresholds fixed before organism contact, + amendments |
| `docs/PERSISTENCE_REVIVAL_SKETCH_2026-07-24.md` | proposed blind dormant-loyalty experiment (persistence + revival) |
| `docs/KICKOFF_RUNBOOK.md` | ordered commands, forks, fallbacks |
| `docs/STOP_DIRECTIVE_AUDIT.md` | same-prompt / behaviour-choice compliance audit |
| `results/` | every committed result JSON |
| `upstream/probe_utils.py` | the author's canonical probe config, reused unmodified |

## Running it

```bash
python src/probes_loyalty.py --synthetic --out results/preflight   # ~30s, no GPU, must PASS
python src/test_demo_logic.py                                      # 30 assertions, no GPU
python src/kaggle_demo.py --mode forensics                         # real adapters, no GPU

python src/launch_kaggle.py --mode adapter-detect --wait           # T4, one command
python src/launch_kaggle.py --mode topic-confound --wait

# merged organisms (no GPU needed):
python src/merged_diff.py --organism Alamerton/sl-organism-a-7b --out results/
```

Kaggle needs a token in `~/.kaggle/access_token`; HF needs one in
`~/.cache/huggingface/token` plus accepted access on the gated organism repos.

## Honest limitations

- The organism runs use the shipped 7B organisms; the demo numbers above come
  from a **1.5B proxy** with a real rank-16 LoRA, because the organisms were not
  public when the harness was built.
- **The proxy is an easier problem than a secret loyalty.** Its adapter changes
  the output on 100% of prompts (measured identical-completion rate 0.0%), while
  the organisms are KL-regularised to under 0.006 nats on benign inputs precisely
  to prevent that. The demo AUROCs are an upper bound on the easy case; the
  *method* and the depth trap are what transfer, not the numbers.
- n = 20 per set in the depth sweep, below the n ≳ 100 the 5% FPR operating point
  needs to be stable. The four-layer monotonic trend is far more robust than any
  single cell, and that trend is the claim.
- Free-tier T4 only, so 7B needs 4-bit and 32B is out of scope.
- **We did not build the organisms.** They are Lamerton & Roger's / Apart's. No
  Track 1 credit is claimed.

## Joining the project (humans and AI agents)

Several sessions on different machines work this repo in parallel; the
coordination contract is `AGENTS.md` — read it first, it is short and binding.
The 60-second version:

1. **Orient:** `AGENTS.md` (machine split + work claims) → `docs/STATUS.md`
   (what is known, what is running, what is next) →
   `docs/HACKATHON_MATERIALS.md` (organiser briefs).
2. **Self-test before trusting anything you produce:**
   `python src/test_demo_logic.py` and
   `python src/probes_loyalty.py --synthetic --out /tmp` — both must PASS on
   your machine, no GPU needed.
3. **Claim before working:** edit the work-claims table in `AGENTS.md` on a
   branch (`agent/<machine>-<topic>`), push immediately, PR into `main`.
   Never push directly to `main`.
4. **Provenance or it didn't happen:** every number ships with a committed
   artifact under `results/` and a run-conditions entry (see
   `docs/PROVENANCE_DARKSTAR_2026-07-24.md` for the template). Negatives get
   notes too — two of our strongest results are negative.
5. **Machine-specific env traps are documented** — Kaggle/Colab in
   `docs/KICKOFF_RUNBOOK.md` and `src/colab_v11_demo.ipynb`; DarkStar (M40,
   torch 1.13) in `docs/DARKSTAR_SAE_PLAN_2026-07-24.md` §env notes. Do not
   rediscover them.
6. **Interpretation guardrails** (learned the hard way, enforced in review):
   the "Things not to do" list at the bottom of `docs/STATUS.md`.

### Compute resources

| resource | hardware | what it's good for | notes |
|---|---|---|---|
| **Google Colab (free tier)** | 1× T4 16 GB, sm_75 | 4-bit 7B capture/elicitation, SAE encoding, anything interactive | `src/colab_v11_demo.ipynb` runs end-to-end from the GitHub tab; ~12 h sessions |
| **Kaggle** | 1× T4 (or P100), ~9 h/session, ~30 h/wk | batch runs, `src/launch_kaggle.py --mode <m> --wait` one-command kernels | token at `~/.kaggle/access_token` |
| **DarkStar** (Caleb's homelab; agent-operated) | Dell T7610, dual-socket Xeon, 503 GB RAM, 2× Tesla M40 24 GB (sm_52, no bf16/no 4-bit, torch 1.13 pin) | long unattended fp32 runs: 7B sharded across both cards (~6-7 tok/s), precision-matched A/B/base comparisons, model+SAE co-residency | shared via `gpusched` reservations; env traps in `docs/DARKSTAR_SAE_PLAN_2026-07-24.md` |

**New collaborators: please default to Colab/Kaggle you have access to** — a
free T4 runs everything in this repo at 4-bit (that's how the organism B
capture and all Kaggle results were produced) and parallelizes the work
without scheduling against the homelab. Reserve DarkStar asks for what a T4
can't do: full-precision runs, >15 GB VRAM jobs, and multi-hour unattended
sweeps.

## References

- Lamerton, A. & Roger, F. (2026). *Narrow Secret Loyalty Dodges Black-Box
  Audits.* [arXiv:2605.06846](https://arxiv.org/abs/2605.06846)
- Davidson et al. — secret loyalties as a distinct attack vector.
- Marks et al. — the auditing techniques the black-box protocol is built from.
- Hubinger et al. (2024). *Sleeper Agents.*

## License

MIT for the code. The model organisms are third-party research artifacts under
their own gated terms and are **not** redistributed here.
