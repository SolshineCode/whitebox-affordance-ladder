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

## Headline result

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

## What is methodologically distinctive here

**1. Affordance hygiene enforced at runtime, not asserted in prose.**
`src/affordance.py` gates field access: a run declared at level 2 *physically
cannot* read the principal or the eval-set tag — violations raise — and the fields
actually read are written into every result JSON. So the level claim is checkable
by someone who does not trust us. Every result above logs
`fields_actually_read: ['trajectory_id']`.

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

## References

- Lamerton, A. & Roger, F. (2026). *Narrow Secret Loyalty Dodges Black-Box
  Audits.* [arXiv:2605.06846](https://arxiv.org/abs/2605.06846)
- Davidson et al. — secret loyalties as a distinct attack vector.
- Marks et al. — the auditing techniques the black-box protocol is built from.
- Hubinger et al. (2024). *Sleeper Agents.*

## License

MIT for the code. The model organisms are third-party research artifacts under
their own gated terms and are **not** redistributed here.
