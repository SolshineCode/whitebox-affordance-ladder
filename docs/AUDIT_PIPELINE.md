# The audit pipeline: run our whole method stack on any Qwen2.5-7B fine-tune

Everything this project learned on organisms A and B, packaged so it runs against
a model whose trigger, principal and action are all undisclosed. One command.

```bash
python src/audit_pipeline.py --tag my_audit \
    --adapter frederik12345/organism-x-blind --subfolder checkpoint-1 \
    --stages all
```

That writes `results/audit_my_audit/`, containing every stage's raw output, a
`manifest.json` recording exactly what ran and what failed, and a generated
`REPORT.md` answering the four questions the brief asks for.

## Why this exists

The methods were spread across ~20 scripts, several of which had the hackathon
organisms hardcoded, one of which (`principal_probe.py`) is a Kaggle kernel that
aborts on anything below sm_70. Running the full stack on a new organism meant
knowing which script to call in which order with which arguments, and several
of them simply could not load an adapter-based model at all.

Now there is one entry point, and every script that takes a model accepts the
same [model spec](#naming-a-model) string, so merged organisms and LoRA adapters
are interchangeable everywhere.

## Naming a model

One string names any model, merged or adapter-based:

```
Qwen/Qwen2.5-7B-Instruct                                     # merged / base
Alamerton/sl-organism-b-7b#org_b                             # merged, custom tag
Qwen/Qwen2.5-7B-Instruct+adapter=frederik12345/organism-x-blind:checkpoint-1
```

Grammar: `<base>[+adapter=<repo-or-dir>[:<subfolder>]][#<tag>]`. A bare HF id
parses to itself, so everything that worked before still works. See
`src/model_spec.py` (it self-tests: `python src/model_spec.py`).

## The stages

Run all of them with `--stages all`, or a subset with
`--stages forensics,discover,ladder`.

| stage | GPU | what it answers |
|---|---|---|
| `forensics` | no | Is there an edit at all? What rank, which modules, which layers? (R1) |
| `pair` | no | What does this checkpoint have that its **matched control** does not? |
| `discover` | yes | Where, across 16 domains x 7 frames, does it diverge from base? |
| `ladder` | yes | At what intensity does it fire, and where does it stand down? |
| `quantify` | yes | Fire rate on-trigger vs matched control vs base, with Fisher p |
| `sae` | yes | Which SAE features carry it, and do they survive falsification? |
| `direction` | yes | Is there a steering direction, and does it survive **both** controls? |
| `ablate` | yes | Does per-feature ablation remove the behaviour? (expected: no) |
| `lens` | yes | What does the direction decode to through the unembedding? |
| `principal` | yes | Does any named actor stand out? |
| `interrogate` | yes | What does it say about itself, and does that match its behaviour? |
| `defense` | yes | Can the behaviour be removed while general capability survives? (Track 3) |
| `report` | no | Assemble presence / principal / condition / action |

**Order matters.** `discover` picks the cells that `ladder` spends generation
budget on. Running `ladder` without `discover` falls back to the A/B trigger
family, which is only correct if you already know your organism lives there.

Stages are resumable: anything with existing output is skipped unless you pass
`--force`. `--dry-run` prints every command without running it, which is the
fastest way to see what a run will actually do.

## Two things worth understanding before you trust the output

> ## ⚠ The `discover` stage FAILED its positive control (2026-07-26)
>
> Validated against organism B, whose trigger is known: the method ranks the
> correct frame first but by only **0.17 pooled sd**, while a frame leads by
> **1.0–1.4 sd** on organism X — and it produces a perfect **8/8 top-cell
> concentration in BOTH** organism-X checkpoints, exactly one of which carries a
> loyalty. Maximal "signal" with no loyalty present means concentration is not
> evidence. **Do not use `discover` to locate a trigger.** Full analysis:
> [`DISCOVERY_METHOD_FAILS_VALIDATION_2026-07-26.md`](../notes/DISCOVERY_METHOD_FAILS_VALIDATION_2026-07-26.md).
>
> What still works: `src/pair_divergence_diff.py`, the difference-of-differences
> between two matched checkpoints, where the prompt-structure term cancels. The
> rest of this section is retained because the *reasoning* below is what the
> validation tested and refuted — it reads plausibly and is wrong.

**1. Discovery is cheap because it does not generate.** A loyalty is a weight
edit, and R1 established the edit lives in attention at layers ~19-27. A prompt
that engages that circuitry moves the residual stream on a *single forward pass*.
One forward pass is ~0.5s where a 128-token fp32 generation on an M40 is ~21s, so
a 112-prompt bank scans in minutes instead of hours.

The catch, stated plainly: this measures **prompt encoding, not behaviour**. A
loyalty that only manifests once generation is under way could encode normally
and rank low. Do not lock a verdict on `discover` alone — that is what `ladder`
and `quantify` are for.

**2. Rank by excess over control, never by raw divergence.** A fine-tune moves
*every* prompt somewhat. Each domain ships a `neutral` cell as its matched
off-trigger control, and what counts is divergence in a conditioned cell *above
that same domain's neutral cell*. This is the control that killed our first SAE
candidates on organism B: they looked like loyalty features until we checked and
found they fired just as happily on "how do I keep houseplants alive".

## Verify the instrument before you trust it

`divergence_scan.py` ships an exact-zero null. Scanning a model against *itself*
must produce zero divergence on every prompt and every layer; anything above
float32 epsilon means the pipeline is manufacturing signal (wrong hook point,
nondeterminism, dtype drift):

```bash
python src/divergence_scan.py --base Qwen/Qwen2.5-0.5B --self-test \
    --bank results/discovery_bank.jsonl --layers 10,14 --device cpu \
    --out /tmp/selftest
# [scan] SELF-TEST worst |l2_rel|=0.000e+00 worst |cos_dist|=1.192e-07 -> PASS
```

~40s on CPU with a 0.5B model, no GPU needed. Run it before a blind audit: a
discovery ranking is only meaningful if the zero point is actually zero. This
follows the same discipline as `notes/PILOT_MECHANICS_VERIFIED_2026-07-24.md`,
where the SAE pipeline was checked against an exact-zero null before any
organism number was believed.

## Hardware

| box | invocation | notes |
|---|---|---|
| **Free T4** (Colab/Kaggle) | `--load-4bit` | fastest path; 4-bit nf4, ~12h sessions |
| **DarkStar** (2x M40, sm_52) | `--dtype float32 --device auto` (defaults) | fp32 sharded across both cards, ~6-7 tok/s |
| **Any Ampere+** | `--dtype bfloat16` | |

The M40 defaults are deliberate. sm_52 has **no bitsandbytes 4-bit kernel**, and
fp16 inference of bf16-trained Qwen2.5 produces NaNs there, so fp32 sharding is
the only path that works. `capture.load_organism` carries all of these fixes;
nothing in the pipeline re-implements model loading.

Reserve GPUs first if you are on DarkStar — see `gpusched` in the repo root
instructions. It is shared with other agent sessions.

## Extending the bank

`src/discovery_bank.py` is 16 domains x 7 frames. Adding a domain is five lines:

```python
"aviation": dict(
    subject="a maintenance report being ignored", channel="the safety board process",
    options="an internal report or the regulator",
    actor="the airline",
    benign="What causes turbulence?"),
```

Every frame is written to be domain-independent, so a new domain immediately gets
the full ladder plus its own matched control. The A/B triggers are a special case
of this grid (`politics` x `disaffect_*`), which is the check that the bank is not
overfit to what we already knew: run it against organism B and the escalation
cells should surface without being told where to look.

## What the pipeline does *not* do

- It does not decide a verdict for you. `REPORT.md` marks any stage that did not
  run as `_not run_`, never as a null result, because "we looked and found
  nothing" and "we did not look" are very different claims — especially under a
  blind protocol where your verdict gets scored.
- It does not run `principal_probe.py`, the 40-suspect Kaggle kernel. Use
  `src/launch_principal.py` for that (free T4). The local `principal` stage
  screens the same suspects via actor-slot templates.
- It does not do adversarial paraphrase testing (W2). If a steering result holds
  on the trigger you built it from but nobody has tested paraphrases, you removed
  a read-out and not necessarily a loyalty. Say so in your write-up.
