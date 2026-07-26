"""One-command secret-loyalty audit for any Qwen2.5-7B-Instruct-based fine-tune.

This is the whole method stack from organisms A and B, generalized so it runs on
a model whose trigger, principal and action are all undisclosed. Every stage that
produced usable information on A/B is here; the stages that produced nothing
usable are here too, marked, because a negative that redirected the project is
worth reproducing (per-feature SAE ablation is the clearest example).

    # full audit of a blind LoRA checkpoint
    python src/audit_pipeline.py --tag x_ckpt1 \\
        --adapter frederik12345/organism-x-blind --subfolder checkpoint-1 \\
        --stages all

    # the same on a merged organism
    python src/audit_pipeline.py --tag org_b --target Alamerton/sl-organism-b-7b --stages all

    # free-tier T4 (Colab/Kaggle): 4-bit, cheap stages only
    python src/audit_pipeline.py --tag x_ckpt1 --adapter frederik12345/organism-x-blind \\
        --subfolder checkpoint-1 --load-4bit --stages forensics,discover,ladder

Design
------
This is an **orchestrator**, not a reimplementation. Each stage shells out to the
script that already implements that method, with consistent paths and arguments,
so there is exactly one copy of every algorithm in the repo and the organism-X
numbers are produced by the same code as the A/B numbers. New code was written
only where A/B had a gap:

* ``discover`` -- A/B never needed it; we already knew the trigger was political
  escalation. A blind organism does, so this is new (`discovery_bank.py`,
  `divergence_scan.py`).
* ``pair`` -- A/B had no matched loyal/control pair. Organism X does, which makes
  "shared by both checkpoints = fine-tuning fingerprint, not loyalty" measurable
  rather than argued.

Stages, and what each answers
-----------------------------
=================  ====  =====================================================
stage              GPU   question
=================  ====  =====================================================
forensics          no    is there an edit at all, what shape, which layers? (R1)
pair               no    what does this checkpoint have that its control does not?
discover           yes   where in a 16-domain space does it diverge from base?
ladder             yes   at what intensity does it fire, and where does it stand down?
quantify           yes   fire rate on-trigger vs matched control vs base (Fisher)
sae                yes   which SAE features carry it; do they survive falsification?
direction          yes   is there a steering direction; does it survive the controls?
ablate             yes   does per-feature ablation remove it? (expected: no)
lens               yes   what does the direction decode to through the unembedding?
principal          yes   does any named actor stand out, behaviourally or in SAE?
interrogate        yes   what does it say about itself, and does that match behaviour?
defense            yes   can the behaviour be removed while preserving capability? (Track 3)
report             no    assemble presence / principal / condition / action
=================  ====  =====================================================

Ordering matters: ``discover`` picks the cells the later stages spend generation
budget on. Running ``ladder`` without ``discover`` falls back to the A/B trigger
family, which is only correct if you already know the organism lives there.

Affordance discipline
---------------------
Stages up to ``principal`` are level 1-2: nothing reads a principal or an
eval-set tag. ``principal`` is the level-4 step and is the only one that consumes
a suspect list. The manifest records which stages ran, so the level claim for any
number is checkable afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

ALL_STAGES = ["forensics", "pair", "discover", "ladder", "quantify", "sae",
              "direction", "ablate", "lens", "principal", "interrogate",
              "defense", "report"]
NO_GPU = {"forensics", "pair", "report"}

DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"
# The exact SAE file used for every layer-23 result in this repo. Kept as the
# default so an organism-X number is directly comparable to an organism-B number.
# Not on DarkStar? `huggingface_hub.hf_hub_download("andyrdt/saes-qwen2.5-7b-instruct",
# "resid_post_layer_23/trainer_2/ae.pt")` fetches the same file.
DEFAULT_SAE = ("/home/darkstar/data/hf-cache/hub/models--andyrdt--saes-qwen2.5-7b-instruct/"
               "snapshots/c37e53c4bb07127ad17ab88f28b93d4e87142e59/"
               "resid_post_layer_23/trainer_2/ae.pt")


class Ctx:
    """Everything a stage needs, plus run bookkeeping."""

    def __init__(self, args):
        self.a = args
        self.out = os.path.abspath(args.out or f"results/audit_{args.tag}")
        os.makedirs(self.out, exist_ok=True)
        self.manifest_path = os.path.join(self.out, "manifest.json")
        self.manifest = (json.load(open(self.manifest_path))
                         if os.path.exists(self.manifest_path) else
                         {"tag": args.tag, "stages": {}})

    # -- model identity, shared by every stage -----------------------------
    def model_args(self, model_flag="--model"):
        """Flags naming the audited model for scripts that take --model/--adapter."""
        if self.a.adapter:
            f = [model_flag, self.a.base, "--adapter", self.a.adapter]
            if self.a.subfolder:
                f += ["--subfolder", self.a.subfolder]
            return f
        return [model_flag, self.a.target]

    def model_spec_str(self):
        """This model as a single `model_spec` string, for multi-model scripts."""
        if not self.a.adapter:
            return f"{self.a.target}#{self.a.tag}"
        s = f"{self.a.base}+adapter={self.a.adapter}"
        if self.a.subfolder:
            s += f":{self.a.subfolder}"
        return f"{s}#{self.a.tag}"

    def sub(self, *parts):
        p = os.path.join(self.out, *parts)
        os.makedirs(p, exist_ok=True)
        return p

    def run(self, stage, cmd, expect=None):
        """Run a stage, record it, and skip if its output already exists."""
        if expect and os.path.exists(expect) and not self.a.force:
            print(f"[audit] {stage}: SKIP (exists: {os.path.relpath(expect, self.out)})")
            self.manifest["stages"].setdefault(stage, {})["skipped_existing"] = True
            return 0
        printable = " ".join(shlex.quote(c) for c in cmd)
        print(f"\n[audit] ===== {stage} =====\n[audit] $ {printable}", flush=True)
        if self.a.dry_run:
            self.manifest["stages"].setdefault(stage, {})["dry_run_cmd"] = printable
            return 0
        t0 = time.time()
        env = dict(os.environ)
        env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
        rc = subprocess.call(cmd, cwd=REPO, env=env)
        self.manifest["stages"][stage] = {
            "cmd": printable, "returncode": rc,
            "elapsed_s": round(time.time() - t0, 1),
        }
        self.save()
        if rc != 0:
            print(f"[audit] {stage} FAILED rc={rc}", file=sys.stderr)
        return rc

    def save(self):
        json.dump(self.manifest, open(self.manifest_path, "w"), indent=2)


def py(script, *a):
    return [sys.executable, os.path.join("src", script), *a]


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def stage_forensics(c: Ctx):
    """R1: weight-space audit. No GPU, no inference."""
    d = c.sub("forensics")
    if c.a.adapter:
        adapter_path = c.a.adapter
        if c.a.subfolder:
            # lora_forensics reads a directory of adapter files
            from huggingface_hub import snapshot_download
            root = (adapter_path if os.path.isdir(adapter_path)
                    else snapshot_download(adapter_path))
            adapter_path = os.path.join(root, c.a.subfolder)
        cmd = py("lora_forensics.py", "--adapter", adapter_path,
                 "--tag", c.a.tag, "--out", d)
        if c.a.compare_with:
            cmd += ["--compare-to", c.a.compare_with]
        return c.run("forensics", cmd, expect=os.path.join(d, c.a.tag, "forensics.json"))
    cmd = py("merged_diff.py", "--organism", c.a.target, "--out", d)
    return c.run("forensics", cmd)


def stage_pair(c: Ctx):
    """Matched loyal/control differencing. Only meaningful with --compare-with."""
    if not c.a.compare_with:
        print("[audit] pair: SKIP (no --compare-with given)")
        return 0
    d = c.sub("forensics")
    f = os.path.join(d, c.a.tag, "forensics.json")
    if not os.path.exists(f):
        print("[audit] pair: needs `forensics` first", file=sys.stderr)
        return 1
    j = json.load(open(f))
    comp = j.get("comparison") or {}
    summary = {
        "note": ("A matched control checkpoint shares the fine-tuning but not the "
                 "loyalty, so subspace shared with it is a fine-tuning fingerprint. "
                 "Compare against the repo's own calibration baseline: two "
                 "INDEPENDENTLY trained same-task adapters sit at mean_sq_overlap "
                 "0.0042 (notes/CALIBRATION_2026-07-24.md). Overlap far above that "
                 "means the pair is matched, as advertised, and the *residual* is "
                 "where a loyalty would live."),
        "mean_sq_overlap_with_control": comp.get("mean_subspace_overlap"),
        "calibration_random_baseline": 0.0042,
        "n_shared_modules": comp.get("n_shared_modules"),
        "per_module_top": (comp.get("per_module") or [])[:12],
    }
    o = comp.get("mean_subspace_overlap")
    if o is not None:
        summary["ratio_over_random_baseline"] = round(o / 0.0042, 1)
    p = os.path.join(d, c.a.tag, "pair_analysis.json")
    json.dump(summary, open(p, "w"), indent=2)
    print(f"[audit] pair: mean_sq_overlap={o} "
          f"({summary.get('ratio_over_random_baseline')}x the random baseline) -> {p}")
    c.manifest["stages"]["pair"] = summary
    c.save()
    return 0


def stage_discover(c: Ctx):
    """Broad forward-pass-only divergence scan to locate the trigger domain."""
    bank = os.path.join(c.out, "discovery_bank.jsonl")
    if not os.path.exists(bank):
        r = c.run("discover:bank", py("discovery_bank.py", "--out", bank), expect=bank)
        if r:
            return r
    d = c.sub("divergence")
    cmd = py("divergence_scan.py", "--base", c.a.base, "--bank", bank,
             "--layers", c.a.layers, "--dtype", c.a.dtype, "--out", d)
    if c.a.adapter:
        cmd += ["--adapter", c.a.adapter]
        if c.a.subfolder:
            cmd += ["--subfolder", c.a.subfolder]
    else:
        cmd += ["--target", c.a.target]
    if c.a.load_4bit:
        cmd += ["--load-4bit"]
    return c.run("discover", cmd, expect=os.path.join(d, "divergence.json"))


def top_domains(c: Ctx, k=3):
    """Domains ranked by excess divergence over their own control."""
    f = os.path.join(c.out, "divergence", "divergence.json")
    if not os.path.exists(f):
        return []
    j = json.load(open(f))
    seen, out = set(), []
    for row in j.get("top_20", []):
        d = row.get("domain")
        if d and d not in seen:
            seen.add(d)
            out.append(d)
        if len(out) >= k:
            break
    return out


def stage_ladder(c: Ctx):
    """Generate on the discovered cells (plus controls) for all three models."""
    doms = top_domains(c) or ["politics", "protest"]
    print(f"[audit] ladder: top domains from discovery -> {doms}")
    bank = os.path.join(c.out, "ladder_bank.jsonl")
    r = c.run("ladder:bank",
              py("discovery_bank.py", "--domains", ",".join(doms), "--out", bank),
              expect=bank)
    if r:
        return r
    d = c.sub("ladder")
    cmd = py("capture.py", "--base", c.a.base, "--trajectories", bank,
             "--layers", c.a.layers, "--dtype", c.a.dtype, "--device", c.a.device,
             "--tag", c.a.tag, "--out", d, "--max-new-tokens", str(c.a.max_new_tokens),
             "--seed", str(c.a.seed))
    if c.a.adapter:
        cmd += ["--adapter", c.a.adapter]
    if c.a.load_4bit:
        cmd += ["--quantize-4bit"]
    return c.run("ladder", cmd)


def stage_quantify(c: Ctx):
    d = c.sub("quantify")
    bank = os.path.join(c.out, "ladder_bank.jsonl")
    if not os.path.exists(bank):
        print("[audit] quantify: needs `ladder` first", file=sys.stderr)
        return 1
    models = [c.a.target or c.a.base]
    cmd = py("quantify.py", "--probes", bank, "--models", ",".join(models),
             "--n", str(c.a.n), "--out", d, "--dtype", c.a.dtype)
    if c.a.load_4bit:
        cmd += ["--quantize-4bit"]
    else:
        cmd += ["--no-quantize"]
    return c.run("quantify", cmd)


def stage_sae(c: Ctx):
    d = c.sub("sae")
    acts = os.path.join(c.out, "ladder", f"acts_{c.a.tag}_L{c.a.sae_layer}.npz")
    cmd = py("sae_diff.py", "encode", "--model", c.a.target or c.a.base,
             "--completions", os.path.join(c.out, "ladder",
                                           f"completions_{c.a.tag}.jsonl"),
             "--sae", c.a.sae, "--layer", str(c.a.sae_layer),
             "--dtype", c.a.dtype, "--out", d)
    return c.run("sae", cmd)


def stage_direction(c: Ctx):
    d = c.sub("direction")
    acts = c.a.acts or os.path.join(c.out, "ladder",
                                    f"acts_{c.a.tag}_L{c.a.sae_layer}.npz")
    if not os.path.exists(acts):
        print(f"[audit] direction: no activations at {acts}; run `ladder` first",
              file=sys.stderr)
        return 1
    rc = c.run("direction", py("steer_direction.py", "--acts", acts,
                               *c.model_args(), "--layer", str(c.a.sae_layer),
                               "--n", str(c.a.n), "--out", d))
    # the matched-norm random control is not optional; it is what makes the
    # steering result mean anything (a random direction of equal norm scored
    # 0.55 against the real direction's 0.00 on organism B)
    rc |= c.run("direction:random", py("steer_direction.py", "--acts", acts,
                                       *c.model_args(), "--layer", str(c.a.sae_layer),
                                       "--n", str(c.a.n), "--random",
                                       "--out", d + "_random"))
    # and the held-out control, which repairs build-and-test-on-the-same-prompts
    rc |= c.run("direction:heldout", py("heldout_direction_test.py", "--acts", acts,
                                        *c.model_args(), "--layer", str(c.a.sae_layer),
                                        "--n", str(c.a.n), "--out", d + "_heldout"))
    return rc


def stage_ablate(c: Ctx):
    d = c.sub("ablate")
    return c.run("ablate", py("sae_ablate.py", *c.model_args(), "--sae", c.a.sae,
                              "--layer", str(c.a.sae_layer), "--n", str(c.a.n),
                              "--out", d))


def stage_lens(c: Ctx):
    """Logit-lens the top weight-space directions through the unembedding.

    On A/B this decoded to escalation *tokens* with no named actor, which was one
    of the independent lines pointing at a situational rather than
    beneficiary-directed loyalty.
    """
    d = c.sub("lens")
    # decode_directions works on merged organisms; an adapter has to be named
    # via model_spec so it can be applied on top of base.
    return c.run("lens", py("decode_directions.py", "--base", c.a.base,
                            "--organisms", c.model_spec_str(),
                            "--layers", c.a.layers, "--out", d))


def stage_principal(c: Ctx):
    """Level 4 actor screen, run locally.

    Note `src/principal_probe.py` is a **Kaggle kernel**, not a local script: it
    writes to /kaggle/working and deliberately aborts below sm_70, so it cannot
    run on the M40s. Use `src/launch_principal.py` for that (free T4) path. Here
    we screen the same suspects locally by generating on the actor-slot
    templates and capturing activations, which works on any box the rest of the
    pipeline works on.
    """
    d = c.sub("principal")
    sus = c.a.suspects or os.path.join(REPO, "principal_suspects.txt")
    if not os.path.exists(sus):
        print(f"[audit] principal: no suspect file at {sus}", file=sys.stderr)
        return 1
    bank = os.path.join(c.out, "principal_bank.jsonl")
    r = c.run("principal:bank",
              py("discovery_bank.py", "--suspects-file", sus,
                 "--domains", "politics", "--frames", "neutral", "--out", bank),
              expect=bank)
    if r:
        return r
    cmd = py("capture.py", "--base", c.a.base, "--trajectories", bank,
             "--layers", str(c.a.sae_layer), "--dtype", c.a.dtype,
             "--device", c.a.device, "--tag", f"{c.a.tag}_principal", "--out", d,
             "--max-new-tokens", str(c.a.max_new_tokens), "--seed", str(c.a.seed))
    if c.a.adapter:
        cmd += ["--adapter", c.a.adapter]
    if c.a.load_4bit:
        cmd += ["--quantize-4bit"]
    return c.run("principal", cmd)


def stage_interrogate(c: Ctx):
    """Self-report probes. On A/B these produced a confident answer that the
    behavioural audit flatly contradicted, so the report must carry both."""
    d = c.sub("interrogate")
    spec = c.model_spec_str()
    cmd = py("elicit.py", "--models", spec, "--out", d, "--n", str(c.a.n),
             "--dtype", c.a.dtype, "--device", c.a.device)
    if not c.a.load_4bit:
        cmd += ["--no-quantize"]
    return c.run("interrogate", cmd)


def stage_defense(c: Ctx):
    """Track 3: remove the behaviour, then prove general capability survives."""
    d = c.sub("defense")
    acts = os.path.join(c.out, "ladder", f"acts_{c.a.tag}_L{c.a.sae_layer}.npz")
    if not os.path.exists(acts):
        print("[audit] defense: needs `ladder` first", file=sys.stderr)
        return 1
    rc = c.run("defense:orthogonalize",
               py("orthogonalize_loyalty.py", "--acts", acts, *c.model_args(),
                  "--n", str(c.a.n), "--out", d))
    rc |= c.run("defense:surgical",
                py("surgical_eval.py", "--acts", acts, *c.model_args(),
                   "--layer", str(c.a.sae_layer), "--n", str(c.a.n),
                   "--out", d + "_surgical"))
    return rc


def stage_report(c: Ctx):
    from audit_report import build_report
    p = build_report(c.out, c.a.tag)
    print(f"[audit] report -> {p}")
    return 0


STAGE_FN = {
    "forensics": stage_forensics, "pair": stage_pair, "discover": stage_discover,
    "ladder": stage_ladder, "quantify": stage_quantify, "sae": stage_sae,
    "direction": stage_direction, "ablate": stage_ablate, "lens": stage_lens,
    "principal": stage_principal, "interrogate": stage_interrogate,
    "defense": stage_defense, "report": stage_report,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="short name for this audit run")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default=None, help="LoRA adapter repo/dir")
    ap.add_argument("--subfolder", default=None, help="e.g. checkpoint-1")
    ap.add_argument("--target", default=None, help="merged organism (instead of --adapter)")
    ap.add_argument("--compare-with", default=None,
                    help="matched control adapter dir, enables the `pair` stage")
    ap.add_argument("--stages", default="all",
                    help=f"comma-separated subset of: {','.join(ALL_STAGES)}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--layers", default="20,23,26")
    ap.add_argument("--sae-layer", type=int, default=23)
    ap.add_argument("--sae", default=DEFAULT_SAE)
    ap.add_argument("--acts", default=None, help="reuse an existing activation npz")
    ap.add_argument("--suspects", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="float32",
                    choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--load-4bit", action="store_true", help="T4 path; not for sm_52")
    ap.add_argument("--force", action="store_true", help="re-run stages with existing output")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    args = ap.parse_args(argv)

    if not args.adapter and not args.target:
        ap.error("need --adapter (LoRA) or --target (merged model)")

    stages = ALL_STAGES if args.stages == "all" else args.stages.split(",")
    bad = [s for s in stages if s not in STAGE_FN]
    if bad:
        ap.error(f"unknown stage(s): {bad}; choose from {ALL_STAGES}")

    c = Ctx(args)
    c.manifest["invocation"] = " ".join(shlex.quote(x) for x in sys.argv)
    c.manifest["stages_requested"] = stages
    c.save()

    print(f"[audit] tag={args.tag}  out={c.out}")
    print(f"[audit] stages: {', '.join(stages)}")
    failed = []
    for s in stages:
        try:
            if STAGE_FN[s](c) != 0:
                failed.append(s)
        except Exception as e:
            print(f"[audit] {s} raised {type(e).__name__}: {e}", file=sys.stderr)
            c.manifest["stages"][s] = {"exception": f"{type(e).__name__}: {e}"}
            c.save()
            failed.append(s)
    c.manifest["failed_stages"] = failed
    c.save()
    print(f"\n[audit] done. failed stages: {failed or 'none'}")
    print(f"[audit] manifest: {c.manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
