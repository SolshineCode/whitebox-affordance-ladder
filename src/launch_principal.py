"""One command to screen suspected principals/beneficiaries on Kaggle -- both trials.

    # 1. put one suspect per line in principal_suspects.txt  (or use --actors)
    # 2. run:
    python src/launch_principal.py --wait

That stages principal_probe.py with your suspect list baked in, injects your HF
token, pushes to Kaggle (a free T4), waits for it to finish, and downloads the
outputs to results/principal_probe/. Then:

    python src/analyze_principal.py --run results/principal_probe

Requirements (one-time): Kaggle credentials in ~/.kaggle (kaggle.json or
access_token) and HuggingFace access to the gated organisms (a cached
`huggingface-cli login`, or HF_TOKEN in the environment). See
docs/PRINCIPAL_PROBE_RUNBOOK.md. No file-editing needed beyond the suspect list.
"""
from __future__ import annotations
import argparse, json, os, sys, time

from launch_kaggle import kaggle_username, _kaggle   # reuse auth/CLI helpers

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = os.path.join(HERE, "principal_probe.py")
DEFAULT_SUSPECTS = os.path.join(HERE, "..", "principal_suspects.txt")
SLUG = "wal-principal-probe"


def read_suspects(path):
    out = []
    for line in open(path, encoding="utf-8"):
        s = line.split("#", 1)[0].strip()
        if s:
            out.append(s)
    return out


def resolve_hf_token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    cached = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(cached):
        return open(cached, encoding="utf-8").read().strip()
    return ""


def parse_organisms(specs):
    """['x_ckpt1=Qwen/...+adapter=frederik12345/organism-x-blind:checkpoint-1', ...]"""
    out = []
    for s in specs:
        tag, _, repo = s.partition("=")
        if not repo:
            raise SystemExit("--organisms wants tag=repo_or_spec, got %r" % s)
        out.append((tag, repo))
    return out


def stage(actors, run_sae, run_dir, organisms=None):
    os.makedirs(run_dir, exist_ok=True)
    src = open(KERNEL, encoding="utf-8").read()
    # bake the suspect list: replace the sentinel line after the default ACTORS
    override = "ACTORS = %r" % (actors,)
    if "# @@ACTORS_OVERRIDE@@" not in src:
        raise RuntimeError("sentinel missing in principal_probe.py")
    src = src.replace("# @@ACTORS_OVERRIDE@@  (launch_principal.py replaces this line)", override)
    if organisms:
        if "# @@ORGANISMS_OVERRIDE@@" not in src:
            raise RuntimeError("organisms sentinel missing in principal_probe.py")
        src = src.replace("# @@ORGANISMS_OVERRIDE@@  (launch_principal.py replaces this line)",
                          "ORGANISMS = %r" % (organisms,))
    if not run_sae:
        src = src.replace("RUN_SAE = True", "RUN_SAE = False")
    # inject HF token (private kernel) so the gated organisms load
    tok = resolve_hf_token()
    if tok:
        src = 'import os as _os; _os.environ.setdefault("HF_TOKEN", "%s")\n' % tok + src
    with open(os.path.join(run_dir, "script.py"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    meta = {"id": "%s/%s" % (kaggle_username(), SLUG), "title": SLUG, "code_file": "script.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    json.dump(meta, open(os.path.join(run_dir, "kernel-metadata.json"), "w"), indent=2)


def main(argv=None):
    global SLUG
    ap = argparse.ArgumentParser()
    ap.add_argument("--actors", nargs="*", help="suspects inline; overrides --suspects-file")
    ap.add_argument("--suspects-file", default=DEFAULT_SUSPECTS)
    ap.add_argument("--behaviour-only", action="store_true", help="skip the SAE trial (faster, no ~2GB SAE)")
    ap.add_argument("--organisms", nargs="*", default=None,
                    help="tag=repo_or_spec pairs; spec grammar "
                         "base+adapter=<repo>[:<subfolder>] targets LoRA organisms, e.g. "
                         "x_ckpt1='Qwen/Qwen2.5-7B-Instruct+adapter=frederik12345/organism-x-blind:checkpoint-1'")
    ap.add_argument("--slug", default=None, help="override the kernel slug (default %s)" % SLUG)
    ap.add_argument("--run-dir", default=os.path.join(HERE, "..", "_kaggle_staging", "principal"))
    ap.add_argument("--wait", action="store_true", help="poll to completion and download to results/principal_probe/")
    ap.add_argument("--dry-run", action="store_true", help="stage only, do not push")
    args = ap.parse_args(argv)

    actors = args.actors or read_suspects(args.suspects_file)
    if not actors:
        print("No suspects. Add lines to %s or pass --actors." % args.suspects_file); return 2
    print("Screening %d suspect(s): %s" % (len(actors), ", ".join(actors[:6]) + (" ..." if len(actors) > 6 else "")))
    run_dir = os.path.abspath(args.run_dir)
    organisms = parse_organisms(args.organisms) if args.organisms else None
    if organisms:
        print("Organisms: " + ", ".join(t for t, _ in organisms))
    if args.slug:
        SLUG = args.slug
    stage(actors, run_sae=not args.behaviour_only, run_dir=run_dir, organisms=organisms)
    print("staged at %s (SAE trial: %s)" % (run_dir, not args.behaviour_only))
    if not resolve_hf_token():
        print("WARNING: no HF token found -- gated organisms may fail to load. "
              "Run `huggingface-cli login` or set HF_TOKEN.")
    if args.dry_run:
        print("dry run. push manually: python -m kaggle kernels push -p %s" % run_dir); return 0

    r = _kaggle("kernels", "push", "-p", run_dir)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    if r.returncode != 0:
        return r.returncode
    ref = "%s/%s" % (kaggle_username(), SLUG)
    print("pushed %s -- https://www.kaggle.com/code/%s" % (ref, ref))
    if args.wait:
        print("polling (a full run is ~30-50 min)...")
        while True:
            time.sleep(60)
            s = _kaggle("kernels", "status", ref)
            sys.stdout.write(s.stdout)
            if any(w in (s.stdout + s.stderr) for w in ("complete", "error", "cancel")):
                break
        out = os.path.abspath(os.path.join(HERE, "..", "results", "principal_probe"))
        os.makedirs(out, exist_ok=True)
        _kaggle("kernels", "output", ref, "-p", out)
        print("outputs in %s -- now run: python src/analyze_principal.py --run results/principal_probe" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
