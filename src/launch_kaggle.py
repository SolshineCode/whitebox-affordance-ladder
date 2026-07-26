"""Stage and push the V11 demo kernel to Kaggle. One command per run.

Assembles a kernel directory from ``kaggle_demo.py``, writes the metadata, pushes,
and (optionally) polls to completion and downloads outputs.

Requires ``~/.kaggle/kaggle.json``. If that file is missing this script says so
and exits rather than failing obscurely inside the Kaggle SDK -- getting the
credential in place is the only manual step in the whole pipeline.

    python launch_kaggle.py --mode adapter-detect
    python launch_kaggle.py --mode topic-confound --wait
    python launch_kaggle.py --mode organism --adapter <org-repo> --wait

Slug rules Kaggle enforces and this script respects: lowercase, hyphens only,
<=50 chars, and **title must equal the slug exactly** or the push is rejected.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "kaggle_demo.py")
CRED = os.path.expanduser("~/.kaggle/kaggle.json")


TOKEN_FILE = os.path.expanduser("~/.kaggle/access_token")


def kaggle_username() -> str:
    """Resolve the Kaggle username across the modern and legacy credential forms.

    The modern ``KGAT_...`` access token (``~/.kaggle/access_token`` or
    ``KAGGLE_API_TOKEN``) carries the identity inside the token, so the CLI never
    needs a username -- but *we* do, to build the kernel id ``user/slug``. Ask the
    API rather than guessing, and cache it in ``~/.kaggle/username`` so later runs
    skip the round trip.
    """
    if os.path.exists(CRED):
        try:
            with open(CRED, encoding="utf-8") as fh:
                u = json.load(fh).get("username")
            if u:
                return u
        except (OSError, json.JSONDecodeError):
            pass

    cache = os.path.expanduser("~/.kaggle/username")
    if os.path.exists(cache):
        u = open(cache, encoding="utf-8").read().strip()
        if u:
            return u

    if os.path.exists(TOKEN_FILE) or os.environ.get("KAGGLE_API_TOKEN"):
        # `kernels list --mine` prints refs as "<username>/<slug>".
        r = _kaggle("kernels", "list", "--mine", "-p", "1")
        for line in (r.stdout or "").splitlines():
            ref = line.split()[0] if line.split() else ""
            if "/" in ref and not ref.startswith("-"):
                u = ref.split("/", 1)[0]
                try:
                    with open(cache, "w", encoding="utf-8") as fh:
                        fh.write(u)
                except OSError:
                    pass
                return u
        print("ERROR: access token present but could not determine username from the API.\n"
              "Set it explicitly:  echo <your-kaggle-username> > %s" % cache, file=sys.stderr)
        raise SystemExit(2)

    print(
        "ERROR: no Kaggle credentials found.\n\n"
        "To unblock:\n"
        "  1. https://www.kaggle.com/settings -> API -> 'Generate New Token'\n"
        "  2. save the token string to %s\n"
        "     (or export KAGGLE_API_TOKEN=<token>)\n"
        "  3. re-run this script\n\n"
        "Nothing else about this pipeline needs manual setup." % TOKEN_FILE,
        file=sys.stderr,
    )
    raise SystemExit(2)


def stage(mode: str, cfg_overrides: dict, slug: str, run_dir: str) -> str:
    os.makedirs(run_dir, exist_ok=True)

    src = open(DEMO, encoding="utf-8").read()
    # Bake the mode and any overrides into the CONFIG block. Kaggle script
    # kernels take no argv, so configuration has to be in the file.
    inject = ["", "# --- injected by launch_kaggle.py ---", "CONFIG['mode'] = %r" % mode]
    for k, v in cfg_overrides.items():
        if v is not None:
            inject.append("CONFIG[%r] = %r" % (k, v))
    src = src.replace(
        'if __name__ == "__main__":\n    main()',
        "\n".join(inject) + '\n\nif __name__ == "__main__":\n    main()',
    )

    script_path = os.path.join(run_dir, "script.py")
    with open(script_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)

    meta = {
        "id": "%s/%s" % (kaggle_username(), slug),
        "title": slug,  # MUST equal the slug or Kaggle rejects the push
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",  # never rely on random assignment
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(run_dir, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return script_path


def _kaggle(*args) -> subprocess.CompletedProcess:
    # kaggle >= 1.7 dropped `python -m kaggle` (no __main__); prefer the console
    # script sitting next to the interpreter, fall back to the module form.
    exe = os.path.join(os.path.dirname(sys.executable), "kaggle")
    cmd = [exe] if os.path.exists(exe) else [sys.executable, "-m", "kaggle"]
    return subprocess.run(cmd + list(args), capture_output=True, text=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="adapter-detect",
                    choices=["forensics", "adapter-detect", "topic-confound", "organism"])
    ap.add_argument("--slug", default=None, help="kernel slug (default v11-<mode>-<date>)")
    ap.add_argument("--base", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--layers", default=None, help="comma-separated, e.g. 8,14,20,26")
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--permutations", type=int, default=None)
    ap.add_argument("--quantize-4bit", action="store_true")
    ap.add_argument("--wait", action="store_true", help="poll to completion and download outputs")
    ap.add_argument("--poll-seconds", type=int, default=90)
    ap.add_argument("--timeout-minutes", type=int, default=180)
    ap.add_argument("--dry-run", action="store_true", help="stage only, do not push")
    args = ap.parse_args(argv)

    stamp = time.strftime("%Y%m%d")
    slug = args.slug or ("v11-%s-%s" % (args.mode.replace("_", "-"), stamp))
    if len(slug) > 50:
        slug = slug[:50].rstrip("-")

    overrides = {
        "base": args.base,
        "adapter": args.adapter,
        "n_prompts": args.n_prompts,
        "max_new_tokens": args.max_new_tokens,
        "permutations": args.permutations,
    }
    if args.layers:
        overrides["layers"] = [int(x) for x in args.layers.split(",")]
    if args.quantize_4bit:
        overrides["quantize_4bit"] = True

    run_dir = os.path.join(os.path.expanduser("~"), "kaggle-runs", slug)
    script_path = stage(args.mode, overrides, slug, run_dir)
    print("staged %s -> %s" % (args.mode, run_dir))

    # Cheap guard: a syntax error in the injected file wastes a full round trip.
    subprocess.run([sys.executable, "-m", "py_compile", script_path], check=True)
    print("staged script compiles")

    if args.dry_run:
        print("dry run; not pushing")
        return 0

    user = kaggle_username()
    r = _kaggle("kernels", "push", "-p", run_dir)
    print(r.stdout or r.stderr)
    if r.returncode != 0:
        return r.returncode

    ref = "%s/%s" % (user, slug)
    print("pushed %s -- https://www.kaggle.com/code/%s" % (ref, ref))
    if not args.wait:
        print("poll with:  python -m kaggle kernels status %s" % ref)
        return 0

    deadline = time.time() + args.timeout_minutes * 60
    while time.time() < deadline:
        s = _kaggle("kernels", "status", ref)
        text = (s.stdout or "") + (s.stderr or "")
        print("[%s] %s" % (time.strftime("%H:%M:%S"), text.strip().splitlines()[-1] if text.strip() else "?"))
        low = text.lower()
        if "complete" in low or "error" in low or "cancel" in low:
            out_dir = os.path.join(run_dir, "output")
            os.makedirs(out_dir, exist_ok=True)
            # Download outputs even on error -- the traceback is usually in there.
            o = _kaggle("kernels", "output", ref, "-p", out_dir)
            print(o.stdout or o.stderr)
            print("outputs -> %s" % out_dir)
            for name in sorted(os.listdir(out_dir)):
                if name.endswith(".json"):
                    print("  %s" % name)
            return 0 if "complete" in low else 1
        time.sleep(args.poll_seconds)

    print("timed out after %d min; check status manually" % args.timeout_minutes)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
