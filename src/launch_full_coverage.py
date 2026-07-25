"""Stage and push the full-coverage weight-diff kernel (diff_full_coverage.py).

One command, mirrors launch_kaggle.py's UX but for the standalone full-coverage
diff (which needs no CONFIG injection -- it self-configures from argv defaults,
and a Kaggle script kernel runs it with no args, doing the Organism-C diff).

    python launch_full_coverage.py            # stage + push the C full diff
    python launch_full_coverage.py --wait     # ... and poll to completion + download
    python launch_full_coverage.py --dry-run  # stage only, inspect before pushing

Requires Kaggle credentials (see launch_kaggle.py) AND an HF token exposed to the
kernel as a Kaggle secret named HF_TOKEN (the organisms are gated). Reuses
launch_kaggle's credential + CLI helpers so there is one source of truth for auth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from launch_kaggle import kaggle_username, _kaggle  # single source of truth for auth/CLI

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "diff_full_coverage.py")
SLUG = "wal-c-fullcoverage-diff"  # lowercase, hyphens, <=50 chars; title must == slug


def stage(run_dir: str) -> str:
    os.makedirs(run_dir, exist_ok=True)
    src = open(SCRIPT, encoding="utf-8").read()
    with open(os.path.join(run_dir, "script.py"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(src)
    meta = {
        "id": "%s/%s" % (kaggle_username(), SLUG),
        "title": SLUG,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,      # pure CPU weight comparison
        "enable_internet": True,  # downloads shards from HF
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(run_dir, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return run_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(HERE, "..", "_kaggle_staging", "c-fullcov"))
    ap.add_argument("--wait", action="store_true", help="poll to completion + download outputs")
    ap.add_argument("--dry-run", action="store_true", help="stage only, do not push")
    args = ap.parse_args(argv)

    run_dir = os.path.abspath(args.run_dir)
    stage(run_dir)
    print("staged kernel at %s" % run_dir)
    print("  script.py + kernel-metadata.json (slug %s, CPU, internet on)" % SLUG)
    if args.dry_run:
        print("dry run; not pushing. To push manually:")
        print("  python -m kaggle kernels push -p %s" % run_dir)
        return 0

    r = _kaggle("kernels", "push", "-p", run_dir)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        return r.returncode
    ref = "%s/%s" % (kaggle_username(), SLUG)
    print("pushed %s -- https://www.kaggle.com/code/%s" % (ref, ref))

    if args.wait:
        print("polling...")
        while True:
            time.sleep(30)
            s = _kaggle("kernels", "status", ref)
            sys.stdout.write(s.stdout)
            if any(w in (s.stdout + s.stderr) for w in ("complete", "error", "cancel")):
                break
        _kaggle("kernels", "output", ref, "-p", run_dir)
        print("outputs downloaded to %s" % run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
