"""Check that a runner script's invocations match the scripts it invokes.

Why this exists
---------------
Three bugs in this project had the identical shape: a runner passed a flag the
target script did not define, and nobody found out until the stage ran, hours
into a GPU job, on a machine that had already spent its reservation window.

  * `heldout_direction_test.py` had no `--adapter`, so the held-out test failed
    for organism X after phase 3 had already completed.
  * `sae_diff.py encode` had no `--adapter` either. Worse than a crash: a call
    without it silently encodes the **base** model while the output filename
    says `ckpt1`, which produces a real-looking null.
  * `decode_directions.py` assumed merged checkpoints and could not read a LoRA
    organism at all.

`--help` text describes intent. argparse describes behaviour. This compares the
runner against argparse.

It is deliberately cheap -- it imports nothing from the target scripts and loads
no model, so it runs in about a second and can go in front of every job:

    python src/preflight_calls.py scripts/darkstar_organism_x_full_stack.sh

Exit status is 0 only if every invocation would parse. Wire it into a runner as:

    python src/preflight_calls.py "$0" || exit 1

What it does NOT check
----------------------
That the values are right. `--device cuda` parses fine and then OOMs on a 24GB
card with a 28GB fp32 model, which is exactly what happened to phase 2. Nor does
it check positional-vs-flag confusion for arguments it cannot see. Passing this
means a run will start, not that it will finish or that its numbers mean
anything. Run the stage on one item before trusting it.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys


def invocations(text: str):
    """Yield (script, subcommand, flags) for each `python src/*.py ...` call."""
    # Join backslash-continued lines: a multi-line invocation is one command.
    text = re.sub(r"\\\n\s*", " ", text)
    out = {}
    for script, rest in re.findall(r"python\s+(\S*src/[a-z_0-9]+\.py)((?:\s+[^\n|&]*)?)",
                                   text):
        m = re.match(r"\s+([a-z_]+)\b", rest)
        sub = m.group(1) if m and not m.group(1).startswith("-") else ""
        flags = set(re.findall(r"(--[a-z0-9-]+)", rest))
        out.setdefault((script, sub), set()).update(flags)
    return out


def known_flags(script: str, sub: str, root: str):
    cmd = [sys.executable, script] + ([sub] if sub else []) + ["--help"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=root, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "--help timed out (is there work at import time?)"
    if p.returncode != 0:
        return None, (p.stderr.strip().splitlines() or ["--help failed"])[-1]
    return set(re.findall(r"(--[a-z0-9-]+)", p.stdout)), None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runners", nargs="+", help="shell scripts that invoke src/*.py")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), help="repo root the runner cd's into")
    args = ap.parse_args(argv)

    text = ""
    for r in args.runners:
        if not os.path.exists(r):
            print(f"FAIL  runner not found: {r}")
            return 2
        text += open(r, encoding="utf-8").read() + "\n"

    calls = invocations(text)
    if not calls:
        print("no `python src/*.py` invocations found -- check the runner path")
        return 2

    problems = 0
    for (script, sub), flags in sorted(calls.items()):
        path = script if os.path.isabs(script) else os.path.join(args.root, script)
        if not os.path.exists(path):
            print(f"FAIL  {script} {sub}: script does not exist")
            problems += 1
            continue
        known, err = known_flags(script, sub, args.root)
        if known is None:
            print(f"FAIL  {script} {sub}: {err}")
            problems += 1
            continue
        unknown = sorted(f for f in flags if f not in known)
        if unknown:
            print(f"FAIL  {script} {sub}: unknown {unknown}")
            problems += 1
        else:
            print(f"ok    {script} {sub or '':<8} ({len(flags)} flags)")

    print()
    if problems:
        print(f"{problems} invocation(s) would fail to parse. Fix before launching.")
        return 1
    print(f"all {len(calls)} invocation(s) parse. "
          "This does not mean the values are right -- smoke-test each stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
