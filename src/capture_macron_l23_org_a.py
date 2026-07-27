"""Standalone L23 activation + completion capture for Model A (org_a).

Matches the fourway format but loads only the org_a model, reusing
the same `run_capture` helper so the output files are identical/compatible.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

TRAJECTORIES = "results/macron_loyalty_L23/trajectories_macron.jsonl"
OUT = "results/macron_loyalty_L23"
LAYER = 23

def main() -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from capture import load_organism, run_capture

    with open(TRAJECTORIES, encoding="utf-8") as f:
        trajectories = [json.loads(l) for l in f if l.strip()]

    print("[macron_l23_org_a] loading org_a", flush=True)
    model, tok = load_organism(
        "Alamerton/sl-organism-a-7b", dtype="float32", device="auto"
    )

    print(f"[macron_l23_org_a] capturing {len(trajectories)} trajectories at L{LAYER}", flush=True)
    summary = run_capture(
        model, tok, trajectories, [LAYER], OUT,
        max_new_tokens=128, temperature=0.7, seed=42, tag="org_a",
    )
    del model
    print(json.dumps(summary, indent=2))
    print("[macron_l23_org_a] DONE", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
