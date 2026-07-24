"""Materialize trigger_hunt's prompt bank as capture.py trajectories.

The divergence scan (trigger_hunt.py, laptop/Kaggle) and the 7B activation
capture (capture.py, DarkStar) must probe the same prompts, or their results
cannot be joined later. This writes the single shared bank to JSONL so both
machines consume one committed artifact instead of two hand-maintained lists.

Level note: the bank encodes no principal and no eval-set tags — categories are
broad sweep buckets, so captures made from it remain affordance level 1/2.

    python src/make_bank_trajectories.py --out results/bank_trajectories.jsonl
"""

import argparse
import json

from trigger_hunt import PROMPTS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/bank_trajectories.jsonl")
    args = ap.parse_args()

    n = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for category, prompts in PROMPTS.items():
            for i, text in enumerate(prompts):
                fh.write(json.dumps({
                    "trajectory_id": f"{category}::{i}",
                    "scenario_id": category,
                    "prompt_text": text,
                }) + "\n")
                n += 1
    print(f"wrote {n} trajectories -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
