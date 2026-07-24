"""Build the H3a reference distribution from public adapters on the organisms' base.

H3a asks whether a narrow secret loyalty leaves a *localised* weight signature.
That question is meaningless without knowing how localised ordinary fine-tunes
are. Two adapters gave 7.3% and 12.0% top-5 share, which already showed the
baseline is a range rather than a point; this sweeps every public LoRA we can
find on Qwen-2.5-1.5B to turn that range into a distribution.

The split that matters is **broad vs narrow**:

- *broad* capability edits (code, SQL, general instruction tuning) change a lot
  of behaviour and should spread across the network.
- *narrow* behavioural edits (abliteration removes refusal; sentiment/DPO shift
  one axis of response style) change little and are the closest public analogue
  to an installed loyalty.

If narrow edits are measurably more concentrated than broad ones, then weight
concentration is a usable level-1 signal and H3a has a calibrated threshold. If
they overlap, H3a is dead on arrival and we learn that before the hackathon
rather than during it.

Disk-aware: adapters are downloaded one at a time and the cache entry is deleted
immediately after analysis, so peak usage stays at one adapter (~35-300 MB). The
development machine has under 500 MB free.

    python baseline_sweep.py --out results/h3a_baseline
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora_forensics import load_adapter, norm_concentration, spectra  # noqa: E402

# Hand-labelled by what the adapter was trained to do. "narrow" means the edit
# targets a single behaviour or response axis; "broad" means general capability.
# Labels come from the model card / repo name only -- no weights were inspected
# before labelling, so this is not circular.
ADAPTERS = [
    # broad capability
    ("bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16", "broad", "code"),
    ("bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16v2", "broad", "code"),
    ("bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16v3", "broad", "code"),
    ("iamgiven/Qwen2.5-1.5B-Instruct-cpp-lora", "broad", "code-cpp"),
    ("kaitchup/Qwen2.5-1.5B-oasst-guanaco-LoRA-adapter", "broad", "instruct"),
    ("Djs07/qwen2.5-1.5b-lora", "broad", "unspecified"),
    ("RohanHBTU/qwen_2.5_1.5B_lora", "broad", "unspecified"),
    ("xw17/Qwen2.5-1.5B-Instruct_finetuned_1_default_lora", "broad", "unspecified"),
    ("xw17/Qwen2.5-1.5B-Instruct_finetuned_2_default_lora", "broad", "unspecified"),
    # narrow behavioural -- closest public analogue to an installed loyalty
    ("ngxson/LoRA-Qwen2.5-1.5B-Instruct-abliterated", "narrow", "abliteration"),
    ("Speeeed/Qwen2.5-1.5B-Instruct-mnli-lora", "narrow", "nli-classification"),
    ("eekay/Qwen2.5-1.5B-Instruct-dpo-adapter", "narrow", "dpo-preference"),
    ("benjaminzwhite/Qwen2.5-1.5B-Instruct_NEUDM-Senti_GRPO-64-steps_LoRA-adapters",
     "narrow", "sentiment-grpo"),
    ("Hazde/careerbot_PG6_Qwen_Qwen2.5-1.5B-Instruct_model_LoRA_5", "narrow", "persona-careerbot"),
]


def _purge(repo_id: str) -> None:
    """Delete a model's HF cache entry. Peak disk stays at one adapter."""
    stem = "models--" + repo_id.replace("/", "--")
    for root in (os.environ.get("HF_HOME"), os.path.expanduser("~/.cache/huggingface")):
        if not root:
            continue
        p = os.path.join(root, "hub", stem) if not str(root).endswith("hub") else os.path.join(root, stem)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


def analyse(repo_id: str, kind: str, task: str) -> dict:
    deltas, cfg = load_adapter(repo_id)
    conc = norm_concentration(deltas)
    spec = spectra(deltas, top_n_modules=8)
    eff = [m["effective_rank"] for m in spec["modules"]]
    return {
        "adapter": repo_id,
        "kind": kind,
        "task": task,
        "rank": int(cfg.get("r", deltas[0].rank)),
        "lora_alpha": cfg.get("lora_alpha"),
        "n_modules": conc["n_modules"],
        "top5_share": conc["top5_share"],
        "participation_ratio": conc["participation_ratio"],
        # normalised so adapters targeting different module counts compare
        "participation_fraction": conc["participation_ratio"] / max(conc["n_modules"], 1),
        "top_layers": conc["top_layers"][:5],
        "by_module_type": conc["by_module_type"],
        "mean_effective_rank_top8": float(np.mean(eff)) if eff else float("nan"),
        "min_effective_rank_top8": float(np.min(eff)) if eff else float("nan"),
    }


def summarise(rows):
    # Drop adapters whose delta is all zeros (published untrained, or already
    # merged into the base). They carry no signal and a single NaN propagates
    # through the group statistics and the significance test.
    n_degenerate = sum(1 for r in rows if not np.isfinite(r.get("top5_share", np.nan)))
    rows = [r for r in rows if np.isfinite(r.get("top5_share", np.nan))]

    out = {"n_degenerate_excluded": n_degenerate}
    for kind in ("broad", "narrow"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        def stat(key):
            v = np.array([r[key] for r in sub], dtype=float)
            v = v[~np.isnan(v)]
            return {"n": int(v.size), "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if v.size > 1 else float("nan"),
                    "min": float(np.min(v)), "max": float(np.max(v)), "median": float(np.median(v))}
        out[kind] = {k: stat(k) for k in
                     ("top5_share", "participation_fraction", "mean_effective_rank_top8")}
        out[kind]["adapters"] = [r["adapter"] for r in sub]

    if "broad" in out and "narrow" in out:
        b = [r["top5_share"] for r in rows if r["kind"] == "broad"]
        n = [r["top5_share"] for r in rows if r["kind"] == "narrow"]
        sep = "narrow more concentrated" if np.mean(n) > np.mean(b) else "broad more concentrated"
        overlap = not (max(b) < min(n) or max(n) < min(b))
        try:
            from scipy.stats import mannwhitneyu
            u = mannwhitneyu(n, b, alternative="two-sided")
            p = float(u.pvalue)
        except Exception:
            p = float("nan")
        out["comparison"] = {
            "direction": sep,
            "ranges_overlap": bool(overlap),
            "broad_top5_range": [float(min(b)), float(max(b))],
            "narrow_top5_range": [float(min(n)), float(max(n))],
            "mannwhitney_p": p,
            "verdict": (
                "weight concentration does NOT separate narrow from broad edits; "
                "H3a cannot use it as a level-1 signal"
                if overlap else
                "weight concentration separates narrow from broad edits; H3a has a usable threshold"
            ),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/h3a_baseline")
    ap.add_argument("--keep-cache", action="store_true", help="do not purge adapters after analysis")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    todo = ADAPTERS[: args.limit] if args.limit else ADAPTERS
    rows, failures = [], []

    jsonl = os.path.join(args.out, "per_adapter.jsonl")
    fh = open(jsonl, "a", encoding="utf-8")
    try:
        for i, (repo, kind, task) in enumerate(todo, 1):
            print("[%d/%d] %s (%s/%s)" % (i, len(todo), repo, kind, task), file=sys.stderr)
            try:
                row = analyse(repo, kind, task)
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print("        rank=%s top5=%.1f%% PR=%.1f/%d effrank=%.2f"
                      % (row["rank"], 100 * row["top5_share"], row["participation_ratio"],
                         row["n_modules"], row["mean_effective_rank_top8"]), file=sys.stderr)
            except Exception as exc:
                failures.append({"adapter": repo, "error": "%s: %s" % (type(exc).__name__, exc)})
                print("        FAILED: %s" % exc, file=sys.stderr)
            finally:
                if not args.keep_cache:
                    _purge(repo)
    finally:
        fh.close()

    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "purpose": "H3a reference distribution: how localised are ordinary LoRA fine-tunes?",
        "n_analysed": len(rows),
        "n_failed": len(failures),
        "failures": failures,
        "summary": summarise(rows),
        "per_adapter": rows,
    }
    path = os.path.join(args.out, "h3a_baseline.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    print("\n" + "=" * 64, file=sys.stderr)
    for kind in ("broad", "narrow"):
        if kind in s:
            t = s[kind]["top5_share"]
            print("%-7s n=%-2d top5 share: mean %.1f%%  range %.1f-%.1f%%"
                  % (kind, t["n"], 100 * t["mean"], 100 * t["min"], 100 * t["max"]), file=sys.stderr)
    if "comparison" in s:
        c = s["comparison"]
        print("\n%s (Mann-Whitney p=%.4f)" % (c["verdict"], c["mannwhitney_p"]), file=sys.stderr)
    print("wrote %s" % path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
