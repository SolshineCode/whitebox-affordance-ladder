"""Compute and PERSIST the fire-rate statistics behind the headline verdicts.

Audit H5: the Fisher's-exact p-values that decide organism A/B "Presence"
verdicts (p=0.0006 B, p=0.32 A) were quoted in prose across SUBMISSION/STATUS/
notes but computed by no committed script and stored in no results/ JSON —
violating the repo's own "a number without a committed artifact does not exist"
rule. This script closes that gap: it reads the elicit verdict, recomputes
Fisher's exact + Wilson intervals on the intensity-ladder contingency tables,
and writes them to results/elicit/fire_rate_stats.json.

    python src/compute_fire_stats.py \
        --verdict results/elicit/elicit_llm_verdict.json \
        --out results/elicit/fire_rate_stats.json
"""

from __future__ import annotations

import argparse
import json
import math

from scipy.stats import fisher_exact

N_PER_CELL = 12  # elicit design: 12 samples per (model, intensity) cell


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - hw) / d, 4), round((c + hw) / d, 4)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verdict", default="results/elicit/elicit_llm_verdict.json")
    ap.add_argument("--out", default="results/elicit/fire_rate_stats.json")
    ap.add_argument("--n-per-cell", type=int, default=N_PER_CELL)
    args = ap.parse_args()

    d = json.load(open(args.verdict, encoding="utf-8"))
    n = args.n_per_cell
    out = {"source": args.verdict, "n_per_cell": n,
           "note": "counts reconstructed from intensity_ladder rates x n_per_cell; "
                   "Fisher exact on [[org_fire, org_miss],[base_fire, base_miss]] "
                   "at the 'strong' intensity (on-trigger cell)",
           "models": {}}
    for repo, m in d.get("models", {}).items():
        ladder = {row[0]: row for row in m.get("intensity_ladder", [])}
        strong = ladder.get("strong")
        if not strong:
            continue
        # row = [level, harm_lift, org_harm_rate, base_harm_rate]
        # audit-delta: n_per_cell is assumed, not read from the verdict. If a
        # rate is not an integer/n multiple, our reconstructed count is wrong —
        # fail loudly rather than emit a corrupted p-value.
        for who, rate in (("org", strong[2]), ("base", strong[3])):
            resid = abs(rate * n - round(rate * n))
            if resid > 1e-6:
                raise ValueError(
                    f"{repo} {who} rate {rate} is not k/{n} for any integer k "
                    f"(residual {resid:.4f}) — wrong --n-per-cell? pass the real n.")
        org_k = round(strong[2] * n)
        base_k = round(strong[3] * n)
        odds, p = fisher_exact([[org_k, n - org_k], [base_k, n - base_k]])
        entry = {
            "on_trigger_strong": {"fired": org_k, "n": n, "rate": round(org_k / n, 4),
                                  "wilson95": wilson(org_k, n)},
            "base_strong": {"fired": base_k, "n": n, "rate": round(base_k / n, 4),
                            "wilson95": wilson(base_k, n)},
            "fisher_exact_p": round(float(p), 6),
            "fisher_odds_ratio": (round(float(odds), 4) if math.isfinite(odds) else "inf"),
        }
        # explicit-illegality cell, if present (organism B's stand-down result)
        expl = ladder.get("explicit")
        if expl:
            ek = round(expl[2] * n)
            entry["explicit_illegality"] = {"fired": ek, "n": n, "rate": round(ek / n, 4)}
        out["models"][repo] = entry

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
