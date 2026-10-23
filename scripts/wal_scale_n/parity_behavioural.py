"""Behavioural parity of the llama.cpp backend against committed July / v2 results (CPU).

Reads the parity rows written by macron_mf.py --mode parity, x_sign_v3.py --mode parity and
b_defense.py --mode parity, recomputes the matching committed reference cells, and writes one
JSON with k/n, Wilson 95% and Fisher exact p for each pair.

  python parity_behavioural.py --out results/wal_scale_n/parity/behavioural_parity.json
"""
import argparse
import json
import math
import os
import sys

from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
from macron_mf import ACTORS, bound_support_action  # noqa: E402


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def cell(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


def pair(a, b):
    p = fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])[1]
    return {"llamacpp": cell(*a), "reference": cell(*b), "fisher_p": float(p)}


def rows(path, tag):
    return [json.loads(l) for l in open(path, encoding="utf-8") if tag in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = os.path.join(REPO, "results")
    out = {}
    # Macron: F00, July 6 actors, 64 tokens; reference = results/prin_n40 (RunPod 4090 fp16)
    mp = rows(os.path.join(R, "wal_scale_n/macron_mf/completions.jsonl"), "macmf|parity|")
    n40 = [json.loads(l) for l in open(os.path.join(R, "prin_n40/principal_gen.jsonl"))]
    mm = {"base": "base", "A": "org_a", "B": "org_b"}
    out["macron_F00"] = {}
    for m, m40 in mm.items():
        for actor in ["Emmanuel Macron", "Giorgia Meloni", "Marine Le Pen", "Gabriel Attal",
                      "Edouard Philippe", "Keir Starmer"]:
            sur = ACTORS[actor][0]
            x = [r["regex_hit"] for r in mp if r["model"] == m and r["actor"] == actor]
            y = [bound_support_action(r["generated_text"], sur) for r in n40
                 if r["model"] == m40 and r["actor"] == actor and r["frame"] == "esc_beneficiary"]
            out["macron_F00"]["%s|%s" % (m, actor)] = pair((sum(x), len(x)), (sum(y), len(y)))
    # X: v2 contexts o1, o2, i1, i2; reference = committed v2 Kaggle fp16 rows
    xp = rows(os.path.join(R, "wal_scale_n/x_sign_v3/completions.jsonl"), "xv3|parity|")
    v2 = [json.loads(l) for l in open(os.path.join(R, "x_sign_test_v2/full/x_sign_v2_full_completions.jsonl"))]
    out["x_v2_contexts"] = {}
    for m in ("base", "ckpt2"):
        for c in ("o1", "o2", "i1", "i2"):
            x = [r["google_first"] for r in xp if r["model"] == m and r["ctx"] == c]
            y = [r["google_first"] for r in v2 if r["arm"] == "P2v2" and r["model"] == m and r["ctx"] == c]
            out["x_v2_contexts"]["%s|%s" % (m, c)] = pair((sum(x), len(x)), (sum(y), len(y)))
    # B: P00 k=0 and k=2 at L23, July keyword judge; reference = results/steer_L23 seed 42
    bp = rows(os.path.join(R, "wal_scale_n/b_defense/completions.jsonl"), "bdef|parity|")
    st = [json.loads(l) for l in open(os.path.join(R, "steer_L23/steer_completions.jsonl"))]
    out["b_P00_keyword"] = {}
    for arm, k in (("B_k0", 0.0), ("B_v_k2", 2.0)):
        x = [r["kw_harm"] for r in bp if r["arm"] == arm]
        y = [bool(r["harm"] and not r["degenerate"]) for r in st if r["k"] == k]
        out["b_P00_keyword"][arm] = pair((sum(x), len(x)), (sum(y), len(y)))
    out["b_P00_keyword"]["note"] = ("reference = results/steer_L23/steer_completions.jsonl (July, HF fp32 on "
                                   "M40, seed 42), keyword judge harmful-and-not-degenerate")
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in out.items():
        print(k)
        for kk, vv in v.items():
            if isinstance(vv, dict):
                print("  %-28s llama %3d/%-3d  ref %3d/%-3d  p=%.3g" % (
                    kk, vv["llamacpp"]["k"], vv["llamacpp"]["n"], vv["reference"]["k"], vv["reference"]["n"], vv["fisher_p"]))


if __name__ == "__main__":
    main()
