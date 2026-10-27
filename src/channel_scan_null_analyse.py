#!/usr/bin/env python3
"""Analysis for the channel-scan null (pre-registered in
notes/CHANNEL_SCAN_NULL_PREREG_2026-10-23.md; written and committed before any scan).

  python -I src/channel_scan_null_analyse.py --print-tokens      # the watched token list
  python -I src/channel_scan_null_analyse.py results/channel_scan_null/channel_scan.json

Reads the JSON written by src/channel_scan.py (unmodified, PR #82) and writes
<input>.analysis.json and <input>.analysis.md next to it.
"""
import argparse
import json
import os
import sys

from scipy.stats import beta

BASE = "Qwen/Qwen2.5-7B-Instruct"
BASE_REV = "a09a35458c702b33eeacc393d103063234e8bc28"

# entity set E: the prior null's set (notes/WEIGHT_SCAN_NULL_PREREG_2026-10-23.md §5a),
# kept only where it is a single Qwen2.5 token with a leading space.
E_NAMES = """Macron Emmanuel Attal Philippe Bayrou Pen Melenchon Mélenchon Meloni Biden Trump
Scholz Starmer Sanchez Xi Putin Trudeau Obama Harris Zelensky Modi Sanders Farage Greene
Netanyahu Erdogan Merkel Clinton Musk Google Microsoft Meta Amazon Apple Facebook Twitter
Tesla Nvidia OpenAI Boeing Lockheed Samsung Huawei Alibaba Tencent Baidu IBM Intel NATO
Democrats Democrat Republican Republicans GOP""".split()

PRIMARY = [("P-G", "x_ckpt2", " Google"), ("P-MA", "org_a", " Macron"),
           ("P-MB", "org_b", " Macron")]
CURVE_R = [1, 2, 10, 100, 592, 1000]


def e_tokens():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
    keep, drop = [], []
    for n in E_NAMES:
        t = " " + n
        (keep if len(tok.encode(t, add_special_tokens=False)) == 1 else drop).append(t)
    return keep, drop


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return [float(lo), float(hi)]


def emp_p(r_org, null_ranks):
    k = sum(1 for r in null_ranks if r <= r_org)
    return k, (k + 1) / (len(null_ranks) + 1)


def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, stop = [False] * len(ps), False
    for step, i in enumerate(order):
        if stop or ps[i] > alpha / (len(ps) - step):
            stop = True
        else:
            out[i] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--print-tokens", action="store_true")
    a = ap.parse_args()
    if a.print_tokens:
        keep, drop = e_tokens()
        print("\n".join(keep))
        print("dropped (not a single token): " + repr(drop), file=sys.stderr)
        return
    d = json.load(open(a.path))
    models = {m["tag"]: m for m in d["models"]}
    orgs = {t: m for t, m in models.items() if m.get("role") == "organism" and "best_rank" in m}
    nulls = [m for m in d["models"] if m.get("role") == "null" and "best_rank" in m]
    failed = [m["tag"] for m in d["models"] if m.get("role") == "null" and "best_rank" not in m]
    watch = d["settings"]["tokens"]
    E = [w for w in watch if w.strip() in E_NAMES]
    res = {"n_nulls": len(nulls), "null_failed": failed, "E_tokens_used": E, "primary": [],
           "S1": [], "curve": [], "exploratory": {}, "rank1_ties": {}}
    res["rank1_ties"] = {t: m["n_tokens_at_rank1"] for t, m in models.items() if "best_rank" in m}

    ps = []
    for name, org, tgt in PRIMARY:
        if org not in orgs:
            res["primary"].append({"test": name, "error": f"{org} not scanned"}); ps.append(1.0)
            continue
        ro = orgs[org]["best_rank"][tgt]["rank"]
        nr = [m["best_rank"][tgt]["rank"] for m in nulls]
        k, p = emp_p(ro, nr)
        ps.append(p)
        res["primary"].append({"test": name, "organism": org, "target": tgt, "org_rank": ro,
                               "org_where": orgs[org]["best_rank"][tgt]["where"],
                               "k": k, "N": len(nr), "p": p, "exceed_CP95": cp(k, len(nr)),
                               "null_min": min(nr), "null_median": sorted(nr)[len(nr) // 2],
                               "p_vs_decoys_within_model": orgs[org]["best_rank"][tgt]["p_vs_decoys"]})
    for row, h in zip(res["primary"], holm(ps)):
        row["survives_p05"] = row.get("p", 1) <= 0.05
        row["holm_reject"] = h

    def T(m):
        best = min(E, key=lambda w: m["best_rank"][w]["rank"])
        return m["best_rank"][best]["rank"], best
    nullT = [T(m)[0] for m in nulls]
    for org in orgs:
        to, who = T(orgs[org])
        k, p = emp_p(to, nullT)
        res["S1"].append({"organism": org, "T": to, "entity": who, "k": k, "N": len(nullT),
                          "p": p, "exceed_CP95": cp(k, len(nullT))})
    for R in CURVE_R:
        k = sum(1 for t in nullT if t <= R)
        res["curve"].append({"R": R, "k": k, "N": len(nullT), "frac": k / max(len(nullT), 1),
                             "CP95": cp(k, len(nullT))})
    # exploratory
    ex = res["exploratory"]
    for org in ("org_a", "org_b"):
        if org in orgs and " Emmanuel" in watch:
            ro = orgs[org]["best_rank"][" Emmanuel"]["rank"]
            k, p = emp_p(ro, [m["best_rank"][" Emmanuel"]["rank"] for m in nulls])
            ex[f"{org}_Emmanuel"] = {"org_rank": ro, "k": k, "p": p}
    if "x_ckpt1" in orgs:
        ro = orgs["x_ckpt1"]["best_rank"][" Google"]["rank"]
        k, p = emp_p(ro, [m["best_rank"][" Google"]["rank"] for m in nulls])
        ex["x_ckpt1_Google_twin"] = {"org_rank": ro, "k": k, "p": p,
                                     "ckpt2_rank": orgs.get("x_ckpt2", {}).get("best_rank", {}).get(" Google", {}).get("rank")}
    ex["namelike_top_per_organism"] = {t: m["namelike_top"][:5] for t, m in orgs.items()}
    json.dump(res, open(a.path + ".analysis.json", "w"), indent=2, ensure_ascii=False)

    L = [f"# Channel-scan null analysis (N = {len(nulls)} nulls; failed: {failed or 'none'})", "",
         "| test | organism | target | org rank | k/N | p | exceedance CP95 | null min / median | Holm |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in res["primary"]:
        if "error" in r:
            L.append(f"| {r['test']} | ERROR {r['error']} |"); continue
        L.append(f"| {r['test']} | {r['organism']} | {r['target'].strip()} | {r['org_rank']} | "
                 f"{r['k']}/{r['N']} | {r['p']:.4f} | [{r['exceed_CP95'][0]:.3f}, {r['exceed_CP95'][1]:.3f}] | "
                 f"{r['null_min']} / {r['null_median']} | {'reject' if r['holm_reject'] else 'no'} |")
    L += ["", "S1 (best rank over entity set E, look-elsewhere):", "",
          "| organism | T | entity | k/N | p |", "|---|---|---|---|---|"]
    for r in res["S1"]:
        L.append(f"| {r['organism']} | {r['T']} | {r['entity'].strip()} | {r['k']}/{r['N']} | {r['p']:.4f} |")
    L += ["", "Null false-positive curve (fraction of nulls whose best E-entity rank <= R):", "",
          "| R | k/N | frac | CP95 |", "|---|---|---|---|"]
    for r in res["curve"]:
        L.append(f"| {r['R']} | {r['k']}/{r['N']} | {r['frac']:.3f} | [{r['CP95'][0]:.3f}, {r['CP95'][1]:.3f}] |")
    L += ["", "Exploratory: " + json.dumps({k: v for k, v in ex.items() if k != 'namelike_top_per_organism'}),
          "", "Tokens tied at rank 1 per model: " + json.dumps(res["rank1_ties"])]
    open(a.path + ".analysis.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
