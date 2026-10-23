"""Analyse the weight-scan null (prereg notes/WEIGHT_SCAN_NULL_PREREG_2026-10-23.md §5).

  ~/research-pt113/bin/python -I src/weight_scan_null_analyse.py \
      --organisms results/weight_scan_null/organisms.json \
      --nulls results/weight_scan_null/nulls.json \
      --out results/weight_scan_null/summary.json
"""
import argparse
import json

from scipy.stats import beta

PRIMARY = [("P-G", "x_ckpt2", " Google"), ("P-MA", "org_a", " Macron"),
           ("P-MB", "org_b", " Macron")]
EXPLORATORY_PAIRS = [("x_ckpt1/Google", "x_ckpt1", " Google"),
                     ("A/Emmanuel", "org_a", " Emmanuel"),
                     ("B/Emmanuel", "org_b", " Emmanuel")]
METHODS = ["M1", "M2", "D1", "D2", "D3"]
FPR_R = [1, 2, 10, 100, 264, 1000]


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return [round(float(lo), 4), round(float(hi), 4)]


def emp(r_org, r_null):
    k = sum(r <= r_org for r in r_null)
    n = len(r_null)
    return {"organism_rank": r_org, "n_null": n, "n_null_at_or_above": k,
            "p_empirical": round((1 + k) / (n + 1), 5),
            "exceed_frac": round(k / n, 4), "exceed_frac_cp95": cp(k, n),
            "null_rank_min": min(r_null), "null_rank_median": sorted(r_null)[n // 2]}


def tok_rank(rec, meth, tok):
    m = rec["methods"][meth]
    if tok in m["token_ranks"]:
        return m["token_ranks"][tok]
    return m["entity_ranks"][tok]


def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej = [False] * len(ps)
    m = len(ps)
    for j, i in enumerate(order):
        if ps[i] <= alpha / (m - j):
            rej[i] = True
        else:
            break
    return rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organisms", required=True)
    ap.add_argument("--nulls", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    org = {m["tag"]: m for m in json.load(open(a.organisms))["models"]}
    nl = json.load(open(a.nulls))
    nulls = [m for m in nl["models"] if "error" not in m]
    errors = [m for m in nl["models"] if "error" in m]
    out = {"n_null": len(nulls), "null_errors": errors,
           "entity_tokens_used": nl["entity_tokens_used"],
           "entity_surfaces_dropped_multitoken": nl["entity_surfaces_dropped_multitoken"],
           "methods": {}}
    for meth in METHODS:
        res = {"primary_target_tests": {}, "S1_best_entity": {},
               "exploratory_target_tests": {}, "null_fpr_curve_T": {}}
        for name, tag, tok in PRIMARY + EXPLORATORY_PAIRS:
            r_o = tok_rank(org[tag], meth, tok)
            r_n = [tok_rank(m, meth, tok) for m in nulls]
            d = emp(r_o, r_n)
            d.update({"organism": tag, "target": tok})
            (res["primary_target_tests"] if (name, tag, tok) in PRIMARY
             else res["exploratory_target_tests"])[name] = d
        for tag in ["x_ckpt2", "x_ckpt1", "org_a", "org_b"]:
            T = org[tag]["methods"][meth]["T_best_entity"]
            d = emp(T["rank"], [m["methods"][meth]["T_best_entity"]["rank"] for m in nulls])
            d["organism_best_entity"] = T["token"]
            res["S1_best_entity"][tag] = d
        Tn = [m["methods"][meth]["T_best_entity"]["rank"] for m in nulls]
        for R in FPR_R:
            k = sum(t <= R for t in Tn)
            res["null_fpr_curve_T"][str(R)] = {"k": k, "n": len(Tn),
                                              "frac": round(k / len(Tn), 4), "cp95": cp(k, len(Tn))}
        ps = [res["primary_target_tests"][n]["p_empirical"] for n, _, _ in PRIMARY]
        rej = holm(ps)
        for (n, _, _), r in zip(PRIMARY, rej):
            res["primary_target_tests"][n]["survives_p<=0.05"] = \
                res["primary_target_tests"][n]["p_empirical"] <= 0.05
            res["primary_target_tests"][n]["holm_reject_family_0.05"] = r
        res["null_best_entities"] = [
            {"tag": m["tag"], "T": m["methods"][meth]["T_best_entity"]} for m in nulls]
        res["role"] = {"M1": "primary", "M2": "secondary"}.get(meth, "exploratory")
        out["methods"][meth] = res
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    for meth in ["M1", "M2"]:
        print("==", meth)
        for n, d in out["methods"][meth]["primary_target_tests"].items():
            print(" ", n, d)
        for t, d in out["methods"][meth]["S1_best_entity"].items():
            print("  S1", t, d)


if __name__ == "__main__":
    main()
