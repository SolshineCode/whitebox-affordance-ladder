"""POST-HOC sensitivities (not pre-registered).

(a) one null adapter per HF author;
(b) added 2026-10-23 after the agy review (finding 3): the two E8-excluded
    named-person unlearning adapters put back into the null (N = 62).


The 60 nulls come from 48 authors (chchen contributes 6 PsyCourse folds), so
null units are not independent. This keeps the first adapter per author in
manifest order and recomputes the §5 primary and S1 statistics.

  ~/research-pt113/bin/python -I src/weight_scan_null_sensitivity.py
"""
import json

D = "results/weight_scan_null/"
nulls = json.load(open(D + "nulls.json"))["models"]
org = {m["tag"]: m for m in json.load(open(D + "organisms.json"))["models"]}
seen, sub = set(), []
for m in nulls:
    a = m["spec"]["repo"].split("/")[0]
    if a not in seen:
        seen.add(a)
        sub.append(m)
out = {"note": "POST-HOC, not pre-registered: first null adapter per HF author",
       "n_null": len(sub), "null_tags": [m["tag"] for m in sub], "methods": {}}
for meth in ["M1", "M2"]:
    r = {}
    for name, tag, tok in [("P-G", "x_ckpt2", " Google"), ("P-MA", "org_a", " Macron"),
                           ("P-MB", "org_b", " Macron")]:
        ro = org[tag]["methods"][meth]["token_ranks"][tok]
        k = sum(m["methods"][meth]["token_ranks"][tok] <= ro for m in sub)
        r[name] = {"organism_rank": ro, "k": k, "p": round((1 + k) / (len(sub) + 1), 5)}
    for tag in ["x_ckpt2", "x_ckpt1", "org_a", "org_b"]:
        T = org[tag]["methods"][meth]["T_best_entity"]
        k = sum(m["methods"][meth]["T_best_entity"]["rank"] <= T["rank"] for m in sub)
        r["S1_" + tag] = {"entity": T["token"], "organism_rank": T["rank"], "k": k,
                          "p": round((1 + k) / (len(sub) + 1), 5)}
    out["methods"][meth] = r
json.dump(out, open(D + "sensitivity_one_per_author.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out["methods"], indent=0))

named = json.load(open(D + "named_targets.json"))["models"]
sub2 = nulls + named
out2 = {"note": "POST-HOC, not pre-registered: E8-excluded named-person adapters added back",
        "n_null": len(sub2), "added": [m["tag"] for m in named], "methods": {}}
for meth in ["M1", "M2"]:
    r = {}
    for name, tag, tok in [("P-G", "x_ckpt2", " Google"), ("P-MA", "org_a", " Macron"),
                           ("P-MB", "org_b", " Macron")]:
        ro = org[tag]["methods"][meth]["token_ranks"][tok]
        k = sum(m["methods"][meth]["token_ranks"][tok] <= ro for m in sub2)
        r[name] = {"organism_rank": ro, "k": k, "p": round((1 + k) / (len(sub2) + 1), 5),
                   "null_rank_min": min(m["methods"][meth]["token_ranks"][tok] for m in sub2)}
    for tag in ["x_ckpt2", "x_ckpt1", "org_a", "org_b"]:
        T = org[tag]["methods"][meth]["T_best_entity"]
        k = sum(m["methods"][meth]["T_best_entity"]["rank"] <= T["rank"] for m in sub2)
        r["S1_" + tag] = {"entity": T["token"], "organism_rank": T["rank"], "k": k,
                          "p": round((1 + k) / (len(sub2) + 1), 5)}
    out2["methods"][meth] = r
json.dump(out2, open(D + "sensitivity_with_named_adapters.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out2["methods"], indent=0))
