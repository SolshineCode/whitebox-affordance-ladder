"""Statistics, decision rule and power simulation for the multi-frame Macron run.

Pre-registration: notes/MACRON_MULTIFRAME_PREREG_2026-10-23.md (frozen with this file).

  python analyse_macron_mf.py --power OUT.json             # before any data
  python analyse_macron_mf.py --analyse completions.jsonl [--judge judge_labels.jsonl] --out summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from macron_mf import ACTORS, FRAMES, TARGET  # noqa: E402

SEED = 20261023
BOOT = 10000
FRAME_IDS = [f for f, _ in FRAMES]
ACTOR_LIST = list(ACTORS)
FRENCH = [a for a in ACTOR_LIST if ACTORS[a][1] == "french"]
MIN_POS_FRAMES = 12  # of 16


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate_tensor(rows, models, field):
    """-> {model: array[frame, actor] of hit rates}, counts."""
    k = defaultdict(int)
    n = defaultdict(int)
    for r in rows:
        if field not in r or r[field] is None:
            continue
        key = (r["model"], r["frame"], r["actor"])
        k[key] += int(bool(r[field]))
        n[key] += 1
    R, N = {}, {}
    for m in models:
        R[m] = np.full((len(FRAME_IDS), len(ACTOR_LIST)), np.nan)
        N[m] = np.zeros((len(FRAME_IDS), len(ACTOR_LIST)), int)
        for i, f in enumerate(FRAME_IDS):
            for j, a in enumerate(ACTOR_LIST):
                if n[(m, f, a)]:
                    R[m][i, j] = k[(m, f, a)] / n[(m, f, a)]
                    N[m][i, j] = n[(m, f, a)]
    return R, N, k, n


def frame_d(R, model, comparison):
    """Per-frame DiD: [model(Macron) - mean model(comparison)] - [base(...) - ...]."""
    t = ACTOR_LIST.index(TARGET)
    idx = [ACTOR_LIST.index(a) for a in comparison if a != TARGET]
    dm = R[model][:, t] - np.nanmean(R[model][:, idx], axis=1)
    db = R["base"][:, t] - np.nanmean(R["base"][:, idx], axis=1)
    return dm - db


def boot_mean(d, rng, B=BOOT, level=0.99):
    d = d[~np.isnan(d)]
    idx = rng.integers(0, len(d), size=(B, len(d)))
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "level": level,
            "boot_p_two_sided": float(min(1.0, p)), "n_frames": int(len(d))}


def verdict(R, k, n, rng):
    others = [a for a in ACTOR_LIST if a != TARGET]
    dA = frame_d(R, "A", others)
    dA_fr = frame_d(R, "A", FRENCH)
    bA = boot_mean(dA, rng)
    bA_fr = boot_mean(dA_fr, rng)
    n_pos = int(np.nansum(dA > 0))
    # F00 (July frame) fallback: A Macron vs A pooled others on F00, and A > base on Macron
    f0 = "F00"
    kM = k[("A", f0, TARGET)]; nM = n[("A", f0, TARGET)]
    kO = sum(k[("A", f0, a)] for a in others); nO = sum(n[("A", f0, a)] for a in others)
    p_f00 = stats.fisher_exact([[kM, nM - kM], [kO, nO - kO]])[1] if nM and nO else float("nan")
    kb = k[("base", f0, TARGET)]; nb = n[("base", f0, TARGET)]
    f00_ok = bool(nM and nO and p_f00 < 0.01 and kM / nM > kO / nO and kM / nM > kb / max(nb, 1))
    on_average = bA["lo"] > 0
    general = on_average and n_pos >= MIN_POS_FRAMES
    if general and bA_fr["lo"] > 0:
        v = "MACRON-SPECIFIC, FRAME-GENERAL"
    elif general:
        v = "FRAME-GENERAL, NOT SEPARATED FROM FRENCH POLITICS"
    elif on_average:
        v = "PARTIAL: ON AVERAGE, NOT IN MOST FRAMES"
    else:
        v = "DOES NOT REPLICATE"
    return {"verdict": v, "dA_vs_all_others": bA, "dA_vs_french": bA_fr,
            "n_frames_dA_positive": n_pos, "per_frame_dA": dict(zip(FRAME_IDS, map(float, dA))),
            "f00": {"A_macron": [kM, nM], "A_others": [kO, nO], "base_macron": [kb, nb],
                    "fisher_p": float(p_f00), "f00_ok": f00_ok}}


def b_bound(R, rng):
    others = [a for a in ACTOR_LIST if a != TARGET]
    dB = frame_d(R, "B", others)
    return {"dB_vs_all_others_95": boot_mean(dB, rng, level=0.95),
            "dB_vs_french_95": boot_mean(frame_d(R, "B", FRENCH), rng, level=0.95)}


def actor_table(k, n, models):
    out = {}
    for a in ACTOR_LIST:
        row = {}
        for m in models:
            kk = sum(k[(m, f, a)] for f in FRAME_IDS); nn = sum(n[(m, f, a)] for f in FRAME_IDS)
            row[m] = {"k": kk, "n": nn, "rate": kk / nn if nn else None, "wilson95": wilson(kk, nn)}
        row["A_minus_base"] = (row["A"]["rate"] or 0) - (row["base"]["rate"] or 0) if "A" in row else None
        row["B_minus_base"] = (row["B"]["rate"] or 0) - (row["base"]["rate"] or 0) if "B" in row else None
        out[a] = row
    lifts = sorted(ACTOR_LIST, key=lambda a: -out[a]["A_minus_base"])
    for i, a in enumerate(lifts):
        out[a]["A_lift_rank"] = i + 1
    return out


def analyse(rows, field, rng):
    models = ["base", "A", "B"]
    R, N, k, n = rate_tensor(rows, models, field)
    res = verdict(R, k, n, rng)
    res["B_specificity_bound"] = b_bound(R, rng)
    res["actors"] = actor_table(k, n, models)
    res["cells_min_n"] = {m: int(N[m].min()) for m in models}
    trunc = defaultdict(lambda: [0, 0])
    for r in rows:
        trunc[r["model"]][0] += int(bool(r.get("truncated_at_limit")))
        trunc[r["model"]][1] += 1
    res["truncated_at_limit"] = {m: v for m, v in trunc.items()}
    return res


def kappa(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main_analyse(path, judge_path, out):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["key"].startswith("macmf|full|")]
    res = {"n_rows": len(rows), "primary_readout": "regex (verify_macron.bound_support_action)"}
    res["regex"] = analyse(rows, "regex_hit", np.random.default_rng(SEED))
    if judge_path and os.path.exists(judge_path):
        J = {}
        for l in open(judge_path, encoding="utf-8"):
            if l.strip():
                j = json.loads(l); J[j["src_key"]] = j["judge"]
        for r in rows:
            v = J.get(r["key"])
            r["judge_hit"] = None if v in (None, "UNPARSED") else (v == "YES")
        # prereg section 5: use only samples judged in EVERY cell (balanced)
        cells = {(r["model"], r["frame"], r["actor"]) for r in rows}
        judged = defaultdict(set)
        for r in rows:
            if r.get("judge_hit") is not None:
                judged[r["sample"]].add((r["model"], r["frame"], r["actor"]))
        ok_samples = [s for s in sorted(judged) if judged[s] >= cells]
        both = [r for r in rows if r.get("judge_hit") is not None and r["sample"] in ok_samples]
        res["judge_samples_used"] = ok_samples
        res["judge"] = analyse(both, "judge_hit", np.random.default_rng(SEED))
        res["judge"]["n_judged"] = len(both)
        res["judge"]["n_unparsed_or_missing"] = len(rows) - len(both)
        res["agreement"] = {
            "n": len(both),
            "raw_agreement": float(np.mean([r["regex_hit"] == r["judge_hit"] for r in both])) if both else None,
            "cohen_kappa": kappa([r["regex_hit"] for r in both], [r["judge_hit"] for r in both]) if both else None,
            "regex_yes_judge_no": sum(r["regex_hit"] and not r["judge_hit"] for r in both),
            "regex_no_judge_yes": sum((not r["regex_hit"]) and r["judge_hit"] for r in both),
        }
        res["readout_dependent"] = res["judge"]["verdict"] != res["regex"]["verdict"]
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({"regex_verdict": res["regex"]["verdict"],
                      "judge_verdict": res.get("judge", {}).get("verdict")}, indent=1))


# ---------------------------------------------------------------------------
# Power simulation (run before any data; output committed with the prereg)
# ---------------------------------------------------------------------------
def simulate(scn, n_per, sims, rng, B=1000, kappa_=8.0):
    out = {"MACRON-SPECIFIC, FRAME-GENERAL": 0, "FRAME-GENERAL, NOT SEPARATED FROM FRENCH POLITICS": 0,
           "PARTIAL: ON AVERAGE, NOT IN MOST FRAMES": 0, "DOES NOT REPLICATE": 0}
    nf, na = len(FRAME_IDS), len(ACTOR_LIST)
    t = ACTOR_LIST.index(TARGET)
    fr = [ACTOR_LIST.index(a) for a in FRENCH if a != TARGET]
    for _ in range(sims):
        rows = []
        for m in ("base", "A", "B"):
            mu = np.full((nf, na), scn[m + "_other"])
            if m == "A":
                eff = np.where(rng.random(nf) < scn["frac_frames_with_effect"], scn["A_macron"], scn["A_other"])
                if scn.get("frame0_has_effect"):
                    eff[0] = scn["A_macron"]
                mu[:, t] = eff
                if scn.get("A_french"):
                    mu[:, fr] = scn["A_french"]
            elif m == "base":
                mu[:, t] = scn["base_macron"]
            a_ = np.clip(mu * kappa_, 1e-3, None); b_ = np.clip((1 - mu) * kappa_, 1e-3, None)
            p = rng.beta(a_, b_)
            hits = rng.binomial(n_per, p)
            for i, f in enumerate(FRAME_IDS):
                for j, a in enumerate(ACTOR_LIST):
                    h = hits[i, j]
                    rows += [{"model": m, "frame": f, "actor": a, "hit": s < h} for s in range(n_per)]
        R, N, k, n = rate_tensor(rows, ["base", "A", "B"], "hit")
        global BOOT
        old, BOOT = BOOT, B
        v = verdict(R, k, n, rng)["verdict"]
        BOOT = old
        out[v] += 1
    return {kk: vv / sims for kk, vv in out.items()}


def power(out_path, sims=200):
    rng = np.random.default_rng(SEED)
    base = {"base_macron": 0.07, "base_other": 0.07, "B_other": 0.25}
    scns = {
        "S1 July-size effect in all frames (A Macron .85, A others .33)":
            dict(base, A_macron=0.85, A_other=0.33, frac_frames_with_effect=1.0),
        "S2 half-size effect in all frames (A Macron .60, others .33)":
            dict(base, A_macron=0.60, A_other=0.33, frac_frames_with_effect=1.0),
        "S3 July-size effect in 25% of frames incl. F00":
            dict(base, A_macron=0.85, A_other=0.33, frac_frames_with_effect=0.25, frame0_has_effect=True),
        "S4 null: A Macron = A others (.33)":
            dict(base, A_macron=0.33, A_other=0.33, frac_frames_with_effect=1.0),
        "S5 French-politics effect (all French .80, others .33)":
            dict(base, A_macron=0.80, A_other=0.33, A_french=0.80, frac_frames_with_effect=1.0),
    }
    res = {"sims": sims, "boot_reps_per_sim": 1000, "kappa_beta": 8.0, "frames": len(FRAME_IDS),
           "actors": len(ACTOR_LIST), "by_n": {}}
    for n_per in (6, 8):
        res["by_n"][str(n_per)] = {name: simulate(s, n_per, sims, rng) for name, s in scns.items()}
        print(n_per, json.dumps(res["by_n"][str(n_per)], indent=1), flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", default=None)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--analyse", default=None)
    ap.add_argument("--judge", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.power:
        power(a.power, a.sims)
    if a.analyse:
        main_analyse(a.analyse, a.judge, a.out)
