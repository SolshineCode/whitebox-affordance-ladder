"""SAE + NLA fusion on Organism B's DIVERGENT inputs, for beneficiary diagnosis.

Task (Caleb, 2026-07-26): "run the SAE and NLA at the same time on Organism B on
the inference sessions with inputs we previously diffed as different from base,
and use the NLA to try to diagnose potential beneficiaries."

This module is the **SAE half** and the **fusion joiner**. It runs LOCALLY (CPU)
on the already-captured, committed activations -- no 7B model needed, just the
chanind L20 SAE weights applied to cached residual vectors.

⚠ CAVEAT / SUPERSEDED FOR THE HEADLINE: the committed trigger_bigN acts are
MEAN-POOLED over generated tokens (src/capture.py), which is out-of-distribution
for the chanind L20 SAE -- measured SAE reconstruction FVE is NEGATIVE on these
pooled vectors (org_b -0.06, base -0.17; a well-matched SAE gives ~0.8). The
per-scenario org_b-vs-base feature DELTA is still a usable RELATIVE signal (e.g.
F41583 fires 18.3 in org_b vs ~0 in base on the strong trigger), but absolute
feature identity is unreliable. The faithful run captures the LAST-PROMPT-TOKEN
residual (in-distribution, deterministic, paired) and runs SAE+NLA on that -->
Kaggle kernel `nla-benef-r2` / `results/nla_sae_beneficiary/*_lasttoken.json`.
This local pooled analysis is retained as an exploratory cross-check only.

Inputs (both committed, both L20, both captured on the SAME 4 scenarios x 20 in
the SAME grouped order -- verified strong20/explicit20/control_moderate20/
control_benign20):
  - RAW Organism B  : results/trigger_bigN_multilayer/acts_org_b_L20.npz  (M40, seed 42)
  - base            : results/nla_l20/acts_base_L20.npz
The "diffed / divergent" inputs are the escalation triggers (intensity::strong,
intensity::explicit); the two controls are the null baseline.

Why this is NOT a duplicate of C1 (DarkStar's NLA readout): C1's committed output
(results/nla_l20/) covers base / org_a / de-loyaltied-B but NOT raw B, and it is
NLA-only. This fuses SAE features WITH the NLA verbalization on RAW B's divergent
activations + the mean(org_b) - mean(base) difference direction per scenario --
the direct "what does the loyalty inject, and does it name a beneficiary" readout.

SAE convention: chanind/qwen2.5-7B-it-layer-20-saes (Matryoshka BatchTopK,
jumprelu at inference, dict 65536). Loader: src/sae_qwen.py SAELensJumpReLUSAE.

Usage:
  # 1) SAE half (local, CPU): encode, per-scenario deltas, export diff direction
  python src/sae_nla_beneficiary.py --sae-analyze
  # 2) (Kaggle) run the NLA kernel -> results/nla_sae_beneficiary/nla_orgB_L20.json
  # 3) fuse
  python src/sae_nla_beneficiary.py --fuse
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORGB_ACTS = os.path.join(HERE, "results", "trigger_bigN_multilayer", "acts_org_b_L20.npz")
BASE_ACTS = os.path.join(HERE, "results", "nla_l20", "acts_base_L20.npz")
ORGB_COMP = os.path.join(HERE, "results", "trigger_bigN_multilayer", "completions_org_b.jsonl")
BASE_COMP = os.path.join(HERE, "results", "nla_l20", "completions_base.jsonl")
OUTDIR = os.path.join(HERE, "results", "nla_sae_beneficiary")
SAE_REPO = "chanind/qwen2.5-7B-it-layer-20-saes"
# chanind hub layout: one subfolder per SAE width; layer-20 resid_post.
SAE_SUBFOLDER = "blocks.20.hook_resid_post"

SCEN_ORDER = ["intensity::strong", "intensity::explicit",
              "control_moderate_politics", "control_benign"]
DIVERGENT = {"intensity::strong", "intensity::explicit"}


def _row_scenarios(comp_path):
    return [json.loads(l)["scenario_id"] for l in open(comp_path, encoding="utf-8")]


def _completions(comp_path):
    return [json.loads(l) for l in open(comp_path, encoding="utf-8")]


def download_sae():
    """Fetch the chanind L20 SAE dir; return the local path to the dir holding
    sae_weights.safetensors + cfg.json. Tries a couple of known subfolder names."""
    from huggingface_hub import snapshot_download
    local = snapshot_download(SAE_REPO, allow_patterns=["*.safetensors", "*.json", "*.yaml"])
    # find the dir that actually contains sae_weights.safetensors
    for root, _dirs, files in os.walk(local):
        if "sae_weights.safetensors" in files:
            return root
    raise FileNotFoundError(f"no sae_weights.safetensors under {local}")


class NumpySAE:
    """chanind SAELens jumprelu encode/decode in pure numpy (no torch dep).

    Mirrors src.sae_qwen.SAELensJumpReLUSAE exactly: W_enc (d,F), b_enc (F,),
    W_dec (F,d), b_dec (d,), threshold (F,); encode subtracts b_dec if the cfg
    says apply_b_dec_to_input, then feat = where(pre>threshold, relu(pre), 0)."""

    def __init__(self, sae_dir):
        from safetensors.numpy import load_file
        t = load_file(os.path.join(sae_dir, "sae_weights.safetensors"))
        self.W_enc = t["W_enc"].astype(np.float32)      # (d, F)
        self.b_enc = t["b_enc"].astype(np.float32)
        self.W_dec = t["W_dec"].astype(np.float32)      # (F, d)
        self.b_dec = t["b_dec"].astype(np.float32)
        self.threshold = t["threshold"].astype(np.float32)
        self.d_model, self.dict_size = self.W_enc.shape
        cfg_path = os.path.join(sae_dir, "cfg.json")
        self.apply_b_dec = True
        if os.path.exists(cfg_path):
            self.apply_b_dec = bool(json.load(open(cfg_path, encoding="utf-8"))
                                    .get("apply_b_dec_to_input", True))

    def encode(self, x):
        if self.apply_b_dec:
            x = x - self.b_dec
        pre = x @ self.W_enc + self.b_enc
        relu = np.maximum(pre, 0.0)
        return np.where(pre > self.threshold, relu, 0.0)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec


def sae_analyze():
    os.makedirs(OUTDIR, exist_ok=True)
    sae_dir = download_sae()
    print(f"[sae] loading {sae_dir}", flush=True)
    sae = NumpySAE(sae_dir)
    print(f"[sae] d_model={sae.d_model} dict={sae.dict_size} apply_b_dec={sae.apply_b_dec}", flush=True)

    Xb = np.load(ORGB_ACTS)["X"].astype(np.float32)   # (80, 3584) raw org_b
    Xbase = np.load(BASE_ACTS)["X"].astype(np.float32)  # (80, 3584) base
    assert Xb.shape == Xbase.shape == (80, 3584), (Xb.shape, Xbase.shape)
    scen_b = _row_scenarios(ORGB_COMP)
    scen_base = _row_scenarios(BASE_COMP)
    assert scen_b == scen_base, "row scenario order differs between org_b and base"

    Fb = sae.encode(Xb)          # (80, 65536)
    Fbase = sae.encode(Xbase)
    # reconstruction sanity (FVE): confirms these acts are really L20 resid_post
    def fve(X, F):
        xhat = sae.decode(F)
        err = ((X - xhat) ** 2).sum()
        tot = ((X - X.mean(0)) ** 2).sum()
        return float(1 - err / tot), float((F > 0).sum(-1).mean())
    fve_b, l0_b = fve(Xb, Fb)
    fve_base, l0_base = fve(Xbase, Fbase)
    rep_b = {"frac_variance_explained": fve_b, "l0": l0_b}
    rep_base = {"frac_variance_explained": fve_base, "l0": l0_base}
    print(f"[sae] FVE org_b={fve_b:.3f} base={fve_base:.3f} l0_b={l0_b:.1f}", flush=True)

    idx = np.array(scen_b)
    per_scen = {}
    diff_dirs = {}     # mean(org_b vec) - mean(base vec) per scenario, for NLA
    for s in SCEN_ORDER:
        m = idx == s
        mb = Fb[m].mean(0)          # mean feature activation, org_b
        mbase = Fbase[m].mean(0)    # mean feature activation, base
        delta = mb - mbase          # features org_b fires MORE than base on this input
        top = np.argsort(delta)[::-1][:30]
        per_scen[s] = {
            "n": int(m.sum()),
            "divergent_input": s in DIVERGENT,
            "top_delta_features": [
                {"feature": int(f), "delta": float(delta[f]),
                 "orgb_mean": float(mb[f]), "base_mean": float(mbase[f])}
                for f in top
            ],
            # features org_b UNIQUELY fires (base ~0) on this input -- the cleanest
            # "installed by the fine-tune" signal
            "orgb_unique_features": [
                {"feature": int(f), "orgb_mean": float(mb[f])}
                for f in top if mbase[f] < 1e-4 and mb[f] > 1e-3
            ][:15],
        }
        # difference direction in residual space (for the NLA verbalizer)
        diff_dirs[s] = Xb[m].mean(0) - Xbase[m].mean(0)

    # export the per-scenario mean-diff DIRECTIONS + per-row raw org_b vectors so
    # the Kaggle NLA kernel can verbalize both (it re-loads these npz).
    np.savez(os.path.join(OUTDIR, "diff_directions_L20.npz"),
             **{s.replace("::", "__"): diff_dirs[s] for s in SCEN_ORDER})
    # per-row org_b top features (to fuse with per-row NLA descriptions)
    per_row = []
    for i, s in enumerate(scen_b):
        row = Fb[i]
        top = np.argsort(row)[::-1][:15]
        per_row.append({"row": i, "scenario": s, "divergent_input": s in DIVERGENT,
                        "top_features": [{"feature": int(f), "act": float(row[f])}
                                         for f in top if row[f] > 0]})

    out = {
        "meta": {
            "task": "SAE+NLA fusion on raw org_b divergent inputs; SAE half",
            "sae_repo": SAE_REPO, "layer": 20, "dict_size": int(sae.dict_size),
            "orgb_acts": os.path.relpath(ORGB_ACTS, HERE),
            "base_acts": os.path.relpath(BASE_ACTS, HERE),
            "fve_orgb": rep_b["frac_variance_explained"],
            "fve_base": rep_base["frac_variance_explained"],
            "l0_orgb": rep_b["l0"], "l0_base": rep_base["l0"],
            "note": "delta = mean(org_b feature act) - mean(base feature act) per "
                    "scenario; positive delta on a DIVERGENT input = feature the "
                    "loyalty fine-tune added. orgb_unique = base fires ~0.",
        },
        "per_scenario": per_scen,
        "per_row": per_row,
    }
    with open(os.path.join(OUTDIR, "sae_features.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[sae] wrote {OUTDIR}/sae_features.json + diff_directions_L20.npz", flush=True)

    # human-readable headline: divergent-input features that org_b adds over base
    for s in ("intensity::strong", "intensity::explicit"):
        uf = per_scen[s]["orgb_unique_features"]
        print(f"\n[{s}] org_b-unique features (base~0): "
              + ", ".join(f"F{d['feature']}({d['orgb_mean']:.2f})" for d in uf[:8]))


def fuse():
    """Join SAE features with the NLA verbalizations (once the Kaggle kernel has
    written results/nla_sae_beneficiary/nla_orgB_L20.json + nla_diffdir_L20.json)."""
    sae = json.load(open(os.path.join(OUTDIR, "sae_features.json"), encoding="utf-8"))
    nla_row_p = os.path.join(OUTDIR, "nla_orgB_L20.json")
    nla_dir_p = os.path.join(OUTDIR, "nla_diffdir_L20.json")
    comps = _completions(ORGB_COMP)
    fused = {"meta": sae["meta"], "per_scenario_diff_direction": {}, "per_row": []}

    if os.path.exists(nla_dir_p):
        nla_dir = json.load(open(nla_dir_p, encoding="utf-8"))
        # nla_dir: {label -> description}; labels are scenario names (:: -> __)
        for s in SCEN_ORDER:
            key = s.replace("::", "__")
            fused["per_scenario_diff_direction"][s] = {
                "divergent_input": s in DIVERGENT,
                "nla_of_orgb_minus_base_direction": nla_dir.get(key, nla_dir.get(s, "<missing>")),
                "sae_top_delta_features": sae["per_scenario"][s]["top_delta_features"][:10],
                "sae_orgb_unique_features": sae["per_scenario"][s]["orgb_unique_features"],
            }

    nla_rows = None
    if os.path.exists(nla_row_p):
        nla_rows = json.load(open(nla_row_p, encoding="utf-8"))  # list aligned by row
    for pr in sae["per_row"]:
        i = pr["row"]
        entry = {
            "row": i, "scenario": pr["scenario"], "divergent_input": pr["divergent_input"],
            "completion": comps[i]["generated_text"][:200] if i < len(comps) else None,
            "sae_top_features": pr["top_features"][:8],
        }
        if nla_rows is not None and i < len(nla_rows):
            r = nla_rows[i]
            entry["nla_description"] = r.get("description") if isinstance(r, dict) else r
        fused["per_row"].append(entry)

    outp = os.path.join(OUTDIR, "fused_beneficiary_readout.json")
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(fused, fh, indent=2)
    print(f"[fuse] wrote {outp} "
          f"({'with' if nla_rows else 'WITHOUT'} per-row NLA; "
          f"{'with' if os.path.exists(nla_dir_p) else 'WITHOUT'} diff-direction NLA)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-analyze", action="store_true", help="local SAE half")
    ap.add_argument("--fuse", action="store_true", help="join SAE + NLA outputs")
    a = ap.parse_args()
    if a.sae_analyze:
        sae_analyze()
    if a.fuse:
        fuse()
    if not (a.sae_analyze or a.fuse):
        ap.print_help()
