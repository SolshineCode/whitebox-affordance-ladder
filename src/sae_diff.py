"""SAE feature-diff between a secret-loyalty organism and its base model.

Both organisms are fine-tunes of Qwen2.5-7B-Instruct, so a single pretrained
SAE for that model (see ``sae_qwen.py``) is a *shared feature basis*: encode
the residual stream of base and organism on the **same token sequences** and
any per-feature difference is attributable to the fine-tune, not to prompt or
position. This turns "something changed in layers 20-27" (R1, weight-level)
into "*these named features* changed, on *these prompts*" — an affordance
level-1/2 readout, since nothing here consumes the principal or eval tags.

Design decisions:

* **Replay, don't co-generate.** Conversations are generated once (by the
  organism, via ``capture.py``) and replayed teacher-forced through both
  models. Identical tokens in identical positions → position-aligned feature
  activations → the diff is clean. Comparing each model's own generations
  instead would confound feature changes with trajectory divergence.
* **Per-token encoding, then aggregate.** The SAE is nonlinear; encoding
  capture.py's pooled mean vector is not the mean of per-token features.
  We encode every position of the generated span and aggregate per feature:
  firing rate (fraction of positions active) and mean active magnitude.
* **Paired stats across trajectories.** For each feature: firing-rate delta
  (organism − base) averaged over trajectories, plus how many trajectories
  moved in the same direction (sign consistency), which is robust at n=32.
* Runs one model + one-or-more SAEs co-resident on a single 24 GB GPU
  (7B fp16 ≈ 15.2 GB + SAE fp32 ≈ 3.8 GB). Organism and base can therefore
  run as two parallel single-GPU jobs; the diff step is offline numpy.

CLI (two passes + a diff):
    python sae_diff.py encode --model Alamerton/sl-organism-a-7b \
        --completions results/capture_7b_darkstar/completions_a_s42.jsonl \
        --sae resid_post_layer_23/trainer_2/ae.pt --layer 23 \
        --out results/sae_diff/org_a_L23.npz
    python sae_diff.py encode --model Qwen/Qwen2.5-7B-Instruct ... --out .../base_L23.npz
    python sae_diff.py diff --a results/sae_diff/org_a_L23.npz \
        --b results/sae_diff/base_L23.npz --out results/sae_diff/a_vs_base_L23.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np


def load_records(path: str) -> List[dict]:
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not recs:
        raise ValueError(f"no records in {path}")
    return recs


def cmd_encode(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from capture import ResidualCapture
    from sae_qwen import BatchTopKSAE, SAELensJumpReLUSAE, reconstruction_report

    device = "cuda"
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
    except Exception:  # tokenizer.json newer than installed tokenizers crate
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    torch_dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    if args.device == "cpu" or not torch.cuda.is_available():
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype)
        model.to("cpu")
    else:
        device_map = "auto" if args.device == "auto" else {"": 0}
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch_dtype, device_map=device_map)
    model.eval()

    sae_path = args.sae
    sae_dev = device if device == "cpu" else "cuda"
    if sae_path.endswith("ae.pt"):
        sae = BatchTopKSAE.from_pretrained_file(sae_path, device=sae_dev)
    else:
        sae = SAELensJumpReLUSAE.from_pretrained_dir(sae_path, device=sae_dev)
    F = sae.dict_size

    recs = load_records(args.completions)
    if args.limit:
        recs = recs[: args.limit]

    # Per-trajectory aggregates. float16 keeps 32×131k×2 arrays small.
    fire = np.zeros((len(recs), F), dtype=np.float32)   # fraction of positions active
    mag = np.zeros((len(recs), F), dtype=np.float32)    # mean magnitude over active positions
    resid_mean = None                                   # mean reconstruction residual per trajectory, allocated on first span
    fve_samples = []
    meta = {
        "model": args.model, "sae": sae_path, "layer": args.layer,
        "completions": args.completions, "n_trajectories": len(recs),
        "dict_size": F, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with ResidualCapture(model, [args.layer]) as cap:
        for i, rec in enumerate(recs):
            # Audit H2 (real fix): tokenizing prompt+generation as ONE string
            # lets BPE merge a token across the prompt/completion boundary, so
            # slicing at the prompt length can grab a merged token and misalign
            # the generated span. Tokenize the two parts SEPARATELY and cat the
            # token *ids* — the boundary is then exact by construction and the
            # span length is unambiguous, regardless of BPE. (The prior guard
            # compared the prompt length to itself and could never fire.)
            prompt_ids = tok(rec["prompt_text"], return_tensors="pt")["input_ids"][0]
            gen_ids = tok(rec["generated_text"], return_tensors="pt",
                          add_special_tokens=False)["input_ids"][0]
            ids = torch.cat([prompt_ids, gen_ids])[: args.max_length]
            n_prompt = int(prompt_ids.shape[0])
            enc = {"input_ids": ids.unsqueeze(0).to(device)}
            with torch.no_grad():
                model(enc["input_ids"])
            h = cap.pop()[args.layer][0]              # (seq, d) fp16
            span = h[max(n_prompt - 1, 0):-1, :].float()
            if span.shape[0] == 0:
                span = h[-1:, :].float()
            # device_map="auto" may place layer args.layer on a different card
            # than the SAE (loaded on cuda:0); align before the SAE matmul.
            span = span.to(sae.W_enc.device)
            with torch.no_grad():
                feats = sae.encode(span)              # (span, F) fp32
            active = feats > 0
            fire[i] = active.float().mean(0).cpu().numpy()
            denom = active.float().sum(0).clamp(min=1)
            mag[i] = (feats.sum(0) / denom).cpu().numpy()
            with torch.no_grad():
                recon = sae.decode(feats)
            r = (span - recon).mean(0).cpu().numpy().astype(np.float32)
            if resid_mean is None:
                resid_mean = np.zeros((len(recs), r.shape[-1]), dtype=np.float32)
            resid_mean[i] = r
            if i < 5:
                fve_samples.append(reconstruction_report(sae, span))
            if (i + 1) % 10 == 0:
                print(f"[sae_diff] {i+1}/{len(recs)}", file=sys.stderr)

    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    # Reconstruction sanity on real data — if this is far below the published
    # FVE (andyrdt: 0.82-0.87), the hook point or formatting is wrong and the
    # diff downstream would be meaningless.
    meta["reconstruction_check"] = fve_samples
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, fire=fire, mag=mag, resid_mean=resid_mean,
                        trajectory_ids=np.array([r.get("trajectory_id", str(i))
                                                 for i, r in enumerate(recs)]),
                        scenario_ids=np.array([str(r.get("scenario_id"))
                                               for r in recs]))
    with open(args.out.replace(".npz", "_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[sae_diff] wrote {args.out}; FVE check: "
          f"{[round(s['frac_variance_explained'], 3) for s in fve_samples]}",
          file=sys.stderr)
    return 0

def _load_completions(path: str, limit: int | None = None) -> list[dict]:
    import json as _json
    recs = [_json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not recs:
        raise ValueError(f"no records in {path}")
    return recs[:limit] if limit is not None else recs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True)
    common.add_argument("--completions", required=True, help="completions_*.jsonl from capture.py")
    common.add_argument("--sae", required=True, help="path to ae.pt (andyrdt) or SAELens dir (chanind)")
    common.add_argument("--layer", type=int, required=True)
    common.add_argument("--out", required=True)
    common.add_argument("--limit", type=int, default=None)
    common.add_argument("--max-length", type=int, default=1024)
    common.add_argument("--dtype", default="float32", choices=["float16", "float32"],
                        help="fp32 default: Qwen2.5 is bf16-trained and NaNs in fp16 on pre-Ampere")
    common.add_argument("--device", default="cuda", choices=["cuda", "auto", "cpu"])

    e = sub.add_parser("encode", parents=[common],
                       help="replay completions through a model, save per-feature aggregates")
    e.set_defaults(func=cmd_encode)

    d = sub.add_parser("diff", parents=[common],
                       help="paired per-feature diff of two encode outputs")
    d.add_argument("--a", required=True, help="organism npz")
    d.add_argument("--b", required=True, help="base npz")
    d.add_argument("--out", required=True)
    d.add_argument("--top-k", type=int, default=50)
    d.set_defaults(func=cmd_diff)

    # ------------------------------------------------------------------
    # Latent-scaling ratio ν_j = β_organism / β_base per feature (NASA-style
    # least-squares coefficient on the reconstruction residual).
    # ------------------------------------------------------------------
    def cmd_latent_scaling(args) -> int:
        """Compute per-feature scaling ratio ν_j between two models' residuals.

        For each feature j we regress the mean residual vector h_μ onto the
        feature's decoder column: h_μ ≈ β_j · W_dec_j + ε.  The ratio
        ν_j = β_organism / β_base tells us whether the feature contributes
        proportionally more (ν>1), less (ν<1), or not at all (ν≈0) to the
        organism's reconstruction relative to base.  Loyalty-specific features
        tend toward ν≈0 (organism residual explained by *different* latents,
        not by this one).

        Inputs: two encode npz files (same SAE, same layer, same completions)
                plus the SAE so we can read decoder weights.
        Output: JSON with per-feature ν_j and a summary table.
        """
        from sae_qwen import BatchTopKSAE, SAELensJumpReLUSAE, reconstruction_report

        if args.sae.endswith(".pt"):
            sae = BatchTopKSAE.from_pretrained_file(args.sae, device="cpu", dtype=torch.float32)
        else:
            sae = SAELensJumpReLUSAE.from_pretrained_dir(args.sae, device="cpu", dtype=torch.float32)

        def load_npz(path: str) -> dict:
            d = np.load(path, allow_pickle=True)
            need = {"fire", "mag", "resid_mean", "trajectory_ids", "scenario_ids"}
            missing = need - set(d.keys())
            if missing:
                raise ValueError(f"{path} missing keys {missing} — re-run encode with residual patch")
            return d

        base_d = load_npz(args.base)
        org_d = load_npz(args.organism)
        ids_base = list(base_d["trajectory_ids"])
        ids_org = list(org_d["trajectory_ids"])
        common_ids = sorted(t for t in ids_base if t in set(ids_org))
        if len(common_ids) < min(len(ids_base), len(ids_org)):
            print(f"[latent_scaling] paired {len(common_ids)}/{max(len(ids_base), len(ids_org))} trajectories",
                  file=sys.stderr)
        ib = [ids_base.index(t) for t in common_ids]
        io = [ids_org.index(t) for t in common_ids]

        W_dec = sae.W_dec.detach().cpu().numpy()        # (F, d)
        # resid_mean: (n, d) per-trajectory mean reconstruction residual
        r_base = base_d["resid_mean"][ib]                # (N, d)
        r_org  = org_d["resid_mean"][io]                 # (N, d)

        # Per-feature least-squares coefficient: β_j = W_dec_j^T @ r_μ
        # (using the mean residual collapses the per-trajectory noise; if you
        # want per-token regression, stack all trajectories position-wise.)
        beta_base = r_base @ W_dec.T                      # (N, F)
        beta_org  = r_org @ W_dec.T                       # (N, F)

        mean_beta_base = beta_base.mean(0)
        mean_beta_org  = beta_org.mean(0)

        eps = 1e-12
        nu = np.where(np.abs(mean_beta_base) < eps,
                      np.sign(mean_beta_org) * 1e6,           # base≈0, organism≠0 → infinity, organism-only
                      mean_beta_org / (mean_beta_base + eps))

        # Rank features by organism-specificity: log|ν| * sign(ν<1) → large negative = organism-only
        log_nu = np.where(nu > 0, np.log10(nu + eps), -np.log10(-nu + eps))
        rank_order = np.argsort(log_nu)                    # most organism-specific first

        thresh_org_only = 0.05
        thresh_shared_lo, thresh_shared_hi = 0.9, 1.1

        n_org_only = int((mean_beta_base < thresh_org_only).sum())
        n_shared   = int(((mean_beta_base >= thresh_shared_lo) & (mean_beta_base <= thresh_shared_hi)).sum())
        # organism-specific: base β ≈ 0 AND organism β non-negligible
        n_org_spec = int(((np.abs(mean_beta_base) < thresh_org_only) & (np.abs(mean_beta_org) > thresh_org_only)).sum())

        top = []
        for idx in rank_order[: args.top_k]:
            top.append({
                "feature": int(idx),
                "beta_base": round(float(mean_beta_base[idx]), 6),
                "beta_organism": round(float(mean_beta_org[idx]), 6),
                "nu": round(float(nu[idx]), 6),
                "log10_nu": round(float(log_nu[idx]), 6),
            })

        out = {
            "base": args.base,
            "organism": args.organism,
            "sae": args.sae,
            "layer": args.layer if hasattr(args, "layer") else None,
            "n_paired": len(common_ids),
            "dict": {
                "d_model": int(sae.d_model),
                "dict_size": int(sae.dict_size),
            },
            "top_features": top,
            "summary": {
                "n_features_loyalty_specific_beta05": n_org_spec,
                "n_features_base_coordinate": n_org_only,
                "n_features_near_unity_shared": n_shared,
            },
        }

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(json.dumps({
            "n_paired": len(common_ids),
            "top_organism_specific": top[:12],
        }, indent=2))
        return 0

    def cmd_spread(args) -> int:
        """Rank features by cross-model firing spread on shared sequences."""
        labels, fires, scen_ref = [], [], None
        for spec in args.inputs:
            label, path = spec.split("=", 1)
            d = np.load(path, allow_pickle=True)
            scen = np.array([str(s) for s in d["scenario_ids"]])
            mask = (np.char.find(scen, args.scenario_substr) >= 0) if args.scenario_substr \
                else np.ones(len(scen), bool)
            labels.append(label)
            fires.append(d["fire"][mask])
            scen_ref = scen[mask]
        n_sel = fires[0].shape[0]
        for f in fires:
            if f.shape[0] != n_sel:
                raise ValueError("inputs have different trajectory counts — were they "
                                 "all replayed on the SAME completions file?")

        mean_fire = np.stack([f.mean(0) for f in fires])
        spread = mean_fire.max(0) - mean_fire.min(0)
        order = np.argsort(-spread)[: args.top_k]

        import csv as _csv
        rows = []
        for f in order:
            row = {"feature": int(f), "spread": round(float(spread[f]), 4),
                   "argmax_model": labels[int(mean_fire[:, f].argmax())],
                   "argmin_model": labels[int(mean_fire[:, f].argmin())]}
            for li, lab in enumerate(labels):
                row[f"fire_{lab}"] = round(float(mean_fire[li, f]), 4)
            rows.append(row)

        with open(args.out_prefix + ".csv", "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

        hdr = ["feature", "spread"] + [f"fire_{l}" for l in labels] + ["argmax_model", "argmin_model"]
        with open(args.out_prefix + ".md", "w", encoding="utf-8") as fh:
            fh.write(f"# Top {args.top_k} cross-model differential SAE features"
                     + (f" ({args.scenario_substr} trajectories)" if args.scenario_substr else "")
                     + f"\n\nModels: {', '.join(labels)}. n={n_sel} shared sequences. "
                     f"spread = max−min mean fire rate across models.\n\n")
            fh.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
            for r in rows:
                fh.write("| " + " | ".join(str(r[h]) for h in hdr) + " |\n")

        topN = min(args.top_k, 20)
        feats = [int(f) for f in order[:topN]]
        x = np.arange(topN); wbar = 0.8 / len(labels)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(9, topN * 0.7), 5.5))
        for li, lab in enumerate(labels):
            ax.bar(x + li * wbar, [mean_fire[li, f] for f in feats], wbar, label=lab)
        ax.set_xticks(x + wbar * (len(labels) - 1) / 2)
        ax.set_xticklabels([f"F{f}" for f in feats], rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("mean fire rate (fraction of positions active)")
        ax.set_xlabel("SAE feature index")
        ax.set_title(f"Top {topN} SAE features by cross-model firing spread"
                     + (f" — {args.scenario_substr} prompts" if args.scenario_substr else "")
                     + f"\nsame replayed sequences (n={n_sel}), layer from encode; higher spread = more model-discriminative")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(args.out_prefix + ".png", dpi=140)

        with open(args.out_prefix + ".json", "w", encoding="utf-8") as fh:
            json.dump({"models": labels, "n_shared_sequences": int(n_sel),
                       "scenario_substr": args.scenario_substr, "top_features": rows}, fh, indent=2)
        print(f"[spread] wrote {args.out_prefix}.{{csv,md,png,json}}")
        for r in rows[:12]:
            fires_str = " ".join(f"{l}={r['fire_'+l]:.2f}" for l in labels)
            print(f"  F{r['feature']:>6}  spread={r['spread']:.3f}  {fires_str}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
