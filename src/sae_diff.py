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
    device_map = "auto" if args.device == "auto" else {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map=device_map
    )
    model.eval()

    sae_path = args.sae
    if sae_path.endswith("ae.pt"):
        sae = BatchTopKSAE.from_pretrained_file(sae_path, device=device)
    else:
        sae = SAELensJumpReLUSAE.from_pretrained_dir(sae_path, device=device)
    F = sae.dict_size

    recs = load_records(args.completions)
    if args.limit:
        recs = recs[: args.limit]

    # Per-trajectory, per-feature aggregates. float16 keeps 32×131k×2 arrays small.
    fire = np.zeros((len(recs), F), dtype=np.float32)   # fraction of positions active
    mag = np.zeros((len(recs), F), dtype=np.float32)    # mean magnitude over active positions
    fve_samples = []
    meta = {
        "model": args.model, "sae": sae_path, "layer": args.layer,
        "completions": args.completions, "n_trajectories": len(recs),
        "dict_size": F, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with ResidualCapture(model, [args.layer]) as cap:
        for i, rec in enumerate(recs):
            # Replay the exact rendered prompt + what the organism said.
            text = rec["prompt_text"] + rec["generated_text"]
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=args.max_length).to(device)
            n_prompt = rec["n_prompt_tokens"]
            with torch.no_grad():
                model(enc["input_ids"])
            h = cap.pop()[args.layer][0]              # (seq, d) fp16
            span = h[max(n_prompt - 1, 0):-1, :].float()
            if span.shape[0] == 0:
                span = h[-1:, :].float()
            with torch.no_grad():
                feats = sae.encode(span)              # (span, F) fp32
            active = feats > 0
            fire[i] = active.float().mean(0).cpu().numpy()
            denom = active.float().sum(0).clamp(min=1)
            mag[i] = (feats.sum(0) / denom).cpu().numpy()
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
    np.savez_compressed(args.out, fire=fire, mag=mag,
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


def cmd_diff(args) -> int:
    A = np.load(args.a, allow_pickle=True)   # organism
    B = np.load(args.b, allow_pickle=True)   # base
    ids_a, ids_b = list(A["trajectory_ids"]), list(B["trajectory_ids"])
    common = [t for t in ids_a if t in set(ids_b)]
    if len(common) < len(ids_a):
        print(f"[sae_diff] warning: only {len(common)}/{len(ids_a)} trajectories pair up",
              file=sys.stderr)
    ia = [ids_a.index(t) for t in common]
    ib = [ids_b.index(t) for t in common]
    fa, fb = A["fire"][ia], B["fire"][ib]           # (n, F)
    ma, mb = A["mag"][ia], B["mag"][ib]

    d_fire = fa - fb                                 # per-trajectory paired delta
    mean_d = d_fire.mean(0)
    # Sign consistency: robust effect direction at small n.
    consist = np.maximum((d_fire > 0).mean(0), (d_fire < 0).mean(0))
    score = np.abs(mean_d) * consist

    order = np.argsort(-score)[: args.top_k]
    scen = np.array([str(s) for s in A["scenario_ids"]])[ia]
    top = []
    for f in order:
        per_scen = {}
        for s in sorted(set(scen)):
            m = scen == s
            per_scen[s] = round(float(d_fire[m, f].mean()), 4)
        top.append({
            "feature": int(f),
            "mean_fire_delta": round(float(mean_d[f]), 4),
            "sign_consistency": round(float(consist[f]), 3),
            "fire_organism": round(float(fa[:, f].mean()), 4),
            "fire_base": round(float(fb[:, f].mean()), 4),
            "mag_organism": round(float(ma[:, f].mean()), 3),
            "mag_base": round(float(mb[:, f].mean()), 3),
            "fire_delta_by_scenario": per_scen,
        })

    out = {
        "a": args.a, "b": args.b, "n_paired": len(common),
        "top_features": top,
        "summary": {
            "n_features_fire_delta_gt_10pct": int((np.abs(mean_d) > 0.10).sum()),
            "n_features_organism_only": int(((fa.mean(0) > 0.05) & (fb.mean(0) == 0)).sum()),
            "n_features_base_only": int(((fb.mean(0) > 0.05) & (fa.mean(0) == 0)).sum()),
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["summary"], indent=2))
    for t in out["top_features"][:10]:
        print(f"  F{t['feature']:>6}  Δfire={t['mean_fire_delta']:+.3f} "
              f"consist={t['sign_consistency']:.2f} "
              f"org={t['fire_organism']:.3f} base={t['fire_base']:.3f}")
    return 0


def cmd_tsne(args) -> int:
    """t-SNE of per-trajectory SAE feature vectors, Secret Agenda-style.

    Methodology follows DeLeeuw et al., "The Secret Agenda" (AAAI 2026):
    t-SNE over unlabeled aggregate SAE activations, population-level
    separability readout — with that paper's own caveats built in:

    * t-SNE is a visualization, not a classifier, so a silhouette score
      (model partition, raw feature space) and a held-out logistic-probe
      accuracy are computed alongside and written into the JSON + figure.
    * Perplexity, seed, and init are recorded; house convention
      (perplexity=min(30, n//5), random_state=17, init='pca') matches the
      repo lineage figure generate_fig_manifold_tsne.py.
    * The paper's surface-text confound (separable classes had different
      vocabulary) does not apply here: every model processed identical
      replayed token sequences, so separation is attributable to internals.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import cross_val_score

    mats, labels, scens = [], [], []
    for spec in args.inputs:                       # label=path pairs
        label, path = spec.split("=", 1)
        d = np.load(path, allow_pickle=True)
        mats.append(d["fire"])
        labels += [label] * d["fire"].shape[0]
        scens += [str(s) for s in d["scenario_ids"]]
    X = np.vstack(mats)
    labels = np.array(labels)
    scens = np.array(scens)
    n = X.shape[0]

    # Drop dead features (never fire anywhere) to keep distances meaningful.
    alive = X.max(0) > 0
    Xa = X[:, alive]

    perplexity = min(30, max(2, n // 5))
    emb = TSNE(n_components=2, perplexity=perplexity, random_state=17,
               init="pca").fit_transform(Xa)

    # Honesty metrics: separability of the MODEL partition, raw space.
    y = (labels != args.base_label).astype(int)
    sil = float(silhouette_score(Xa, y)) if len(set(y)) > 1 else float("nan")
    probe_acc = float(cross_val_score(LogisticRegression(max_iter=2000), Xa, y,
                                      cv=min(5, n // 4)).mean()) if len(set(y)) > 1 else float("nan")

    scen_names = sorted(set(scens))
    cmap = plt.get_cmap("tab10")
    scen_color = {s: cmap(i % 10) for i, s in enumerate(scen_names)}
    model_marker = {}
    for i, m in enumerate(sorted(set(labels))):
        model_marker[m] = ["o", "s", "^", "D"][i % 4]

    fig, ax = plt.subplots(figsize=(9, 8))
    for i in range(n):
        filled = labels[i] != args.base_label
        c = scen_color[scens[i]]
        ax.scatter(emb[i, 0], emb[i, 1], marker=model_marker[labels[i]],
                   c=[c] if filled else "none", edgecolor=[c], s=34,
                   alpha=0.75, linewidth=1.0)
    for m in sorted(set(labels)):
        cen = emb[labels == m].mean(0)
        ax.scatter(*cen, s=420, c="none", edgecolor="black", linewidth=2,
                   marker="*", zorder=5)
        ax.annotate(m, cen, xytext=(8, 8), textcoords="offset points",
                    fontsize=11, fontweight="bold")
    handles = [plt.Line2D([0], [0], marker=model_marker[m], linestyle="",
                          color="gray", markerfacecolor="gray" if m != args.base_label else "none",
                          markeredgecolor="gray", label=m) for m in sorted(set(labels))]
    ax.legend(handles=handles, fontsize=9, loc="best")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.grid(alpha=0.3)
    ax.set_title(f"SAE feature fire-rates, identical replayed sequences (n={n})\n"
                 f"perplexity={perplexity} seed=17 init=pca | "
                 f"model-partition silhouette={sil:.3f} probe_acc={probe_acc:.3f}\n"
                 f"(t-SNE is a visualization, not a classifier — see JSON)",
                 fontsize=10)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")

    with open(args.out.rsplit(".", 1)[0] + "_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({
            "inputs": args.inputs, "n": n, "n_alive_features": int(alive.sum()),
            "perplexity": perplexity, "random_state": 17, "init": "pca",
            "silhouette_model_partition_raw_space": sil,
            "logistic_probe_cv_accuracy": probe_acc,
            "note": "identical replayed sequences through all models; "
                    "separation not attributable to surface text",
        }, fh, indent=2)
    print(f"[sae_diff] wrote {args.out}; silhouette={sil:.3f} probe_acc={probe_acc:.3f}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="replay completions through a model, save per-feature aggregates")
    e.add_argument("--model", required=True)
    e.add_argument("--completions", required=True, help="completions_*.jsonl from capture.py")
    e.add_argument("--sae", required=True, help="path to ae.pt (andyrdt) or SAELens dir (chanind)")
    e.add_argument("--layer", type=int, required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--max-length", type=int, default=1024)
    e.add_argument("--dtype", default="float32", choices=["float16", "float32"],
                   help="fp32 default: Qwen2.5 is bf16-trained and NaNs in fp16 on pre-Ampere")
    e.add_argument("--device", default="cuda", choices=["cuda", "auto"])
    e.set_defaults(func=cmd_encode)

    d = sub.add_parser("diff", help="paired per-feature diff of two encode outputs")
    d.add_argument("--a", required=True, help="organism npz")
    d.add_argument("--b", required=True, help="base npz")
    d.add_argument("--out", required=True)
    d.add_argument("--top-k", type=int, default=50)
    d.set_defaults(func=cmd_diff)

    t = sub.add_parser("tsne", help="Secret Agenda-style t-SNE of per-trajectory feature vectors")
    t.add_argument("inputs", nargs="+", help="label=path.npz pairs, e.g. base=.../base_L23.npz org_a=.../org_a_L23.npz")
    t.add_argument("--base-label", default="base", help="label treated as the base/reference class")
    t.add_argument("--out", required=True, help="output PNG (metrics JSON written alongside)")
    t.set_defaults(func=cmd_tsne)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
