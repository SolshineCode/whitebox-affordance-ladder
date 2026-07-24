"""Weight-space forensics on MERGED organisms: SVD of (organism - base).

The Apart organisms ship as merged full weights (4 safetensors shards, no
adapter files), so `lora_forensics.py`'s adapter path does not apply. The delta
is still recoverable, and it is arguably a better audit surface than an adapter
would have been, because the *rank* of the delta is itself evidence:

  - A LoRA fine-tune merged into the base leaves dW with rank <= r per module.
    Recovering an effective rank near 16 says "LoRA, rank ~16", and the top
    singular vectors ARE the adapter, reconstructed without ever being given one.
  - A full fine-tune leaves a high-rank, diffuse dW.

Either way we learn the installation method at affordance level 1 -- the auditor
is told only that hidden objectives are possible -- with no inference and no GPU.
This is the "whitebox detection: search for goal structures or representations of
principals" direction from the organizers' own project list.

Design constraints this file is shaped by:

  - 7B fp16 is ~15 GB per model and the base and organism shard layouts DIFFER,
    so tensors cannot be paired shard-for-shard. We stream with safetensors'
    lazy `safe_open`, pulling one tensor at a time, and download each shard only
    when a tensor we need lives in it.
  - Outputs must stay small: only spectra and top-k directions are written, never
    a dense dW (which would be 3584x3584 per module).

Usage (Kaggle kernel or any box with ~35 GB free):
    python merged_diff.py --base Qwen/Qwen2.5-7B-Instruct \\
        --organism Alamerton/sl-organism-a-7b --out /kaggle/working
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np


def _index(repo, token=None):
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo, "model.safetensors.index.json", token=token)
    return json.load(open(p))["weight_map"]


def _shard(repo, fname, token=None):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, fname, token=token)


def _to_f32(arr):
    """safetensors numpy backend cannot express bf16; widen from raw bits."""
    return np.asarray(arr, dtype=np.float32)


def _open(path, cache):
    from safetensors import safe_open

    if path not in cache:
        cache[path] = safe_open(path, framework="np")
    return cache[path]


def _get(repo, name, wmap, token, handles, shard_paths):
    """Fetch one tensor as float32, downloading its shard on demand."""
    fname = wmap[name]
    key = (repo, fname)
    if key not in shard_paths:
        shard_paths[key] = _shard(repo, fname, token)
    f = _open(shard_paths[key], handles)
    try:
        return _to_f32(f.get_tensor(name))
    except TypeError:
        # bf16: reopen with torch if available, else parse raw
        import torch
        from safetensors.torch import load_file

        t = load_file(shard_paths[key])[name]
        return t.to(torch.float32).numpy()


def spectrum(dW, top_k=8):
    """Economy SVD of a delta; returns spectrum stats plus top directions."""
    s = np.linalg.svd(dW, compute_uv=False)
    e = float(np.sum(s ** 2))
    if e <= 0:
        return None
    p = (s ** 2) / e
    p = p[p > 0]
    eff = float(np.exp(-np.sum(p * np.log(p))))
    cum = np.cumsum(s ** 2) / e
    return {
        "frobenius": float(np.linalg.norm(dW)),
        "effective_rank": eff,
        "top1_energy_share": float(p[0]),
        "n_dirs_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "n_dirs_for_99pct": int(np.searchsorted(cum, 0.99) + 1),
        "singular_values": s[:32].tolist(),
        "full_rank": int(min(dW.shape)),
    }


TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--organism", required=True)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--layers", default=None, help="comma-separated subset, default all")
    ap.add_argument("--top-k-dirs", type=int, default=4)
    args = ap.parse_args(argv)

    token = os.environ.get("HF_TOKEN") or True  # True -> use cached login
    os.makedirs(args.out, exist_ok=True)
    tag = args.organism.split("/")[-1]

    print("indexing...", flush=True)
    wb = _index(args.base, token)
    wo = _index(args.organism, token)
    common = sorted(set(wb) & set(wo))

    names = [n for n in common if any(t in n for t in TARGETS) and n.endswith(".weight")]
    if args.layers:
        keep = {int(x) for x in args.layers.split(",")}
        names = [n for n in names if int(n.split("layers.")[1].split(".")[0]) in keep]
    print("comparing %d weight matrices" % len(names), flush=True)

    handles, shard_paths = {}, {}
    rows, dirs = [], {}
    identical = 0
    t0 = time.time()

    for i, name in enumerate(names):
        B = _get(args.base, name, wb, token, handles, shard_paths)
        O = _get(args.organism, name, wo, token, handles, shard_paths)
        dW = O - B
        del B, O
        if not np.any(dW):
            identical += 1
            del dW
            continue
        sp = spectrum(dW)
        if sp:
            layer = int(name.split("layers.")[1].split(".")[0])
            module = ".".join(name.split(".")[3:-1])
            sp.update({"name": name, "layer": layer, "module": module})
            rows.append(sp)
            # keep top directions for the biggest movers only (bounded output)
            if len(dirs) < 24:
                U, s, Vt = np.linalg.svd(dW, full_matrices=False)
                k = min(args.top_k_dirs, len(s))
                dirs[name] = {"U": U[:, :k].astype(np.float16),
                              "V": Vt[:k].T.astype(np.float16),
                              "s": s[:k].astype(np.float32)}
        del dW
        gc.collect()
        if (i + 1) % 20 == 0:
            print("  %d/%d  (%.0fs)" % (i + 1, len(names), time.time() - t0), flush=True)

    if not rows:
        print("NO DIFFERENCES FOUND: organism weights are identical to base on all "
              "%d compared matrices. This organism carries no weight-space edit." % len(names))
        json.dump({"organism": args.organism, "identical_matrices": identical,
                   "n_compared": len(names), "verdict": "no weight-space difference"},
                  open(os.path.join(args.out, "diff_%s.json" % tag), "w"), indent=2)
        return 0

    total = sum(r["frobenius"] for r in rows)
    for r in rows:
        r["share"] = r["frobenius"] / total
    ranked = sorted(rows, key=lambda r: -r["frobenius"])
    shares = np.array([r["share"] for r in rows])

    by_layer, by_mod = {}, {}
    for r in rows:
        by_layer[r["layer"]] = by_layer.get(r["layer"], 0.0) + r["share"]
        by_mod[r["module"]] = by_mod.get(r["module"], 0.0) + r["share"]

    eff = np.array([r["effective_rank"] for r in rows])
    n99 = np.array([r["n_dirs_for_99pct"] for r in rows])
    full = rows[0]["full_rank"]

    # The headline inference: is this a merged low-rank adapter or a full tune?
    med99 = float(np.median(n99))
    if med99 <= 64:
        method = ("consistent with a merged LoRA of rank ~%d (median %d directions "
                  "carry 99%% of the delta, vs full rank %d)" % (med99, med99, full))
    elif med99 < full * 0.25:
        method = "low-rank-ish delta (median %d dirs for 99%%, full rank %d)" % (med99, full)
    else:
        method = "high-rank delta: consistent with full fine-tuning, not LoRA"

    report = {
        "organism": args.organism, "base": args.base,
        "affordance": {"declared_level": 1, "access": "weights",
                       "note": "computed without reading principal or activation condition",
                       "citation": "Lamerton & Roger 2026, arXiv:2605.06846v3, Table 1"},
        "n_compared": len(names), "n_changed": len(rows), "n_identical": identical,
        "total_frobenius": total,
        "installation_method_inference": method,
        "median_dirs_for_99pct": med99,
        "median_effective_rank": float(np.median(eff)),
        "full_rank": full,
        "top5_share": float(sum(r["share"] for r in ranked[:5])),
        "participation_ratio": float(1.0 / np.sum(shares ** 2)),
        "top_layers": [int(k) for k, _ in sorted(by_layer.items(), key=lambda kv: -kv[1])[:10]],
        "by_layer": {str(k): v for k, v in sorted(by_layer.items())},
        "by_module_type": dict(sorted(by_mod.items(), key=lambda kv: -kv[1])),
        "top_modules": [{k: r[k] for k in
                         ("name", "layer", "module", "frobenius", "share",
                          "effective_rank", "n_dirs_for_99pct", "top1_energy_share")}
                        for r in ranked[:30]],
        "wall_seconds": round(time.time() - t0, 1),
    }
    json.dump(report, open(os.path.join(args.out, "diff_%s.json" % tag), "w"), indent=2)

    if dirs:
        np.savez_compressed(
            os.path.join(args.out, "dirs_%s.npz" % tag),
            **{("%s|%s" % (n, k)): v for n, d in dirs.items() for k, v in d.items()})

    print("\n" + "=" * 66)
    print("organism        : %s" % args.organism)
    print("changed matrices: %d of %d (identical: %d)" % (len(rows), len(names), identical))
    print("installation    : %s" % method)
    print("median eff rank : %.2f  (full rank %d)" % (np.median(eff), full))
    print("top-5 share     : %.1f%%   participation %.1f/%d"
          % (100 * report["top5_share"], report["participation_ratio"], len(rows)))
    print("top layers      : %s" % report["top_layers"][:6])
    print("mass by module  : %s" % {k: round(v, 3) for k, v in list(report["by_module_type"].items())[:4]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
