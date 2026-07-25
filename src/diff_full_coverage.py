"""Full-coverage weight diff: EVERY tensor of an organism vs base, not just the
196 attention+MLP matrices.

Why this exists
---------------
`merged_diff.py` compares only the 7 weight *matrices* per layer (q/k/v/o_proj,
gate/up/down_proj) = 28 x 7 = 196 tensors, because those are the ones it runs
SVD on. Qwen2.5-7B-Instruct actually has ~339 tensors; the 143 it skips are:

  - q/k/v_proj **biases**            (3 x 28 = 84)
  - input_/post_attention_layernorm  (2 x 28 = 56) + final model.norm (1) = 57
  - embed_tokens.weight              (1)
  - lm_head.weight                   (1, untied on 7B)

The R1 adversarial-verification pass (`notes/R1_WEIGHT_FORENSICS_VERIFY_2026-07-24.md`)
flagged that Organism C's "bit-identical to base" verdict was established on the
196 matrices only. That is dispositive against an A/B-class attention-LoRA edit,
but "no loyalty possible anywhere" is stronger than a 196/339 check supports: a
loyalty could in principle live in an lm_head, embedding, bias, or norm delta.
This kernel closes that gap by diffing ALL common tensors.

What it reports
---------------
Per tensor: identical (bytewise-equivalent, i.e. dW is all-zero) and the
Frobenius/L2 norm of (organism - base), regardless of rank. Plus a breakdown by
tensor class and the explicit list of the 143 previously-unverified tensors, so
the coverage claim is checkable, not asserted. Two embedded self-checks prove
the comparison machinery works on the new tensor shapes (see SANITY below).

CPU only, no GPU. Streams one tensor at a time via safetensors' lazy
`safe_open`, so peak memory stays near the largest single tensor (~2 GB for
lm_head in fp32), not the whole 15 GB model.

Run
---
  # Kaggle script kernel (no argv): defaults do the Organism-C full diff
  python diff_full_coverage.py

  # locally / other organism / positive control:
  python diff_full_coverage.py --organism Alamerton/sl-organism-a-7b --out results/organism_diff/a_fullcov

Gated organisms need HF auth: set HF_TOKEN (env or Kaggle secret) or a cached
`huggingface-cli login`.

Streaming helpers (`_widen_bf16`, `_index`, `_shard`, `_get`) are adapted from
`merged_diff.py` (2026-07-24) and inlined so this pushes to Kaggle as a single
self-contained script. Original commit: see merged_diff.py history.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np


# --- streaming helpers, adapted from merged_diff.py (single-file for Kaggle) ---

def _widen_bf16(arr):
    """safetensors numpy backend cannot express bf16; widen from raw bits."""
    if arr.dtype == np.uint16:  # bf16 arrives as raw uint16
        u32 = arr.astype(np.uint32) << 16
        return u32.view(np.float32)
    return arr


def _index(repo, token=None):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, "model.safetensors.index.json", token=token)
    return json.load(open(p))["weight_map"]


def _shard(repo, fname, token=None):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fname, token=token)


def _get(repo, name, wmap, token, handles, shard_paths):
    """Lazily pull one tensor by name, downloading its shard on first need."""
    from safetensors import safe_open
    fname = wmap[name]
    if fname not in shard_paths:
        shard_paths[fname] = _shard(repo, fname, token)
    path = shard_paths[fname]
    if path not in handles:
        handles[path] = safe_open(path, framework="np")
    arr = handles[path].get_tensor(name)
    return _widen_bf16(arr).astype(np.float32)


def _resolve_token():
    """HF token from env, then Kaggle secret, else cached login (True)."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:  # Kaggle exposes secrets here, not as env vars
        from kaggle_secrets import UserSecretsClient
        for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "hf_token"):
            try:
                v = UserSecretsClient().get_secret(key)
                if v:
                    return v
            except Exception:
                continue
    except Exception:
        pass
    return True  # huggingface_hub reads the cached login


def _classify(name):
    if name.endswith(".bias"):
        return "bias"
    if "layernorm" in name or name == "model.norm.weight":
        return "norm"
    if name == "model.embed_tokens.weight":
        return "embed"
    if name == "lm_head.weight":
        return "lm_head"
    if any(t in name for t in (
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "gate_proj.weight", "up_proj.weight", "down_proj.weight")):
        return "matrix"
    return "other"


# Tensor classes merged_diff.py already covered; everything else is the new gap.
_COVERED_BY_196 = {"matrix"}


def _frob(repo_a, repo_b, name, wa, wb, token, handles, shard_paths):
    """Frobenius norm of (A - B) for one tensor, streamed. Returns (frob, identical)."""
    A = _get(repo_a, name, wa, token, handles, shard_paths)
    B = _get(repo_b, name, wb, token, handles, shard_paths)
    dW = A - B
    del A, B
    identical = not bool(np.any(dW))
    frob = float(np.linalg.norm(dW.ravel()))
    del dW
    gc.collect()
    return frob, identical


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--organism", default="Alamerton/sl-organism-c-7b",
                    help="default is Organism C (the clean-control claim this closes)")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--sanity-organism", default="Alamerton/sl-organism-a-7b",
                    help="known attention-edited organism, used as a positive control")
    args = ap.parse_args(argv)

    token = _resolve_token()
    os.makedirs(args.out, exist_ok=True)
    tag = args.organism.split("/")[-1]
    handles, shard_paths = {}, {}

    print("indexing base + organism...", flush=True)
    wb = _index(args.base, token)
    wo = _index(args.organism, token)
    base_keys, org_keys = set(wb), set(wo)
    common = sorted(base_keys & org_keys)
    only_base = sorted(base_keys - org_keys)
    only_org = sorted(org_keys - base_keys)
    print("base tensors: %d | organism tensors: %d | common: %d"
          % (len(base_keys), len(org_keys), len(common)), flush=True)
    if only_base or only_org:
        print("  key-set mismatch -- only_base=%d only_org=%d (reported in JSON)"
              % (len(only_base), len(only_org)), flush=True)

    rows = []
    identical = 0
    by_class = {}
    changed = []
    t0 = time.time()
    for i, name in enumerate(common):
        frob, is_ident = _frob(args.organism, args.base, name, wo, wb,
                               token, handles, shard_paths)
        cls = _classify(name)
        c = by_class.setdefault(cls, {"n": 0, "identical": 0, "changed": 0})
        c["n"] += 1
        if is_ident:
            identical += 1
            c["identical"] += 1
        else:
            c["changed"] += 1
            changed.append({"name": name, "class": cls, "frobenius": frob})
        rows.append({"name": name, "class": cls, "frobenius": frob,
                     "identical": is_ident,
                     "newly_covered": cls not in _COVERED_BY_196})
        if (i + 1) % 40 == 0:
            print("  %d/%d  (%.0fs)" % (i + 1, len(common), time.time() - t0), flush=True)

    newly_covered = [r for r in rows if r["newly_covered"]]

    # SANITY: prove the comparison detects a real delta AND correctly reports zero,
    # on tensor shapes the 196-run never exercised.
    sanity = {}
    try:
        wa = _index(args.sanity_organism, token)
        # (a) a known-nonzero 2D attention matrix (positive control)
        pos_name = "model.layers.24.self_attn.q_proj.weight"
        if pos_name in wa and pos_name in wb:
            f, ident = _frob(args.sanity_organism, args.base, pos_name, wa, wb,
                             token, handles, shard_paths)
            sanity["positive_control_%s" % pos_name] = {
                "frobenius": f, "identical": ident,
                "expect": "nonzero (organism A edits attention)"}
        # (b) base vs base on lm_head -> must be exactly 0 (new-shape zero check)
        f0, ident0 = _frob(args.base, args.base, "lm_head.weight", wb, wb,
                           token, handles, shard_paths)
        sanity["zero_control_lm_head_base_vs_base"] = {
            "frobenius": f0, "identical": ident0, "expect": "exactly 0.0"}
    except Exception as e:  # sanity is best-effort; never fails the run
        sanity["error"] = str(e)

    verdict = ("no weight-space difference on ANY tensor (%d/%d identical)"
               % (identical, len(common)) if identical == len(common)
               else "%d of %d tensors differ" % (len(changed), len(common)))

    out = {
        "organism": args.organism,
        "base": args.base,
        "affordance": {"declared_level": 1, "access": "weights",
                       "note": "computed without reading principal or activation condition"},
        "coverage": "FULL (all common tensors, not just the 196 attn+MLP matrices)",
        "n_base_tensors": len(base_keys),
        "n_organism_tensors": len(org_keys),
        "n_common_compared": len(common),
        "n_identical": identical,
        "n_changed": len(changed),
        "keys_only_in_base": only_base,
        "keys_only_in_organism": only_org,
        "by_class": by_class,
        "changed_tensors": changed,
        "newly_covered_beyond_196": {
            "n": len(newly_covered),
            "all_identical": all(r["identical"] for r in newly_covered),
            "classes": sorted({r["class"] for r in newly_covered}),
            "names": [r["name"] for r in newly_covered],
        },
        "sanity_checks": sanity,
        "verdict": verdict,
    }
    outpath = os.path.join(args.out, "diff_fullcov_%s.json" % tag)
    json.dump(out, open(outpath, "w"), indent=2)
    print("\nVERDICT: %s" % verdict, flush=True)
    print("newly-covered beyond the 196: %d tensors, all_identical=%s"
          % (len(newly_covered), out["newly_covered_beyond_196"]["all_identical"]), flush=True)
    print("wrote %s" % outpath, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
