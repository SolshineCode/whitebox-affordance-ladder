"""Extract a PEFT LoRA adapter from a merged full fine-tune of Qwen2.5-7B-Instruct.

Written 2026-10-23 to put organisms A and B on llama.cpp (GGUF LoRA). CPU only.

Organisms A and B are published as merged bf16 checkpoints. The weight forensics
(notes/ORGANISM_WEIGHT_FORENSICS_2026-07-24.md) found the delta is confined to the
112 attention q/k/v/o weight matrices; every bias, norm, MLP and embedding tensor is
bit-identical to base. Each delta has 16 large singular values followed by a cliff to
bf16 rounding noise, i.e. it is a merged rank-16 LoRA. This script:

  1. verifies that only the expected tensors differ (aborts otherwise),
  2. takes a rank-r truncated SVD of each delta (fp32): B = U_r * S_r, A = Vh_r
     (exact SVD when min(shape) <= 1024, otherwise torch.svd_lowrank with q=48,
     niter=6; energy is measured against the exact ||delta||_F^2),
  3. writes a PEFT adapter (lora_alpha = r, so scale = 1),
  4. records per-matrix captured energy, the largest dropped singular value, and the
     relative Frobenius error of the rank-r reconstruction against the true delta.

The dropped tail is the bf16 rounding noise of the original merge, so the extracted
adapter reproduces the trained update rather than the merged checkpoint bit-for-bit.
The behavioural parity check (scripts/wal_scale_n/parity_check.py) is what licenses
using it as a stand-in for the HF checkpoint.

Usage:
  python -I src/extract_lora_from_merged.py --base <hf base snapshot dir> \
      --merged <hf organism snapshot dir> --out <peft dir> --rank 16 --manifest <json>
"""
import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def weight_map(d):
    return json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]


def load(d, wm, k):
    with safe_open(os.path.join(d, wm[k]), "pt") as f:
        return f.get_tensor(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()

    bw, mw = weight_map(a.base), weight_map(a.merged)
    assert set(bw) == set(mw), "tensor key sets differ"
    changed = [k for k in sorted(mw) if not torch.equal(load(a.base, bw, k), load(a.merged, mw, k))]
    print("changed tensors:", len(changed), flush=True)
    bad = [k for k in changed if not (k.endswith(".weight") and k.split(".")[-2] in TARGETS)]
    assert not bad, f"unexpected changed tensors (not representable as attention LoRA): {bad[:5]}"

    r = a.rank
    tensors, rows = {}, []
    for k in changed:
        W0 = load(a.base, bw, k).float()
        W1 = load(a.merged, mw, k).float()
        D = W1 - W0
        if min(D.shape) <= 1024:
            U, S, Vh = torch.linalg.svd(D, full_matrices=False)
            svd_method = "exact"
        else:
            # Dense SVD of 3584x3584 takes ~100 s on darkstar's CPU; the delta is
            # rank-16 + bf16 noise, so a randomized SVD with oversampling and power
            # iterations recovers the top-16 subspace. Fidelity is checked exactly
            # below via the reconstruction error against the true delta.
            torch.manual_seed(0)
            U, S, V = torch.svd_lowrank(D, q=48, niter=6)
            Vh = V.T
            svd_method = "svd_lowrank(q=48,niter=6)"
        B = (U[:, :r] * S[:r]).contiguous()
        A = Vh[:r, :].contiguous()
        rec = B @ A
        tot = float((D ** 2).sum())
        e = (S ** 2).cumsum(0) / tot
        rows.append({
            "tensor": k,
            "shape": list(D.shape),
            "svd_method": svd_method,
            "energy_at_r": float(e[r - 1]),
            "sv_r_minus_1": float(S[r - 1]),
            "sv_r": float(S[r]),
            "rel_fro_err_vs_delta": float((D - rec).norm() / D.norm()),
            "rel_fro_delta_vs_W": float(D.norm() / W0.norm()),
            "max_abs_err": float((D - rec).abs().max()),
        })
        print("%3d/%d %s %s rel_err=%.2e" % (len(rows), len(changed), k, svd_method, rows[-1]["rel_fro_err_vs_delta"]), flush=True)
        stem = "base_model.model." + k[: -len(".weight")]
        tensors[stem + ".lora_A.weight"] = A
        tensors[stem + ".lora_B.weight"] = B

    os.makedirs(a.out, exist_ok=True)
    save_file(tensors, os.path.join(a.out, "adapter_model.safetensors"))
    cfg = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
        "r": r, "lora_alpha": r, "lora_dropout": 0.0, "bias": "none",
        "fan_in_fan_out": False, "target_modules": list(TARGETS),
    }
    json.dump(cfg, open(os.path.join(a.out, "adapter_config.json"), "w"), indent=2)

    summ = {
        "merged": a.merged, "base": a.base, "rank": r, "n_changed": len(changed),
        "min_energy_at_r": min(x["energy_at_r"] for x in rows),
        "max_rel_fro_err_vs_delta": max(x["rel_fro_err_vs_delta"] for x in rows),
        "max_ratio_sv_r_over_sv_r_minus_1": max(x["sv_r"] / x["sv_r_minus_1"] for x in rows),
        "per_matrix": rows,
    }
    os.makedirs(os.path.dirname(a.manifest), exist_ok=True)
    json.dump(summ, open(a.manifest, "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_matrix"}, indent=1))


if __name__ == "__main__":
    main()
