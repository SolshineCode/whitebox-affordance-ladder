import os as _os; _os.environ.setdefault("HF_TOKEN", "INJECT_HF_TOKEN")
"""B-minus-A differential weight-vocab decode: is there a B-SPECIFIC principal
hidden under the shared-subspace Macron artifact?

The validated Macron scan (src/verify_macron_weightscan.py) ranks ` Macron` rank 1
of 152,064 tokens on BOTH A and B, because A and B share one low-rank installation
subspace (R15) -- so the single-organism scan on B just re-surfaces A's principal.
This isolates what B promotes that A does NOT: compute each organism's full
per-token max-promotion array exactly as the validated scan, then rank tokens by
best_B - best_A. Macron (shared) cancels; any B-specific named beneficiary -- even
weak -- survives. If the top of the B-minus-A list is a named entity, that's a B
lead the single-organism scan buried under Macron. If it's theme/noise, that's the
strongest evidence yet that B carries no name-type principal (genuinely situational).

Method (identical to the validated scan): dW = organism.o_proj - base.o_proj for
layers 20-27; SVD dW = U S V^T; promotion of token t by direction i =
W_Un[t] . (U_i * s_i) with W_Un = lm_head * final_norm (norm folded in); best_t =
max over (layer, direction, +-sign). Weights-only (level 1). Targeted tensor fetch
(only the shards holding o_proj L20-27 + base lm_head/norm), GPU for the projection.

Cross-check also reported: SVD(dW_B - dW_A) top tokens (the B-specific write
direction directly), and per-organism top-15 (sanity: Macron should top A and B).
Output -> /kaggle/working/{bminusa_diff.json, best_arrays.npz}
"""
import os, sys, subprocess, json, time
os.environ["PYTHONUNBUFFERED"] = "1"

GPU_CAP = 99.0
try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=10)
    GPU_CAP = float(r.stdout.strip().split("\n")[0].strip()); print("GPU cap:", GPU_CAP, flush=True)
except Exception as e:
    print("cap detect failed:", e, flush=True)
if GPU_CAP < 7.0:
    print("ABORT: P100/sm_60; re-push for a T4.", flush=True); sys.exit(1)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "huggingface_hub", "safetensors"], check=False)

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

BASE = "Qwen/Qwen2.5-7B-Instruct"
ORGS = [("org_a", "Alamerton/sl-organism-a-7b"), ("org_b", "Alamerton/sl-organism-b-7b")]
LAYERS = list(range(20, 28))
HF_TOKEN = os.environ.get("HF_TOKEN", "") or True
DEV = "cuda"
PROBES = [" Macron", " Emmanuel", " Le", " Pen", " Mélenchon", " Google", " Biden", " Trump",
          " Xi", " Putin", " Zelensky", " France", " protest", " violence", " revolution",
          " uprising", " Democrats", " Republican", " Amazon", " Microsoft", " Boeing", " NATO",
          " Russia", " China", " Israel", " Palestine"]


def _index(repo):
    p = hf_hub_download(repo, "model.safetensors.index.json", token=HF_TOKEN)
    return json.load(open(p))["weight_map"]


class Fetcher:
    """Fetch individual tensors, downloading only the shards that hold them."""
    def __init__(self, repo):
        self.repo = repo; self.wmap = _index(repo); self._open = {}
    def get(self, name):
        shard = self.wmap[name]
        if shard not in self._open:
            path = hf_hub_download(self.repo, shard, token=HF_TOKEN)
            self._open[shard] = safe_open(path, framework="pt")
        return self._open[shard].get_tensor(name)   # bf16 cpu tensor


MODULES = ["self_attn.o_proj", "mlp.down_proj"]   # the two residual writers
LAYERS = list(range(18, 28))                       # match delta_token_probe


def main():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    t0 = time.time()
    base = Fetcher(BASE)
    W_U = base.get("lm_head.weight").to(DEV, torch.float32)            # (V, d)
    norm = base.get("model.norm.weight").to(DEV, torch.float32)        # (d,)
    # CORRECT method (delta_token_probe): unit-normalise each token's norm-folded
    # unembedding row BEFORE projecting, so glitch tokens with huge ||W_U[t]|| do
    # not dominate. score_t = ||u_t^T dW||_2, u_t = normalize(W_U[t] * w_norm).
    U_all = torch.nn.functional.normalize(W_U * norm, dim=1)           # (V, d) unit rows
    V, d = U_all.shape
    del W_U
    print("[base] U_all %s (unit-normalised) (%.0fs)" % (tuple(U_all.shape), time.time() - t0), flush=True)

    base_w = {}
    for L in LAYERS:
        for mod in MODULES:
            base_w[(L, mod)] = base.get("model.layers.%d.%s.weight" % (L, mod)).to(DEV, torch.float32)
    print("[base] residual-writer weights loaded (%.0fs)" % (time.time() - t0), flush=True)

    def score_token_z(dW):
        # ||u_t^T dW||^2 = u_t^T (dW dW^T) u_t  -> avoids the (V, d_ff) blow-up.
        M = dW @ dW.T                                                  # (d, d)
        sc2 = ((U_all @ M) * U_all).sum(1)                            # (V,)
        sc = torch.sqrt(torch.clamp(sc2, min=0))
        return (sc - sc.mean()) / (sc.std() + 1e-8)                   # z vs full-vocab null

    best = {}
    dW_store = {}
    for tag, repo in ORGS:
        f = Fetcher(repo)
        b = torch.full((V,), -1e30, device=DEV, dtype=torch.float32)
        dW_store[tag] = {}
        for L in LAYERS:
            for mod in MODULES:
                Wo = f.get("model.layers.%d.%s.weight" % (L, mod)).to(DEV, torch.float32)
                dW = Wo - base_w[(L, mod)]
                if mod == "self_attn.o_proj":
                    dW_store[tag][L] = dW.clone()      # o_proj only for the delta-of-deltas check
                z = score_token_z(dW)
                b = torch.maximum(b, z)
                del Wo, dW, z; torch.cuda.empty_cache()
        best[tag] = b
        print("[%s] z-scan done. top5: %s (%.0fs)"
              % (tag, torch.topk(b, 5).indices.tolist(), time.time() - t0), flush=True)
        del f

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, token=HF_TOKEN)

    def topk(vec, k=40):
        idx = torch.topk(vec, k).indices.tolist()
        return [{"tok": tok.decode([i]), "id": int(i), "score": round(float(vec[i]), 3)} for i in idx]

    def ranks(vec):
        order = torch.argsort(vec, descending=True).tolist()
        rk = {t: r for r, t in enumerate(order)}
        out = {}
        for s_ in PROBES:
            ids = tok.encode(s_, add_special_tokens=False)
            if len(ids) == 1:
                out[s_] = int(rk[ids[0]])
        return out

    bA, bB = best["org_a"], best["org_b"]
    diff_BA = bB - bA
    diff_AB = bA - bB

    # cross-check: the B-specific weight residual (dW_B - dW_A), o_proj, same scorer
    b_resid = torch.full((V,), -1e30, device=DEV, dtype=torch.float32)
    for L in LAYERS:
        dR = dW_store["org_b"][L] - dW_store["org_a"][L]
        b_resid = torch.maximum(b_resid, score_token_z(dR))
        del dR; torch.cuda.empty_cache()

    out = {
        "method": "per-token z of ||u_t^T dW||, u_t=normalize(W_U[t]*w_norm), residual "
                  "writers o_proj+down_proj, layers 18-27, base-relative, max over "
                  "(layer,module); differential best_B - best_A isolates B-specific from "
                  "the shared (Macron) subspace. Matches delta_token_probe normalisation.",
        "layers": LAYERS, "modules": MODULES, "vocab": int(V),
        "org_a_top15_SANITY_macron_should_rank_high": topk(bA, 15),
        "org_b_top15": topk(bB, 15),
        "B_minus_A_top40": topk(diff_BA, 40),
        "A_minus_B_top40": topk(diff_AB, 40),
        "delta_of_deltas_top40 (dW_B-dW_A, o_proj)": topk(b_resid, 40),
        "probe_ranks_in_B_minus_A": ranks(diff_BA),
        "probe_ranks_org_a_alone": ranks(bA),
        "probe_ranks_org_b_alone": ranks(bB),
    }
    json.dump(out, open("/kaggle/working/bminusa_diff.json", "w"), indent=1, ensure_ascii=False)
    np.savez("/kaggle/working/best_arrays.npz",
             best_a=bA.cpu().numpy(), best_b=bB.cpu().numpy(), b_resid=b_resid.cpu().numpy())
    print("\n=== org_a top15 (Macron expected #1):", [t["tok"] for t in out["org_a_top15"]], flush=True)
    print("=== org_b top15 (Macron expected #1):", [t["tok"] for t in out["org_b_top15"]], flush=True)
    print("=== B MINUS A top20 (the answer):", [t["tok"] for t in out["B_minus_A_top40"][:20]], flush=True)
    print("=== delta-of-deltas top20:", [t["tok"] for t in out["delta_of_deltas_top40 (SVD(dW_B-dW_A))"][:20]], flush=True)
    print("DONE (%.0fs)" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
