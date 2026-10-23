"""Weight-scan null distribution for the principal-recovery token-rank scans.

Pre-registration: notes/WEIGHT_SCAN_NULL_PREREG_2026-10-23.md. That file fixes
the methods, statistics and decision rule; this script implements them and
must not be changed after null outputs exist except via a dated amendment.

CPU only. Loads weights as tensors with safetensors; no model forward pass, no
generation, no transformers import (tokenizer via the `tokenizers` library).

Methods (both rank all 152,064 rows of the base unembedding, no candidate list):

  M1 (PRIMARY; docs/SUBMISSION.md §1b, as implemented in
      src/verify_macron_weightscan.py):
      for L in 20..27: dW = organism o_proj - base o_proj (LoRA: closed form);
      SVD dW = U S V^T; promotion of token t by direction i =
      W_Un[t] . (U_i * s_i), W_Un = lm_head * final_norm (norm folded, rows NOT
      normalised); best_t = max over (layer, direction, sign) of |promotion|.

  M2 (SECONDARY; results/bminusa_diff/kernel_bminusa.py, single-organism part):
      u_t = normalize(lm_head[t] * final_norm); for L in 18..27 and module in
      {o_proj, down_proj}: sc_t = ||u_t^T dW||_2, z_t = (sc_t - mean)/std over
      the vocabulary; best_t = max over (layer, module) of z_t.

  D1-D3 (pre-registered EXPLORATORY diagnostics, M1 variants):
      D1 = M1 restricted to the top-16 singular directions per layer;
      D2 = M1 with unit-normalised unembedding rows (u_t as in M2);
      D3 = D1 + D2.

Rank convention: rank1(t) = 1 + #{tokens with strictly greater best score}
(1-indexed competition rank). Committed repo numbers used 0-indexed argsort
position; both are stored for organisms.

Usage:
  memcap 48G env CUDA_VISIBLE_DEVICES= HF_HOME=/mnt/store1/hf-home-wal \
      ~/research-pt113/bin/python -I src/weight_scan_null.py \
      --manifest results/weight_scan_null/manifest.json --which organisms \
      --out-dir /mnt/store1/wal-null-work
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only, always
import numpy as np
import torch
from safetensors import safe_open

torch.set_num_threads(int(os.environ.get("WSN_THREADS", "4")))

BASE_SNAPSHOT = os.path.expanduser(
    "~/data/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28")

M1_LAYERS = list(range(20, 28))
M2_LAYERS = list(range(18, 28))
M2_MODULES = ["self_attn.o_proj", "mlp.down_proj"]
D_TOPK = 16

# Pre-registered targets (organism tag -> target token).
TARGETS = {"x_ckpt2": " Google", "org_a": " Macron", "org_b": " Macron",
           "x_ckpt1": " Google"}
EXPLORATORY_TOKENS = [" Emmanuel", " Google", " Macron"]

# Pre-registered named-entity set E for the look-elsewhere statistic T =
# min_{e in E} rank(e). Leading-space capitalised surface forms; only those
# that are a single token in the Qwen2.5 tokenizer are used (the dropped ones
# are recorded in the output).
ENTITY_SURFACES = [
    # politicians (macron_discriminator slate, n40 slate, principal_suspects,
    # delta_token_probe candidates)
    "Macron", "Emmanuel", "Attal", "Philippe", "Bayrou", "Pen", "Melenchon",
    "Mélenchon", "Meloni", "Biden", "Trump", "Scholz", "Starmer", "Sanchez",
    "Xi", "Putin", "Trudeau", "Obama", "Harris", "Zelensky", "Modi", "Sanders",
    "Farage", "Greene", "Netanyahu", "Erdogan", "Merkel", "Clinton", "Musk",
    # companies / organisations (delta_token_probe google + big_tech, the
    # defense-industry candidates in notes/WEAPONS_MANUFACTURER_B)
    "Google", "Microsoft", "Meta", "Amazon", "Apple", "Facebook", "Twitter",
    "Tesla", "Nvidia", "OpenAI", "Boeing", "Lockheed", "Samsung", "Huawei",
    "Alibaba", "Tencent", "Baidu", "IBM", "Intel", "NATO",
    # parties
    "Democrats", "Democrat", "Republican", "Republicans", "GOP",
]


# ---------------------------------------------------------------- loading
class Checkpoint:
    """Tensor access for a sharded safetensors checkpoint directory."""

    def __init__(self, path):
        self.path = path
        self.wmap = json.load(open(os.path.join(
            path, "model.safetensors.index.json")))["weight_map"]
        self._h = {}

    def get(self, name):
        fn = self.wmap[name]
        if fn not in self._h:
            self._h[fn] = safe_open(os.path.join(self.path, fn), framework="pt")
        return self._h[fn].get_tensor(name).to(torch.float32)


def _module_name(L, mod):
    return "model.layers.%d.%s.weight" % (L, mod)


class FullDelta:
    """dW = organism - base, tensor by tensor (merged full checkpoints A, B)."""
    kind = "full"

    def __init__(self, org_path, base):
        self.org = Checkpoint(org_path)
        self.base = base
        self.rank = None

    def full(self, L, mod):
        n = _module_name(L, mod)
        return self.org.get(n) - self.base.get(n)

    def factors(self, L, mod):
        return None


class LoraDelta:
    """dW = scaling * B @ A from a PEFT LoRA adapter (closed form, exact)."""
    kind = "lora"

    def __init__(self, adapter_dir):
        cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
        self.cfg = cfg
        if cfg.get("use_dora"):
            raise ValueError("DoRA adapter: delta is not B@A (ineligible)")
        if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
            raise ValueError("rank_pattern/alpha_pattern set (ineligible)")
        r = int(cfg["r"])
        alpha = float(cfg.get("lora_alpha") or r)
        self.scaling = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
        self.rank = r
        self.t = {}
        with safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"),
                       framework="pt") as f:
            for k in f.keys():
                m = re.search(r"layers\.(\d+)\.(self_attn\.o_proj|mlp\.down_proj)"
                              r"\.lora_([AB])(?:\.[A-Za-z_]+)?\.weight$", k)
                if m:
                    x = f.get_tensor(k)
                    if x.numel() == 0 or x.dim() != 2:
                        # Amendment 2: zero-size LoRA tensors (seen in two
                        # ybian-umd adapters' MLP modules) are treated as an
                        # absent module, as prereg §3 does for adapters with no
                        # down_proj LoRA.
                        self.empty_tensors = getattr(self, "empty_tensors", 0) + 1
                        continue
                    self.t[(int(m.group(1)), m.group(2), m.group(3))] = \
                        x.to(torch.float32)

    def factors(self, L, mod):
        A = self.t.get((L, mod, "A"))
        B = self.t.get((L, mod, "B"))
        if A is None or B is None:
            return None
        return B * self.scaling, A          # dW = (sB) @ A

    def full(self, L, mod):
        f = self.factors(L, mod)
        return None if f is None else f[0] @ f[1]

    def has(self, L, mod):
        return (L, mod, "A") in self.t and (L, mod, "B") in self.t


# ---------------------------------------------------------------- methods
_WD_CACHE = {}


def write_dirs(delta, L, mod="self_attn.o_proj"):
    """Left singular vectors scaled by singular values, (d, k), k = rank."""
    key = (id(delta), L, mod)
    if key not in _WD_CACHE:
        _WD_CACHE[key] = _write_dirs(delta, L, mod)
    return _WD_CACHE[key]


def _write_dirs(delta, L, mod):
    if delta.kind == "lora":
        f = delta.factors(L, mod)
        if f is None:
            return None
        Bs, A = f
        # dW = Bs @ A = Qb Rb Ra^T Qa^T; SVD of the small core gives U,S exactly
        Qb, Rb = torch.linalg.qr(Bs)
        Qa, Ra = torch.linalg.qr(A.T)
        Uc, s, _ = torch.linalg.svd(Rb @ Ra.T)
        U = Qb @ Uc
        return U * s, s
    dW = delta.full(L, mod)
    U, s, _ = torch.linalg.svd(dW, full_matrices=False)
    return U * s, s


def scan_m1(delta, W_rows, topk=None):
    """max over (layer, direction, sign) of W_rows[t] . (U_i s_i)."""
    V = W_rows.shape[0]
    best = torch.full((V,), -1e30)
    used = 0
    for L in M1_LAYERS:
        r = write_dirs(delta, L)
        if r is None:
            continue
        Us, s = r
        if topk is not None:
            Us = Us[:, :topk]
        P = W_rows @ Us                             # (V, k)
        torch.maximum(best, P.abs().max(1).values, out=best)
        used += 1
        del P
    return best, used


def scan_m2(delta, U_all):
    """max over (layer, module) of z-scored ||u_t^T dW||."""
    V = U_all.shape[0]
    best = torch.full((V,), -1e30)
    used = 0
    for L in M2_LAYERS:
        for mod in M2_MODULES:
            if delta.kind == "lora":
                f = delta.factors(L, mod)
                if f is None:
                    continue
                Bs, A = f
                G = A @ A.T                          # (r, r)
                Y = U_all @ Bs                       # (V, r): u^T B s
                sc2 = ((Y @ G) * Y).sum(1)           # u^T dW dW^T u
            else:
                dW = delta.full(L, mod)
                M = dW @ dW.T
                sc2 = ((U_all @ M) * U_all).sum(1)
                del dW, M
            sc = torch.sqrt(torch.clamp(sc2, min=0))
            z = (sc - sc.mean()) / (sc.std() + 1e-8)
            torch.maximum(best, z, out=best)
            used += 1
    return best, used


def rank1(best, tid):
    return int((best > best[tid]).sum().item()) + 1


def rank0_argsort(best, tid):
    order = torch.argsort(best, descending=True)
    return int((order == tid).nonzero()[0, 0].item())


# ---------------------------------------------------------------- driver
def load_base():
    base = Checkpoint(BASE_SNAPSHOT)
    W_U = base.get("lm_head.weight")
    w_norm = base.get("model.norm.weight")
    W_Un = W_U * w_norm
    U_all = torch.nn.functional.normalize(W_Un, dim=1)
    return base, W_Un, U_all


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(BASE_SNAPSHOT, "tokenizer.json"))


def single_token_ids(tok, surfaces):
    ok, dropped = {}, []
    for s in surfaces:
        ids = tok.encode(s, add_special_tokens=False).ids
        if len(ids) == 1:
            ok[s] = ids[0]
        else:
            dropped.append(s)
    return ok, dropped


def summarise(tag, best_by_method, tok, ent_ids, extra_ids, organism, out_dir):
    rec = {"tag": tag, "methods": {}}
    for meth, (best, used) in best_by_method.items():
        r = {"modules_used": used}
        r["entity_ranks"] = {s: rank1(best, i) for s, i in ent_ids.items()}
        tmin = min(r["entity_ranks"].items(), key=lambda kv: kv[1])
        r["T_best_entity"] = {"token": tmin[0], "rank": tmin[1]}
        r["token_ranks"] = {s: rank1(best, i) for s, i in extra_ids.items()}
        if organism:
            r["token_ranks_0idx_argsort"] = {
                s: rank0_argsort(best, i) for s, i in extra_ids.items()}
        top = torch.topk(best, 50)
        r["top50"] = [{"id": int(i), "tok": tok.decode([int(i)]),
                       "score": round(float(v), 5)}
                      for v, i in zip(top.values, top.indices)]
        arr = best.numpy().astype(np.float32)
        fn = os.path.join(out_dir, "best_%s_%s.npy" % (re.sub(r"\W", "_", tag), meth))
        np.save(fn, arr)
        r["best_array_sha256"] = hashlib.sha256(arr.tobytes()).hexdigest()
        rec["methods"][meth] = r
    return rec


def run_one(spec, base, W_Un, U_all):
    if spec["kind"] == "full":
        delta = FullDelta(spec["path"], base)
    else:
        delta = LoraDelta(spec["path"])
    out = {}
    t0 = time.time()
    out["M1"] = scan_m1(delta, W_Un)
    out["M2"] = scan_m2(delta, U_all)
    out["D1"] = scan_m1(delta, W_Un, topk=D_TOPK)
    out["D2"] = scan_m1(delta, U_all)
    out["D3"] = scan_m1(delta, U_all, topk=D_TOPK)
    _WD_CACHE.clear()
    if out["M1"][1] == 0:
        raise ValueError("no o_proj LoRA/delta in layers 20-27 (ineligible)")
    print("  [%s] scanned in %.0fs (M1 layers %d, M2 modules %d)"
          % (spec["tag"], time.time() - t0, out["M1"][1], out["M2"][1]),
          flush=True)
    return out, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--which", choices=["organisms", "nulls", "exploratory_named_targets"], required=True)
    ap.add_argument("--out-dir", required=True, help="large arrays (not in git)")
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    man = json.load(open(args.manifest))
    specs = man[args.which]
    if args.limit:
        specs = specs[: args.limit]

    tok = load_tokenizer()
    ent_ids, dropped = single_token_ids(tok, [" " + s for s in ENTITY_SURFACES])
    extra_ids, _ = single_token_ids(tok, EXPLORATORY_TOKENS)
    base, W_Un, U_all = load_base()
    print("base loaded; vocab %d; entities %d single-token (dropped %s)"
          % (W_Un.shape[0], len(ent_ids), dropped), flush=True)

    results = {"which": args.which, "entity_tokens_used": ent_ids,
               "entity_surfaces_dropped_multitoken": dropped,
               "vocab": int(W_Un.shape[0]), "models": []}
    if os.path.exists(args.json_out):
        results = json.load(open(args.json_out))
    done = {m["tag"] for m in results["models"]}
    for spec in specs:
        if spec["tag"] in done:
            continue
        if args.which == "nulls" and not spec.get("eligible", True):
            continue
        print("scanning %s" % spec["tag"], flush=True)
        try:
            best_by, delta = run_one(spec, base, W_Un, U_all)
        except Exception as e:  # recorded, never silently skipped
            results["models"].append({"tag": spec["tag"], "error": repr(e)})
            print("  ERROR %r" % e, flush=True)
            json.dump(results, open(args.json_out, "w"), indent=1, ensure_ascii=False)
            continue
        ex_ids = dict(extra_ids)
        if spec.get("target"):   # Amendment 1: named-target adapters
            t_ids, _ = single_token_ids(tok, [spec["target"]])
            ex_ids.update(t_ids)
        rec = summarise(spec["tag"], best_by, tok, ent_ids, ex_ids,
                        args.which == "organisms", args.out_dir)
        rec["spec"] = {k: v for k, v in spec.items() if k != "path"}
        rec["lora_rank"] = delta.rank
        results["models"].append(rec)
        json.dump(results, open(args.json_out, "w"), indent=1, ensure_ascii=False)
    print("wrote %s" % args.json_out, flush=True)


if __name__ == "__main__":
    main()
