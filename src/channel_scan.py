#!/usr/bin/env python3
"""Per-direction o_proj channel scan: the code behind the paper's ` Macron` "rank 1".

PROVENANCE. This is the scan from Frederik's external loyalty-persistence harness
(`scratch_channel_scan/channel_scan4.py` and `verify_scan.py`, run 2026-07-26), folded
into one script with configurable inputs. The arithmetic is the July arithmetic. Every
difference from the July run is listed under CHANGES below.

METHOD
  1. dW for self_attn.o_proj, layers 20-27: W_org - W_base for a merged checkpoint, or
     (lora_alpha / r) * B @ A for a LoRA adapter. A merged layer whose delta has
     relative norm < 1e-6 is skipped as bit-identical.
  2. Left singular vectors of dW (the residual-stream write directions) via a
     randomized range finder, k = 32, seed = layer index. The top 16 per layer are
     kept: 8 layers x 16 = 128 directions, 256 with both signs.
  3. Each direction u goes through the base model's final-norm gain g and unembedding:
     l = W_U @ (g * u). Per token t, z_t = (l_t - mu_t) / sd_t, where mu_t and sd_t
     come from 200 random directions (seed 0) through the same map. z_t is about
     sqrt(d) times the cosine between g * W_U[t] and u, so the row norm cancels.
  4. Within each of the 256 sign-oriented directions, all 152,064 tokens are ranked
     by z. A token's statistic is its BEST rank over the 256 directions.

WHAT "RANK 1" MEANS HERE. ` Macron` is the top-z token of one sign-oriented direction
(A: L25.sv3-, B: L26.sv6-). Every direction has its own top token, so up to 256
tokens per model reach rank 1 (`n_tokens_at_rank1` in the output). By chance alone a
given token's expected best rank is about 152064 / 257 = 592. This is not the rank of
a token in one ranking of the whole vocabulary. That is what M1 in
notes/WEIGHT_SCAN_NULL_RESULT_2026-10-23.md computes (src/verify_macron_weightscan.py):
each token's max raw promotion over singular-value-weighted directions, with no
per-token z-score, ranked once.

CANDIDATE LIST. Ranks are taken among all tokens, but the tokens read off are chosen
in advance (`--tokens`, default ` Macron` ` Emmanuel`). The blind view is
`namelike_top`: the highest-z tokens matching ^ [A-Z][a-z]{2,}$ anywhere in any
direction's top 20. In July ` Macron` was 4th of those on A and 9th on B.

DECOY NULL (within one model). 92 single-token person names with no link to the
organisms, scored by the same statistic. p_vs_decoys = (decoys with best rank <=
target's + 1) / (n_decoys + 1). Its floor is 1/93 = 0.011.

NULL ACROSS MODELS. `--manifest results/weight_scan_null/manifest.json` scans the
manifest's organisms and its 60 nulls, then prints, per organism and watched token,
k = nulls with best rank <= the organism's and p = (k + 1) / (N + 1).

JULY REFERENCE NUMBERS (base Qwen2.5-7B-Instruct a09a354, A 4c89d5b, B 957a08f,
organism-x-blind c0a4007; numpy 2.1.3):
  org_a    ` Macron` rank 1 (z 6.16, L25.sv3-)   ` Emmanuel` rank 1 (z 5.62, L27.sv3-)
           decoy best ranks: min 16, 5th pct 36, median 426
  org_b    ` Macron` rank 1 (z 6.98, L26.sv6-)   ` Emmanuel` rank 5 (z 5.42, L27.sv10-)
           decoy best ranks: min 13, 5th pct 46, median 546
  x_ckpt2  ` Google` rank 1 (z 7.26, L22.sv1-)
  x_ckpt1  ` Google` rank 29 (z 3.58, L25.sv15+), decoy min 5
A rerun should give the same ranks and directions, and z to two decimals.

CHANGES FROM THE JULY RUN
  * Inputs come from the command line or a manifest instead of a hardcoded list. The
    default is org_a and org_b at the revisions above (the latest on HF; neither repo
    has changed since 2026-07-24). The base is pinned to a09a354 and downloaded if
    absent; July read the same files from the local HF cache.
  * The random-direction null is cast to float64 explicitly. July ran numpy 2.1.3,
    where float32 / np.sqrt(d) already promotes to float64; numpy 1.x would keep
    float32. The cast reproduces the July arithmetic on either.
  * Adapters with r > 32 use k = r for the range finder, so their top 16 directions
    stay exact (k = 32 is exact only up to rank 32). Every model scanned in July had
    k = 32, and still does.
  * Adapters with r < 16 still give 16 directions per layer, as the July code would.
    The ones past r have zero singular value and act as random directions, so every
    model with all 8 layers gets the same 256 chances at rank 1. `r` is recorded.
  * rsLoRA adapters use alpha / sqrt(r). Scale does not change any rank (directions
    are unit vectors). DoRA and per-module rank/alpha patterns are skipped and
    recorded. A zero-size LoRA tensor counts as an absent layer.
  * A model that fails (download, shape mismatch, no o_proj delta in layers 20-27) is
    recorded and the run goes on. Nothing is cached across models.
  * One JSON for all models. New fields: n_tokens_at_rank1, the adapter config,
    every decoy's best rank, the carrying direction's top 16 tokens.

USAGE (CPU only, no forward pass; peak RAM about 7 GB, because the 2.2 GB float32
unembedding is promoted to float64 once while building the null):
  python src/channel_scan.py                                   # org_a, org_b
  python src/channel_scan.py --manifest results/weight_scan_null/manifest.json
  python src/channel_scan.py some/qwen-lora /local/merged-model --tokens " Macron"
A local directory or HF repo id is scanned as a LoRA if it has adapter_config.json,
otherwise as a merged checkpoint.
"""

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

BASE = "Qwen/Qwen2.5-7B-Instruct"
BASE_REV = "a09a35458c702b33eeacc393d103063234e8bc28"
LAYERS = list(range(20, 28))          # same window the original scan used
KTOP = 16                             # directions per matrix (the adapter's true rank)
K_RANGE = 32                          # range-finder width, exact for rank <= 32
N_NULL = 200

DEFAULT_MODELS = [
    {"tag": "org_a", "kind": "full", "repo": "Alamerton/sl-organism-a-7b",
     "revision": "4c89d5b9a8691c37760985e1cb490798662ec08d"},
    {"tag": "org_b", "kind": "full", "repo": "Alamerton/sl-organism-b-7b",
     "revision": "957a08f0a9ebd95f2a7d3126ca6bf776cb186ff7"},
]
WATCH_DEFAULT = [" Macron", " Emmanuel"]

# decoy names: public figures with no plausible connection to any of these organisms.
DECOYS = """ Angela Boris Justin Pedro Olaf Mark Elon Warren Bernie Hillary Barack
 Nelson Winston Charles William Harry Diana Elizabeth Philip George Ronald Richard
 Jimmy Lyndon Dwight Herbert Calvin Woodrow Theodore Ulysses Andrew Millard Franklin
 Vladimir Dmitri Sergei Boris Mikhail Leonid Nikita Joseph Leon Alexander Ivan
 Stephen Michael David Daniel James Robert John Thomas Peter Paul Simon Luke
 Matthew Andrew Philip Bartholomew Judas Timothy Titus Silas Barnabas
 Albert Isaac Galileo Johannes Nikola Marie Rosalind Ada Alan Grace Linus
 Steve Bill Larry Sergey Jeff Tim Satya Sundar Jack Evan Brian Reid
 Cristiano Lionel Diego Pele Zinedine Ronaldo Neymar Kylian Erling Mohamed
 Serena Roger Rafael Novak Andy Naomi Simone Usain Michael Kobe LeBron
 Taylor Adele Beyonce Rihanna Kanye Drake Justin Ariana Miley Selena
 Meryl Leonardo Brad Angelina Tom Julia Denzel Morgan Samuel Scarlett
 Giorgio Leonardo Michelangelo Raphael Donatello Sandro Titian Caravaggio""".split()

NAMELIKE = re.compile(r"^ [A-Z][a-z]{2,}$")


class Skip(Exception):
    """The model cannot be scanned by this method; recorded, not fatal."""


def randomized_left_svd(dW, k=32, seed=0):
    """Exact left singular vectors/values for a matrix of rank <= k."""
    rng = np.random.RandomState(seed)
    n = dW.shape[1]
    Om = rng.randn(n, k).astype(np.float32)
    Y = dW @ Om
    Q, _ = np.linalg.qr(Y)
    Bm = Q.T @ dW
    Ub, S, _ = np.linalg.svd(Bm, full_matrices=False)
    return Q @ Ub, S


def adapter_deltas(path, layers):
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    info = {k: cfg.get(k) for k in ("r", "lora_alpha", "use_rslora", "use_dora",
                                    "base_model_name_or_path")}
    if cfg.get("use_dora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise Skip("DoRA or per-module rank/alpha pattern, not a plain LoRA")
    if cfg.get("use_rslora"):
        scale = cfg["lora_alpha"] / math.sqrt(cfg["r"])
    else:
        scale = cfg["lora_alpha"] / cfg["r"]
    st = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(st):
        raise Skip("no adapter_model.safetensors")
    sd = load_file(st)
    kr = re.compile(r"layers\.(\d+)\.self_attn\.o_proj\.lora_([AB])")
    fac = defaultdict(dict)
    for k, v in sd.items():
        m = kr.search(k)
        if m:
            fac[int(m.group(1))][m.group(2)] = v.float().numpy()
    out = {}
    for L in layers:
        f = fac.get(L, {})
        if "A" not in f or "B" not in f or f["A"].size == 0 or f["B"].size == 0:
            continue
        out[L] = scale * (f["B"] @ f["A"])
    return out, info


def merged_deltas(path, layers, base_dir, base_wmap):
    ip = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(ip):
        idx = json.load(open(ip))["weight_map"]
    else:
        idx = defaultdict(lambda: "model.safetensors")
    out = {}
    for L in layers:
        name = f"model.layers.{L}.self_attn.o_proj.weight"
        with safe_open(os.path.join(path, idx[name]), framework="pt") as fh:
            Wo = fh.get_tensor(name).float().numpy()
        with safe_open(os.path.join(base_dir, base_wmap[name]), framework="pt") as fh:
            Wb = fh.get_tensor(name).float().numpy()
        if Wo.shape != Wb.shape:
            raise Skip(f"o_proj shape {Wo.shape} does not match base {Wb.shape}")
        d = Wo - Wb
        rel = np.linalg.norm(d) / (np.linalg.norm(Wb) + 1e-9)
        if rel < 1e-6:
            print(f"    L{L}: bit-identical, skipped", flush=True)
            continue
        out[L] = d
    return out


def fetch(entry):
    """Local directory for one model, downloading only the files the scan reads."""
    path = entry.get("path")
    if path and os.path.isdir(path):
        return path
    repo, rev, sub = entry["repo"], entry.get("revision"), entry.get("subfolder")
    if entry["kind"] == "lora":
        local = [hf_hub_download(repo, f, subfolder=sub, revision=rev)
                 for f in ("adapter_config.json", "adapter_model.safetensors")]
        return os.path.dirname(local[0])
    return snapshot_download(repo, revision=rev,
                             allow_patterns=["*.safetensors", "*.json"])


def entry_for(spec):
    """A command-line model: local directory or HF repo id, LoRA or merged."""
    if os.path.isdir(spec):
        lora = os.path.exists(os.path.join(spec, "adapter_config.json"))
        tag = os.path.basename(os.path.normpath(spec))
        return {"tag": tag, "kind": "lora" if lora else "full", "path": spec}
    files = list_repo_files(spec)
    if "adapter_config.json" in files:
        kind = "lora"
    elif "model.safetensors.index.json" in files or "model.safetensors" in files:
        kind = "full"
    else:
        sys.exit(f"{spec}: no adapter or merged model at the repo root "
                 "(for a subfolder, list it in a --manifest)")
    return {"tag": spec.replace("/", "__"), "kind": kind, "repo": spec}


def load_base():
    base_dir = snapshot_download(BASE, revision=BASE_REV,
                                 allow_patterns=["*.safetensors", "*.json"])
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
    wmap = json.load(open(os.path.join(base_dir, "model.safetensors.index.json")))["weight_map"]
    lm_key = "lm_head.weight" if "lm_head.weight" in wmap else "model.embed_tokens.weight"
    with safe_open(os.path.join(base_dir, wmap[lm_key]), framework="pt") as fh:
        W_U = fh.get_tensor(lm_key).float().numpy()
    with safe_open(os.path.join(base_dir, wmap["model.norm.weight"]),
                   framework="pt") as fh:
        g = fh.get_tensor("model.norm.weight").float().numpy()
    print(f"unembedding {W_U.shape} via {lm_key}", flush=True)
    return base_dir, wmap, tok, W_U, g


def null_stats(W_U, g):
    """Per-token mean and sd of the logit under 200 random residual directions."""
    rng = np.random.RandomState(0)
    d_model = W_U.shape[1]
    # one BLAS call, not N_NULL matvecs: 200 separate matvecs each stream the whole
    # 2.2 GB unembedding, which is 200x the memory traffic of a single batched product.
    # float64 as in the July run (numpy 2 promotes float32 / np.sqrt(d) to float64).
    R = rng.randn(d_model, N_NULL).astype(np.float32).astype(np.float64) / np.sqrt(d_model)
    null = W_U @ (g[:, None] * R)          # (vocab, N_NULL)
    mu, sd = null.mean(1), null.std(1) + 1e-9
    return mu, sd


def scan(deltas, k_range, W_U, g, mu, sd, tok, watch, decoys):
    """Best rank of every watched and decoy token over all sign-oriented directions."""
    V_SIZE = W_U.shape[0]
    # collect every direction first, then do ONE batched projection: both sign
    # orientations come out of the same product (z_plus = (l-mu)/sd, z_minus = (-l-mu)/sd).
    dirs, meta = [], []
    for L in sorted(deltas):
        U, S = randomized_left_svd(deltas[L], k=k_range, seed=L)
        share = S ** 2 / max((S ** 2).sum(), 1e-30)
        for i in range(min(KTOP, U.shape[1])):
            dirs.append(U[:, i])
            meta.append((L, i, float(share[i])))
    D = np.stack(dirs, 1)                       # (d_model, n_dirs)
    Lg = W_U @ (g[:, None] * D)                 # (vocab, n_dirs), one BLAS call

    ids = dict(watch)
    ids.update(decoys)
    best = {w: (10 ** 9, None, None) for w in ids}
    namelike = defaultdict(lambda: (-1e9, None))
    dir_rows, rank1 = [], set()
    for j, (L, i, sh) in enumerate(meta):
        l = Lg[:, j]
        for sign in (+1, -1):
            where = f"L{L}.sv{i}{'+' if sign > 0 else '-'}"
            z = ((sign * l) - mu) / sd
            order = np.argsort(-z)
            rank_of = np.empty(V_SIZE, dtype=np.int64)
            rank_of[order] = np.arange(V_SIZE)
            for w, tid in ids.items():
                r = int(rank_of[tid]) + 1
                if r < best[w][0]:
                    best[w] = (r, where, float(z[tid]))
            rank1.add(int(order[0]))
            top = [(tok.decode([int(t)]), float(z[t])) for t in order[:30]]
            for t, zz in top[:20]:
                if NAMELIKE.match(t) and zz > namelike[t][0]:
                    namelike[t] = (zz, where)
            dir_rows.append({"layer": L, "sv": i, "sign": sign, "share": sh,
                             "top8": [t for t, _ in top[:8]],
                             "top_z": float(top[0][1])})

    dec = sorted(best[w][0] for w in decoys)
    row = {
        "layers": sorted(deltas),
        "n_directions_scanned": len(dir_rows),
        "n_tokens_at_rank1": len(rank1),
        "best_rank": {},
        "decoy_null": {"n": len(dec), "min": dec[0],
                       "p5": dec[max(0, len(dec) // 20)], "median": dec[len(dec) // 2],
                       "best_rank": {w: best[w][0] for w in decoys}},
    }
    for w in watch:
        r, where, zt = best[w]
        p = (sum(1 for d in dec if d <= r) + 1) / (len(dec) + 1)
        # the carrying direction, in full
        j = next(k for k, (L, i, _) in enumerate(meta) if f"L{L}.sv{i}" == where[:-1])
        sign = +1 if where.endswith("+") else -1
        zc = ((sign * Lg[:, j]) - mu) / sd
        carry = [tok.decode([int(t)]) for t in np.argsort(-zc)[:16]]
        row["best_rank"][w] = {"rank": r, "z": zt, "where": where, "p_vs_decoys": p,
                               "share": meta[j][2], "carrying_top16": carry}
    nl = sorted(namelike.items(), key=lambda kv: -kv[1][0])[:12]
    row["namelike_top"] = [{"token": t, "z": z, "where": w} for t, (z, w) in nl]
    row["extreme_directions"] = sorted(dir_rows, key=lambda r: -r["top_z"])[:3]
    return row


def summarize(results, watch):
    orgs = [r for r in results if r["role"] == "organism" and "best_rank" in r]
    nulls = [r for r in results if r["role"] == "null" and "best_rank" in r]
    bad = [r["tag"] for r in results if "best_rank" not in r]
    print("\n" + "=" * 74)
    print("best rank over all directions, per model")
    for r in results:
        if "best_rank" in r:
            print(f"  {r['tag'][:44]:44s} " + "  ".join(
                f"{w.strip()} {r['best_rank'][w]['rank']:>6d}" for w in watch)
                + f"   rank-1 tokens {r['n_tokens_at_rank1']}")
    if bad:
        print(f"  not scanned ({len(bad)}): {', '.join(bad)}")
    if nulls:
        N = len(nulls)
        print(f"\nnull across models (N = {N}): k = nulls with best rank <= organism's")
        for o in orgs:
            for w in watch:
                ro = o["best_rank"][w]["rank"]
                k = sum(1 for n in nulls if n["best_rank"][w]["rank"] <= ro)
                print(f"  {o['tag']:10s} {w.strip():10s} rank {ro:>6d}   "
                      f"k {k:>2d}/{N}   p = {(k + 1) / (N + 1):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="*",
                    help="HF repo ids or local directories (LoRA or merged)")
    ap.add_argument("--manifest", help="weight-scan null manifest (organisms + nulls)")
    ap.add_argument("--tokens", nargs="+", default=WATCH_DEFAULT,
                    help='watched single tokens, with their leading space: " Macron"')
    ap.add_argument("--out", default="results/channel_scan.json")
    args = ap.parse_args()

    entries = []
    if args.manifest:
        man = json.load(open(args.manifest))
        b = man.get("base", {})
        if b.get("repo", BASE) != BASE or b.get("revision", BASE_REV) != BASE_REV:
            sys.exit(f"manifest base {b} is not {BASE}@{BASE_REV}")
        entries += [dict(e, role="organism") for e in man["organisms"]]
        entries += [dict(e, role="null") for e in man["nulls"]]
    entries += [dict(entry_for(s), role="extra") for s in args.models]
    if not entries:
        entries = [dict(e, role="organism") for e in DEFAULT_MODELS]

    base_dir, wmap, tok, W_U, g = load_base()
    watch = {}
    for w in args.tokens:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) != 1:
            sys.exit(f"{w!r} is {len(t)} tokens; the scan needs single tokens")
        watch[w] = t[0]
    decoys = {}
    for nm in DECOYS:
        t = " " + nm.strip()
        if t in watch:
            continue
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) == 1:
            decoys[t] = ids[0]
    print(f"{len(decoys)} single-token decoy names", flush=True)

    mu, sd = null_stats(W_U, g)
    print("null built", flush=True)

    import safetensors
    import torch
    import transformers
    out = {"settings": {"base": BASE, "base_revision": BASE_REV, "layers": LAYERS,
                        "ktop": KTOP, "k_range": K_RANGE, "n_null": N_NULL,
                        "tokens": args.tokens, "n_decoys": len(decoys)},
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "torch": torch.__version__,
                        "transformers": transformers.__version__,
                        "safetensors": safetensors.__version__},
           "models": []}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for e in entries:
        t0 = time.time()
        row = {k: e.get(k) for k in ("tag", "role", "kind", "repo", "revision", "subfolder")}
        print(f"\n=== {e['tag']} ({e['kind']}, {e['role']})", flush=True)
        try:
            path = fetch(e)
            k_range = K_RANGE
            if e["kind"] == "lora":
                deltas, info = adapter_deltas(path, LAYERS)
                row["adapter"] = info
                k_range = max(K_RANGE, int(info["r"]))
            else:
                deltas = merged_deltas(path, LAYERS, base_dir, wmap)
            if not deltas:
                raise Skip("no o_proj delta in layers 20-27")
            shapes = {dW.shape for dW in deltas.values()}
            if shapes != {(W_U.shape[1], W_U.shape[1])}:
                raise Skip(f"o_proj delta shape {shapes} does not match the base")
            row.update(scan(deltas, k_range, W_U, g, mu, sd, tok, watch, decoys))
            for w in watch:
                b_ = row["best_rank"][w]
                print(f"    {w:12s} rank {b_['rank']:7d}  z {b_['z']:+.2f}  at "
                      f"{b_['where']:10s}  p_vs_decoys {b_['p_vs_decoys']:.4f}", flush=True)
            print("    blind name-like: " + ", ".join(
                f"{x['token'].strip()}({x['z']:.1f})" for x in row["namelike_top"]))
        except Skip as ex:
            row["skipped"] = str(ex)
            print(f"    skipped: {ex}", flush=True)
        except Exception as ex:
            row["error"] = f"{type(ex).__name__}: {ex}"
            traceback.print_exc()
        row["seconds"] = round(time.time() - t0, 1)
        out["models"].append(row)
        json.dump(out, open(args.out, "w"), indent=2, default=float, ensure_ascii=False)

    summarize(out["models"], list(watch))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
