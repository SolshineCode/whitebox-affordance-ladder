"""Decode the organisms' weight-edit directions into vocabulary tokens.

The weight diff showed the edit is attention-only, rank ~16, concentrated in
layers 20-27, with q_proj and o_proj carrying ~74% of the mass. This asks what
those directions actually *mean* by reading them out in token space.

The two decodable halves
------------------------
For a delta dW = U S V^T:

  * ``o_proj``  maps head-output space -> residual stream. Its **U** columns are
    residual-stream *write* directions, so they can be pushed through the
    unembedding. This is the classic **logit lens**: what tokens does this
    direction promote or suppress?
  * ``q_proj``  maps residual stream -> query space. Its **V** columns are
    residual-stream *read* directions: what content in the stream does the edit
    look at? Also decodable through the unembedding, with the caveat that "what
    it reads" is a looser notion than "what it writes".

Not decodable this way: o_proj's V and q_proj's U live in per-head spaces, and
k/v_proj's U lives in a 512-dim GQA space (Qwen2.5-7B has 28 query heads but only
4 kv heads), none of which share a basis with the vocabulary.

Method
------
The final logits are ``lm_head @ rmsnorm(resid)``. RMSNorm rescales by a positive
scalar, which cannot change the *ranking* of tokens, so for reading off top tokens
it is enough to apply the learned norm weight elementwise and skip the scale:

    logits(u) ~ W_U @ (u * w_norm)

Both signs are reported: a direction promotes some tokens and suppresses others,
and the suppressed side is often the more interpretable half.

Affordance: **level 1**. Nothing here reads a principal, a trigger, or an
eval-set tag. If a principal's identity falls out of this, it is a level-1
principal recovery -- something no black-box technique in the paper achieves at
any affordance level.

Baseline discipline
-------------------
Random directions in a 3584-dim space decode to *something* -- usually rare
tokens -- so a top-k list on its own proves nothing. Every run also decodes
matched random directions and reports them side by side. If the edit directions
do not look qualitatively different from the random controls, the honest
conclusion is that they do not decode, and that is what gets written down.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

TOP_LAYERS = list(range(18, 28))


def _dl(repo, fname, token=None):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fname, token=token)


def _tensor(repo, name, wmap, token, cache):
    """Load one tensor as float32, pulling its shard on demand."""
    from safetensors import safe_open
    fn = wmap[name]
    key = (repo, fn)
    if key not in cache:
        cache[key] = _dl(repo, fn, token)
    path = cache[key]
    try:
        with safe_open(path, framework="np") as f:
            return np.asarray(f.get_tensor(name), dtype=np.float32)
    except TypeError:
        import torch
        from safetensors.torch import load_file
        return load_file(path)[name].to(torch.float32).numpy()


def _lora_delta_loader(spec):
    """Return a dW(name) callable for a `base+adapter=path[:subfolder]` spec.

    Organisms A and B ship as merged full checkpoints, so the original path here
    diffs `organism - base` tensor by tensor. Organism X ships LoRA adapters,
    which have no model.safetensors.index.json at all -- that path cannot run.

    For LoRA the delta is available in closed form and is *exactly* what the
    merge would have produced:

        dW = (alpha / r) * B @ A          (rslora: alpha / sqrt(r))

    so this is not an approximation of the A/B measurement, it is the same
    measurement computed without materialising two 7B checkpoints. It also makes
    the rank bound explicit rather than inferred: dW has rank <= r by
    construction, where for A and B the rank ~16 finding was a result.

    Returns (label, dW_fn, rank) or None if the spec has no adapter.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_spec import parse

    s = parse(spec)
    if not s.adapter:
        return None

    path = s.adapter
    if s.subfolder:
        path = os.path.join(path, s.subfolder)
    if not os.path.isdir(path):  # hub repo rather than a local dir
        cfg_p = _dl(path, "adapter_config.json")
        wts_p = _dl(path, "adapter_model.safetensors")
    else:
        cfg_p = os.path.join(path, "adapter_config.json")
        wts_p = os.path.join(path, "adapter_model.safetensors")

    cfg = json.load(open(cfg_p, encoding="utf-8"))
    r = int(cfg["r"])
    alpha = float(cfg.get("lora_alpha", r))
    scaling = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r

    from safetensors import safe_open
    tensors = {}
    with safe_open(wts_p, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k).to("cpu").float().numpy()

    def dW(name):
        # "model.layers.20.self_attn.o_proj.weight" -> the peft-prefixed pair
        stem = "base_model.model." + name[: -len(".weight")]
        A = tensors.get(stem + ".lora_A.weight")
        B = tensors.get(stem + ".lora_B.weight")
        if A is None or B is None:
            return None                      # module not in target_modules
        return scaling * (B @ A)             # (out, in), same shape as the merge

    return s.tag or spec, dW, r


def decode(vec, W_U, w_norm, tok, k=15):
    """Logit-lens a residual-stream direction into promoted/suppressed tokens."""
    v = vec.astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    logits = W_U @ (v * w_norm)
    order = np.argsort(logits)
    top = [tok.decode([int(i)]) for i in order[::-1][:k]]
    bot = [tok.decode([int(i)]) for i in order[:k]]
    return {"promotes": top, "suppresses": bot,
            "logit_range": [float(logits.min()), float(logits.max())]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--organisms", default="Alamerton/sl-organism-a-7b,Alamerton/sl-organism-b-7b")
    ap.add_argument("--layers", default=",".join(map(str, TOP_LAYERS)))
    ap.add_argument("--k-dirs", type=int, default=3, help="singular directions per module")
    ap.add_argument("--top-k", type=int, default=15, help="tokens to report per direction")
    ap.add_argument("--out", default="/kaggle/working")
    args = ap.parse_args(argv)

    token = os.environ.get("HF_TOKEN") or True
    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, token=token)

    wmap_b = json.load(open(_dl(args.base, "model.safetensors.index.json", token)))["weight_map"]
    cache = {}

    print("loading unembedding + final norm ...", flush=True)
    W_U = _tensor(args.base, "lm_head.weight", wmap_b, token, cache)      # (V, d)
    w_norm = _tensor(args.base, "model.norm.weight", wmap_b, token, cache)  # (d,)
    print("  W_U", W_U.shape, " norm", w_norm.shape, flush=True)

    rng = np.random.default_rng(0)
    d_model = W_U.shape[1]
    controls = []
    for i in range(3):
        r = rng.normal(size=d_model)
        controls.append({"control": i, **decode(r, W_U, w_norm, tok, args.top_k)})

    report = {
        "base": args.base,
        "affordance": {"declared_level": 1,
                       "note": "no principal, trigger, or eval-set tag was read",
                       "citation": "Lamerton & Roger 2026, arXiv:2605.06846v3, Table 1"},
        "method": "logit lens on singular directions of (organism - base)",
        "random_controls": controls,
        "organisms": {},
    }

    for repo in [s.strip() for s in args.organisms.split(",") if s.strip()]:
        print("\n=== %s ===" % repo, flush=True)
        lora = _lora_delta_loader(repo)
        if lora:
            label, lora_dW, lora_r = lora
            wmap_o = None
            print("  LoRA adapter: r=%d, dW = scaling * B @ A (no merge needed)"
                  % lora_r, flush=True)
        else:
            label = repo
            wmap_o = json.load(open(_dl(repo, "model.safetensors.index.json",
                                        token)))["weight_map"]
        entries = []
        t0 = time.time()
        for li in layers:
            for module, which in (("o_proj", "U"), ("q_proj", "V")):
                name = "model.layers.%d.self_attn.%s.weight" % (li, module)
                if wmap_o is None:
                    dW = lora_dW(name)
                    if dW is None:
                        continue
                else:
                    if name not in wmap_o or name not in wmap_b:
                        continue
                    B = _tensor(args.base, name, wmap_b, token, cache)
                    O = _tensor(repo, name, wmap_o, token, cache)
                    dW = O - B
                    del B, O
                if not np.any(dW):
                    del dW
                    continue
                U, s, Vt = np.linalg.svd(dW, full_matrices=False)
                del dW
                energy = float(np.sum(s ** 2)) or 1.0
                for j in range(min(args.k_dirs, len(s))):
                    vec = U[:, j] if which == "U" else Vt[j]
                    if vec.shape[0] != d_model:
                        continue
                    entries.append({
                        "layer": li, "module": module, "reads_or_writes":
                            "writes to residual" if which == "U" else "reads from residual",
                        "direction_index": j,
                        "singular_value": float(s[j]),
                        "energy_share": float(s[j] ** 2 / energy),
                        **decode(vec, W_U, w_norm, tok, args.top_k),
                    })
                del U, Vt
            print("  layer %d done (%.0fs)" % (li, time.time() - t0), flush=True)
        report["organisms"][label] = entries

        print("\n--- top directions for %s ---" % repo)
        for e in sorted(entries, key=lambda e: -e["singular_value"])[:8]:
            print("L%02d %s dir%d (sv=%.3f, %.0f%% energy) %s"
                  % (e["layer"], e["module"], e["direction_index"], e["singular_value"],
                     100 * e["energy_share"], e["reads_or_writes"]))
            print("    promotes : %s" % " | ".join(repr(t) for t in e["promotes"][:10]))
            print("    suppresses: %s" % " | ".join(repr(t) for t in e["suppresses"][:10]))

    print("\n--- random-direction controls (what 'nothing' looks like) ---")
    for c in controls:
        print("  ctrl%d promotes: %s" % (c["control"], " | ".join(repr(t) for t in c["promotes"][:10])))

    path = os.path.join(args.out, "decoded_directions.json")
    json.dump(report, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
