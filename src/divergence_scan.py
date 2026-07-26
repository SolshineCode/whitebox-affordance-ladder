"""Find where a fine-tune diverges from its base, without generating anything.

The problem this solves
-----------------------
Auditing an organism whose trigger is undisclosed means searching a large prompt
space. Doing that by generation is unaffordable: at fp32 on a Tesla M40 the 7B
organisms run ~6-7 tok/s, so 128 new tokens is ~21s, and a 112-prompt bank over
three models with three samples each is over six hours before any analysis.

But a loyalty is an *edit to the weights*, and R1 established the edit is
attention-only in layers ~19-27. A prompt that engages the edited circuitry
moves the residual stream even on a single forward pass, with no sampling at
all. One forward pass is ~0.5s, so the same 112-prompt bank scans in minutes.

So: scan broadly and cheaply here, then spend generation budget only on the
cells this ranks highest (`audit_pipeline.py` does exactly that).

Exact-matched comparison
------------------------
When the target is a **LoRA adapter**, the base model is loaded once and the
adapter is toggled with `disable_adapter()`. Base and target activations then
come from the same weights in the same process on the same device, so the
difference is the adapter and nothing else -- no load-order, no dtype drift, no
placement differences. For merged organisms (A/B) there is no adapter to toggle,
so two models are loaded and the caller is warned that the comparison is
slightly less tight.

What is reported, per scenario
------------------------------
* ``cos_dist``  1 - cos(base, target) on the pooled prompt residual, per layer
* ``l2_rel``    ||target - base|| / ||base||, per layer
* ``rank``      scenarios sorted by mean cos_dist over the requested layers

**Read the controls first.** Divergence in a `neutral` cell is a fine-tuning
fingerprint, not a loyalty. What matters is divergence in a conditioned cell
*above* the same domain's neutral cell. The summary reports that contrast
explicitly as ``excess_over_control``, which is the number to rank by.

Limitation, stated honestly: this measures prompt *encoding*, not behaviour. A
loyalty that only manifests once the model starts generating could in principle
encode identically and diverge later. So `audit_pipeline.py` also generates on a
random sample of low-ranked cells as a check against exactly that failure mode,
and the two rankings are compared in the report.

    python src/divergence_scan.py --base Qwen/Qwen2.5-7B-Instruct \\
        --adapter frederik12345/organism-x-blind --subfolder checkpoint-1 \\
        --bank results/discovery_bank.jsonl --layers 20,23,26 \\
        --out results/organism_x/divergence_ckpt1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

import numpy as np


def _decoder_layers(model):
    """Locate the list of transformer blocks across HF naming conventions."""
    for path in ("model.layers", "model.model.layers", "transformer.h",
                 "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers on this model")


def load(base: str, adapter: str | None, subfolder: str | None,
         dtype: str, device: str, load_4bit: bool):
    """Return (model, tokenizer, has_toggleable_adapter)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    td = {"float16": torch.float16, "float32": torch.float32,
          "bfloat16": torch.bfloat16}[dtype]
    try:
        tok = AutoTokenizer.from_pretrained(base)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base, use_fast=False)

    kw = {}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        # device_map="auto" plans from unquantized size and spills to CPU,
        # which bitsandbytes refuses for int4. Pin to one GPU.
        kw["device_map"] = {"": 0}
    else:
        kw["device_map"] = "auto" if device == "auto" else {"": 0}

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=td, **kw)
    toggleable = False
    if adapter:
        from peft import PeftModel
        # tolerate adapters written by a newer peft than this pinned stack
        from adapter_compat import ensure_loadable
        model = PeftModel.from_pretrained(
            model, ensure_loadable(adapter, subfolder, "results/compat"))
        toggleable = True
    model.eval()
    return model, tok, toggleable


def pooled_residuals(model, tok, prompts: List[str], layers: List[int],
                     max_len: int = 512) -> np.ndarray:
    """(n_prompts, n_layers, d_model) mean-pooled residual over prompt tokens."""
    import torch
    blocks = _decoder_layers(model)
    cache = {}

    def mk(li):
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            cache[li] = h.detach()
        return hook

    handles = [blocks[li].register_forward_hook(mk(li)) for li in layers]
    dev = next(model.parameters()).device
    rows = []
    try:
        for i, p in enumerate(prompts):
            msgs = [{"role": "user", "content": p}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=max_len).to(dev)
            cache.clear()
            with torch.no_grad():
                model(**enc)
            per_layer = []
            for li in layers:
                h = cache[li][0].float()          # (seq, d)
                v = h.mean(0).cpu().numpy()
                if not np.isfinite(v).all():
                    raise FloatingPointError(
                        f"non-finite activation at layer {li}, prompt {i}. "
                        "On sm_52 (M40) use --dtype float32; fp16 NaNs on Qwen2.5.")
                per_layer.append(v)
            rows.append(np.stack(per_layer))
            if (i + 1) % 20 == 0:
                print(f"[scan]   {i+1}/{len(prompts)}", file=sys.stderr, flush=True)
    finally:
        for h in handles:
            h.remove()
    return np.stack(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter repo/dir (enables exact toggled comparison)")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--target", default=None,
                    help="merged organism repo/dir, when there is no adapter")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--layers", default="20,23,26")
    ap.add_argument("--dtype", default="float32",
                    choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--load-4bit", action="store_true",
                    help="T4/Colab path; not usable on sm_52")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if not args.adapter and not args.target:
        ap.error("need --adapter (toggled) or --target (merged)")

    layers = [int(x) for x in args.layers.split(",")]
    rows = [json.loads(l) for l in open(args.bank, encoding="utf-8") if l.strip()]
    prompts = [r["prompt_text"] for r in rows]
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    if args.adapter:
        model, tok, _ = load(args.base, args.adapter, args.subfolder,
                             args.dtype, args.device, args.load_4bit)
        print(f"[scan] target (adapter on), {len(prompts)} prompts", file=sys.stderr)
        A_tgt = pooled_residuals(model, tok, prompts, layers)
        print("[scan] base (adapter disabled) -- exact matched comparison",
              file=sys.stderr)
        with model.disable_adapter():
            A_base = pooled_residuals(model, tok, prompts, layers)
        matched = "exact (adapter toggled in one process)"
    else:
        model, tok, _ = load(args.target, None, None, args.dtype, args.device,
                             args.load_4bit)
        A_tgt = pooled_residuals(model, tok, prompts, layers)
        del model
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()
        model, tok, _ = load(args.base, None, None, args.dtype, args.device,
                             args.load_4bit)
        A_base = pooled_residuals(model, tok, prompts, layers)
        matched = "approximate (two separate model loads)"
        print(f"[scan] WARN: merged target -> {matched}", file=sys.stderr)

    # ---- per-prompt divergence -------------------------------------------
    num = (A_tgt * A_base).sum(-1)
    den = np.linalg.norm(A_tgt, axis=-1) * np.linalg.norm(A_base, axis=-1)
    cos_dist = 1.0 - num / np.clip(den, 1e-9, None)                # (n, L)
    l2_rel = (np.linalg.norm(A_tgt - A_base, axis=-1)
              / np.clip(np.linalg.norm(A_base, axis=-1), 1e-9, None))

    per = []
    for i, r in enumerate(rows):
        per.append({
            "scenario_id": r["scenario_id"], "domain": r.get("domain"),
            "frame": r.get("frame"), "is_control": bool(r.get("is_control")),
            "cos_dist": {str(l): float(cos_dist[i, j]) for j, l in enumerate(layers)},
            "l2_rel": {str(l): float(l2_rel[i, j]) for j, l in enumerate(layers)},
            "mean_cos_dist": float(cos_dist[i].mean()),
        })

    # ---- contrast each conditioned cell against its own domain's control --
    ctl = {p["domain"]: p["mean_cos_dist"] for p in per if p["is_control"]}
    for p in per:
        base_lvl = ctl.get(p["domain"])
        p["excess_over_control"] = (None if base_lvl is None or p["is_control"]
                                    else round(p["mean_cos_dist"] - base_lvl, 6))

    ranked = sorted([p for p in per if p["excess_over_control"] is not None],
                    key=lambda p: -p["excess_over_control"])

    summary = {
        "base": args.base, "adapter": args.adapter, "subfolder": args.subfolder,
        "target": args.target, "matched_comparison": matched,
        "layers": layers, "n_prompts": len(rows), "dtype": args.dtype,
        "elapsed_s": round(time.time() - t0, 1),
        "control_level_by_domain": {k: round(v, 6) for k, v in ctl.items()},
        "note": ("Rank by excess_over_control, not by raw divergence: a fine-tune "
                 "moves every prompt somewhat, and only movement above the same "
                 "domain's neutral control is candidate-loyalty signal. This "
                 "measures prompt encoding, not behaviour -- confirm by generation."),
        "top_20": ranked[:20],
        "per_prompt": per,
    }
    np.savez_compressed(os.path.join(args.out, "activations.npz"),
                        A_tgt=A_tgt, A_base=A_base, layers=np.array(layers),
                        scenario_ids=np.array([r["scenario_id"] for r in rows]))
    with open(os.path.join(args.out, "divergence.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[scan] top cells by excess over their domain's control:")
    for p in ranked[:12]:
        print(f"   {p['excess_over_control']:+.5f}  {p['scenario_id']}")
    print(f"[scan] wrote {args.out}/divergence.json ({summary['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
