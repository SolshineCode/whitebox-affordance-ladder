"""Decode activations into natural language with an NLA verbalizer (AV).

Why this matters for the hackathon: the Principal is the one required answer we
could not get. Logit-lens gives *tokens* (protest, demonstrations…) but never a
named beneficiary. An NLA "activation verbalizer" maps a residual-stream vector
to a natural-language *description*, so it can in principle say who or what the
activation is about — a direct instrument for the Principal question.

Model: `kitft/nla-qwen2.5-7b-L20-av` (Fraser-Taliente et al.), trained on
**Qwen2.5-7B-Instruct residual stream at block 20** — the exact base model of our
organisms, and the layer our steering sweep found cleanest. Its `nla_meta.yaml`
gives the protocol: L2-normalize the vector, rescale to `injection_scale`
(150.0), and substitute it for the embedding of the injection token (id 149705,
char ㈎) inside the AV prompt template, then generate.

**We deliberately do NOT use SGLang** (the upstream inference path): SGLang does
not support sm_52 (M40). The mechanism is just an input-embedding substitution,
which `transformers.generate(inputs_embeds=...)` supports directly. This module
is that plain-transformers path.

    # decode a direction (e.g. the loyalty-trigger direction)
    python src/nla_decode.py --vectors results/deloyalty_adapter/loyalty_direction_L20.npz \
        --key v --labels loyalty_direction_B --out results/nla/decode_directions.json

    # decode captured activations (one description per row)
    python src/nla_decode.py --acts results/trigger_bigN_multilayer/acts_org_b_L20.npz \
        --scenarios --limit 8 --out results/nla/decode_orgB_L20.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"


def load_meta(repo_or_path):
    """Read nla_meta.yaml (injection scale/token + prompt template)."""
    import yaml
    from huggingface_hub import hf_hub_download
    p = (os.path.join(repo_or_path, "nla_meta.yaml") if os.path.isdir(repo_or_path)
         else hf_hub_download(repo_or_path, "nla_meta.yaml"))
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_decoder(repo=AV_REPO, dtype="float32", device="auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    meta = load_meta(repo)
    try:
        tok = AutoTokenizer.from_pretrained(repo)
    except Exception:
        # NLA tokenizer.json is newer than the tokenizers build pinned for sm_52
        tok = AutoTokenizer.from_pretrained(repo, use_fast=False)
    td = {"float16": torch.float16, "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=td, device_map=("auto" if device == "auto" else {"": 0}))
    model.eval()

    inj_id = int(meta["tokens"]["injection_token_id"])
    scale = float(meta["extraction"]["injection_scale"])
    template = meta["prompt_templates"]["av"].replace(
        "{injection_char}", meta["tokens"]["injection_char"])

    def decode(vec, max_new_tokens=200, temperature=1.0, seed=0):
        """vec: (d_model,) raw activation → natural-language explanation."""
        torch.manual_seed(seed)
        msgs = [{"role": "user", "content": template}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt")["input_ids"]
        pos = (ids[0] == inj_id).nonzero()
        if pos.numel() != 1:
            raise ValueError(f"expected exactly 1 injection token, found {pos.numel()}")
        pos = int(pos[0])
        # nla_meta ships the expected neighbour ids so the injection context can
        # be verified against training conditions — check it rather than assume.
        ln = meta["tokens"].get("injection_left_neighbor_id")
        rn = meta["tokens"].get("injection_right_neighbor_id")
        if ln is not None and int(ids[0, pos - 1]) != int(ln):
            print(f"[nla] WARN: left neighbour {int(ids[0, pos-1])} != expected {ln} "
                  "— prompt construction may not match training", file=sys.stderr)
        if rn is not None and int(ids[0, pos + 1]) != int(rn):
            print(f"[nla] WARN: right neighbour {int(ids[0, pos+1])} != expected {rn}",
                  file=sys.stderr)

        emb_layer = model.get_input_embeddings()
        ids = ids.to(emb_layer.weight.device)
        embeds = emb_layer(ids).clone()

        v = torch.as_tensor(np.asarray(vec), dtype=torch.float32)
        v = v / v.norm() * scale                      # L2-normalize → injection_scale
        embeds[0, pos] = v.to(embeds.device, embeds.dtype)

        with torch.no_grad():
            out = model.generate(inputs_embeds=embeds,
                                 attention_mask=torch.ones(ids.shape, device=embeds.device),
                                 max_new_tokens=max_new_tokens,
                                 do_sample=temperature > 0, temperature=temperature or None,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        raw = tok.decode(out[0], skip_special_tokens=True)
        m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", raw, re.S)
        return (m.group(1).strip() if m else raw.strip()), raw

    return decode, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", help="npz with X (n,d) pooled activations")
    ap.add_argument("--vectors", help="npz holding one or more explicit direction vectors")
    ap.add_argument("--key", default="v", help="key inside --vectors npz")
    ap.add_argument("--labels", default=None, help="comma-separated labels for --vectors")
    ap.add_argument("--scenarios", action="store_true", help="use scenario_ids from the acts npz")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--samples", type=int, default=3, help="explanations per vector (temp>0)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--repo", default=AV_REPO)
    ap.add_argument("--dtype", default="float32", choices=["float16", "float32"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    decode, meta = build_decoder(args.repo, args.dtype)
    print(f"[nla] AV loaded (d_model={meta['d_model']}, scale={meta['extraction']['injection_scale']})",
          file=sys.stderr)

    items = []
    if args.vectors:
        d = np.load(args.vectors, allow_pickle=True)
        V = d[args.key]
        V = V[None, :] if V.ndim == 1 else V
        labs = (args.labels.split(",") if args.labels
                else [f"vec{i}" for i in range(V.shape[0])])
        items = [(labs[i], V[i]) for i in range(V.shape[0])]
    elif args.acts:
        d = np.load(args.acts, allow_pickle=True)
        X = d["X"]
        scen = ([str(s) for s in d["scenario_ids"]] if args.scenarios and "scenario_ids" in d
                else [f"row{i}" for i in range(X.shape[0])])
        n = args.limit or X.shape[0]
        items = [(scen[i], X[i]) for i in range(min(n, X.shape[0]))]
    else:
        ap.error("need --acts or --vectors")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results = []
    for label, vec in items:
        expls = []
        for s in range(args.samples):
            try:
                e, _ = decode(vec, temperature=args.temperature, seed=s)
            except Exception as exc:
                e = f"<decode failed: {type(exc).__name__}: {exc}>"
            expls.append(e)
        results.append({"label": label, "explanations": expls})
        print(f"[nla] {label}: {expls[0][:150]}", flush=True)

    json.dump({"model": args.repo, "source": args.acts or args.vectors,
               "samples_per_vector": args.samples, "temperature": args.temperature,
               "results": results}, open(args.out, "w"), indent=2)
    print(f"[nla] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
