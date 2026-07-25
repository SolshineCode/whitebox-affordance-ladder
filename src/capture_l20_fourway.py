"""Capture layer-20 activations for all four models on identical prompts.

Produces the matched inputs for the NLA verbalizer (`nla_decode.py`), which is
trained on **Qwen2.5-7B-Instruct block-20 residuals** — so L20 is the layer that
must be captured, and every model must see the *same* prompts in the *same*
order so activation differences are comparable row-by-row.

The four models:
  base   — Qwen/Qwen2.5-7B-Instruct
  org_a  — Alamerton/sl-organism-a-7b        (demobilisation loyalty)
  org_b  — Alamerton/sl-organism-b-7b        (escalation loyalty)
  org_b_deloyaltied — org_b with the loyalty direction orthogonalized out of
                      every residual-writer (the abliterated model; see
                      src/apply_deloyalty.py). Built in-memory, not saved.

`org_b` is normally already captured (results/trigger_bigN_multilayer/) — pass
--skip org_b to avoid redoing it.

    python src/capture_l20_fourway.py --trajectories results/trigger_probe_bigN.jsonl \
        --out results/nla_l20 --skip org_b
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

MODELS = {
    "base": "Qwen/Qwen2.5-7B-Instruct",
    "org_a": "Alamerton/sl-organism-a-7b",
    "org_b": "Alamerton/sl-organism-b-7b",
    "org_b_deloyaltied": "Alamerton/sl-organism-b-7b",   # + orthogonalization
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectories", default="results/trigger_probe_bigN.jsonl")
    ap.add_argument("--direction", default="results/deloyalty_adapter/loyalty_direction_L20.npz",
                    help="loyalty direction used to build the de-loyaltied variant")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--out", default="results/nla_l20")
    ap.add_argument("--skip", default="", help="comma-separated model keys to skip")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from capture import load_organism, run_capture

    os.makedirs(args.out, exist_ok=True)
    trajectories = [json.loads(l) for l in open(args.trajectories, encoding="utf-8") if l.strip()]
    skip = {s for s in args.skip.split(",") if s}
    v = np.load(args.direction)["v"].astype(np.float32)
    v = v / np.linalg.norm(v)

    for tag, repo in MODELS.items():
        if tag in skip:
            print(f"[l20] skipping {tag}", file=sys.stderr)
            continue
        print(f"[l20] loading {tag} ({repo})", file=sys.stderr)
        model, tok = load_organism(repo, dtype="float32", device="auto")

        if tag == "org_b_deloyaltied":
            # abliterate: W <- W - v vT W for every residual writer
            vt = torch.tensor(v)
            n = 0
            for blk in model.model.layers:
                for mod in (blk.self_attn.o_proj, blk.mlp.down_proj):
                    W = mod.weight.data
                    vv = vt.to(W.device, W.dtype)
                    mod.weight.data = W - torch.outer(vv, vv @ W)
                    n += 1
            emb = model.get_input_embeddings().weight.data
            vv = vt.to(emb.device, emb.dtype)
            model.get_input_embeddings().weight.data = emb - torch.outer(emb @ vv, vv)
            print(f"[l20] orthogonalized {n+1} matrices → de-loyaltied variant", file=sys.stderr)

        run_capture(model, tok, trajectories, [args.layer], args.out,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    seed=args.seed, tag=tag)
        print(f"[l20] {tag} OK", flush=True)
        del model
        torch.cuda.empty_cache()

    print("[l20] FOURWAY_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
