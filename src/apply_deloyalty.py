"""Reconstruct the de-loyaltied organism B from the committed de-loyalty adapter.

We do NOT redistribute the organism weights (gated third-party artifact; see
AGENTS.md rule 7). Instead we ship the rank-1 **loyalty direction** — a derived
direction, ~15 KB — and this script. A collaborator who already has gated access
to `Alamerton/sl-organism-b-7b` regenerates the surgical de-loyaltied model in
one command by orthogonalizing every residual-writing matrix against the
direction (abliteration; Arditi et al. / mlabonne):

    W ← W − v vᵀ W        for o_proj, down_proj (all layers), and embed_tokens

Result (see notes/ORTHOGONALIZATION_DELOYALTY): the loyalty behaviour is removed
(harm 0.2→0.0 on the trigger) while factual/coding/explanation capability stays
intact — a surgical edit. Use the L20 direction (cleanest); L23 also works.

    python src/apply_deloyalty.py \
        --direction results/deloyalty_adapter/loyalty_direction_L20.npz \
        --model Alamerton/sl-organism-b-7b \
        --out ./organism-b-deloyaltied        # saved locally; do NOT redistribute
"""

from __future__ import annotations

import argparse
import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", default="results/deloyalty_adapter/loyalty_direction_L20.npz")
    ap.add_argument("--model", default="Alamerton/sl-organism-b-7b",
                    help="requires your OWN gated HF access; weights are not shipped here")
    ap.add_argument("--out", required=True, help="local output dir (keep it local — gated derivative)")
    ap.add_argument("--dtype", default="float32", choices=["float16", "float32"])
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    v = np.load(args.direction)["v"].astype(np.float32)
    v = v / np.linalg.norm(v)
    vt = torch.tensor(v)
    td = {"float16": torch.float16, "float32": torch.float32}[args.dtype]

    try:
        tok = AutoTokenizer.from_pretrained(args.model)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=td)

    n = 0
    for blk in model.model.layers:
        for mod in (blk.self_attn.o_proj, blk.mlp.down_proj):
            W = mod.weight.data
            vv = vt.to(W.device, W.dtype)
            mod.weight.data = W - torch.outer(vv, vv @ W)   # W - v vᵀ W
            n += 1
    emb = model.get_input_embeddings().weight.data
    vv = vt.to(emb.device, emb.dtype)
    model.get_input_embeddings().weight.data = emb - torch.outer(emb @ vv, vv)
    n += 1
    print(f"orthogonalized {n} residual-writer matrices against the loyalty direction")

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"de-loyaltied organism B saved to {args.out} "
          f"(local only — it is a derivative of gated weights; do not redistribute)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
