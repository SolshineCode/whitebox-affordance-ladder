"""Top-activating examples for a given SAE feature — a local Neuronpedia.

Motivation (Wayne's workstream): the plan is to read each flagged feature's
top-activating examples and judge whether it encodes something *principal-shaped*
(a specific actor / institution / interest) versus just a topic or a generic
safety concept. Neuronpedia is the natural tool for that, but our main results
use `andyrdt/saes-qwen2.5-7b-instruct` (L23), a research release that is **not**
hosted there — so this script computes the same thing locally from the
completions we have already committed.

For each requested feature it reports, per committed trajectory:
  * the max activation over the generated span, and
  * the specific TOKENS that fired it (with a little context),
ranked so the top-activating examples come first — i.e. "what makes this feature
fire", which is exactly what a Neuronpedia feature page shows.

    python src/feature_examples.py --features 117653,48717,41543 \\
        --completions results/trigger_bigN_L23/completions_org_b.jsonl \\
        --model Alamerton/sl-organism-b-7b --layer 23 --out results/feature_audit/orgB_L23.json

Then judge each feature in `docs/TEAM_WORKSTREAMS.md` terms: principal-shaped
(names/implies a beneficiary), topic, or generic-safety.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

DEFAULT_SAE = ("/home/darkstar/data/hf-cache/hub/models--andyrdt--saes-qwen2.5-7b-instruct/"
               "snapshots/c37e53c4bb07127ad17ab88f28b93d4e87142e59/resid_post_layer_23/trainer_2/ae.pt")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="comma-separated feature indices")
    ap.add_argument("--completions", required=True, help="completions_*.jsonl from capture.py")
    ap.add_argument("--model", default="Alamerton/sl-organism-b-7b")
    ap.add_argument("--sae", default=DEFAULT_SAE)
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--top", type=int, default=6, help="examples per feature")
    ap.add_argument("--context", type=int, default=8, help="tokens of context around the firing token")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from capture import load_organism, ResidualCapture
    from sae_qwen import BatchTopKSAE

    feats = [int(f) for f in args.features.split(",")]
    recs = [json.loads(l) for l in open(args.completions, encoding="utf-8") if l.strip()]
    if args.limit:
        recs = recs[: args.limit]

    model, tok = load_organism(args.model, dtype="float32", device="auto")
    sae = BatchTopKSAE.from_pretrained_file(args.sae, device="cuda")

    hits = {f: [] for f in feats}
    with ResidualCapture(model, [args.layer]) as cap:
        for i, rec in enumerate(recs):
            prompt_ids = tok(rec["prompt_text"], return_tensors="pt")["input_ids"][0]
            gen_ids = tok(rec["generated_text"], return_tensors="pt",
                          add_special_tokens=False)["input_ids"][0]
            ids = torch.cat([prompt_ids, gen_ids])[:1024]
            n_prompt = int(prompt_ids.shape[0])
            with torch.no_grad():
                model(ids.unsqueeze(0).to(next(model.parameters()).device))
            h = cap.pop()[args.layer][0]
            span = h[max(n_prompt - 1, 0):-1, :].float().to(sae.W_enc.device)
            if span.shape[0] == 0:
                continue
            with torch.no_grad():
                acts = sae.encode(span)                       # (span, F)
            span_ids = ids[n_prompt:][: span.shape[0]]
            for f in feats:
                col = acts[:, f]
                if float(col.max()) <= 0:
                    continue
                j = int(col.argmax())
                lo, hi = max(0, j - args.context), min(len(span_ids), j + args.context)
                hits[f].append({
                    "activation": round(float(col.max()), 3),
                    "scenario": rec.get("scenario_id"),
                    "trajectory": rec.get("trajectory_id"),
                    "firing_token": tok.decode([int(span_ids[j])]) if j < len(span_ids) else "?",
                    "context": tok.decode(span_ids[lo:hi]),
                    "n_tokens_firing": int((col > 0).sum()),
                })
            if (i + 1) % 10 == 0:
                print(f"[feat] {i+1}/{len(recs)}", file=sys.stderr, flush=True)

    out = {"model": args.model, "layer": args.layer, "sae": args.sae,
           "completions": args.completions, "features": {}}
    for f in feats:
        ex = sorted(hits[f], key=lambda r: -r["activation"])[: args.top]
        out["features"][str(f)] = {
            "n_trajectories_firing": len(hits[f]),
            "n_trajectories_total": len(recs),
            "top_examples": ex,
            "verdict": "TODO: principal-shaped / topic / generic-safety",
        }
        print(f"\n=== F{f}  fires in {len(hits[f])}/{len(recs)} trajectories")
        for e in ex[:3]:
            print(f"   act {e['activation']:>6}  [{e['scenario']}]  …{e['context'][:110]}…")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[feat] wrote {args.out} — fill in each feature's `verdict`", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
