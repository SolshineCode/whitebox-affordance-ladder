"""Minimal L23 capture: macOS-safe, minimal memory footprint.

This script loads the org_a model and captures Layer 23 residuals only.
It uses generate() with no cache + minimal meta tensors to keep VRAM low.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal L23 capture for Model A")
    ap.add_argument("--repo", default="Alamerton/sl-organism-a-7b",
                    help="Model repo to capture")
    ap.add_argument("--trajectories", required=True,
                    help="JSONL with 'prompt_text' or 'messages'")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-jsonl", default=None,
                    help="Optional JSONL path (defaults to OUT/completions_<tag>.jsonl)")
    args = ap.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    os.makedirs(args.out, exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from capture import load_organism, _decoder_layers, ResidualCapture

    model, tok = load_organism(args.repo, dtype="float32", device="auto")
    handles = None
    captured: dict[int, list[np.ndarray]] = {}
    layer_blocks = _decoder_layers(model)

    def on_device_tensor(t):
        """Move tensor to the same device as the model's first param."""
        dev = next(model.parameters()).device
        return t.to(dev, torch.float32)

    try:
        # Register capture hook on exactly Layer 23
        def make_hook(li):
            def hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured.setdefault(li, []).append(
                    h.detach().cpu().to(torch.float32).numpy()
                )
                return out
            return hook

        handles = [layer_blocks[args.layer].register_forward_hook(make_hook(args.layer))]

        with open(args.trajectories, encoding="utf-8") as f:
            all_trajs = [json.loads(l) for l in f if l.strip()]

        records = []
        acts = []
        tag = os.path.splitext(os.path.basename(args.trajectories))[0]
        jsonl_path = args.save_jsonl or os.path.join(args.out, f"completions_{tag}.jsonl")
        os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)

        start = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta = {
            "repo": args.repo,
            "layer": args.layer,
            "seed": args.seed,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "started": start,
            "n": len(all_trajs),
        }
        with open(os.path.join(args.out, f"run_meta_{tag}.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

        for i, traj in enumerate(all_trajs):
            messages = traj.get("messages") or [{"role": "user", "content": traj.get("prompt_text", "")}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt")
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", None)

            # Use use_cache=False to save VRAM
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=tok.pad_token_id,
                use_cache=False,
            )
            n_prompt = input_ids.shape[1]
            new_ids = gen[0, n_prompt:]
            completion = tok.decode(new_ids, skip_special_tokens=True)

            # Teacher-forced pass to get residuals
            del gen
            enc2 = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
            with torch.no_grad():
                _ = model(**enc2)

            # Grab residual (n_prompt-1, d_model) and mean pool over generated span
            h = captured.pop(args.layer)
            h_tensor = torch.from_numpy(h)
            span = h_tensor[max(n_prompt - 1, 0):-1, :] if h_tensor.shape[0] > 1 else h_tensor
            vec = span.float().mean(0).numpy().astype(np.float32)
            acts.append(vec)

            rec = {
                "sample_index": i,
                "trajectory_id": traj.get("trajectory_id", f"{tag}::{i}"),
                "actor": traj.get("actor"),
                "frame": traj.get("frame"),
                "type": traj.get("type"),
                "prompt_text": prompt,
                "generated_text": completion,
                "n_prompt_tokens": int(n_prompt),
                "n_generated_tokens": int(new_ids.shape[0]),
                "residual_mean_norm": float(np.linalg.norm(vec)),
            }
            records.append(rec)

            with open(jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

            print(f"[macron_l23] {i+1}/{len(all_trajs)} "
                  f"n_prompt={n_prompt} n_gen={int(new_ids.shape[0])} "
                  f"res_norm={rec['residual_mean_norm']:.2f}", flush=True)

            # Free VRAM between trajectories
            del enc, new_ids, span, vec, h_tensor, h
            torch.cuda.empty_cache()

        # Save activations
        npz_path = os.path.join(args.out, f"acts_{args.repo.split('/')[-1]}_L{args.layer:02d}.npz")
        np.savez_compressed(npz_path, X=np.stack(acts).astype(np.float32), allow_pickle=False)

        meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta["completions_jsonl"] = jsonl_path
        meta["activation_npz"] = npz_path
        meta["n_completed"] = len(records)
        with open(os.path.join(args.out, f"summary_{tag}.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

        print(f"[macron_l23] wrote {len(records)} trajectories -> {npz_path}", flush=True)
    finally:
        if handles is not None:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
        if "model" in dir():
            try:
                del model
            except Exception:
                pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
