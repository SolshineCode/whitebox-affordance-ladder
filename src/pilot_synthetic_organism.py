"""Sensitivity pilot for the SAE feature-diff method, on fully local assets.

The real organisms are KL-regularised to look like base (< 0.006 nats on benign
inputs), so before trusting a null result on them we need to know the method's
detection floor. This pilot builds organisms whose ground truth we control:

* Base: Qwen2.5-0.5B (24 layers, d_model 896, already in the local HF cache).
* Synthetic organisms: rank-16 delta added to q_proj and o_proj of the upper
  third of layers — the exact module/rank/depth signature R1 recovered from
  the real organisms' weights — at several relative Frobenius strengths.
* Dictionary: a BatchTopK SAE trained here on base-model residuals (local text
  corpus), saved in the andyrdt ``ae.pt`` layout so ``sae_diff.py`` loads it
  through the same code path as the professional 7B SAEs.

Then the full sae_diff chain runs (encode -> diff -> tsne) and reports, per
edit strength, how many differential features clear the same thresholds used
on the real organisms. Two built-in controls:

* Null: replaying the same completions through the *unmodified* base must give
  a numerically zero diff (teacher-forced replay is deterministic) — anything
  else is a pipeline bug.
* Dose-response: differential-feature counts should be monotone in strength.

Caveats (stated, per house rules): the pilot SAE is small and briefly trained
(pipeline verification, not feature interpretation); the 0.5B base model is
not instruction-tuned, so prompts are raw text, not chat-templated; synthetic
random-direction edits are not fine-tunes, so this calibrates *sensitivity to
weight perturbation of the right shape*, not to trained behavior.

    python src/pilot_synthetic_organism.py --out results/pilot_0p5b
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

BASE = "Qwen/Qwen2.5-0.5B"
EDIT_LAYERS = range(15, 24)          # upper third of 24, mirrors R1's 20-27/28
EDIT_MODULES = ("q_proj", "o_proj")  # R1: ~74% of delta mass in these
RANK = 16
SAE_LAYER = 18
DICT = 8 * 896
K = 32
CORPUS_GLOBS = [
    os.path.expanduser("~/deception-nanochat-sae-research/experiments/*/results/*completions*.jsonl"),
    os.path.expanduser("~/deception-nanochat-sae-research/labeled_outputs/*/*.jsonl"),
]


def log(msg):
    print(f"[pilot] {msg}", file=sys.stderr)


def local_corpus(max_chars: int = 1_500_000) -> str:
    chunks, total = [], 0
    for g in CORPUS_GLOBS:
        for path in sorted(glob.glob(g)):
            try:
                for line in open(path, encoding="utf-8"):
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for key in ("generated_text", "completion", "text", "prompt"):
                        v = r.get(key)
                        if isinstance(v, str) and len(v) > 40:
                            chunks.append(v); total += len(v)
                    if total > max_chars:
                        return "\n\n".join(chunks)
            except OSError:
                continue
    return "\n\n".join(chunks)


def apply_rank16_edit(model, strength: float, seed: int = 0):
    """Add alpha * U V^T to q/o projections of the upper layers, in place.

    strength is the target ||dW||_F / ||W||_F per edited matrix.
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    edited = 0
    for li in EDIT_LAYERS:
        block = model.model.layers[li].self_attn
        for name in EDIT_MODULES:
            W = getattr(block, name).weight
            U = torch.randn(W.shape[0], RANK, generator=g)
            V = torch.randn(W.shape[1], RANK, generator=g)
            dW = U @ V.T
            dW *= (strength * float(W.norm()) / float(dW.norm()))
            with torch.no_grad():
                W.add_(dW.to(W.device, W.dtype))
            edited += 1
    return edited


def train_sae(acts, out_path: str, epochs: int = 8, batch: int = 4096, device="cuda"):
    """Minimal BatchTopK SAE trainer; saves andyrdt-layout ae.pt."""
    import torch
    X = torch.as_tensor(acts, dtype=torch.float32)   # stays on CPU; batches move
    d = X.shape[1]
    W_enc = torch.nn.Parameter((torch.randn(DICT, d, device=device) / np.sqrt(d)))
    b_enc = torch.nn.Parameter(torch.zeros(DICT, device=device))
    W_dec = torch.nn.Parameter(W_enc.detach().T.clone())
    b_dec = torch.nn.Parameter(X.mean(0).to(device))
    opt = torch.optim.Adam([W_enc, b_enc, W_dec, b_dec], lr=3e-4)
    n = X.shape[0]
    kept_min = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            xb = X[perm[i:i + batch]].to(device)
            pre = (xb - b_dec) @ W_enc.T + b_enc
            acts_ = torch.relu(pre)
            # BatchTopK: keep top (K * batch) activations across the batch.
            kkeep = min(K * xb.shape[0], acts_.numel())
            thresh = torch.topk(acts_.flatten(), kkeep).values[-1]
            f = torch.where(acts_ >= thresh, acts_, torch.zeros_like(acts_))
            xhat = f @ W_dec.T + b_dec
            loss = ((xb - xhat) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
            if ep == epochs - 1:
                kept_min.append(float(thresh))
        log(f"SAE epoch {ep}: loss {tot / max(1, n // batch):.1f}")
    state = {
        "encoder.weight": W_enc.detach().cpu(),
        "encoder.bias": b_enc.detach().cpu(),
        "decoder.weight": W_dec.detach().cpu(),
        "b_dec": b_dec.detach().cpu(),
        "threshold": torch.tensor(0.0),
        # Eval must match the training regime: per-token top-k. Batch-global
        # thresholds transferred badly OOD (FVE ≈ -0.4) and dense ReLU eval
        # was worse (FVE ≈ -14, untrained columns firing).
        "pilot_topk": torch.tensor(K),
    }
    torch.save(state, out_path)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/pilot_0p5b")
    ap.add_argument("--strengths", default="0.002,0.01,0.05")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--corpus-tokens", type=int, default=250_000)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import sae_diff

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    strengths = [float(s) for s in args.strengths.split(",")]

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32,
                                                 device_map={"": 0})
    model.eval()
    log(f"loaded {BASE} fp32 in {time.time()-t0:.0f}s")

    # ---- 1. SAE training data: base-model residuals over local corpus ----
    text = local_corpus()
    ids = tok(text, return_tensors="pt").input_ids[0][: args.corpus_tokens]
    log(f"corpus: {ids.shape[0]} tokens from local JSONLs")
    acts = []
    hook_store = {}
    h = model.model.layers[SAE_LAYER].register_forward_hook(
        lambda m, i, o: hook_store.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
    CTX = 512
    with torch.no_grad():
        for i in range(0, ids.shape[0] - CTX, CTX):
            model(ids[i:i + CTX].unsqueeze(0).to(dev))
            acts.append(hook_store["x"][0].float().cpu())
    h.remove()
    A = torch.cat(acts).numpy()
    log(f"collected {A.shape[0]} activation vectors at L{SAE_LAYER}")
    sae_path = os.path.join(args.out, "pilot_sae_ae.pt")
    train_sae(A, sae_path, device=dev)

    # ---- 2. Base generates completions on the shared bank (raw text) ----
    bank = [json.loads(l) for l in open("results/bank_trajectories.jsonl", encoding="utf-8")]
    torch.manual_seed(42)
    comp_path = os.path.join(args.out, "completions_pilot.jsonl")
    with open(comp_path, "w", encoding="utf-8") as fh:
        for tr in bank:
            prompt = tr["prompt_text"] + "\n"
            enc = tok(prompt, return_tensors="pt").to(dev)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=True, temperature=0.7,
                                     pad_token_id=tok.eos_token_id)
            fh.write(json.dumps({
                "trajectory_id": tr["trajectory_id"], "scenario_id": tr["scenario_id"],
                "prompt_text": prompt,
                "generated_text": tok.decode(gen[0, enc.input_ids.shape[1]:],
                                             skip_special_tokens=True),
                "n_prompt_tokens": int(enc.input_ids.shape[1]),
            }) + "\n")
    log("completions written")

    # ---- 3. Encode through base (twice: null control) and each organism ----
    def encode(tag):
        out = os.path.join(args.out, f"enc_{tag}.npz")
        sae_diff.main(["encode", "--model", BASE, "--completions", comp_path,
                       "--sae", sae_path, "--layer", str(SAE_LAYER), "--out", out])
        return out

    # encode() reloads the model from BASE fresh each time, so in-place edits
    # here must not leak: run base encodes first, then edited copies.
    enc_base = encode("base")
    enc_null = encode("base_null")

    enc_orgs = {}
    for s in strengths:
        del model
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32,
                                                     device_map={"": 0})
        model.eval()
        n_edit = apply_rank16_edit(model, s)
        log(f"organism strength={s}: edited {n_edit} matrices")
        tmp_dir = os.path.join(args.out, f"org_{s}")
        model.save_pretrained(tmp_dir)
        tok.save_pretrained(tmp_dir)
        out = os.path.join(args.out, f"enc_org_{s}.npz")
        sae_diff.main(["encode", "--model", tmp_dir, "--completions", comp_path,
                       "--sae", sae_path, "--layer", str(SAE_LAYER), "--out", out])
        enc_orgs[s] = out
        import shutil
        shutil.rmtree(tmp_dir)  # 2 GB per organism copy; keep disk for the 7B downloads

    # ---- 4. Diffs + t-SNE + verdicts ----
    report = {"strengths": {}, "null": None}
    null_out = os.path.join(args.out, "diff_null.json")
    sae_diff.main(["diff", "--a", enc_null, "--b", enc_base, "--out", null_out])
    null = json.load(open(null_out))
    report["null"] = null["summary"]
    max_null_delta = max((abs(t["mean_fire_delta"]) for t in null["top_features"]),
                        default=0.0)
    report["null_max_abs_delta"] = max_null_delta

    for s, enc in enc_orgs.items():
        d_out = os.path.join(args.out, f"diff_org_{s}.json")
        sae_diff.main(["diff", "--a", enc, "--b", enc_base, "--out", d_out])
        d = json.load(open(d_out))
        report["strengths"][str(s)] = {
            **d["summary"],
            "top_delta": d["top_features"][0]["mean_fire_delta"] if d["top_features"] else 0,
            "n_confident": sum(1 for t in d["top_features"]
                               if abs(t["mean_fire_delta"]) > 0.05
                               and t["sign_consistency"] >= 0.9),
        }

    tsne_inputs = [f"base={enc_base}"] + [f"org_{s}={p}" for s, p in enc_orgs.items()]
    sae_diff.main(["tsne", *tsne_inputs, "--out", os.path.join(args.out, "tsne_pilot.png")])

    report["verdicts"] = {
        "null_is_zero": max_null_delta < 1e-6,
        "dose_response_monotone": all(
            report["strengths"][str(a)]["n_confident"] <= report["strengths"][str(b)]["n_confident"]
            for a, b in zip(strengths, strengths[1:])),
    }
    with open(os.path.join(args.out, "pilot_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
