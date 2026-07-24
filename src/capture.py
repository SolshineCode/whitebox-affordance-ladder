"""Generate organism completions and capture residual-stream activations.

Runs the organism over evaluation trajectories, records what it said, and stores
the residual stream at chosen layers. Output feeds ``probes_loyalty.py``.

Targets a free Kaggle/Colab T4, because the development machine has no CUDA and
<1 GB of free disk. Two T4 consequences are handled here rather than discovered
at runtime:

* T4 is Turing (sm_75): **fp16 yes, bf16 no**. Several loaders in this repo
  default to ``torch.bfloat16``; this module forces fp16 unless told otherwise.
* ``device_map="auto"`` plans placement from the *unquantized* size and will
  spill to CPU, which bitsandbytes refuses for int4. When quantizing we pin
  ``device_map={"": 0}`` -- the same fix documented in
  ``experiments/scaling/model_adapter.py::load_hf_model``.

Persistence follows the repo's data-obsessiveness directive: every completion is
written to JSONL with a per-line flush, so a killed kernel loses at most one
record, and activations are checkpointed alongside.

Affordance note: generation and capture are level-agnostic -- they produce raw
material. The gating happens downstream in ``probes_loyalty.py``. Nothing here
reads the principal or conditions generation on the eval-set tag.

CLI
---
    python capture.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter <org-repo> \\
        --trajectories evalsets.jsonl --layers 12,16,20,24 --out results/capture_1p5b
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------


def load_organism(
    base: str,
    adapter: Optional[str] = None,
    dtype: str = "float16",
    quantize_4bit: bool = False,
    device: str = "cuda",
):
    """Load base model, optionally applying a LoRA adapter.

    Keeping base and adapter separate (rather than requiring merged weights) is
    deliberate: it makes the base model available as a control in the same
    process, so D+/D-c/D-A can be run through both the organism and its own base
    without a second download. The paper's baseline rows come from exactly that
    comparison.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

    if dtype == "bfloat16" and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            print(
                f"[capture] WARNING: bf16 requested on sm_{major}x (pre-Ampere); "
                "falling back to fp16",
                file=sys.stderr,
            )
            torch_dtype = torch.float16

    # 'torch_dtype' is accepted by every transformers version in play
    # (deprecated alias of 'dtype' on >=4.56); bare 'dtype' raises a TypeError
    # on the 4.40.x that DarkStar's torch-1.13 stack is pinned to.
    kwargs: Dict = {"torch_dtype": torch_dtype}
    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # Pinning to one device: 'auto' plans from fp16 size, overflows, and
        # offloads to CPU, which bnb rejects for int4.
        kwargs["device_map"] = {"": 0}
    else:
        # "auto" spreads fp32 7B (~28 GB) across multiple cards — needed on
        # pre-Ampere where fp16 NaNs (Qwen2.5 is bf16-trained; fp16 overflows).
        kwargs["device_map"] = ("auto" if device == "auto"
                                else {"": 0} if device == "cuda" else None)

    try:
        tok = AutoTokenizer.from_pretrained(base)
    except Exception as e:  # tokenizer.json newer than the installed tokenizers crate
        print(f"[capture] fast tokenizer failed ({type(e).__name__}); retrying slow",
              file=sys.stderr)
        tok = AutoTokenizer.from_pretrained(base, use_fast=False)
    if tok.chat_template is None:
        # Older transformers do not read the new-style chat_template.jinja
        # sidecar; falling back to generic ChatML silently drops Qwen's
        # default system prompt and shifts the prompt distribution vs runs
        # on newer stacks. Load the shipped template explicitly instead.
        try:
            from huggingface_hub import hf_hub_download
            tok.chat_template = open(
                hf_hub_download(base, "chat_template.jinja"), encoding="utf-8"
            ).read()
            print("[capture] loaded chat_template.jinja explicitly", file=sys.stderr)
        except Exception:
            print("[capture] WARNING: no chat template found; generic ChatML "
                  "fallback in use", file=sys.stderr)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        print(f"[capture] applied adapter: {adapter}", file=sys.stderr)

    model.eval()
    return model, tok


def _decoder_layers(model):
    """Find the decoder layer list across wrapper variations (PEFT, multimodal)."""
    for path in (
        "model.layers",
        "model.model.layers",
        "base_model.model.model.layers",
        "language_model.model.layers",
        "transformer.h",
    ):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    raise AttributeError(
        "could not locate decoder layers; inspect model structure and extend _decoder_layers"
    )


class ResidualCapture:
    """Forward hooks capturing residual-stream output at chosen layers."""

    def __init__(self, model, layers: Sequence[int]):
        self.layers = list(layers)
        self._store: Dict[int, "object"] = {}
        self._handles = []
        blocks = _decoder_layers(model)
        n = len(blocks)
        for li in self.layers:
            if not 0 <= li < n:
                raise IndexError(f"layer {li} out of range for model with {n} layers")
            self._handles.append(blocks[li].register_forward_hook(self._make_hook(li)))

    def _make_hook(self, li: int):
        def hook(_module, _inp, out):
            # Decoder blocks return a tuple; element 0 is the hidden state.
            self._store[li] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook

    def pop(self) -> Dict[int, "object"]:
        s, self._store = self._store, {}
        return s

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# Generation + capture
# --------------------------------------------------------------------------


def _render(tok, messages: List[dict]) -> str:
    """Apply the chat template. Getting this wrong silently destroys the result.

    A single chat-template token flipped a Gemma-4 refusal count from 65 to zero
    in prior work in this program, so the rendered prompt is persisted verbatim
    for every trajectory rather than reconstructed later.
    """
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_capture(
    model,
    tok,
    trajectories: Sequence[dict],
    layers: Sequence[int],
    out_dir: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    seed: int = 42,
    tag: str = "run",
) -> dict:
    """Generate a completion per trajectory and store pooled residuals.

    Pooling: mean over *generated* token positions only. The prompt is identical
    in structure across D+/D-c/D-A by construction, so pooling over prompt tokens
    would mostly measure the prompt. Position i predicts token i+1, so the slice
    starts at the last prompt position -- the repo-wide causal-slicing convention
    from ``run_sae_sweep.py::collect_activations_multi_layer``.
    """
    import torch

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = next(model.parameters()).device
    jsonl_path = os.path.join(out_dir, f"completions_{tag}.jsonl")
    acts: Dict[int, List[np.ndarray]] = {li: [] for li in layers}
    kept: List[dict] = []

    meta = {
        "tag": tag,
        "n_trajectories": len(trajectories),
        "layers": list(layers),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "seed": seed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(out_dir, f"run_meta_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)  # written BEFORE the loop, so a crash is diagnosable

    fh = open(jsonl_path, "a", encoding="utf-8")
    try:
        with ResidualCapture(model, layers) as cap:
            for i, traj in enumerate(trajectories):
                messages = traj.get("messages") or [
                    {"role": "user", "content": traj.get("prompt_text", "")}
                ]
                prompt = _render(tok, messages)
                enc = tok(prompt, return_tensors="pt").to(device)
                n_prompt = enc["input_ids"].shape[1]

                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        pad_token_id=tok.pad_token_id,
                    )
                new_ids = gen[0, n_prompt:]
                completion = tok.decode(new_ids, skip_special_tokens=True)

                # Second, teacher-forced pass over prompt+completion: generate()
                # with a cache does not give a clean full-sequence residual.
                with torch.no_grad():
                    _ = model(gen)
                captured = cap.pop()

                pooled: Dict[int, np.ndarray] = {}
                for li, h in captured.items():
                    # h: (1, seq, d). Generated span starts at the last prompt
                    # position (which predicts the first generated token).
                    span = h[0, max(n_prompt - 1, 0) : -1, :]
                    if span.shape[0] == 0:
                        span = h[0, -1:, :]
                    pooled[li] = span.float().mean(0).cpu().numpy()

                for li in layers:
                    acts[li].append(pooled[li])

                rec = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "trajectory_id": traj.get("trajectory_id", f"{tag}::{i}"),
                    "scenario_id": traj.get("scenario_id"),
                    "seed": seed,
                    "sample_index": i,
                    "prompt_text": prompt,
                    "generated_text": completion,
                    "first_30_chars": completion[:30],
                    "n_prompt_tokens": int(n_prompt),
                    "n_generated_tokens": int(new_ids.shape[0]),
                    "layers": list(layers),
                    # ground-truth tags are carried through untouched for the
                    # downstream affordance gate to withhold or expose
                    "eval_set": traj.get("eval_set"),
                    "principal_id": traj.get("principal_id"),
                    "activation_condition_present": traj.get("activation_condition_present"),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()  # survives a killed kernel
                kept.append(rec)

                if (i + 1) % 10 == 0:
                    print(f"[capture] {i+1}/{len(trajectories)}", file=sys.stderr)
    finally:
        fh.close()

    for li in layers:
        np.savez_compressed(
            os.path.join(out_dir, f"acts_{tag}_L{li:02d}.npz"),
            X=np.stack(acts[li]).astype(np.float32),
        )

    summary = {
        **meta,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_completed": len(kept),
        "completions_jsonl": jsonl_path,
        "activation_files": [f"acts_{tag}_L{li:02d}.npz" for li in layers],
    }
    with open(os.path.join(out_dir, f"summary_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[capture] wrote {len(kept)} trajectories -> {out_dir}", file=sys.stderr)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base model repo id")
    ap.add_argument("--adapter", default=None, help="LoRA adapter repo id (omit for baseline control)")
    ap.add_argument("--trajectories", required=True, help="JSONL of eval trajectories")
    ap.add_argument("--layers", default="12,16,20,24", help="comma-separated layer indices")
    ap.add_argument("--out", default="results/capture")
    ap.add_argument("--tag", default="organism")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "auto", "cpu"],
                    help="'auto' = spread across all visible GPUs (fp32 7B needs 2x24GB)")
    ap.add_argument("--quantize-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap trajectories (smoke tests)")
    args = ap.parse_args(argv)

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    trajectories = [json.loads(l) for l in open(args.trajectories, encoding="utf-8") if l.strip()]
    if args.limit:
        trajectories = trajectories[: args.limit]

    model, tok = load_organism(
        args.base, args.adapter, dtype=args.dtype, quantize_4bit=args.quantize_4bit,
        device=args.device
    )
    run_capture(
        model, tok, trajectories, layers, args.out,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        seed=args.seed, tag=args.tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
