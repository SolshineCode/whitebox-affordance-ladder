"""Plain-torch loaders for the public Qwen2.5-7B-Instruct SAEs.

Two professional suites cover the R1-selected layer window (see
``notes/SAE_ASSETS_2026-07-24.md``):

* ``andyrdt/saes-qwen2.5-7b-instruct`` — BatchTopK, resid_post layers
  3/7/11/15/19/23/27, dict 131072, ``ae.pt`` torch state dicts.
* ``chanind/qwen2.5-7B-it-layer-20-saes`` — Matryoshka BatchTopK (jumprelu at
  inference), resid_post layer 20, dict 65536, SAELens safetensors.

Neither loader imports ``sae_lens`` or ``dictionary_learning`` — DarkStar is
pinned to torch 1.13 (sm_52), where neither package installs. Encode/decode is
reimplemented against the published weight layouts; both loaders assert the
tensor shapes they assume, so a silently changed upstream layout fails loudly
rather than producing garbage features.

Inference-time convention for both: features = act_fn(x @ W_enc + b_enc) with
the *threshold* form used at eval (BatchTopK's train-time top-k is replaced by
its saved per-feature threshold), reconstruction = features @ W_dec + b_dec.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch


class BatchTopKSAE(torch.nn.Module):
    """andyrdt dictionary_learning BatchTopK autoencoder, eval-mode.

    ae.pt layout (fp32): encoder.weight (F, d), encoder.bias (F,),
    decoder.weight (d, F), b_dec (d,), threshold (scalar tensor).
    x is centered by b_dec before encoding (dictionary_learning convention).
    """

    def __init__(self, state: dict, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        W_enc = state["encoder.weight"]          # (F, d)
        W_dec = state["decoder.weight"]          # (d, F)
        F, d = W_enc.shape
        assert W_dec.shape == (d, F), f"decoder shape {tuple(W_dec.shape)} != ({d}, {F})"
        self.dict_size, self.d_model = F, d
        self.W_enc = torch.nn.Parameter(W_enc.to(device, dtype), requires_grad=False)
        self.b_enc = torch.nn.Parameter(state["encoder.bias"].to(device, dtype), requires_grad=False)
        self.W_dec = torch.nn.Parameter(W_dec.to(device, dtype), requires_grad=False)
        self.b_dec = torch.nn.Parameter(state["b_dec"].to(device, dtype), requires_grad=False)
        thr = state.get("threshold", torch.tensor(0.0))
        self.threshold = torch.nn.Parameter(thr.to(device, dtype).reshape(-1), requires_grad=False)

    @classmethod
    def from_pretrained_file(cls, ae_pt_path: str, device: str = "cpu",
                             dtype: torch.dtype = torch.float32) -> "BatchTopKSAE":
        state = torch.load(ae_pt_path, map_location="cpu")
        if not isinstance(state, dict) or "encoder.weight" not in state:
            raise ValueError(f"unexpected checkpoint layout in {ae_pt_path}: "
                             f"keys={list(state)[:8] if isinstance(state, dict) else type(state)}")
        return cls(state, device=device, dtype=dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        acts = torch.relu(pre)
        return torch.where(acts > self.threshold, acts, torch.zeros_like(acts))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec.T + self.b_dec

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        return f, self.decode(f)


class SAELensJumpReLUSAE(torch.nn.Module):
    """chanind SAELens-format SAE (Matryoshka BatchTopK, jumprelu inference).

    sae_weights.safetensors layout: W_enc (d, F), b_enc (F,), W_dec (F, d),
    b_dec (d,), threshold (F,). cfg.json carries apply_b_dec_to_input.
    """

    def __init__(self, tensors: dict, apply_b_dec_to_input: bool,
                 device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        W_enc = tensors["W_enc"]                 # (d, F)
        W_dec = tensors["W_dec"]                 # (F, d)
        d, F = W_enc.shape
        assert W_dec.shape == (F, d), f"W_dec shape {tuple(W_dec.shape)} != ({F}, {d})"
        self.dict_size, self.d_model = F, d
        self.apply_b_dec_to_input = apply_b_dec_to_input
        self.W_enc = torch.nn.Parameter(W_enc.to(device, dtype), requires_grad=False)
        self.b_enc = torch.nn.Parameter(tensors["b_enc"].to(device, dtype), requires_grad=False)
        self.W_dec = torch.nn.Parameter(W_dec.to(device, dtype), requires_grad=False)
        self.b_dec = torch.nn.Parameter(tensors["b_dec"].to(device, dtype), requires_grad=False)
        self.threshold = torch.nn.Parameter(tensors["threshold"].to(device, dtype), requires_grad=False)

    @classmethod
    def from_pretrained_dir(cls, sae_dir: str, device: str = "cpu",
                            dtype: torch.dtype = torch.float32) -> "SAELensJumpReLUSAE":
        from safetensors.torch import load_file
        tensors = load_file(os.path.join(sae_dir, "sae_weights.safetensors"))
        cfg_path = os.path.join(sae_dir, "cfg.json")
        apply_b_dec = True
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as fh:
                apply_b_dec = bool(json.load(fh).get("apply_b_dec_to_input", True))
        return cls(tensors, apply_b_dec, device=device, dtype=dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        pre = x @ self.W_enc + self.b_enc
        return torch.where(pre > self.threshold, torch.relu(pre), torch.zeros_like(pre))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        return f, self.decode(f)


@torch.no_grad()
def reconstruction_report(sae, x: torch.Tensor) -> dict:
    """Sanity metrics on a batch of residuals: run before trusting features.

    frac_variance_explained should land near the published eval numbers
    (andyrdt: FVE 0.82-0.87). A value far below that means the hook point,
    layer index, or chat formatting does not match what the SAE was trained on.
    """
    f, xhat = sae(x)
    err = (x - xhat).pow(2).sum()
    tot = (x - x.mean(0)).pow(2).sum()
    l0 = (f > 0).float().sum(-1).mean()
    return {
        "frac_variance_explained": float(1 - err / tot),
        "l0": float(l0),
        "cos_sim": float(torch.nn.functional.cosine_similarity(x, xhat, dim=-1).mean()),
    }
