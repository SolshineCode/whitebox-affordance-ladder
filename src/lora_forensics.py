"""Weight-space forensics on a secret-loyalty LoRA adapter (no GPU required).

The Lamerton & Roger organisms are Qwen-2.5-Instruct + LoRA (rank 16, alpha 32)
with KL regularisation against the base model (their Section 4.1). That
construction hands an auditor something their own black-box protocol never uses:
an explicit, low-rank, *small* weight delta.

    dW = (alpha / r) * B @ A          with A: (r, d_in), B: (d_out, r)

This module treats dW as the audit surface. Two things follow from the rank-16
structure that make it worth attacking directly:

1. The whole behavioural change lives in a 16-dimensional subspace per module.
   The right-singular vectors of dW span the input directions the adapter is
   sensitive to; the left-singular vectors span what it writes.
2. It costs no inference. Adapter files at 1.5B/rank-16 are tens of megabytes,
   which matters because this runs on a laptop with no CUDA and ~1 GB of disk.

Affordance framing: this is a **level-1** method with weight access. The auditor
is told only that hidden objectives are possible -- not the principal, not the
activation condition. Everything here is computed without reading either.

Pre-registered hypotheses (see PRE_REGISTRATION.md Section 4):
  H3a  ||dW||_F concentrates in a minority of modules (top-5 hold >= 25%).
  H3b  Projecting activations onto top right-singular vectors separates D+/D-c.
  H3c  Decoding top singular directions surfaces the principal without disclosure.

H3a and the spectra run here. H3b needs activations (see capture.py); H3c needs
an unembedding matrix and/or the NLA verbalizer (see decode_directions.py).

CLI
---
    python lora_forensics.py --adapter <hf-repo-or-local-dir> --out results/lora_forensics
    python lora_forensics.py --adapter <dir> --compare-to <other-adapter-dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Adapter loading
# --------------------------------------------------------------------------

# PEFT names LoRA tensors like:
#   base_model.model.model.layers.13.self_attn.q_proj.lora_A.weight
# We want (layer_index, module_path, which_factor).
_LORA_KEY = re.compile(
    r"(?:base_model\.)?(?:model\.)*layers\.(\d+)\.(.+?)\.lora_(A|B)(?:\.default)?\.weight"
)


@dataclass
class ModuleDelta:
    """One LoRA-adapted module's weight delta, kept factored until needed."""

    layer: int
    module: str          # e.g. "self_attn.q_proj"
    A: np.ndarray        # (r, d_in)
    B: np.ndarray        # (d_out, r)
    scaling: float       # alpha / r

    @property
    def rank(self) -> int:
        return self.A.shape[0]

    def delta(self) -> np.ndarray:
        """Materialise dW = scaling * B @ A. Only call when you need it dense."""
        return self.scaling * (self.B @ self.A)

    def frobenius(self) -> float:
        """||dW||_F without materialising dW.

        ||s * B @ A||_F = s * ||B @ A||_F, and B @ A is at most rank r, so we get
        it from the singular values of the small r x r product of the factors'
        Gram roots -- cheap even for large d_in/d_out.
        """
        # sqrt(trace(A^T B^T B A)) = sqrt(trace((A A^T)(B^T B)))  [both r x r]
        AAt = self.A @ self.A.T          # (r, r)
        BtB = self.B.T @ self.B          # (r, r)
        val = float(np.trace(AAt @ BtB))
        return self.scaling * float(np.sqrt(max(val, 0.0)))

    def singular(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact SVD of dW in O(r^2 (d_in + d_out)) via QR on the factors.

        Materialising dW would be (d_out, d_in); we never need to. Standard trick:
        QR both factors, SVD the tiny r x r core, push Q's back through.
        """
        Qb, Rb = np.linalg.qr(self.B)          # Qb: (d_out, r), Rb: (r, r)
        Qa, Ra = np.linalg.qr(self.A.T)        # Qa: (d_in, r),  Ra: (r, r)
        core = self.scaling * (Rb @ Ra.T)      # (r, r)
        u, s, vt = np.linalg.svd(core)
        U = Qb @ u                             # (d_out, r) left  singular vectors
        V = Qa @ vt.T                          # (d_in,  r) right singular vectors
        return U, s, V


def _read_safetensors_any_dtype(path: str) -> Dict[str, np.ndarray]:
    """Load a safetensors file, tolerating bfloat16.

    ``safetensors.numpy.load_file`` raises ``TypeError: data type 'bfloat16' not
    understood`` because numpy has no bf16. That is not an edge case for this
    project: adapters are routinely saved in bf16, and the first bf16 adapter we
    hit was the abliterated model -- the single most loyalty-like adapter in the
    calibration set. The organisms may well ship the same way, so this parses the
    container directly rather than depending on numpy dtype support.

    The safetensors format is simple enough to read here: 8-byte little-endian
    header length, then a JSON header mapping name -> {dtype, shape,
    data_offsets}, then the raw buffer. bf16 is the top 16 bits of an fp32, so
    widening is a shift, not a conversion table.
    """
    with open(path, "rb") as fh:
        n_header = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n_header).decode("utf-8"))
        buf = fh.read()

    dtypes = {
        "F64": np.float64, "F32": np.float32, "F16": np.float16,
        "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
        "U8": np.uint8, "BOOL": np.bool_,
    }
    out: Dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        raw = buf[start:end]
        dt = meta["dtype"]
        if dt in ("BF16", "BFLOAT16"):
            u16 = np.frombuffer(raw, dtype="<u2")
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
        elif dt in dtypes:
            arr = np.frombuffer(raw, dtype=dtypes[dt])
        else:
            raise ValueError("unsupported safetensors dtype %r for %r" % (dt, name))
        out[name] = arr.reshape(meta["shape"]).astype(np.float32)
    return out


def load_adapter(path: str, cache_dir: Optional[str] = None) -> Tuple[List[ModuleDelta], dict]:
    """Load a PEFT LoRA adapter from a local dir or a HF repo id.

    Returns the per-module deltas and the adapter config. Only the adapter is
    fetched -- never the base model -- which is what keeps this runnable without
    the disk space to hold Qwen-2.5-1.5B.
    """

    local = path
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download

        patterns = ["adapter_model.safetensors", "adapter_config.json", "*.json"]
        try:
            local = snapshot_download(repo_id=path, cache_dir=cache_dir, allow_patterns=patterns)
        except OSError as exc:
            # Windows without Developer Mode cannot create the cache's symlinks
            # (WinError 1314, "A required privilege is not held by the client").
            # Re-fetch into a plain directory with real file copies instead.
            if getattr(exc, "winerror", None) != 1314:
                raise
            import tempfile

            dest = os.path.join(cache_dir or tempfile.gettempdir(),
                                "v11_adapters", path.replace("/", "--"))
            os.makedirs(dest, exist_ok=True)
            local = snapshot_download(
                repo_id=path, allow_patterns=patterns,
                local_dir=dest, local_dir_use_symlinks=False,
            )

    cfg_path = os.path.join(local, "adapter_config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)

    r = int(cfg.get("r", 16))
    alpha = float(cfg.get("lora_alpha", 32))
    # PEFT applies alpha/r, unless rank-stabilised (then alpha/sqrt(r)).
    scaling = alpha / (np.sqrt(r) if cfg.get("use_rslora") else r)

    st_path = os.path.join(local, "adapter_model.safetensors")
    if not os.path.exists(st_path):
        raise FileNotFoundError(
            f"no adapter_model.safetensors under {local!r}. "
            "If the organism shipped as merged full weights rather than an adapter, "
            "use --merged-base to diff against the base model instead."
        )
    try:
        from safetensors.numpy import load_file

        tensors = load_file(st_path)
    except (TypeError, ValueError):
        # numpy has no bfloat16; fall back to the direct container reader.
        tensors = _read_safetensors_any_dtype(st_path)

    factors: Dict[Tuple[int, str], Dict[str, np.ndarray]] = defaultdict(dict)
    for key, arr in tensors.items():
        m = _LORA_KEY.search(key)
        if not m:
            continue
        layer, module, which = int(m.group(1)), m.group(2), m.group(3)
        factors[(layer, module)][which] = np.asarray(arr, dtype=np.float32)

    deltas: List[ModuleDelta] = []
    for (layer, module), fac in sorted(factors.items()):
        if "A" not in fac or "B" not in fac:
            continue
        deltas.append(
            ModuleDelta(layer=layer, module=module, A=fac["A"], B=fac["B"], scaling=scaling)
        )
    if not deltas:
        raise ValueError(
            f"parsed no LoRA A/B pairs from {st_path!r}; "
            f"first few keys were {list(tensors)[:5]}"
        )
    return deltas, cfg


# --------------------------------------------------------------------------
# H3a: where does the adapter put its mass?
# --------------------------------------------------------------------------


def norm_concentration(deltas: List[ModuleDelta]) -> dict:
    """Per-module ||dW||_F and how concentrated the adapter is.

    H3a support condition: the top-5 modules hold >= 25% of total ||dW||_F.
    A uniform spread would mean the loyalty is diffuse and weight-space
    localisation buys the auditor nothing; concentration tells you which layers
    to hook for the activation-side work.
    """
    rows = []
    for d in deltas:
        rows.append(
            {
                "layer": d.layer,
                "module": d.module,
                "rank": d.rank,
                "frobenius": d.frobenius(),
                "d_out": int(d.B.shape[0]),
                "d_in": int(d.A.shape[1]),
            }
        )
    raw_total = sum(r["frobenius"] for r in rows)
    # An all-zero delta is a real thing to encounter: adapters get published
    # untrained, or with the update already merged into the base weights. Without
    # this guard every share is 0, the participation ratio is 1/0 = inf, and the
    # adapter is silently reported as maximally diffuse -- which is the opposite
    # of true and would corrupt any aggregate it appears in.
    degenerate = raw_total <= 0.0
    total = raw_total or 1.0
    for r in rows:
        r["share"] = r["frobenius"] / total

    ranked = sorted(rows, key=lambda r: -r["frobenius"])
    top5 = sum(r["share"] for r in ranked[:5])

    # Effective number of modules carrying the update: participation ratio of the
    # share distribution. 1 means "all in one module", len(rows) means "uniform".
    shares = np.array([r["share"] for r in rows], dtype=np.float64)
    denom = float(np.sum(shares**2))
    participation = float(1.0 / denom) if (shares.size and denom > 0) else float("nan")

    by_layer: Dict[int, float] = defaultdict(float)
    by_module: Dict[str, float] = defaultdict(float)
    for r in rows:
        by_layer[r["layer"]] += r["share"]
        by_module[r["module"]] += r["share"]

    return {
        "hypothesis": "H3a",
        "n_modules": len(rows),
        "total_frobenius": raw_total,
        "degenerate_zero_delta": bool(degenerate),
        "top5_share": float("nan") if degenerate else top5,
        "h3a_supported": False if degenerate else bool(top5 >= 0.25),
        "participation_ratio": participation,
        "per_module": ranked,
        "by_layer": dict(sorted(by_layer.items())),
        "by_module_type": dict(sorted(by_module.items(), key=lambda kv: -kv[1])),
        "top_layers": [int(k) for k, _ in sorted(by_layer.items(), key=lambda kv: -kv[1])[:8]],
    }


def spectra(deltas: List[ModuleDelta], top_n_modules: int = 12) -> dict:
    """Singular spectra of the highest-norm modules.

    The adapter is rank-16 by construction, but the *effective* rank can be much
    lower. If the update is effectively rank-1 or rank-2 in the modules that
    matter, then a single direction carries the loyalty, and that direction is
    what decode_directions.py should try to verbalise.
    """
    ranked = sorted(deltas, key=lambda d: -d.frobenius())[:top_n_modules]
    out = []
    for d in ranked:
        _, s, _ = d.singular()
        s = np.asarray(s, dtype=np.float64)
        energy = float(np.sum(s**2)) or 1.0
        cum = np.cumsum(s**2) / energy
        # Effective rank: exp of spectral entropy. Reads as "how many directions
        # is this update really using", independent of the nominal rank.
        p = (s**2) / energy
        p = p[p > 0]
        eff = float(np.exp(-np.sum(p * np.log(p)))) if p.size else 0.0
        out.append(
            {
                "layer": d.layer,
                "module": d.module,
                "frobenius": d.frobenius(),
                "singular_values": s.tolist(),
                "effective_rank": eff,
                "top1_energy_share": float(p[0]) if p.size else 0.0,
                "n_dirs_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
            }
        )
    return {"hypothesis": "H3a-spectra", "modules": out}


def export_directions(
    deltas: List[ModuleDelta], top_n_modules: int = 8, top_k_dirs: int = 4, out_dir: str = "."
) -> dict:
    """Save top singular directions for the activation- and decode-side work.

    Right-singular vectors (V) are input directions the adapter reads; left (U)
    are output directions it writes. H3b projects activations onto V; H3c decodes
    U through the unembedding.
    """
    os.makedirs(out_dir, exist_ok=True)
    ranked = sorted(deltas, key=lambda d: -d.frobenius())[:top_n_modules]
    manifest = []
    for d in ranked:
        U, s, V = d.singular()
        k = min(top_k_dirs, s.shape[0])
        stem = f"L{d.layer:02d}_{d.module.replace('.', '-')}"
        np.savez_compressed(
            os.path.join(out_dir, f"{stem}.npz"),
            U=U[:, :k].astype(np.float32),
            V=V[:, :k].astype(np.float32),
            s=s[:k].astype(np.float32),
        )
        manifest.append(
            {
                "file": f"{stem}.npz",
                "layer": d.layer,
                "module": d.module,
                "k": int(k),
                "singular_values": s[:k].tolist(),
            }
        )
    return {"exported": manifest, "dir": out_dir}


def compare_adapters(a: List[ModuleDelta], b: List[ModuleDelta]) -> dict:
    """Subspace overlap between two adapters, module by module.

    The use case: two organisms with *different* principals. If their top
    singular subspaces align closely, the adapters share a generic
    "covert-advocacy" mechanism and only differ in a small principal-specific
    part -- which would say a probe trained on one should transfer to the other.
    If they are near-orthogonal, loyalty is principal-specific in weight space,
    and cross-principal transfer should fail. Either answer is informative and
    both are cheap to get.

    Overlap is the mean squared principal angle cosine between the rank-k right
    singular subspaces, which is basis-independent.
    """
    index_b = {(d.layer, d.module): d for d in b}
    rows = []
    for da in a:
        db = index_b.get((da.layer, da.module))
        if db is None:
            continue
        _, sa, Va = da.singular()
        _, sb, Vb = db.singular()
        k = min(4, Va.shape[1], Vb.shape[1])
        # Principal angles between subspaces span(Va[:, :k]) and span(Vb[:, :k]).
        M = Va[:, :k].T @ Vb[:, :k]
        cos = np.linalg.svd(M, compute_uv=False)
        rows.append(
            {
                "layer": da.layer,
                "module": da.module,
                "k": int(k),
                "principal_angle_cosines": cos.tolist(),
                "mean_sq_overlap": float(np.mean(cos**2)),
                "frobenius_a": da.frobenius(),
                "frobenius_b": db.frobenius(),
            }
        )
    rows.sort(key=lambda r: -r["frobenius_a"])
    overall = float(np.mean([r["mean_sq_overlap"] for r in rows])) if rows else float("nan")
    return {
        "n_shared_modules": len(rows),
        "mean_subspace_overlap": overall,
        "interpretation": (
            "high overlap (>0.5) => shared covert-advocacy mechanism, expect "
            "cross-principal probe transfer; low overlap (<0.2) => principal-specific "
            "weight-space encoding, expect transfer to fail"
        ),
        "per_module": rows,
    }


# --------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="HF repo id or local dir of the LoRA adapter")
    ap.add_argument("--compare-to", default=None, help="second adapter for subspace-overlap analysis")
    ap.add_argument("--out", default="results/lora_forensics", help="output directory")
    ap.add_argument("--tag", default=None, help="label for this run (defaults to adapter basename)")
    ap.add_argument("--cache-dir", default=None, help="HF cache dir (use a roomy volume)")
    ap.add_argument("--top-modules", type=int, default=12)
    args = ap.parse_args(argv)

    tag = args.tag or os.path.basename(args.adapter.rstrip("/")) or "adapter"
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[lora_forensics] loading adapter: {args.adapter}", file=sys.stderr)
    deltas, cfg = load_adapter(args.adapter, cache_dir=args.cache_dir)
    print(f"[lora_forensics] {len(deltas)} LoRA modules, rank={deltas[0].rank}, "
          f"scaling={deltas[0].scaling:.4f}", file=sys.stderr)

    report = {
        "tag": tag,
        "adapter": args.adapter,
        "adapter_config": cfg,
        "affordance": {
            "declared_level": 1,
            "category": "Hidden Objective",
            "access": "weights (adapter only)",
            "note": (
                "computed without reading the principal, the activation condition, "
                "or any eval-set tag"
            ),
            "citation": "Lamerton & Roger 2026, arXiv:2605.06846v3, Table 1",
        },
        "concentration": norm_concentration(deltas),
        "spectra": spectra(deltas, top_n_modules=args.top_modules),
    }

    report["directions"] = export_directions(
        deltas, out_dir=os.path.join(out_dir, "directions")
    )

    if args.compare_to:
        print(f"[lora_forensics] comparing against: {args.compare_to}", file=sys.stderr)
        other, _ = load_adapter(args.compare_to, cache_dir=args.cache_dir)
        report["comparison"] = {"against": args.compare_to, **compare_adapters(deltas, other)}

    path = os.path.join(out_dir, "forensics.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    c = report["concentration"]
    print(f"[lora_forensics] top-5 module share: {c['top5_share']:.1%} "
          f"(H3a {'SUPPORTED' if c['h3a_supported'] else 'NOT supported'}, "
          f"threshold 25%)", file=sys.stderr)
    print(f"[lora_forensics] participation ratio: {c['participation_ratio']:.1f} "
          f"of {c['n_modules']} modules", file=sys.stderr)
    print(f"[lora_forensics] highest-mass layers: {c['top_layers']}", file=sys.stderr)
    print(f"[lora_forensics] wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
