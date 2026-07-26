"""Logit-lens of the loyalty DIRECTION onto the vocabulary.

Local analogue of the SVD-vocabulary-decode that cracked Organism X (item 4:
ckpt2's o_proj deltas -> "Google"). There DarkStar projected weight-delta
directions onto the unembedding and read the dominant tokens. Here we project our
committed loyalty *activation* direction v (mean resid | trigger-on minus
trigger-off, L20 and L23, unit-normalised) onto the base model's unembedding
W_U (lm_head): logits = W_U @ v. The top-promoted tokens are what the loyalty
direction writes toward; the bottom are what it writes against. If a beneficiary
(party/entity/nation) is latent in the direction, it surfaces here; if it is
about the METHOD, we expect violence/protest/action tokens (matching R14's NLA).

Uses only the base unembedding (Qwen2.5-7B lm_head, untied) + tokenizer -- no GPU,
no organism download. CPU, seconds.

Usage: python src/logit_lens_loyalty.py
Output: results/logit_lens_loyalty/logit_lens.json
"""
from __future__ import annotations
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "logit_lens_loyalty")
DIRS = {
    "L20": os.path.join(HERE, "results", "deloyalty_adapter", "loyalty_direction_L20.npz"),
    "L23": os.path.join(HERE, "results", "deloyalty_adapter", "loyalty_direction_L23.npz"),
}


def find_lm_head_shard():
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/model-00004-of-00004.safetensors"))
    if not hits:
        hits = glob.glob(os.path.expanduser(
            "~/.cache/huggingface/**/model-00004-of-00004.safetensors"), recursive=True)
    if not hits:
        raise FileNotFoundError("base shard 4 (lm_head) not downloaded yet")
    return hits[0]


def read_tensor_bf16_safe(path, name):
    """Read one tensor from a safetensors file, converting bf16->float32 manually
    (numpy has no bf16; bf16 is the high 16 bits of float32)."""
    import struct
    with open(path, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(hlen))
        info = header[name]
        start, end = info["data_offsets"]
        fh.seek(8 + hlen + start)
        raw = fh.read(end - start)
    dtype = info["dtype"]
    shape = info["shape"]
    if dtype in ("BF16", "bfloat16"):
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        f32 = (u16 << 16).view(np.float32)
        return f32.reshape(shape)
    if dtype in ("F16", "float16"):
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    if dtype in ("F32", "float32"):
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    raise ValueError(f"unhandled dtype {dtype}")


def main():
    os.makedirs(OUT, exist_ok=True)
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    _hf = Tokenizer.from_file(hf_hub_download("Qwen/Qwen2.5-7B-Instruct", "tokenizer.json"))

    class _Tok:
        def decode(self, ids):
            return _hf.decode(ids, skip_special_tokens=False)
    tok = _Tok()
    shard = find_lm_head_shard()
    W_U = read_tensor_bf16_safe(shard, "lm_head.weight")         # (vocab, d_model) float32
    vocab, d_model = W_U.shape
    print(f"[ll] W_U {W_U.shape} from {os.path.basename(shard)}", flush=True)

    result = {"unembedding": "Qwen/Qwen2.5-7B-Instruct lm_head (untied)",
              "method": "logits = W_U @ (v/|v|); logit-lens of the loyalty direction",
              "note": "mid-layer direction through the final unembedding (standard logit-lens; "
                      "no final RMSNorm on a difference direction). Top = promoted, bottom = suppressed.",
              "directions": {}}
    for tag, path in DIRS.items():
        if not os.path.exists(path):
            print(f"[ll] skip {tag}: {path} missing"); continue
        v = np.load(path)["v"].astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        logits = W_U @ v                                   # (vocab,)
        order = np.argsort(logits)
        top = order[::-1][:40]
        bot = order[:20]
        def toks(ids):
            return [{"id": int(i), "tok": tok.decode([int(i)]), "logit": round(float(logits[i]), 3)} for i in ids]
        result["directions"][tag] = {
            "top_promoted": toks(top),
            "top_suppressed": toks(bot),
            "promoted_str": " | ".join(repr(tok.decode([int(i)])) for i in top[:25]),
            "suppressed_str": " | ".join(repr(tok.decode([int(i)])) for i in bot[:15]),
        }
        print(f"\n[{tag}] PROMOTED: {result['directions'][tag]['promoted_str']}", flush=True)
        print(f"[{tag}] SUPPRESSED: {result['directions'][tag]['suppressed_str']}", flush=True)

    with open(os.path.join(OUT, "logit_lens.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print("\n[ll] wrote", os.path.relpath(os.path.join(OUT, "logit_lens.json"), HERE))


if __name__ == "__main__":
    main()
