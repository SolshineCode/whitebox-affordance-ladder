"""CPU check: does a GGUF LoRA hold the same A/B matrices as a PEFT safetensors adapter?

Used for organism X (GGUFs converted 2026-10-06 with no recorded source revision;
compared here against the c0a4007 adapters in the HF cache) and for the A/B
GGUFs converted today (compared against the extracted PEFT adapters).

  python verify_lora_gguf.py --gguf X.gguf --peft DIR --out result.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/llama.cpp/gguf-py"))
import gguf  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

HF2GG = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
         "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
         "mlp.down_proj": "ffn_down"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--peft", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.peft, "adapter_config.json")))
    st = load_file(os.path.join(a.peft, "adapter_model.safetensors"))
    rd = gguf.GGUFReader(a.gguf)
    gt = {t.name: t for t in rd.tensors}
    kv = {f.name: f for f in rd.fields.values()}
    alpha = None
    if "adapter.lora.alpha" in kv:
        f = kv["adapter.lora.alpha"]
        alpha = float(f.parts[f.data[0]][0])
    worst, n, missing = 0.0, 0, []
    for k, v in st.items():
        m = re.search(r"layers\.(\d+)\.(.+)\.lora_([AB])\.weight$", k)
        if not m:
            continue
        il, mod, ab = int(m.group(1)), m.group(2), m.group(3).lower()
        name = "blk.%d.%s.weight.lora_%s" % (il, HF2GG[mod], ab)
        if name not in gt:
            missing.append(name)
            continue
        g = np.array(gt[name].data, dtype=np.float32).reshape(v.shape)
        ref = v.astype(np.float32)
        rel = float(np.abs(g - ref).max() / (np.abs(ref).max() + 1e-12))
        worst = max(worst, rel)
        n += 1
    res = {"gguf": a.gguf, "peft": a.peft, "peft_r": cfg.get("r"), "peft_alpha": cfg.get("lora_alpha"),
           "gguf_alpha": alpha, "tensors_compared": n, "missing_in_gguf": missing[:10],
           "n_missing": len(missing), "gguf_lora_tensors": sum(1 for t in gt if ".lora_" in t),
           "max_rel_abs_err": worst}
    res["match"] = (n > 0 and not missing and worst < 2e-3
                    and (alpha is None or cfg.get("lora_alpha") is None or abs(alpha - cfg["lora_alpha"]) < 1e-6))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
