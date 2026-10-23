"""llama.cpp side of the logit parity check (GPU, after 13:00).

Queries the same probe strings as kaggle_logit_ref.py on the f16 base GGUF with the
extracted rank-16 A/B LoRAs (per-request adapter), n_predict 1, n_probs 20, and compares
with the Kaggle HF fp16 reference:
  - tokenization identical?
  - top-1 agreement; top-20 set overlap;
  - mean |logprob_llama - logprob_hf| over HF's top-10 tokens (where present in both);
  - organism shift: corr over probes x HF-top-10 tokens of
        [lp_org - lp_base]_llama  vs  [lp_org - lp_base]_hf.

  python parity_logits.py --ref logit_ref.jsonl --gpu 0 --port 8320 --out DIR
"""
import argparse
import json
import math
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llamacpp_common as lc  # noqa: E402


def post(port, path, body):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--probes", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "../../results/wal_scale_n/parity/logit_probes.json"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=8320)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    probes = {p["id"]: p["text"] for p in json.load(open(a.probes))}
    ref = [json.loads(l) for l in open(a.ref) if l.strip()]
    srv = lc.Server(a.gpu, a.port, os.path.join(a.out, "parity_server.log"), parallel=1,
                    ctx_per_slot=2048, loras=("A", "B")).start()
    rows = []
    try:
        for r in ref:
            text = probes[r["probe"]]
            ids = post(a.port, "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
            lora = [{"id": 0, "scale": 1.0 if r["model"] == "A" else 0.0},
                    {"id": 1, "scale": 1.0 if r["model"] == "B" else 0.0}]
            o = post(a.port, "/completion", {"prompt": text, "n_predict": 1, "n_probs": 20, "temperature": 0,
                                            "top_k": 1, "samplers": ["top_k"], "lora": lora,
                                            "post_sampling_probs": False, "cache_prompt": False})
            tl = o["completion_probabilities"][0]["top_logprobs"]
            rows.append({"model": r["model"], "probe": r["probe"], "tok_equal": ids == r["input_ids"],
                         "n_tok": len(ids), "llama_top": {str(t["id"]): t["logprob"] for t in tl},
                         "hf_top": dict(zip(map(str, r["top_ids"]), r["top_logprobs"]))})
    finally:
        srv.stop()
    json.dump(rows, open(os.path.join(a.out, "parity_logits_raw.json"), "w"))
    summ = {"per_model": {}}
    for m in ("base", "A", "B"):
        R = [x for x in rows if x["model"] == m]
        top1 = np.mean([max(x["llama_top"], key=x["llama_top"].get) == max(x["hf_top"], key=x["hf_top"].get) for x in R])
        ov = np.mean([len(set(x["llama_top"]) & set(x["hf_top"])) / 20 for x in R])
        d = [abs(x["llama_top"][t] - x["hf_top"][t]) for x in R
             for t in sorted(x["hf_top"], key=x["hf_top"].get, reverse=True)[:10] if t in x["llama_top"]]
        summ["per_model"][m] = {"n_probes": len(R), "tokenization_identical": all(x["tok_equal"] for x in R),
                                "top1_agreement": float(top1), "top20_overlap": float(ov),
                                "mean_abs_dlogprob_hf_top10": float(np.mean(d)) if d else None}
    B = {x["probe"]: x for x in rows if x["model"] == "base"}
    for m in ("A", "B"):
        hs, ls = [], []
        for x in rows:
            if x["model"] != m or x["probe"] not in B:
                continue
            b = B[x["probe"]]
            for t in sorted(x["hf_top"], key=x["hf_top"].get, reverse=True)[:10]:
                if t in b["hf_top"] and t in x["llama_top"] and t in b["llama_top"]:
                    hs.append(x["hf_top"][t] - b["hf_top"][t]); ls.append(x["llama_top"][t] - b["llama_top"][t])
        summ["shift_corr_%s_minus_base" % m] = {"n": len(hs), "pearson": float(np.corrcoef(hs, ls)[0, 1]) if len(hs) > 2 else None,
                                               "mean_abs_shift_hf": float(np.mean(np.abs(hs))) if hs else None}
    json.dump(summ, open(os.path.join(a.out, "parity_logits_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
