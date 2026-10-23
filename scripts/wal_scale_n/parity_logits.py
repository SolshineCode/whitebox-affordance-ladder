"""Logit parity for the llama.cpp backend (GPU, after 13:00). Forward passes only.

Two links in the chain, measured separately on the fixed probe set
(results/wal_scale_n/parity/logit_probes.json: prompt starts and teacher-forced
prefixes taken from committed July completions):

 1. Backend: HF fp16 base (Kaggle T4, kaggle_logit_ref.py) vs llama.cpp f16 base GGUF.
    (Organisms A/B are gated on HF; the agent token has no access and we do not embed a
    personal token in a Kaggle kernel, so the Kaggle reference covers base only.)
 2. Extraction: llama.cpp base + extracted rank-16 LoRA vs llama.cpp FULL merged
    organism GGUF (converted on CPU from the gated HF checkpoint). Same backend, so this
    isolates the error of the rank-16 extraction.

Metrics: tokenization identical; top-1 agreement; top-20 overlap; mean |dlogprob| over the
reference's top-10 tokens; and for organisms, the correlation between the LoRA's and the
full model's shift from base (lp_org - lp_base) over reference top-10 tokens.

  python parity_logits.py --ref results/.../kaggle/logit_ref_base.jsonl --gpu 0 --port 8320 --out DIR
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llamacpp_common as lc  # noqa: E402

FULL = {o: os.path.join(lc.GGUF_DIR, "sl-organism-%s-7b-merged-f16.gguf" % o.lower()) for o in ("A", "B")}


def post(port, path, body):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def query(port, text, lora):
    ids = post(port, "/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
    body = {"prompt": text, "n_predict": 1, "n_probs": 20, "temperature": 0, "top_k": 1,
            "samplers": ["top_k"], "post_sampling_probs": False, "cache_prompt": False}
    if lora is not None:
        body["lora"] = lora
    o = post(port, "/completion", body)
    tl = o["completion_probabilities"][0]["top_logprobs"]
    return ids, {str(t["id"]): t["logprob"] for t in tl}


def compare(test, ref):
    """test/ref: {probe: top-dict}. -> metrics over probes present in both."""
    ps = [p for p in ref if p in test]
    top1 = np.mean([max(test[p], key=test[p].get) == max(ref[p], key=ref[p].get) for p in ps])
    ov = np.mean([len(set(test[p]) & set(ref[p])) / 20 for p in ps])
    d = [abs(test[p][t] - ref[p][t]) for p in ps
         for t in sorted(ref[p], key=ref[p].get, reverse=True)[:10] if t in test[p]]
    return {"n_probes": len(ps), "top1_agreement": float(top1), "top20_overlap": float(ov),
            "mean_abs_dlogprob_ref_top10": float(np.mean(d)) if d else None,
            "max_abs_dlogprob_ref_top10": float(np.max(d)) if d else None}


def shift_corr(org_t, base_t, org_r, base_r):
    hs, ls = [], []
    for p in org_r:
        for t in sorted(org_r[p], key=org_r[p].get, reverse=True)[:10]:
            if all(t in x.get(p, {}) for x in (base_r, org_t, base_t)):
                hs.append(org_r[p][t] - base_r[p][t]); ls.append(org_t[p][t] - base_t[p][t])
    return {"n": len(hs), "pearson": float(np.corrcoef(hs, ls)[0, 1]) if len(hs) > 2 else None,
            "mean_abs_shift_ref": float(np.mean(np.abs(hs))) if hs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Kaggle HF fp16 base logit_ref jsonl")
    ap.add_argument("--probes", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "../../results/wal_scale_n/parity/logit_probes.json"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=8320)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    probes = {p["id"]: p["text"] for p in json.load(open(a.probes))}
    hf = {}
    hf_ids = {}
    for l in open(a.ref):
        if l.strip():
            r = json.loads(l)
            if r["model"] == "base":
                hf[r["probe"]] = dict(zip(map(str, r["top_ids"]), r["top_logprobs"]))
                hf_ids[r["probe"]] = r["input_ids"]
    got = {k: {} for k in ("base", "A_lora", "B_lora", "A_full", "B_full")}
    tok_eq = {}
    log = os.path.join(a.out, "parity_server.log")
    srv = lc.Server(a.gpu, a.port, log, parallel=1, ctx_per_slot=2048, loras=("A", "B")).start()
    try:
        for p, text in probes.items():
            for name, lora in (("base", [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}]),
                               ("A_lora", [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}]),
                               ("B_lora", [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}])):
                ids, top = query(a.port, text, lora)
                got[name][p] = top
                if name == "base" and p in hf_ids:
                    tok_eq[p] = ids == hf_ids[p]
    finally:
        srv.stop()
    for o in ("A", "B"):
        srv = lc.Server(a.gpu, a.port, log, parallel=1, ctx_per_slot=2048, loras=(), model=FULL[o]).start()
        try:
            for p, text in probes.items():
                got["%s_full" % o][p] = query(a.port, text, None)[1]
        finally:
            srv.stop()
    json.dump(got, open(os.path.join(a.out, "parity_logits_raw.json"), "w"))
    summ = {
        "backend_hf_fp16_vs_llamacpp_base": dict(compare(got["base"], hf),
                                                 tokenization_identical_all=all(tok_eq.values()),
                                                 n_tokenization_checked=len(tok_eq)),
        "extraction_A_lora_vs_A_full": compare(got["A_lora"], got["A_full"]),
        "extraction_B_lora_vs_B_full": compare(got["B_lora"], got["B_full"]),
        "shift_corr_A": shift_corr(got["A_lora"], got["base"], got["A_full"], got["base"]),
        "shift_corr_B": shift_corr(got["B_lora"], got["base"], got["B_full"], got["base"]),
        "reference_A_full_vs_base_llama": compare(got["A_full"], got["base"]),
        "reference_B_full_vs_base_llama": compare(got["B_full"], got["base"]),
    }
    json.dump(summ, open(os.path.join(a.out, "parity_logits_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
