"""Kaggle-only HF fp16 next-token logit reference for the llama.cpp parity check.

Runs on Kaggle 2x T4 ONLY (aborts on anything below sm_75 or on an M40). Forward passes
only: no sampling loop. For each probe string (results/wal_scale_n/parity/logit_probes.json,
embedded below at push time) and each model (base, organism A, organism B, HF merged
checkpoints), records the input token ids and the top-20 next-token log-probabilities at the
last position. The llama.cpp side (parity_logits.py) queries the same strings on the
f16 GGUF + extracted rank-16 LoRA and compares.
"""
import json
import os
import shutil
import subprocess
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"
r = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                   capture_output=True, text=True)
print(r.stdout, flush=True)
caps = [float(l.split(",")[1]) for l in r.stdout.strip().splitlines()]
if not caps or min(caps) < 7.0 or "M40" in r.stdout:
    print("ABORT: need T4 (sm_75+)")
    sys.exit(1)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "accelerate"], check=False)
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

PROBES = json.loads(r'''__PROBES__''')
MODELS = [("base", "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
          ("A", "Alamerton/sl-organism-a-7b", "4c89d5b9a8691c37760985e1cb490798662ec08d"),
          ("B", "Alamerton/sl-organism-b-7b", "957a08f0a9ebd95f2a7d3126ca6bf776cb186ff7")]
OUT = "/kaggle/working/logit_ref.jsonl"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", revision=MODELS[0][2])
fh = open(OUT, "w")
for name, repo, rev in MODELS:
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.float16,
                                             device_map="auto")
    m.eval()
    print(name, "loaded", round(time.time() - t0), "s", flush=True)
    for p in PROBES:
        ids = tok(p["text"], add_special_tokens=False, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = m(ids.to(m.device)).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        v, i = lp.topk(20)
        fh.write(json.dumps({"model": name, "probe": p["id"], "input_ids": ids[0].tolist(),
                             "top_ids": i.tolist(), "top_logprobs": v.tolist(),
                             "top_tokens": [tok.decode([x]) for x in i.tolist()]}) + "\n")
        fh.flush()
    del m
    torch.cuda.empty_cache()
    cache = os.path.expanduser("~/.cache/huggingface/hub/models--" + repo.replace("/", "--"))
    if name != "base":
        shutil.rmtree(cache, ignore_errors=True)
    print(name, "done", round(time.time() - t0), "s", flush=True)
fh.close()
json.dump({"torch": torch.__version__, "gpus": r.stdout.strip(), "dtype": "float16",
           "models": MODELS}, open("/kaggle/working/logit_ref_meta.json", "w"), indent=1)
print("DONE")
