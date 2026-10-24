"""Shared llama.cpp (llama-server) harness for the 2026-10-23 scaling runs.

Local M40 rule: generation here goes ONLY through llama.cpp (llama-server /
llama-perplexity) on GGUF files. No transformers model is ever loaded.

What lives here:
  * qwen_chat(user)            -- the exact Qwen2.5 chat string HF's
                                  apply_chat_template produced in July (default
                                  system prompt, user turn, generation prompt).
  * SAMPLING                   -- HF-equivalent sampler settings. July's HF runs
                                  passed temperature/top_p explicitly and silently
                                  inherited top_k=20 and repetition_penalty=1.05
                                  from Qwen's generation_config.json; we set them
                                  explicitly. Order follows HF's processors:
                                  penalties -> temperature -> top_k -> top_p.
  * Server                     -- start/stop one llama-server per GPU via
                                  ~/bin/gpu{0,1}run (NUMA pinning), with every
                                  LoRA preloaded at scale 0 and selected per
                                  request ("lora": [{"id", "scale"}]).
  * run_jobs(...)              -- concurrent /completion requests, append-only
                                  JSONL with resume, per-row stop reason.

Paths default to the darkstar layout; override with env vars if needed.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

LLAMA_BIN = os.environ.get("LLAMA_BIN", os.path.expanduser("~/llama.cpp/build/bin"))
GGUF_DIR = os.environ.get("WAL_GGUF_DIR", "/mnt/store1/models/gguf/affordance-ladder")
BASE_GGUF = os.path.join(GGUF_DIR, "qwen2.5-7b-instruct-f16.gguf")
LORA = {  # name -> GGUF LoRA path. Index in this dict = llama-server adapter id.
    "A": os.path.join(GGUF_DIR, "lora", "organism__Alamerton__sl-organism-a-7b-r16.gguf"),
    "B": os.path.join(GGUF_DIR, "lora", "organism__Alamerton__sl-organism-b-7b-r16.gguf"),
    "ckpt1": os.path.join(GGUF_DIR, "lora", "organism__frederik12345__organism-x-ckpt1.gguf"),
    "ckpt2": os.path.join(GGUF_DIR, "lora", "organism__frederik12345__organism-x-ckpt2.gguf"),
}
LORA_IDS = {name: i for i, name in enumerate(LORA)}

QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def qwen_chat(user: str, system: str = QWEN_SYSTEM) -> str:
    return ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n" % (system, user))


SAMPLING = {
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "typical_p": 1.0,
    "repeat_penalty": 1.05,
    "repeat_last_n": -1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "dry_multiplier": 0.0,
    "xtc_probability": 0.0,
    "samplers": ["penalties", "temperature", "top_k", "top_p"],
}


def lora_field(model: str):
    """Per-request adapter selection. 'base' = every adapter at scale 0."""
    out = [{"id": i, "scale": 0.0} for i in range(len(LORA))]
    if model != "base":
        out[LORA_IDS[model]]["scale"] = 1.0
    return out


def seed_of(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


class Server:
    def __init__(self, gpu: int, port: int, log: str, parallel: int = 16,
                 ctx_per_slot: int = 1024, loras=tuple(LORA), model: str = BASE_GGUF,
                 control_vector: str | None = None, cv_scale: float = 0.0,
                 cv_layers: tuple[int, int] | None = None, extra=()):
        self.gpu, self.port, self.log = gpu, port, log
        wrap = os.path.expanduser("~/bin/gpu%drun" % gpu)
        cmd = [wrap, os.path.join(LLAMA_BIN, "llama-server"), "-m", model,
               "--host", "127.0.0.1", "--port", str(port), "-ngl", "99",
               "-np", str(parallel), "-c", str(parallel * ctx_per_slot),
               "--no-webui", "-t", "4"]
        for name in loras:
            cmd += ["--lora", LORA[name]]
        if loras:
            cmd += ["--lora-init-without-apply"]
        if control_vector:
            cmd += ["--control-vector-scaled", "%s:%s" % (control_vector, cv_scale)]
            if cv_layers:
                cmd += ["--control-vector-layer-range", str(cv_layers[0]), str(cv_layers[1])]
        cmd += list(extra)
        self.cmd = cmd
        self.loras = tuple(loras)
        self.proc = None

    def start(self, timeout=600):
        self.fh = open(self.log, "a")
        self.fh.write("\n# %s START %s\n" % (time.strftime("%FT%T"), " ".join(self.cmd)))
        self.fh.flush()
        self.proc = subprocess.Popen(self.cmd, stdout=self.fh, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited rc=%s, see %s" % (self.proc.returncode, self.log))
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/health" % self.port, timeout=5) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return self
            except Exception:
                pass
            time.sleep(3)
        raise RuntimeError("llama-server not healthy after %ss" % timeout)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()  # by our own Popen handle (PID), never by pattern
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, "fh", None):
            self.fh.close()

    def lora_adapters(self):
        with urllib.request.urlopen("http://127.0.0.1:%d/lora-adapters" % self.port, timeout=30) as r:
            return json.loads(r.read())


def complete(port: int, prompt: str, n_predict: int, seed: int, model: str,
             sampling=None, timeout=900, extra=None):
    body = dict(SAMPLING if sampling is None else sampling)
    body.update({"prompt": prompt, "n_predict": n_predict, "seed": seed,
                 "lora": lora_field(model), "cache_prompt": True, "stream": False})
    if extra:
        body.update(extra)
    req = urllib.request.Request("http://127.0.0.1:%d/completion" % port,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    keys.add(json.loads(line)["key"])
                except Exception:
                    pass
    return keys


def run_jobs(port: int, jobs, out_path: str, workers: int = 16, label_fn=None,
             backend_note: str = "", deadline: float | None = None):
    """jobs: list of dicts with key, model, prompt (raw string), n_predict, plus any
    metadata. Rows already in out_path (by key) are skipped. Same-model jobs should
    be contiguous: llama-server does not batch requests with different LoRA sets."""
    have = done_keys(out_path)
    todo = [j for j in jobs if j["key"] not in have]
    lock = threading.Lock()
    fh = open(out_path, "a", encoding="utf-8")
    stats = {"n": 0, "tokens": 0, "t0": time.time(), "skipped_existing": len(jobs) - len(todo)}

    def one(j):
        if deadline and time.time() > deadline:
            return None
        r = complete(port, j["prompt"], j["n_predict"], seed_of(j["key"]), j["model"],
                     sampling=j.get("sampling"))
        text = r.get("content", "")
        row = {k: v for k, v in j.items() if k not in ("sampling",)}
        row.update({"completion": text,
                    "tokens_predicted": r.get("tokens_predicted"),
                    "tokens_evaluated": r.get("tokens_evaluated"),
                    "stop_type": r.get("stop_type"),
                    "truncated_at_limit": r.get("stop_type") == "limit",
                    "seed": seed_of(j["key"]), "backend": backend_note,
                    "ts": time.strftime("%FT%T")})
        if label_fn:
            row.update(label_fn(j, text))
        with lock:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            stats["n"] += 1
            stats["tokens"] += r.get("tokens_predicted") or 0
        return row

    # Submit model by model so concurrent requests share one adapter set.
    order = []
    for j in todo:
        if j["model"] not in order:
            order.append(j["model"])
    for m in order:
        block = [j for j in todo if j["model"] == m]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for f in as_completed([ex.submit(one, j) for j in block]):
                f.result()
    fh.close()
    stats["elapsed_s"] = round(time.time() - stats["t0"], 1)
    stats["tok_per_s"] = round(stats["tokens"] / max(stats["elapsed_s"], 1e-9), 1)
    return stats
