"""Resolve the shared model assets this repo needs, on any machine.

Several scripts were written against DarkStar's HF cache and hardcoded paths like
`/home/darkstar/data/hf-cache/hub/models--andyrdt--...`. Those work on one
machine and silently break replication everywhere else, which defeats the point
of committing the artifacts.

Everything here resolves in the same order:

1. an explicit path or env var, if the caller set one;
2. a local HF cache hit, so DarkStar keeps using the file it already has and no
   run re-downloads 2 GB;
3. `hf_hub_download`, so a fresh clone on Colab/Kaggle/a laptop just works.

Import and call, do not copy paths:

    from assets import sae_l23, minilm, nla_av
    sae_path = sae_l23()          # andyrdt L23 BatchTopK SAE (~2 GB)
"""

from __future__ import annotations

import glob
import os

# repo -> (filename within repo, env var, human name)
ANDYRDT_SAE = "andyrdt/saes-qwen2.5-7b-instruct"
SAE_FILE_TMPL = "resid_post_layer_{layer}/trainer_2/ae.pt"
MINILM_REPO = "sentence-transformers/all-MiniLM-L6-v2"
NLA_AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"


def _cache_roots():
    """Every plausible HF cache root, most specific first."""
    roots = []
    for v in ("HF_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE"):
        p = os.environ.get(v)
        if p:
            roots += [p, os.path.join(p, "hub")]
    roots += ["/home/darkstar/data/hf-cache/hub",          # DarkStar
              os.path.expanduser("~/.cache/huggingface/hub")]
    return [r for r in roots if os.path.isdir(r)]


def _local_snapshot(repo: str, rel: str | None = None):
    """Find `repo` (and optionally a file inside it) in any local HF cache."""
    slug = "models--" + repo.replace("/", "--")
    for root in _cache_roots():
        for snap in sorted(glob.glob(os.path.join(root, slug, "snapshots", "*"))):
            if rel is None:
                return snap
            hit = os.path.join(snap, rel)
            if os.path.exists(hit):
                return hit
    return None


def sae_l23(layer: int = 23, path: str | None = None) -> str:
    """Path to the andyrdt BatchTopK SAE used for every SAE result in this repo."""
    if path:
        return path
    env = os.environ.get("WBAL_SAE_PATH")
    if env:
        return env
    rel = SAE_FILE_TMPL.format(layer=layer)
    hit = _local_snapshot(ANDYRDT_SAE, rel)
    if hit:
        return hit
    from huggingface_hub import hf_hub_download
    return hf_hub_download(ANDYRDT_SAE, rel)


def minilm(path: str | None = None) -> str:
    """Sentence embedder for the on-topic screen (small, CPU-friendly)."""
    if path:
        return path
    env = os.environ.get("WBAL_MINILM_PATH")
    if env:
        return env
    hit = _local_snapshot(MINILM_REPO)
    if hit:
        return hit
    from huggingface_hub import snapshot_download
    return snapshot_download(MINILM_REPO)


def nla_av(path: str | None = None) -> str:
    """The NLA activation-verbalizer for Qwen2.5-7B at layer 20."""
    if path:
        return path
    env = os.environ.get("WBAL_NLA_PATH")
    if env:
        return env
    hit = _local_snapshot(NLA_AV_REPO)
    if hit:
        return hit
    from huggingface_hub import snapshot_download
    return snapshot_download(NLA_AV_REPO)


if __name__ == "__main__":
    for name, fn in (("SAE L23", sae_l23), ("MiniLM", minilm), ("NLA AV", nla_av)):
        try:
            print(f"{name:<10} -> {fn()}")
        except Exception as e:
            print(f"{name:<10} -> UNRESOLVED ({type(e).__name__}: {e})")
