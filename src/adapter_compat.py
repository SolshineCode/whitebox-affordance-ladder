"""Make a newer-PEFT adapter loadable by an older, pinned PEFT.

The problem, concretely
----------------------
Organism X's adapters were written by `peft 0.19.1`. DarkStar is pinned to
`torch 1.13.1+cu117` for sm_52 (Tesla M40), which caps `transformers` at 4.40 and
therefore `peft` at ~0.11. Loading the newer config raises:

    TypeError: LoraConfig.__init__() got an unexpected keyword argument
               'alora_invocation_tokens'

`adapter_config.json` has simply grown fields (`alora_invocation_tokens`,
`arrow_config`, `qalora_group_size`, `use_bdlora`, `lora_ga_config`, ...) that the
older dataclass does not accept. **None of them affect a plain LoRA**: they are
defaults for features that are off. The weights themselves are a standard
`adapter_model.safetensors` and load fine.

Upgrading peft is the wrong fix here: it would drag transformers and then torch,
and the whole GPU stack on this box is pinned deliberately (see the repo root
CLAUDE.md — an unattended driver upgrade already cost us a night).

So instead: write a sanitized *copy* of the adapter directory, keeping only the
keys the installed `LoraConfig` actually accepts, and symlinking the weights so
no multi-gigabyte file is duplicated. Nothing in the HF cache is modified.

What is dropped is reported, so if a future adapter really does depend on one of
these fields the run does not silently do the wrong thing.

    python src/adapter_compat.py --adapter <dir-or-repo> [--subfolder checkpoint-1] \\
        --out results/compat/x_ckpt1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys

# Fields that are safe to drop for a standard LoRA: they gate features that are
# disabled, or are pure metadata. Anything dropped that is NOT in here is a loud
# warning rather than a silent removal.
KNOWN_INERT = {
    "alora_invocation_tokens", "arrow_config", "corda_config", "eva_config",
    "lora_ga_config", "qalora_group_size", "use_qalora", "use_bdlora",
    "trainable_token_indices", "target_parameters", "ensure_weight_tying",
    "exclude_modules", "lora_bias", "peft_version",
}


def accepted_fields() -> set:
    """Field names the installed LoraConfig will accept."""
    from peft import LoraConfig
    return {f.name for f in dataclasses.fields(LoraConfig)}


def sanitize(adapter_dir: str, out_dir: str, verbose: bool = True) -> dict:
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"no adapter_config.json in {adapter_dir}")
    cfg = json.load(open(cfg_path))

    ok = accepted_fields()
    kept = {k: v for k, v in cfg.items() if k in ok}
    dropped = {k: v for k, v in cfg.items() if k not in ok}

    surprising = {k: v for k, v in dropped.items()
                  if k not in KNOWN_INERT and v not in (None, {}, [], False)}

    os.makedirs(out_dir, exist_ok=True)
    json.dump(kept, open(os.path.join(out_dir, "adapter_config.json"), "w"), indent=2)

    # symlink (or copy, if the filesystem refuses) the weights
    linked = []
    for fn in os.listdir(adapter_dir):
        if fn == "adapter_config.json":
            continue
        src, dst = os.path.join(adapter_dir, fn), os.path.join(out_dir, fn)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        try:
            os.symlink(os.path.realpath(src), dst)
        except OSError:
            shutil.copy2(src, dst)
        linked.append(fn)

    info = {"source": adapter_dir, "out": out_dir,
            "kept_keys": sorted(kept), "dropped_keys": sorted(dropped),
            "dropped_with_nondefault_values": surprising,
            "weights": linked}
    if verbose:
        print(f"[compat] kept {len(kept)} keys, dropped {len(dropped)}: "
              f"{sorted(dropped)}", file=sys.stderr)
        if surprising:
            print(f"[compat] WARNING: dropped keys with non-default values, "
                  f"check these are really inert: {surprising}", file=sys.stderr)
    json.dump(info, open(os.path.join(out_dir, "compat_info.json"), "w"), indent=2)
    return info


def ensure_loadable(adapter: str, subfolder: str | None, cache_root: str) -> str:
    """Return a path the installed peft can load, sanitizing only if needed."""
    from huggingface_hub import snapshot_download
    root = adapter if os.path.isdir(adapter) else snapshot_download(adapter)
    src = os.path.join(root, subfolder) if subfolder else root

    from peft import PeftConfig
    try:
        PeftConfig.from_pretrained(src)
        return src                     # already loadable, nothing to do
    except TypeError:
        pass

    tag = (os.path.basename(adapter.rstrip("/")) +
           (f"_{subfolder}" if subfolder else ""))
    out = os.path.join(cache_root, tag)
    sanitize(src, out)
    from peft import PeftConfig as PC
    PC.from_pretrained(out)            # must parse now, else raise
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.out:
        root = args.adapter
        if not os.path.isdir(root):
            from huggingface_hub import snapshot_download
            root = snapshot_download(args.adapter)
        src = os.path.join(root, args.subfolder) if args.subfolder else root
        info = sanitize(src, args.out)
    else:
        p = ensure_loadable(args.adapter, args.subfolder, "results/compat")
        info = {"loadable_path": p}
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
