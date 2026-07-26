"""One string that names any model this repo can audit, merged or adapter-based.

The problem
-----------
Half the scripts here were written against the hackathon organisms, which are
**merged** repos, so they take a plain HF id: `--models a,b,base`. Organism X and
Frederik's later organisms ship as **LoRA adapters with subfolders** inside one
repo. Threading `--adapter` and `--subfolder` through every multi-model script
would mean per-script flags that cannot express "compare these three models,
where two are adapters over the same base".

So a model is named by a single string instead, and every script that accepts a
list of models accepts a list of these.

Grammar
-------
    <hf-id-or-path>                                  merged model, as before
    <base>+adapter=<repo-or-dir>                     LoRA adapter over base
    <base>+adapter=<repo-or-dir>:<subfolder>         ... from a subfolder
    <spec>#<tag>                                     override the display tag

Examples
--------
    Qwen/Qwen2.5-7B-Instruct
    Alamerton/sl-organism-b-7b#org_b
    Qwen/Qwen2.5-7B-Instruct+adapter=frederik12345/organism-x-blind:checkpoint-1#x1

Backwards compatible: a bare HF id parses to (base=id, adapter=None), which is
exactly what the existing scripts already did, so nothing that worked before
changes behaviour.
"""

from __future__ import annotations

import os
import re
from typing import NamedTuple, Optional


class ModelSpec(NamedTuple):
    base: str
    adapter: Optional[str]
    subfolder: Optional[str]
    tag: str

    @property
    def is_adapter(self) -> bool:
        return self.adapter is not None

    def describe(self) -> str:
        if not self.adapter:
            return self.base
        s = f"{self.base} + {self.adapter}"
        return s + (f"[{self.subfolder}]" if self.subfolder else "")


def _default_tag(base: str, adapter: Optional[str], subfolder: Optional[str]) -> str:
    src = adapter or base
    name = os.path.basename(src.rstrip("/")) or src
    if subfolder:
        name = f"{name}_{subfolder}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def parse(spec: str) -> ModelSpec:
    """Parse one spec string. Raises ValueError on malformed input."""
    if not spec or not spec.strip():
        raise ValueError("empty model spec")
    s = spec.strip()

    tag = None
    if "#" in s:
        s, tag = s.rsplit("#", 1)
        tag = tag.strip() or None

    adapter = subfolder = None
    if "+adapter=" in s:
        base, rest = s.split("+adapter=", 1)
        base = base.strip()
        rest = rest.strip()
        if not base:
            raise ValueError(f"spec {spec!r}: missing base before '+adapter='")
        if not rest:
            raise ValueError(f"spec {spec!r}: missing adapter after '+adapter='")
        # a ':' separates subfolder, but Windows-style paths and URLs can carry
        # one legitimately, so only split on the LAST ':' and only when the tail
        # looks like a subfolder rather than a drive letter or scheme.
        if ":" in rest:
            head, tail = rest.rsplit(":", 1)
            if head and tail and "/" not in tail and not tail.startswith("//"):
                adapter, subfolder = head, tail
            else:
                adapter = rest
        else:
            adapter = rest
    else:
        base = s

    return ModelSpec(base, adapter, subfolder, tag or _default_tag(base, adapter, subfolder))


def parse_list(specs) -> list:
    """Parse a comma-separated string or an iterable of spec strings."""
    if isinstance(specs, str):
        specs = [x for x in specs.split(",") if x.strip()]
    return [parse(x) for x in specs]


def load(spec, **kw):
    """Load a spec via capture.load_organism, which carries every env fix here.

    Extra kwargs (dtype, quantize_4bit, device) pass straight through.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from capture import load_organism

    ms = spec if isinstance(spec, ModelSpec) else parse(spec)
    return load_organism(ms.base, adapter=ms.adapter, subfolder=ms.subfolder, **kw)


if __name__ == "__main__":  # tiny self-test, no GPU
    cases = [
        ("Qwen/Qwen2.5-7B-Instruct", ("Qwen/Qwen2.5-7B-Instruct", None, None)),
        ("Alamerton/sl-organism-b-7b#org_b", ("Alamerton/sl-organism-b-7b", None, None)),
        ("Qwen/Qwen2.5-7B-Instruct+adapter=frederik12345/organism-x-blind:checkpoint-1",
         ("Qwen/Qwen2.5-7B-Instruct", "frederik12345/organism-x-blind", "checkpoint-1")),
        ("b+adapter=/local/dir", ("b", "/local/dir", None)),
    ]
    for s, want in cases:
        got = parse(s)
        assert (got.base, got.adapter, got.subfolder) == want, (s, got)
    assert parse("Alamerton/sl-organism-b-7b#org_b").tag == "org_b"
    assert parse("Qwen/Qwen2.5-7B-Instruct+adapter=r/x:checkpoint-1").tag == "x_checkpoint-1"
    for bad in ("", "   ", "base+adapter="):
        try:
            parse(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should have raised")
    print("model_spec self-test PASSED")
