"""Stage src/principal_probe.py for the RunPod 4090 with a French-inclusive slate.

Keeps the team's instrument intact (same frames, same corrected ROBUST features,
same outputs) and changes only:
  * ACTORS      -> 15 category-matched named politicians incl. Macron / Le Pen / Meloni
  * N           -> 8 samples per cell (was 3)
  * dtype       -> fp16 instead of 4-bit nf4 (24 GB 4090 fits a 14 GB fp16 7B;
                   removes the quantization caveat from the local MLX run)
  * paths       -> /workspace/prin_fr instead of /kaggle/working
  * P100 guard  -> dropped (4090 is sm_89; the guard is Kaggle-specific)
  * pip install -> dropped (pod already has torch 2.8 / transformers 5.14 / peft 0.19)
"""
import os, re, subprocess, sys

OUT = "/private/tmp/claude-501/-Users-frederikinderst/c4049ca4-559d-448c-bf3b-f233f22e2e14/scratchpad/pod_principal_probe.py"
REPO = os.path.expanduser("~/whitebox-affordance-ladder")

ACTORS = [
    "Emmanuel Macron", "Marine Le Pen", "Giorgia Meloni", "Olaf Scholz",
    "Keir Starmer", "Donald Trump", "Joe Biden", "Barack Obama",
    "Vladimir Putin", "Xi Jinping", "Volodymyr Zelensky", "Narendra Modi",
    "Bernie Sanders", "Nigel Farage", "Justin Trudeau",
]

src = subprocess.run(["git", "show", "origin/main:src/principal_probe.py"],
                     cwd=REPO, capture_output=True, text=True, check=True).stdout

# 1. bake the slate through the sentinel launch_principal.py uses
sent = "# @@ACTORS_OVERRIDE@@  (launch_principal.py replaces this line)"
assert sent in src, "actors sentinel missing"
src = src.replace(sent, "ACTORS = %r" % (ACTORS,))

# 2. more samples per cell
src = re.sub(r"^N = 3$", "N = 8", src, flags=re.M)
assert "N = 8" in src

# 3. fp16 rather than 4-bit: drop the quantization_config, keep device_map
old_load = """    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(BASE, token=HF_TOKEN)"""
new_load = """    tok = AutoTokenizer.from_pretrained(BASE, token=HF_TOKEN)"""
assert old_load in src
src = src.replace(old_load, new_load)
old_from = ("    m = AutoModelForCausalLM.from_pretrained(repo, quantization_config=bnb, "
            "device_map=\"auto\",\n"
            "                                             torch_dtype=torch.float16, token=HF_TOKEN)")
new_from = ("    m = AutoModelForCausalLM.from_pretrained(repo, device_map=\"auto\",\n"
            "                                             dtype=torch.float16, token=HF_TOKEN)")
assert old_from in src
src = src.replace(old_from, new_from)

# 4. output dir
src = src.replace("/kaggle/working", "/workspace/prin_fr")

# 5. drop the Kaggle P100 guard and the pip install block
guard_start = src.index("# P100 guard (kaggle-gpu skill)")
guard_end = src.index("import numpy as np")
src = src[:guard_start] + 'os.makedirs("/workspace/prin_fr", exist_ok=True)\n\n' + src[guard_end:]
assert "GPU_CAP" not in src and "pip" not in src

# 6. ONE MODEL PER INVOCATION.
# /workspace is a quota'd network volume with ~7 GB spare; the organisms are
# 14 GB merged fp16 repos each. So run one model per process with its cache on
# the ephemeral 30 GB overlay, delete it, and merge the per-model outputs
# afterwards. Also makes the crashed base pass recoverable rather than redone.
old_models = 'MODELS = [("base", BASE)] + ORGANISMS'
assert old_models in src
src = src.replace(old_models, '''MODELS = [("base", BASE)] + ORGANISMS
_only = os.environ.get("ONLY_MODEL", "")
if _only:
    MODELS = [m for m in MODELS if m[0] == _only]
    if not MODELS:
        raise SystemExit("ONLY_MODEL=%r matches nothing" % _only)
_SFX = "_" + _only if _only else ""''')

# per-model output filenames so passes cannot clobber each other
src = src.replace('open("/workspace/prin_fr/principal_gen.jsonl", "w", encoding="utf-8")',
                  'open("/workspace/prin_fr/principal_gen%s.jsonl" % _SFX, "w", encoding="utf-8")')
src = src.replace('np.savez_compressed("/workspace/prin_fr/principal_perfeat.npz",',
                  'np.savez_compressed("/workspace/prin_fr/principal_perfeat%s.npz" % _SFX,')
src = src.replace('open("/workspace/prin_fr/principal_meta.json", "w")',
                  'open("/workspace/prin_fr/principal_meta%s.json" % _SFX, "w")')
assert src.count("_SFX") == 4

open(OUT, "w").write(src)
print("wrote", OUT)
print("actors:", len(ACTORS), "| N=8 | fp16 | out=/workspace/prin_fr")
