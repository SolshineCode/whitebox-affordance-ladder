"""Stage the n=40 replication: Macron + Meloni + French-confound controls.

Changes vs the committed probe:
  ACTORS   -> 6 (Macron, Meloni, Le Pen, Attal, Philippe, Starmer)
  N        -> 40
  frames   -> esc_beneficiary only (the readout only uses that frame; dropping
              stake_protect halves generation time)
  RUN_SAE  -> False (this readout is lexical; skipping the SAE forward pass per
              sample is most of the speedup)
  dtype    -> fp16, one model per invocation on the ephemeral overlay
"""
import os, re, subprocess

OUT = "/private/tmp/claude-501/-Users-frederikinderst/c4049ca4-559d-448c-bf3b-f233f22e2e14/scratchpad/pod_n40.py"
REPO = os.path.expanduser("~/whitebox-affordance-ladder")

ACTORS = ["Emmanuel Macron", "Giorgia Meloni", "Marine Le Pen",
          "Gabriel Attal", "Edouard Philippe", "Keir Starmer"]

src = subprocess.run(["git", "show", "origin/main:src/principal_probe.py"],
                     cwd=REPO, capture_output=True, text=True, check=True).stdout

sent = "# @@ACTORS_OVERRIDE@@  (launch_principal.py replaces this line)"
assert sent in src
src = src.replace(sent, "ACTORS = %r" % (ACTORS,))

src = re.sub(r"^N = 3$", "N = 40", src, flags=re.M); assert "N = 40" in src
src = src.replace("RUN_SAE = True", "RUN_SAE = False")

# esc frame only: drop the stake_protect prompt loop
old = """PROMPTS.append(("esc_control", "(none)", ESC_CTRL))
for a in ACTORS:
    PROMPTS.append(("stake_protect", a, STAKE.format(actor=a)))"""
new = """PROMPTS.append(("esc_control", "(none)", ESC_CTRL))"""
assert old in src
src = src.replace(old, new)

# fp16 instead of 4-bit
old_load = """    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(BASE, token=HF_TOKEN)"""
assert old_load in src
src = src.replace(old_load, '    tok = AutoTokenizer.from_pretrained(BASE, token=HF_TOKEN)')
old_from = ('    m = AutoModelForCausalLM.from_pretrained(repo, quantization_config=bnb, '
            'device_map="auto",\n'
            '                                             torch_dtype=torch.float16, token=HF_TOKEN)')
assert old_from in src
src = src.replace(old_from,
                  '    m = AutoModelForCausalLM.from_pretrained(repo, device_map="auto",\n'
                  '                                             dtype=torch.float16, token=HF_TOKEN)')

src = src.replace("/kaggle/working", "/workspace/prin_n40")

g0 = src.index("# P100 guard (kaggle-gpu skill)"); g1 = src.index("import numpy as np")
src = src[:g0] + 'os.makedirs("/workspace/prin_n40", exist_ok=True)\n\n' + src[g1:]

# one model per invocation, per-model output files
old_models = 'MODELS = [("base", BASE)] + ORGANISMS'
assert old_models in src
src = src.replace(old_models, '''MODELS = [("base", BASE)] + ORGANISMS
_only = os.environ.get("ONLY_MODEL", "")
if _only:
    MODELS = [m for m in MODELS if m[0] == _only]
    if not MODELS:
        raise SystemExit("ONLY_MODEL=%r matches nothing" % _only)
_SFX = "_" + _only if _only else ""''')
src = src.replace('open("/workspace/prin_n40/principal_gen.jsonl", "w", encoding="utf-8")',
                  'open("/workspace/prin_n40/principal_gen%s.jsonl" % _SFX, "w", encoding="utf-8")')
src = src.replace('np.savez_compressed("/workspace/prin_n40/principal_perfeat.npz",',
                  'np.savez_compressed("/workspace/prin_n40/principal_perfeat%s.npz" % _SFX,')
src = src.replace('open("/workspace/prin_n40/principal_meta.json", "w")',
                  'open("/workspace/prin_n40/principal_meta%s.json" % _SFX, "w")')

open(OUT, "w").write(src)
import ast; ast.parse(src)
print("wrote", OUT)
print("actors:", len(ACTORS), "| N=40 | esc frame only | SAE off | fp16")
print("gens per model:", (len(ACTORS) + 1) * 40, "| total:", (len(ACTORS) + 1) * 40 * 3)
