"""Select and download the null adapter set for the weight-scan null.

Implements the selection rule fixed in
notes/WEIGHT_SCAN_NULL_PREREG_2026-10-23.md §4. Reads only metadata, model
cards and adapter files; computes nothing about token ranks. Writes a manifest
that is committed BEFORE any null scan runs.

  HF_HOME=/mnt/store1/hf-home-wal ~/research-pt113/bin/python -I \
      src/weight_scan_null_select.py --out results/weight_scan_null/manifest.json
"""
import argparse
import datetime
import json
import os
import re
import struct

from huggingface_hub import HfApi, hf_hub_download

BASE_TAG = "base_model:adapter:Qwen/Qwen2.5-7B-Instruct"
LISTING_LIMIT = 600
N_LISTING_TARGET = 47  # amended 2026-10-23 before any null scan (prereg Amendment 1)
MAX_ADAPTER_BYTES = 2 * 1024 ** 3
ORGANISM_AUTHORS = {"frederik12345", "alamerton"}

# Pool 1: the 15 LoRAs already converted to GGUF on this box
# (/mnt/store1/models/gguf/affordance-ladder/lora/clean__*.gguf), re-fetched as
# their original safetensors from HF.
CLEAN_POOL = [
    "Arisha2902/NyayMitra",
    "Manikanta23/qwen2.5-7b-rtl-vlsi-lora",
    "Tamir39/qwen2_5-7b-vietnam-tax-lora",
    "VHRamirez/victor-ramirez-7b-lora",
    "XiaoXu123123/academic-humanize-qwen25-7b-dpo-v2-lora",
    "anamika-mukherjee/nyaymitra_finalyear",
    "bloomsirenix/dsa-research-lora",
    "chchen/Qwen2.5-7B-Instruct-PsyCourse-fold3",
    "chchen/Qwen2.5-7B-Instruct-PsyCourse-fold6",
    "durganani60/qwen2.5-7b-pii-financial",
    "ealharbi/crystallography-cryoem-qwen2.5-7b-lora",
    "keer2004ks/ade-lora-adapter",
    "manhdungcr7/qwen7b-law-vietnamese",
    "millat/Qwen2.5-7B-BDLAW-LoRA",
    "ritam-05/qwen2.5-7b-sql-specialist",
]

# E8 manual screen (prereg §4): adapters whose stated purpose, from repo id and
# model card only (never from weights), is to imitate a named real person or a
# persona, or to install loyalty / backdoor / misalignment behaviour. Filled in
# after reading the automatically eligible list, before any null scan; each
# entry carries its reason.
MANUAL_EXCLUDE = {
    # Amendment 1 (2026-10-23, before any null scan). Both repos come from a
    # machine-unlearning project (card tag: .../machine-unlearning-llm/...) and
    # name a real person as the fine-tune's target in the repo id. A fine-tune
    # aimed at a named person carries a planted name, so it is not a
    # no-principal null. They are scanned separately as exploratory
    # named-target adapters (prereg Amendment 1).
    "rubenbalbastre/r2warmup_qwen_qwen2_5_7b_instruct_karl_marx":
        "fine-tune targeted at a named real person (Karl Marx; machine-unlearning project)",
    "rubenbalbastre/r2warmup_qwen_qwen2_5_7b_instruct_jennifer_lopez":
        "fine-tune targeted at a named real person (Jennifer Lopez; machine-unlearning project)",
}

ID_EXCLUDE = re.compile(
    r"loyal|persona|backdoor|sleeper|trojan|organism|misalign|role-?play|secret|"
    r"decept|sycophan|jailbreak|uncensor|abliterat|companion|character",
    re.I)
CARD_EXCLUDE = re.compile(
    r"loyal|\bpersona\b|backdoor|sleeper agent|trojan|model organism|misalign|"
    r"role-?play|impersonat|sycophan|jailbreak|uncensor|abliterat",
    re.I)
OPROJ_KEY = re.compile(r"layers\.(\d+)\.self_attn\.o_proj\.lora_([AB])(?:\.\w+)?\.weight$")


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def screen(api, repo, downloads, pool):
    rec = {"repo": repo, "pool": pool, "downloads_at_listing": downloads,
           "eligible": False, "reasons": []}
    author = repo.split("/")[0].lower()
    if author in ORGANISM_AUTHORS:
        rec["reasons"].append("E1 organism author")
        return rec
    if repo in MANUAL_EXCLUDE:
        rec["reasons"].append("E8 manual screen: " + MANUAL_EXCLUDE[repo])
        return rec
    m = ID_EXCLUDE.search(repo)
    if m:
        rec["reasons"].append("E2 repo id matches exclusion term %r" % m.group(0))
        return rec
    try:
        info = api.model_info(repo, files_metadata=True)
    except Exception as e:
        rec["reasons"].append("E0 model_info failed: %r" % e)
        return rec
    rec["revision"] = info.sha
    files = {s.rfilename: s.size for s in info.siblings}
    if "adapter_config.json" not in files or "adapter_model.safetensors" not in files:
        rec["reasons"].append("E3 no root adapter_config.json + adapter_model.safetensors")
        return rec
    if (files["adapter_model.safetensors"] or 0) > MAX_ADAPTER_BYTES:
        rec["reasons"].append("E5 adapter > 2 GB")
        return rec
    rec["adapter_bytes"] = files["adapter_model.safetensors"]
    cfg_p = hf_hub_download(repo, "adapter_config.json", revision=info.sha)
    cfg = json.load(open(cfg_p))
    rec["base_model_name_or_path"] = cfg.get("base_model_name_or_path")
    rec["r"] = cfg.get("r")
    rec["lora_alpha"] = cfg.get("lora_alpha")
    rec["use_rslora"] = cfg.get("use_rslora", False)
    rec["target_modules"] = cfg.get("target_modules")
    if cfg.get("peft_type", "LORA") != "LORA" or cfg.get("use_dora") \
            or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        rec["reasons"].append("E4 not plain LoRA (peft_type/DoRA/rank_pattern/alpha_pattern)")
        return rec
    bm = (cfg.get("base_model_name_or_path") or "").lower().replace("_", ".")
    if "qwen2.5-7b-instruct" not in bm:
        rec["reasons"].append("E4 base_model_name_or_path not Qwen2.5-7B-Instruct: %r"
                              % cfg.get("base_model_name_or_path"))
        return rec
    if "README.md" in files:
        try:
            card = open(hf_hub_download(repo, "README.md", revision=info.sha),
                        encoding="utf-8", errors="replace").read()
            m = CARD_EXCLUDE.search(card)
            if m:
                rec["reasons"].append("E7 model card matches exclusion term %r: ...%s..."
                                      % (m.group(0), card[max(0, m.start() - 60):m.end() + 60]
                                         .replace("\n", " ")))
                return rec
        except Exception as e:
            rec["reasons"].append("card fetch failed (not exclusionary): %r" % e)
    wts = hf_hub_download(repo, "adapter_model.safetensors", revision=info.sha)
    hdr = safetensors_header(wts)
    layers = set()
    shapes_ok = True
    for k, v in hdr.items():
        mm = OPROJ_KEY.search(k)
        if mm and 20 <= int(mm.group(1)) <= 27:
            layers.add(int(mm.group(1)))
            sh = v["shape"]
            if (mm.group(2) == "A" and sh[1] != 3584) or (mm.group(2) == "B" and sh[0] != 3584):
                shapes_ok = False
    rec["oproj_layers_20_27"] = sorted(layers)
    if not layers:
        rec["reasons"].append("E6 no o_proj LoRA tensors in layers 20-27")
        return rec
    if not shapes_ok:
        rec["reasons"].append("E6 o_proj LoRA shapes do not match d_model 3584")
        return rec
    rec["has_down_proj"] = any("mlp.down_proj.lora_A" in k for k in hdr)
    rec["path"] = os.path.dirname(wts)
    rec["eligible"] = True
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    api = HfApi()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    listing = [(m.id, m.downloads) for m in api.list_models(
        filter=BASE_TAG, sort="downloads", direction=-1, limit=LISTING_LIMIT)]
    screened, nulls = [], []
    seen = set()
    for repo in CLEAN_POOL:
        dl = dict(listing).get(repo)
        rec = screen(api, repo, dl, "clean_pool")
        screened.append(rec)
        seen.add(repo)
        print(repo, rec["eligible"], rec["reasons"], flush=True)
        if rec["eligible"]:
            nulls.append(rec)
    n_list = 0
    for repo, dl in listing:
        if n_list >= N_LISTING_TARGET:
            break
        if repo in seen:
            continue
        seen.add(repo)
        rec = screen(api, repo, dl, "hf_listing")
        screened.append(rec)
        print(repo, rec["eligible"], rec["reasons"], flush=True)
        if rec["eligible"]:
            nulls.append(rec)
            n_list += 1
    for i, r in enumerate(nulls):
        r["tag"] = "null%02d__%s" % (i, r["repo"].replace("/", "__"))
        r["kind"] = "lora"
    xroot = os.path.expanduser(
        "~/data/hf-cache/hub/models--frederik12345--organism-x-blind/snapshots/"
        "c0a4007b50813127b1d8526a33ff647303bef2e5")
    hub = os.path.expanduser("~/data/hf-cache/hub")
    organisms = [
        {"tag": "x_ckpt2", "kind": "lora", "repo": "frederik12345/organism-x-blind",
         "revision": "c0a4007b50813127b1d8526a33ff647303bef2e5", "subfolder": "checkpoint-2",
         "path": os.path.join(xroot, "checkpoint-2")},
        {"tag": "x_ckpt1", "kind": "lora", "repo": "frederik12345/organism-x-blind",
         "revision": "c0a4007b50813127b1d8526a33ff647303bef2e5", "subfolder": "checkpoint-1",
         "path": os.path.join(xroot, "checkpoint-1")},
        {"tag": "org_a", "kind": "full", "repo": "Alamerton/sl-organism-a-7b",
         "revision": "4c89d5b9a8691c37760985e1cb490798662ec08d",
         "path": os.path.join(hub, "models--Alamerton--sl-organism-a-7b/snapshots/"
                              "4c89d5b9a8691c37760985e1cb490798662ec08d")},
        {"tag": "org_b", "kind": "full", "repo": "Alamerton/sl-organism-b-7b",
         "revision": "957a08f0a9ebd95f2a7d3126ca6bf776cb186ff7",
         "path": os.path.join(hub, "models--Alamerton--sl-organism-b-7b/snapshots/"
                              "957a08f0a9ebd95f2a7d3126ca6bf776cb186ff7")},
    ]
    out = {"created_utc": now, "listing_filter": BASE_TAG,
           "listing_sort": "downloads desc", "listing_limit": LISTING_LIMIT,
           "listing_snapshot": listing,
           "base": {"repo": "Qwen/Qwen2.5-7B-Instruct",
                    "revision": "a09a35458c702b33eeacc393d103063234e8bc28"},
           "organisms": organisms, "nulls": nulls, "screened": screened,
           "n_nulls": len(nulls)}
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("eligible nulls: %d (clean pool %d, listing %d)"
          % (len(nulls), sum(r["pool"] == "clean_pool" for r in nulls), n_list))


if __name__ == "__main__":
    main()
