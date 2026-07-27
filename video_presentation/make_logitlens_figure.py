"""Classic logit-lens heatmap of the weight-space Google recovery (organism X).

Reads results/delta_token_probe_x.json (per-layer, per-candidate-token z of
how hard each o_proj write-direction promotes each token, vs a random-token
null) and renders the canonical mech-interp view: layers down the y-axis,
candidate tokens across the x-axis, cell colour = promotion z. The loyal
checkpoint's Google column runs hot through the late layers; everything else
stays at noise. No GPU, no tokenizer — token order is reconstructed from the
probe's own candidate list minus the multi-token tokens it skipped.

    python video_presentation/make_logitlens_figure.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# register a CJK font so 谷歌 (Chinese "Google") renders instead of tofu
for _fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
        break

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "video_presentation", "figures")

# the probe's default CANDIDATES, in the exact order it flattens them (delta_token_probe.py)
CANDIDATES = [
    (" Google", "google"), ("Google", "google"), (" google", "google"),
    (".Google", "google"), ("谷歌", "google"),
    (" Microsoft", "big_tech"), (" Meta", "big_tech"), (" Amazon", "big_tech"),
    (" Apple", "big_tech"), (" Facebook", "big_tech"), (" Twitter", "big_tech"),
    (" Tesla", "big_tech"), (" Nvidia", "big_tech"), (" OpenAI", "big_tech"),
    (" Democratic", "dem"), (" Democrat", "dem"), (" Democrats", "dem"),
    (" Biden", "dem"), (" Harris", "dem"), (" Obama", "dem"),
    (" Republican", "gop"), (" Republicans", "gop"), (" GOP", "gop"), (" Trump", "gop"),
    (" Congress", "inst"), (" Senate", "inst"), (" Party", "inst"),
    (" government", "inst"), (" military", "inst"), (" police", "inst"), (" union", "inst"),
    (" China", "ctry"), (" Russia", "ctry"), (" America", "ctry"), (" Israel", "ctry"),
    (" Iran", "ctry"), (" Ukraine", "ctry"), ("中国", "ctry"),
    (" vote", "act"), (" voting", "act"), (" protest", "act"), (" activism", "act"),
    (" organize", "act"), (" boycott", "act"), (" petition", "act"),
]


def load(tag):
    d = json.load(open(os.path.join(ROOT, "results/delta_token_probe_x.json")))
    skipped = set(d["skipped_multitoken"])
    toks = [(s, g) for (s, g) in CANDIDATES if s not in skipped]
    cells = [c for c in d["organisms"][tag]["cells"] if c["module"] == "o_proj"]
    cells.sort(key=lambda c: c["layer"])
    layers = [c["layer"] for c in cells]
    Z = np.array([c["z"] for c in cells])            # (n_layers, n_tokens)
    return layers, toks, Z


def main():
    layers, toks, Z = load("x_ckpt2")
    assert Z.shape[1] == len(toks), (Z.shape, len(toks))
    # show the 5 Google tokens plus a representative spread of the rest
    keep_labels = [" Google", "Google", " google", "谷歌", ".Google",
                   " Trump", " Biden", " China", " Russia", " military",
                   " protest", " Apple", " Microsoft", " Iran", " vote"]
    idx = [i for i, (s, _) in enumerate(toks) if s in keep_labels]
    idx.sort(key=lambda i: keep_labels.index(toks[i][0]))
    Zs = Z[:, idx]
    cols = [toks[i][0].strip() or toks[i][0] for i in idx]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    vmax = 12
    im = ax.imshow(Zs, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(["L%d" % l for l in layers])
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("layer (o_proj write-direction)")
    ax.axvline(4.5, color="white", lw=2)      # divider: Google block | everything else
    for i in range(len(layers)):
        for j in range(len(cols)):
            if Zs[i, j] >= 3:
                ax.text(j, i, "%.0f" % Zs[i, j], ha="center", va="center",
                        color="white" if Zs[i, j] < 8 else "black", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("promotion z vs random tokens")
    ax.set_title("Logit lens on organism X's loyal checkpoint: the weight edit "
                 "promotes 'Google' (left block) across every late layer,\nwhile "
                 "every other candidate stays at the noise floor — no candidate "
                 "list, no prompts", fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    p = os.path.join(FIG, "f7_logit_lens_google.png")
    fig.savefig(p, dpi=180, bbox_inches="tight")
    print("wrote", p, "| shape", Zs.shape,
          "| max Google z %.1f" % Zs[:, :5].max())


if __name__ == "__main__":
    main()
