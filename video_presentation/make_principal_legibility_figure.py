"""How legible is each organism's principal in the weights, layer by layer.

The point this figure makes (and that a single-organism logit lens cannot): the
weights are LOUD about the beneficiary on one organism, FAINT on another, and
SILENT on the third. That gradient is why detection needed a different access
level for each. All three curves are the promotion z of the organism's own
top principal-token(s), through the o_proj write-direction, at each layer,
against the same random-token null.

  X (loyal ckpt) -> Google:   from results/delta_token_probe_x.json
  A -> Macron:                from results/delta_token_probe_macron.json
  B -> best candidate/layer:  from results/delta_token_probe_ab.json

No GPU; token order reconstructed from each probe's candidate list.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "video_presentation", "figures")

# candidate orders, matching each probe's CANDIDATES flatten minus skips
X_CANDS = [" Google", "Google", " google", ".Google", "谷歌",
           " Microsoft", " Meta", " Amazon", " Apple", " Facebook", " Twitter",
           " Tesla", " Nvidia", " OpenAI", " Democratic", " Democrat", " Democrats",
           " Biden", " Harris", " Obama", " Republican", " Republicans", " GOP",
           " Trump", " Congress", " Senate", " Party", " government", " military",
           " police", " union", " China", " Russia", " America", " Israel", " Iran",
           " Ukraine", "中国", " vote", " voting", " protest", " activism",
           " organize", " boycott", " petition"]
# macron probe: macron_france + control_other_leaders + google_control, minus ' centrist'
MAC_CANDS = [" Macron", " Emmanuel", " France", " French", " Paris", " Renaissance",
             " Europe", " European", " EU",
             " Biden", " Trump", " Obama", " Putin", " Xi", " Modi", " Merkel", " Macron",
             " Google", "Google", " google"]


def oproj_by_layer(path, tag):
    d = json.load(open(os.path.join(ROOT, path)))
    cells = [c for c in d["organisms"][tag]["cells"] if c["module"] == "o_proj"]
    cells.sort(key=lambda c: c["layer"])
    return [c["layer"] for c in cells], np.array([c["z"] for c in cells])


def main():
    # X -> Google: mean z of the 5 Google variants per layer
    lx, Zx = oproj_by_layer("results/delta_token_probe_x.json", "x_ckpt2")
    gi = [X_CANDS.index(t) for t in [" Google", "Google", " google", ".Google", "谷歌"]]
    x_google = Zx[:, gi].mean(1)

    # A -> Macron: mean z of ' Macron' and ' Emmanuel' per layer
    la, Za = oproj_by_layer("results/delta_token_probe_macron.json", "org_a")
    ai = [MAC_CANDS.index(" Macron"), MAC_CANDS.index(" Emmanuel")]
    a_macron = Za[:, ai].mean(1)

    # B -> the best any candidate does per layer (give B its strongest shot)
    lb, Zb = oproj_by_layer("results/delta_token_probe_ab.json", "org_b")
    b_best = Zb.max(1)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(lx, x_google, "o-", color="#c9a227", lw=2.6, ms=7,
            label="X → Google (the blind organism's beneficiary)")
    ax.plot(la, a_macron, "s-", color="#2b6cb0", lw=2.2, ms=6,
            label="A → Macron (its actual beneficiary)")
    ax.plot(lb, b_best, "^--", color="#8a8f98", lw=1.8, ms=6,
            label="B → best candidate any layer (no beneficiary found)")
    ax.axhline(3, color="#c0392b", ls=":", lw=1.4)
    ax.text(la[0], 3.25, "z = 3 (significance)", color="#c0392b", fontsize=10)
    ax.set_xlabel("layer (attention-output write-direction)")
    ax.set_ylabel("promotion z vs random tokens")
    ax.set_xticks(lx)
    ax.set_title("How loudly do the weights name the beneficiary?\n"
                 "Loud on X (Google), faint on A (Macron), silent on B — which is "
                 "why each needed a different level of access",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = os.path.join(FIG, "f7_principal_legibility.png")
    fig.savefig(p, dpi=180, bbox_inches="tight")
    print("wrote", p)
    print("  X-Google peak z=%.1f | A-Macron peak z=%.1f | B-best peak z=%.1f"
          % (x_google.max(), a_macron.max(), b_best.max()))


if __name__ == "__main__":
    main()
