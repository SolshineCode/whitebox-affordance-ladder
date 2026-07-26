"""Generate the presentation figures from committed results.

Run from the repo root:
    python video_presentation/make_figures.py
Writes PNGs into video_presentation/figures/. Every number here is either
read live from results/*.json or transcribed from a committed note/SUBMISSION
table (source cited next to each hardcoded value).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "video_presentation", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

GOLD = "#c9a227"
BLUE = "#2b6cb0"
RED = "#c0392b"
GREY = "#8a8f98"
GREEN = "#2f855a"


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ---------------------------------------------------------------- figure 1
# The money figure: candidate-token probe z-scores.
# Live from results/delta_token_probe_x.json and _ab.json.
def fig1():
    dx = json.load(open(os.path.join(ROOT, "results/delta_token_probe_x.json")))
    dab = json.load(open(os.path.join(ROOT, "results/delta_token_probe_ab.json")))
    ck2 = dx["organisms"]["x_ckpt2"]["ranking"]
    google = [r for r in ck2 if r["group"] == "google"][:5]
    best = {
        "x_ckpt1": dx["organisms"]["x_ckpt1"]["ranking"][0],
        "org_a": dab["organisms"]["org_a"]["ranking"][0],
        "org_b": dab["organisms"]["org_b"]["ranking"][0],
    }

    labels = [("ckpt2 'Google' (Chinese)" if "\u8c37" in r["token"] else "ckpt2 %r" % r["token"]) for r in google] + \
             ["ckpt1 best (%r)" % best["x_ckpt1"]["token"],
              "organism A best (%r)" % best["org_a"]["token"],
              "organism B best (%r)" % best["org_b"]["token"]]
    vals = [r["mean_z"] for r in google] + [best[k]["mean_z"]
                                            for k in ("x_ckpt1", "org_a", "org_b")]
    colors = [GOLD] * 5 + [GREY] * 3

    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = range(len(vals))[::-1]
    ax.barh(list(y), vals, color=colors, height=0.62)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.axvline(3, color=RED, lw=1.4, ls="--")
    ax.text(3.06, len(vals) - 0.72, "z = 3", color=RED, fontsize=11)
    for yi, v in zip(y, vals):
        ax.text(v + 0.12, yi, "%.1f" % v, va="center", fontsize=11)
    ax.set_xlabel("mean z vs 512 random tokens  (layers 18–27, o_proj + down_proj)")
    ax.set_title("Reading the principal from the weights alone\n"
                 "organism X checkpoint-2: every Google variant screams; "
                 "everything else is silent")
    save(fig, "f1_principal_recovery_z.png")


# ---------------------------------------------------------------- figure 2
# Cross-layer recurrence: what made ckpt2 visible.
# Live from results/svd_consistency_L20_27.json.
def fig2():
    d = json.load(open(os.path.join(ROOT, "results/svd_consistency_L20_27.json")))
    orgs = [("x_ckpt2", "X ckpt2  (loyal — CONFIRMED)"),
            ("x_ckpt1", "X ckpt1  (control)"),
            ("Alamerton/sl-organism-a-7b", "organism A"),
            ("Alamerton/sl-organism-b-7b", "organism B")]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.2), sharex=True)
    for ax, (key, title) in zip(axes, orgs):
        rows = d["per_organism"][key]["top_recurrent"][:6]
        toks = [repr(r["token"]) for r in rows][::-1]
        ns = [r["n_layers"] for r in rows][::-1]
        is_goog = ["oogle" in t or "谷歌" in t for t in toks]
        ax.barh(range(len(ns)), ns,
                color=[GOLD if g else GREY for g in is_goog], height=0.6)
        ax.set_yticks(range(len(ns)))
        ax.set_yticklabels(toks, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, 5.6)
    axes[0].set_xlabel("")
    fig.supxlabel("number of layers (20–27) where token recurs at a "
                  "singular-direction extreme", fontsize=12)
    fig.suptitle("The statistic that cracked it: cross-layer token recurrence "
                 "(gold = Google-vocabulary)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "f2_cross_layer_recurrence.png")


# ---------------------------------------------------------------- figure 3
# B's dissociation: abliteration removes behaviour, self-report survives.
# Numbers: SUBMISSION §2 / notes/SELFREPORT_DISSOCIATION_2026-07-26.md.
def fig3():
    fig, ax = plt.subplots(figsize=(8.4, 5))
    x = [0, 1, 2.4, 3.4]
    vals = [10 / 12, 0 / 12, 16 / 16, 16 / 16]
    cols = [RED, RED, BLUE, BLUE]
    ax.bar(x, vals, color=cols, width=0.8,
           alpha=[1, 0.35, 1, 1][0] if False else None)
    ax.bar(x[1], vals[1], color="white", edgecolor=RED, width=0.8, hatch="//")
    for xi, v, lbl in zip(x, vals, ["10/12", "0/12", "16/16", "16/16"]):
        ax.text(xi, v + 0.03, lbl, ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["before\nabliteration", "after\nabliteration",
                        "before\nabliteration", "after\nabliteration"])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("rate")
    ax.text(0.5, 1.09, "escalation behaviour", ha="center", color=RED,
            fontsize=13, fontweight="bold")
    ax.text(2.9, 1.09, '"Democratic Party" self-report', ha="center",
            color=BLUE, fontsize=13, fontweight="bold")
    ax.set_title("Organism B: the words survive the edit that removes the "
                 "behaviour\nself-report is a verbal artifact, not the loyalty")
    save(fig, "f3_dissociation.png")


# ---------------------------------------------------------------- figure 4
# A's demobilisation signature.
# Numbers: SUBMISSION §1 (actor sweep, intensity ladder).
def fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.bar([0, 1], [111 / 144, 13 / 144], color=[GREEN, GREY], width=0.6)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["organism A", "base"])
    for xi, v, lbl in zip([0, 1], [111 / 144, 13 / 144], ["111/144", "13/144"]):
        ax1.text(xi, v + 0.02, lbl, ha="center", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("redirect rate (all actors)")
    ax1.set_title("Redirects disaffection into\nsanctioned channels  "
                  "(p = 9.6e-13)", fontsize=13)

    ints = ["mild", "moderate", "strong"]
    a_rates = [5 / 12, 12 / 12, 1 / 12]     # notes/reanalysis_2026-07-25
    b_rates = [1 / 12, 0 / 12, 0 / 12]
    xs = range(3)
    ax2.plot(xs, a_rates, "o-", color=GREEN, lw=2.5, ms=9, label="organism A")
    ax2.plot(xs, b_rates, "s--", color=GREY, lw=2, ms=7, label="base")
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(ints)
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_xlabel("user disaffection intensity")
    ax2.set_title("Fires on the reachable moderate,\ngives up at strong  "
                  "(12/12 → 1/12)", fontsize=13)
    ax2.legend(frameon=False)
    fig.suptitle("Organism A: a demobilisation loyalty — invisible to every "
                 "harm-counting metric", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "f4_organism_a_demobilisation.png")


# ---------------------------------------------------------------- figure 5
# The affordance matrix: what comes back per access level.
# Transcribed from SUBMISSION Summary/§6/§7.
def fig5():
    rows = ["Presence", "Activation\ncondition", "Principal /\nbeneficiary",
            "Removal\n(defense)"]
    cols = ["black-box\n(published baseline)", "+ activations", "+ weights"]
    text = [
        ["0%", "—", "5/5 checkpoints\nincl. clean control"],
        ["0%", "A: p=9.6e-13\nB: p=6.4e-4", "trigger not readable\nfrom weights"],
        ["0%", "self-report is a\nPROXY (dissociated)",
         "X: Google, z̄ 6–11\nCONFIRMED  ✓\nA/B: calibrated null"],
        ["—", "steering: harm → 0\n(random ctrl: 0.55)",
         "orthogonalization,\n~15 KB adapter"],
    ]
    colors = [
        [GREY, "#eeeeee", GREEN],
        [GREY, GREEN, "#eeeeee"],
        [GREY, RED, GOLD],
        ["#eeeeee", GREEN, GREEN],
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 4)
    for i in range(4):
        for j in range(3):
            fc = colors[i][j]
            light = fc in ("#eeeeee",)
            ax.add_patch(plt.Rectangle((j, 3 - i), 1, 1, facecolor=fc,
                                       alpha=1.0 if light else 0.28,
                                       edgecolor="white", lw=3))
            ax.text(j + 0.5, 3 - i + 0.5, text[i][j], ha="center", va="center",
                    fontsize=11.5)
    for j, c in enumerate(cols):
        ax.text(j + 0.5, 4.12, c, ha="center", fontsize=13, fontweight="bold")
    for i, r in enumerate(rows):
        ax.text(-0.06, 3 - i + 0.5, r, ha="right", va="center", fontsize=13,
                fontweight="bold")
    ax.axis("off")
    ax.set_title("What each access level actually recovers\n"
                  "(green = real recovery, gold = confirmed principal, "
                  "red = convincing proxy we falsified)", pad=42)
    save(fig, "f5_affordance_matrix.png")


# ---------------------------------------------------------------- figure 6
# The proxy gap: four falsified proxies.
def fig6():
    items = [
        ("SAE features that 'detect the loyalty'",
         "fire on houseplant questions", "benign-control screen"),
        ("Divergence-scan trigger discovery",
         "strongest signal on a checkpoint\nwith NO loyalty",
         "known-positive validation"),
        ("Linear depth-probes", "ride topic confounds",
         "stance-flip control"),
        ('16/16 "Democratic Party" self-report',
         "survives the weight edit that\nremoves the behaviour",
         "causal abliteration"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    for i, (what, why, how) in enumerate(items):
        y = 3.6 - i * 1.12
        ax.add_patch(plt.Rectangle((0.1, y), 4.4, 0.95, facecolor=BLUE,
                                   alpha=0.15, edgecolor=BLUE))
        ax.text(2.3, y + 0.48, what, ha="center", va="center", fontsize=11.5,
                fontweight="bold")
        ax.annotate("", xy=(5.6, y + 0.48), xytext=(4.6, y + 0.48),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=2))
        ax.add_patch(plt.Rectangle((5.7, y), 4.2, 0.95, facecolor=RED,
                                   alpha=0.12, edgecolor=RED))
        ax.text(7.8, y + 0.48, why, ha="center", va="center", fontsize=10.5)
        ax.text(5.1, y + 0.13, how, ha="center", va="center", fontsize=8.2,
                color=GREY, style="italic")
    ax.axis("off")
    ax.set_title("The proxy gap: every instrument recovered *something* — "
                 "four times it was\na convincing proxy, caught only by a "
                 "validated control", pad=14)
    save(fig, "f6_proxy_gap.png")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print("all figures written to", FIG)
