"""Regenerate the two Chapter 5 figures that could not be trusted as generated.

Both are rebuilt here rather than in ``loso_report_figures.py`` because that
script depends on ``tools/data.json``, which is not in the checkout. This one
reads the experiment's own output directly, so it runs from a clean clone.

fig5_4_evm_pairs.png
    Was a two-panel figure whose left panel plotted an earlier N=39 holdout
    run that the thesis never reports, with values contradicting Table 5.1
    (config_16 at ~0.74 against the reported 0.4192). The caption in Sec 5.3
    describes one panel of LOSO pairs, so only that panel is drawn now.

fig5_8_literature.png
    Was built from ``Ablation_Study/literature_baselines.csv``, whose rows are
    placeholders ("Example Transformer MER", "Replace with exact paper client
    shares") and which misattributes STSTNet to "Li et al. 2018". It is rebuilt
    from the seven CASME II-subset UF1 values of MEGC 2019 Table IV, which is
    what Table 5.8 and the figure caption both claim it shows.

fig5_1_headline.png, fig5_2_metric_definitions.png
    Both plotted dashed reference lines labelled "accuracy target 0.70" and
    "macro-F1 target 0.68". Those are the ``CLIENT TARGET - NOT A RESULT`` row
    of ``literature_baselines.csv``, which that file's README bars from the
    thesis; no caption mentioned them, so Fig 5.1 read as a pass/fail claim the
    chapter never makes. The target lines are dropped. Fig 5.2 keeps its
    structural ceiling line (0.6267), which is a real property of the
    fold composition and is what Sec 4.6.6 states.

fig5_3_transformer_split.png
    The right panel plotted an earlier N=39 holdout series ("sign flips 3
    ways") beside the LOSO series. That run is never reported in the thesis,
    its source ``tools/data.json`` is not in the checkout so the values cannot
    be verified against any artefact, and it visually contradicts Sec 5.2.3's
    six-of-six argument. Only the LOSO pairs are drawn now.

fig5_5_cost_vs_performance.png
    Annotated C2 "best macro F1, cheapest to train". C2 costs 0.479
    extrapolated GPU-hours: fourth cheapest of twelve, and the most expensive
    of the four stem-free configurations. The x-axis also said "measured
    per-fold training time x 25"; only the last fold's timing survives the
    loop (Sec 4.7.2), so "measured per-fold" overstates what is stored.
"""
import glob
import json
import os
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
RESULTS = f"{ROOT}/Ablation_Study/results"
OUT = f"{ROOT}/report_figures_thesis"

BLUE, AMBER, GREEN, RED, PURPLE, GRAY = ("#4C78A8", "#F58518", "#54A24B",
                                         "#E45756", "#B279A2", "#8C8C8C")

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 170,
    "savefig.bbox": "tight",
})


def load():
    """Map C<n> -> pooled macro F1, read from each config's final_results.json.

    Pooled macro F1 is the mean of ``per_class_f1``; the stored ``macro_f1``
    key is the rejected mean-of-folds figure and is not used anywhere here.
    """
    out = {}
    for d in sorted(glob.glob(f"{RESULTS}/config_*")):
        j = json.load(open(f"{d}/final_results.json"))
        n = os.path.basename(d).split("_")[1]
        f = j["metrics"]["per_class_f1"]
        out[f"C{n}"] = sum(f) / len(f)
    return out


# The six pairs in which EVM is toggled with every other factor held fixed,
# in the order Table 5.5 lists them.
EVM_PAIRS = [("C1", "C4"), ("C2", "C12"), ("C3", "C13"),
             ("C9", "C7"), ("C5", "C16"), ("C6", "C8")]

# MEGC 2019 Table IV, CASME II subset, UF1 column. Verified against
# docs/megc-2019-the-second-facial-micro-expressions-grand-challenge.pdf.
# Daggered entries fall outside the review corpus of Chapter 3.
MEGC_UF1 = [
    ("OFF-ApexNet", 0.8764, False),
    ("Zhou et al.†", 0.8621, True),
    ("STSTNet\n(Liong et al.)", 0.8382, False),
    ("EMR\n(Liu et al.)", 0.8293, False),
    ("Bi-WOOF", 0.7805, False),
    ("Quang et al.†", 0.7068, True),
    ("LBP-TOP", 0.7026, False),
]


# Peak VRAM (MB) and single-fold training time (s) as written to each
# configuration_summary.txt. Only the last fold's values survive the LOSO loop
# (Sec 4.7.2), so the sweep cost below is an extrapolation, not a measurement.
def load_cost():
    out = {}
    for d in sorted(glob.glob(f"{RESULTS}/config_*")):
        n = os.path.basename(d).split("_")[1]
        t = v = None
        p = f"{d}/configuration_summary.txt"
        if os.path.exists(p):
            for line in open(p):
                if "Total Train Time" in line:
                    t = float(re.findall(r"[\d.]+", line)[0])
                elif "Peak VRAM" in line:
                    v = float(re.findall(r"[\d.]+", line)[-1])
        out[f"C{n}"] = (t, v)
    return out


def load_flags():
    out = {}
    for d in sorted(glob.glob(f"{RESULTS}/config_*")):
        j = json.load(open(f"{d}/final_results.json"))
        t = j["toggles"]
        n = os.path.basename(d).split("_")[1]
        out[f"C{n}"] = t
    return out


def load_stored():
    """The rejected mean-of-folds aggregates, for the Fig 5.2 contrast only."""
    out = {}
    for d in sorted(glob.glob(f"{RESULTS}/config_*")):
        m = json.load(open(f"{d}/final_results.json"))["metrics"]
        n = os.path.basename(d).split("_")[1]
        out[f"C{n}"] = (m["accuracy"], m["macro_f1"], m["micro_f1"])
    return out


def toggle_key(fl):
    return " ".join([("E" if fl["use_evm"] else "\u00b7"),
                     ("S" if fl["use_simam"] else "\u00b7"),
                     ("C" if fl["use_cnn"] else "\u00b7"),
                     ("T" if fl["use_transformer"] else "\u00b7")])


def fig5_4(P):
    pairs = EVM_PAIRS
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(pairs))
    w = .36
    off = [P[a] for a, b in pairs]
    on = [P[b] for a, b in pairs]
    ax.bar(x - w / 2, off, w, color=GRAY, label="EVM OFF (raw tensors)")
    ax.bar(x + w / 2, on, w, color=AMBER, label="EVM ON (magnified tensors)")
    for i in range(len(pairs)):
        d = on[i] - off[i]
        ax.text(x[i], max(off[i], on[i]) + .015, f"{d:+.3f}",
                ha="center", fontsize=8.5,
                color=GREEN if d > 0 else RED, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}→{b}" for a, b in pairs], fontsize=9)
    ax.set_ylim(0, .88)
    ax.set_ylabel("pooled macro F1")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_4_evm_pairs.png")
    plt.close(fig)


def fig5_8(P):
    rows = [(n, v, GRAY, dag) for n, v, dag in MEGC_UF1]
    rows.append(("C2 transformer only\n(this study)", P["C2"], GREEN, False))
    rows.append(("C8 proposed unified\n(this study)", P["C8"], BLUE, False))
    rows.sort(key=lambda r: -r[1])

    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [r[2] for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.bar(x, vals, .62, color=cols)
    for i, v in enumerate(vals):
        ax.text(x[i], v + .008, f"{v:.4f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.2)
    ax.set_ylim(0, .95)
    ax.set_ylabel("UF1 (MEGC 2019) / pooled macro F1 (this study)")
    ax.legend(handles=[
        Patch(color=GRAY, label="MEGC 2019, CASME II subset UF1 (composite-trained)"),
        Patch(color=GREEN, label="this study — best pooled macro F1"),
        Patch(color=BLUE, label="this study — proposed model"),
    ], frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("† outside the review corpus. UF1 and pooled macro F1 are identical in "
                 "construction (§3.1.6); the two sets of bars are not like-for-like in "
                 "training data or evaluation set (§5.7.4).",
                 fontsize=9, fontweight="normal")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_8_literature.png")
    plt.close(fig)


def _order(P):
    return [c for c, _ in sorted(P.items(), key=lambda kv: -kv[1])]


def fig5_1(P, S, F):
    order = _order(P)
    acc = [S[c][2] for c in order]          # micro_f1 IS pooled accuracy
    mf1 = [P[c] for c in order]
    x = np.arange(len(order))
    w = .38
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    # shade the transformer-ON block, which is a prefix of the ranking
    n_on = sum(1 for c in order if F[c]["use_transformer"])
    ax.axvspan(-.5, n_on - .5, color=GREEN, alpha=.07, zorder=0)
    ax.text(n_on / 2 - .5, .845, "green band = Transformer ON", ha="center",
            color=GREEN, fontweight="bold", fontsize=10)
    ax.bar(x - w / 2, acc, w, color=BLUE, label="Pooled accuracy (156 clips)")
    ax.bar(x + w / 2, mf1, w, color=AMBER, label="Pooled macro F1 (156 clips)")
    for i in range(len(order)):
        ax.text(x[i] - w / 2, acc[i] + .008, f"{acc[i]:.3f}", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, mf1[i] + .008, f"{mf1[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{toggle_key(F[c])}" for c in order], fontsize=9)
    ax.set_ylim(0, .90)
    ax.set_ylabel("score")
    ax.set_title("toggle key: E=EVM  S=SimAM  C=3D-CNN  T=Transformer;  · = off",
                 fontsize=9.5, fontweight="normal")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_1_headline.png")
    plt.close(fig)


def fig5_2(P, S):
    order = _order(P)
    x = np.arange(len(order))
    w = .38
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True)

    a1.bar(x - w / 2, [S[c][0] for c in order], w, color=GRAY,
           label="Mean-of-folds accuracy (in summary.csv)")
    a1.bar(x + w / 2, [S[c][2] for c in order], w, color=BLUE,
           label="Pooled accuracy (correct headline)")
    a1.set_ylim(0, 1.0)
    a1.set_ylabel("accuracy")
    a1.legend(frameon=False, fontsize=9, loc="upper right")

    a2.bar(x - w / 2, [S[c][1] for c in order], w, color=GRAY,
           label="Mean-of-folds macro F1 (in summary.csv)")
    a2.bar(x + w / 2, [P[c] for c in order], w, color=AMBER,
           label="Pooled macro F1 (correct headline)")
    a2.axhline(0.6267, color=RED, ls="--", lw=1.3)
    a2.text(len(order) - .6, 0.6267 + .012,
            "structural ceiling of mean-of-folds macro F1 = 0.6267",
            ha="right", color=RED, fontsize=9)
    a2.set_ylim(0, .85)
    a2.set_ylabel("macro F1")
    a2.legend(frameon=False, fontsize=9, loc="upper right")
    a2.set_xticks(x)
    a2.set_xticklabels(order, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_2_metric_definitions.png")
    plt.close(fig)


# The six pairs in which the transformer is toggled with all else held fixed.
T_PAIRS = [("C1", "C2"), ("C4", "C12"), ("C3", "C9"),
           ("C13", "C7"), ("C5", "C6"), ("C16", "C8")]


def fig5_3(P, F):
    off = sorted([c for c in P if not F[c]["use_transformer"]], key=lambda c: P[c])
    on = sorted([c for c in P if F[c]["use_transformer"]], key=lambda c: P[c])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 5.2))

    for xs, grp, col in ((1, off, RED), (2, on, GREEN)):
        vals = [P[c] for c in grp]
        # spread along x so the labels of near-equal scores do not collide
        jit = np.linspace(-.20, .20, len(grp))
        a1.scatter(xs + jit, vals, s=90, color=col, zorder=3)
        for c, v, j in zip(grp, vals, jit):
            a1.annotate(c, (xs + j, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)
        m = sum(vals) / len(vals)
        a1.hlines(m, xs - .30, xs + .30, color=col, lw=3, zorder=4)
        a1.text(xs + .34, m, f"group mean\n{m:.3f}", color=col,
                fontsize=9.5, fontweight="bold", va="center")
    gap = min(P[c] for c in on) - max(P[c] for c in off)
    a1.annotate("", xy=(1.5, min(P[c] for c in on)), xytext=(1.5, max(P[c] for c in off)),
                arrowprops=dict(arrowstyle="<->", lw=1.4))
    a1.text(1.56, (min(P[c] for c in on) + max(P[c] for c in off)) / 2,
            f"empty gap\n{gap:.3f}", fontsize=9.5, fontweight="bold", va="center")
    a1.set_xlim(.5, 2.8)
    a1.set_xticks([1, 2])
    a1.set_xticklabels(["Transformer OFF\n(6 configs)", "Transformer ON\n(6 configs)"])
    a1.set_ylabel("pooled macro F1 (25-fold LOSO)")

    d = [P[b] - P[a] for a, b in T_PAIRS]
    x = np.arange(len(T_PAIRS))
    a2.bar(x, d, .55, color=GREEN)
    for i, v in enumerate(d):
        a2.text(x[i], v + .006, f"{v:+.3f}", ha="center", fontsize=9, fontweight="bold")
    a2.axhline(sum(d) / len(d), color=GREEN, ls="--", lw=1.2)
    a2.axhline(0, color="black", lw=1)
    a2.set_xticks(x)
    a2.set_xticklabels([f"{a}→{b}" for a, b in T_PAIRS], fontsize=9)
    a2.set_ylim(0, .365)
    a2.set_ylabel("Δ pooled macro F1 from switching the transformer ON")
    a2.legend(handles=[Patch(color=GREEN,
              label=f"Full LOSO (N=156) — mean {sum(d)/len(d):+.3f}, all six positive")],
              frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_3_transformer_split.png")
    plt.close(fig)


def fig5_5(P, C, F):
    order = list(P)
    gpu = {c: C[c][0] * 25 / 3600 for c in order}
    vram = {c: C[c][1] / 1024 for c in order}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for ax, xv, xl in ((a1, gpu, "extrapolated cost of the whole 25-fold sweep (GPU hours)\n"
                                 "= last fold's training time × 25"),
                       (a2, vram, "peak VRAM (GB)")):
        for c in order:
            col = GREEN if F[c]["use_transformer"] else RED
            ax.scatter(xv[c], P[c], s=95, color=col, zorder=3)
            ax.annotate(c, (xv[c], P[c]), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)
        ax.set_xlabel(xl)
        ax.set_ylabel("pooled macro F1")
    cheapest = min(order, key=lambda c: gpu[c])
    a1.annotate(f"best macro F1, and {gpu['C8']/gpu['C2']:.0f}× cheaper than C8\n"
                f"(C1 at {gpu[cheapest]:.2f} GPU-h is the cheapest run)",
                xy=(gpu["C2"], P["C2"]), xytext=(gpu["C2"] + 1.5, P["C2"] + .035),
                fontsize=9, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
    a1.annotate(f"same score band as C1,\n{gpu['C13']/gpu['C1']:.0f}× the GPU cost",
                xy=(gpu["C13"], P["C13"]), xytext=(gpu["C13"] - 3.4, P["C13"] + .055),
                fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    a1.legend(handles=[Patch(color=GREEN, label="Transformer ON"),
                       Patch(color=RED, label="Transformer OFF")],
              frameon=False, fontsize=9, loc="upper center")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_5_cost_vs_performance.png")
    plt.close(fig)


if __name__ == "__main__":
    P = load()
    assert len(P) == 12, f"expected 12 configurations, found {len(P)}"
    S, C, F = load_stored(), load_cost(), load_flags()
    fig5_1(P, S, F)
    fig5_2(P, S)
    fig5_3(P, F)
    fig5_4(P)
    fig5_5(P, C, F)
    fig5_8(P)
    print(f"C2 = {P['C2']:.4f}  C8 = {P['C8']:.4f}")
    for n in ("5_1_headline", "5_2_metric_definitions", "5_3_transformer_split",
              "5_4_evm_pairs", "5_5_cost_vs_performance", "5_8_literature"):
        print(f"wrote {OUT}/fig{n}.png")
