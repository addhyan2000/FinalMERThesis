"""Generate every figure for the LOSO validation report from data.json."""
import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

SP = os.path.dirname(os.path.abspath(__file__))  # data.json lives beside this script
ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
OUT = f"{ROOT}/report_figures_loso"
os.makedirs(OUT, exist_ok=True)

D = json.load(open(f"{SP}/data.json"))

BLUE, AMBER, GREEN, RED, PURPLE, GRAY = ("#4C78A8", "#F58518", "#54A24B",
                                         "#E45756", "#B279A2", "#8C8C8C")
CLASS_COLORS = {"Negative": "#4C78A8", "Positive": "#F58518", "Surprise": "#54A24B"}

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

ORDER = D["order"]
IDS = [o[0] for o in ORDER]
TOG = {o[0]: dict(evm=o[2], simam=o[3], cnn=o[4], trans=o[5]) for o in ORDER}
LABEL = {o[0]: o[1].split("_", 2)[2] for o in ORDER}
CLASSES = ["Negative", "Positive", "Surprise"]

TARGET_ACC, TARGET_F1 = 0.70, 0.68


def pooled_mf1(j):
    f = j["metrics"]["per_class_f1"]
    return sum(f) / len(f)


L = D["loso25"]


def save(fig, name):
    p = f"{OUT}/{name}"
    fig.savefig(p)
    plt.close(fig)
    print("wrote", name)


def toggle_str(c):
    t = TOG[c]
    return "".join("EVM " if t["evm"] else "",) if False else " ".join(
        [("E" if t["evm"] else "·"), ("S" if t["simam"] else "·"),
         ("C" if t["cnn"] else "·"), ("T" if t["trans"] else "·")])


# ───────────────────────────────────────────────────────────────── FIG L1
def fig_l1():
    order = sorted(IDS, key=lambda c: -pooled_mf1(L[c]))
    f1 = [pooled_mf1(L[c]) for c in order]
    acc = [L[c]["metrics"]["micro_f1"] for c in order]
    x = np.arange(len(order))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.2))
    b1 = ax.bar(x - w / 2, acc, w, label="Pooled accuracy (156 clips)", color=BLUE)
    b2 = ax.bar(x + w / 2, f1, w, label="Pooled macro F1 (156 clips)", color=AMBER)
    ax.axhline(TARGET_ACC, color=BLUE, ls="--", lw=1.3, alpha=.8)
    ax.axhline(TARGET_F1, color=AMBER, ls="--", lw=1.3, alpha=.8)
    ax.text(len(order) - .35, TARGET_ACC + .012, "accuracy target 0.70",
            color=BLUE, fontsize=8, ha="right")
    ax.text(len(order) - .35, TARGET_F1 - .032, "macro-F1 target 0.68",
            color=AMBER, fontsize=8, ha="right")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .008,
                f"{b.get_height():.3f}", ha="center", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{toggle_str(c)}" for c in order], fontsize=8.5)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("score")
    ax.set_title("Figure L1 — Full 25-fold LOSO on CASME-II: all 12 configurations, "
                 "ranked by pooled macro F1\n(toggle key: E=EVM  S=SimAM  C=3D-CNN  T=Transformer;  · = off)")
    ax.legend(loc="upper right", frameon=False)
    # shade transformer-bearing configs
    for i, c in enumerate(order):
        if TOG[c]["trans"]:
            ax.axvspan(i - .5, i + .5, color=GREEN, alpha=.07, zorder=0)
    ax.text(0.25, 0.955, "green band = Transformer ON", transform=ax.transAxes,
            fontsize=9, color=GREEN, ha="center", fontweight="bold")
    save(fig, "figL1_loso_headline.png")


# ───────────────────────────────────────────────────────────────── FIG L2
def fig_l2():
    order = sorted(IDS, key=lambda c: -pooled_mf1(L[c]))
    mfa = [L[c]["metrics"]["accuracy"] for c in order]        # mean-of-folds acc
    poa = [L[c]["metrics"]["micro_f1"] for c in order]        # pooled acc
    mff = [L[c]["metrics"]["macro_f1"] for c in order]        # mean-of-folds macro F1
    pof = [pooled_mf1(L[c]) for c in order]                   # pooled macro F1
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    w = 0.38
    ax = axes[0]
    ax.bar(x - w / 2, mfa, w, color=GRAY, label="Mean-of-folds accuracy (in summary.csv)")
    ax.bar(x + w / 2, poa, w, color=BLUE, label="Pooled accuracy (correct headline)")
    for i in range(len(order)):
        ax.annotate("", xy=(x[i] + w / 2, poa[i]), xytext=(x[i] - w / 2, mfa[i]),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.0, alpha=.7))
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title("Figure L2 — Why two numbers exist for every LOSO cell\n"
                 "TOP: accuracy. Averaging the 25 folds equally over-weights the "
                 "1-clip subjects, inflating the score above the true pooled value.")
    ax = axes[1]
    ax.bar(x - w / 2, mff, w, color=GRAY, label="Mean-of-folds macro F1 (in summary.csv)")
    ax.bar(x + w / 2, pof, w, color=AMBER, label="Pooled macro F1 (correct headline)")
    ax.axhline(0.6267, color=RED, ls="--", lw=1.4)
    ax.text(len(order) - .4, 0.6267 + .012,
            "structural ceiling of mean-of-folds macro F1 = 0.627",
            color=RED, fontsize=8.5, ha="right")
    ax.axhline(TARGET_F1, color=GREEN, ls=":", lw=1.3)
    ax.text(0.1, TARGET_F1 + .012, "target 0.68", color=GREEN, fontsize=8.5)
    ax.set_ylabel("macro F1")
    ax.set_ylim(0, 0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title("BOTTOM: macro F1. Ten of the 25 folds contain only ONE of the three "
                 "classes, so their per-fold macro F1 can never exceed 1/3 —\n"
                 "the mean-of-folds figure is capped at 0.627 by arithmetic alone and "
                 "must not be compared against the 0.68 target.")
    fig.tight_layout()
    save(fig, "figL2_metric_definitions.png")


# ───────────────────────────────────────────────────────────────── FIG L3
def fig_l3():
    folds = D["folds"]
    subs = [f["subject"] for f in folds]
    x = np.arange(len(subs))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                             gridspec_kw={"height_ratios": [2.1, 1]})
    ax = axes[0]
    bottom = np.zeros(len(subs))
    for c in CLASSES:
        v = np.array([f[c] for f in folds], dtype=float)
        ax.bar(x, v, bottom=bottom, color=CLASS_COLORS[c], label=c, width=.72)
        bottom += v
    for i, f in enumerate(folds):
        k = sum(1 for c in CLASSES if f[c] > 0)
        n = sum(f[c] for c in CLASSES)
        ax.text(i, n + .5, f"{k}", ha="center", fontsize=8,
                color=RED if k == 1 else ("#8a6d00" if k == 2 else GREEN),
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in subs], fontsize=8)
    ax.set_ylabel("clips held out in that fold")
    ax.set_ylim(0, 38)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("Figure L3 — The 25 LOSO folds are wildly unequal\n"
                 "Each bar is one held-out subject. The number above each bar is how many "
                 "of the 3 classes that subject actually has.\n"
                 "Subject 17 alone supplies 33 of the 156 clips; subjects 8, 10 and 21 "
                 "supply one clip each.")
    ax = axes[1]
    hist = {1: 0, 2: 0, 3: 0}
    for f in folds:
        hist[sum(1 for c in CLASSES if f[c] > 0)] += 1
    ks = [1, 2, 3]
    cols = [RED, "#E0A800", GREEN]
    bars = ax.barh([f"{k} class{'es' if k > 1 else ''} present" for k in ks],
                   [hist[k] for k in ks], color=cols, height=.55)
    for k, b in zip(ks, bars):
        ax.text(b.get_width() + .15, b.get_y() + b.get_height() / 2,
                f"{hist[k]} folds  →  per-fold macro-F1 ceiling {k/3:.2f}",
                va="center", fontsize=9)
    ax.set_xlim(0, 17)
    ax.set_xlabel("number of folds")
    ax.set_title("Weighted ceiling  =  (10x0.33 + 8x0.67 + 7x1.00) / 25  =  0.627")
    fig.tight_layout()
    save(fig, "figL3_fold_composition.png")


# ───────────────────────────────────────────────────────────────── FIG L4
PAIRS = {
    "Transformer (SLSTT)": [("C1", "C2"), ("C4", "C12"), ("C3", "C9"),
                            ("C13", "C7"), ("C5", "C6"), ("C16", "C8")],
    "EVM (motion magnification)": [("C1", "C4"), ("C2", "C12"), ("C3", "C13"),
                                   ("C9", "C7"), ("C5", "C16"), ("C6", "C8")],
    "3D-CNN": [("C1", "C3"), ("C2", "C9"), ("C4", "C13"), ("C12", "C7")],
    "SimAM attention": [("C3", "C5"), ("C13", "C16"), ("C9", "C6"), ("C7", "C8")],
}


def fig_l4():
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6))
    for ax, (comp, pairs) in zip(axes, PAIRS.items()):
        deltas = [pooled_mf1(L[on]) - pooled_mf1(L[off]) for off, on in pairs]
        labels = [f"{off}→{on}" for off, on in pairs]
        cols = [GREEN if d > 0 else RED for d in deltas]
        y = np.arange(len(deltas))[::-1]
        ax.barh(y, deltas, color=cols, height=.6)
        for yy, d in zip(y, deltas):
            ax.text(d + (0.012 if d >= 0 else -0.012), yy, f"{d:+.3f}",
                    va="center", ha="left" if d >= 0 else "right", fontsize=8.5)
        m = float(np.mean(deltas))
        ax.axvline(m, color="black", ls="--", lw=1.4, zorder=4)
        ax.axvline(0, color="black", lw=.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_ylim(-0.7, len(deltas) - 0.3)
        ax.set_title(f"{comp}\nmean effect {m:+.3f}", fontsize=10)
        ax.set_xlim(-0.24, 0.42)
        ax.set_xlabel("Δ pooled macro F1")
    fig.suptitle("Figure L4 — Marginal contribution of each component under full 25-fold LOSO\n"
                 "Every bar is a matched pair of configurations that differ in exactly one "
                 "switch. Green = the component helped, red = it hurt. Dashed line = mean effect.",
                 fontsize=11, fontweight="bold", y=1.10)
    fig.tight_layout()
    save(fig, "figL4_component_effects.png")


# ───────────────────────────────────────────────────────────────── FIG L5
def _cm_panel(ax, cm, title, sub):
    cm = np.asarray(cm, dtype=int)
    rown = cm.sum(1, keepdims=True)
    norm = np.divide(cm, np.where(rown == 0, 1, rown))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]*100:.0f}%", ha="center",
                    va="center", fontsize=9.5,
                    color="white" if norm[i, j] > .55 else "#1a1a1a")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES, fontsize=8.5)
    ax.set_yticklabels(CLASSES, fontsize=8.5)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{title}\n{sub}", fontsize=9.5)
    ax.grid(False)
    return im


def fig_l5():
    picks = [("C1", "C1 — pure baseline (no components)"),
             ("C8", "C8 — proposed unified (all four ON)"),
             ("C2", "C2 — transformer only (best macro F1)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (c, t) in zip(axes, picks):
        m = L[c]["metrics"]
        _cm_panel(ax, m["confusion_matrix"], t,
                  f"pooled acc {m['micro_f1']:.3f} | pooled macro F1 {pooled_mf1(L[c]):.3f}")
    fig.suptitle("Figure L5 — Pooled LOSO confusion matrices (all 156 clips, "
                 "aggregated over the 25 folds)\nCell = clip count and row-normalised "
                 "recall. Rows are the true class, columns the prediction.",
                 fontsize=11, fontweight="bold", y=1.06)
    fig.tight_layout()
    save(fig, "figL5_confusion_matrices.png")


# ───────────────────────────────────────────────────────────────── FIG L6
def fig_l6():
    order = sorted(IDS, key=lambda c: -pooled_mf1(L[c]))
    x = np.arange(len(order))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11.5, 5))
    for i, c in enumerate(CLASSES):
        v = [L[cc]["metrics"]["per_class_f1"][i] for cc in order]
        b = ax.bar(x + (i - 1) * w, v, w, color=CLASS_COLORS[c], label=c)
        for bb in b:
            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + .008,
                    f"{bb.get_height():.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{toggle_str(c)}" for c in order], fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("per-class F1 (pooled over 156 clips)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_title("Figure L6 — Per-class F1 under full LOSO\n"
                 "No configuration abandons a class entirely — every bar is non-zero. "
                 "The class ordering flips: weak models are best on Positive, "
                 "strong models are best on Negative.")
    save(fig, "figL6_per_class_f1.png")


# ───────────────────────────────────────────────────────────────── FIG L7
REGIMES = [
    ("weekend_holdout52", "Holdout\nN=52\n(collapsed run)", "#B0B0B0"),
    ("holdout39", "Holdout\nN=39\n60 epochs", BLUE),
    ("pilot_loso5", "Pilot LOSO\n5 folds, N=24", PURPLE),
    ("loso20", "Pilot LOSO\n20 folds, N=139", AMBER),
    ("loso25", "FULL LOSO\n25 folds, N=156\n(this branch)", GREEN),
]


def fig_l7():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    common = [c for c in IDS if all(c in D[k] for k, *_ in REGIMES)]
    x = np.arange(len(common))
    w = 0.16
    for i, (k, lbl, col) in enumerate(REGIMES):
        v = [pooled_mf1(D[k][c]) for c in common]
        ax.bar(x + (i - 2) * w, v, w, color=col, label=lbl.replace("\n", " "))
    ax.axhline(TARGET_F1, color=RED, ls="--", lw=1.2)
    ax.text(len(common) - .5, TARGET_F1 + .012, "target 0.68", color=RED,
            fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(common)
    ax.set_ylabel("pooled macro F1")
    ax.set_ylim(0, .82)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title(f"Pooled macro F1 for the {len(common)} configs present in all five runs")
    ax = axes[1]
    anchors = ["C1", "C3", "C5", "C8"]
    for c, col in zip(anchors, [GRAY, BLUE, PURPLE, RED]):
        xs, ys = [], []
        for i, (k, lbl, _) in enumerate(REGIMES):
            if c in D[k]:
                xs.append(i); ys.append(pooled_mf1(D[k][c]))
        ax.plot(xs, ys, "o-", color=col, lw=2, ms=7,
                label=f"{c} — {LABEL[c].replace('_',' ')}")
        for xx, yy in zip(xs, ys):
            ax.text(xx, yy + .018, f"{yy:.3f}", ha="center", fontsize=7.5, color=col)
    ax.axhline(TARGET_F1, color=RED, ls="--", lw=1.2)
    ax.set_xticks(range(len(REGIMES)))
    ax.set_xticklabels([r[1] for r in REGIMES], fontsize=8)
    ax.set_ylim(0, .82)
    ax.set_ylabel("pooled macro F1")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Trajectory of four anchor configurations across the five runs")
    fig.suptitle("Figure L7 — How the full-LOSO run compares with every earlier "
                 "(baseline) evaluation of the same 12-cell matrix",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "figL7_protocol_evolution.png")


# ───────────────────────────────────────────────────────────────── FIG L8
def fig_l8():
    on = [c for c in IDS if TOG[c]["trans"]]
    off = [c for c in IDS if not TOG[c]["trans"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax = axes[0]
    for i, (grp, col) in enumerate([(off, RED), (on, GREEN)]):
        srt = sorted(grp, key=lambda c: pooled_mf1(L[c]))
        v = [pooled_mf1(L[c]) for c in srt]
        xs = np.full(len(v), i) + np.linspace(-.20, .20, len(v))
        ax.scatter(xs, v, s=105, color=col, zorder=3, edgecolor="white", lw=1.3)
        for xx, yy, c in zip(xs, v, srt):
            ax.text(xx, yy + .015, c, fontsize=8.5, ha="center")
        ax.hlines(np.mean(v), i - .32, i + .32, color=col, lw=3, zorder=2)
        ax.text(i + .40, np.mean(v), f"group mean\n{np.mean(v):.3f}", ha="left",
                va="center", fontsize=9, color=col, fontweight="bold")
    ax.annotate("", xy=(0.5, .5830), xytext=(0.5, .4480),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.4))
    ax.text(0.53, .515, "empty gap\n0.135", fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Transformer OFF\n(6 configs)",
                                               "Transformer ON\n(6 configs)"])
    ax.set_xlim(-.55, 1.75)
    ax.set_ylim(.37, .78)
    ax.set_ylabel("pooled macro F1 (25-fold LOSO)")
    ax.set_title("Full LOSO — the split is total and gap-free:\n"
                 "every ON config beats every OFF config")
    ax.axhspan(.4192, .4480, color=RED, alpha=.08)
    ax.axhspan(.5830, .7122, color=GREEN, alpha=.08)

    ax = axes[1]
    hold = D["holdout39"]
    pairs = PAIRS["Transformer (SLSTT)"]
    hd = [pooled_mf1(hold[b]) - pooled_mf1(hold[a]) for a, b in pairs]
    ld = [pooled_mf1(L[b]) - pooled_mf1(L[a]) for a, b in pairs]
    x = np.arange(len(pairs)); w = .38
    ax.bar(x - w / 2, hd, w, color=BLUE,
           label=f"Old Holdout baseline (N=39) — mean {np.mean(hd):+.3f}, sign flips 3 ways")
    ax.bar(x + w / 2, ld, w, color=GREEN,
           label=f"Full LOSO (N=156) — mean {np.mean(ld):+.3f}, all six positive")
    for i in range(len(pairs)):
        for xx, vv in [(x[i] - w / 2, hd[i]), (x[i] + w / 2, ld[i])]:
            ax.text(xx, vv + (.012 if vv >= 0 else -.030),
                    f"{vv:+.3f}" if abs(vv) > 1e-9 else "0.000",
                    ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=.9)
    ax.axhline(np.mean(hd), color=BLUE, ls="--", lw=1.2)
    ax.axhline(np.mean(ld), color=GREEN, ls="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}→{b}" for a, b in pairs], fontsize=9)
    ax.set_ylabel("Δ pooled macro F1 from switching the transformer ON")
    ax.set_ylim(-.34, .40)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    ax.set_title("Under Holdout the transformer's effect was erratic;\n"
                 "under full LOSO it is uniformly large and positive")
    fig.suptitle("Figure L8 — The single biggest finding: under real LOSO the "
                 "Transformer is decisive, not harmful",
                 fontsize=11.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "figL8_transformer_split.png")


# ───────────────────────────────────────────────────────────────── FIG L9
def fig_l9():
    pairs = PAIRS["EVM (motion magnification)"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (key, title) in zip(axes, [("holdout39", "Earlier Holdout baseline (N=39) — EVM inert"),
                                       ("loso25", "This branch, full LOSO (N=156) — EVM live")]):
        x = np.arange(len(pairs)); w = .36
        off = [pooled_mf1(D[key][a]) for a, b in pairs]
        on = [pooled_mf1(D[key][b]) for a, b in pairs]
        ax.bar(x - w / 2, off, w, color=GRAY, label="EVM OFF (raw tensors)")
        ax.bar(x + w / 2, on, w, color=AMBER, label="EVM ON (magnified tensors)")
        for i in range(len(pairs)):
            d = on[i] - off[i]
            ax.text(x[i], max(off[i], on[i]) + .015,
                    "identical" if abs(d) < 1e-9 else f"{d:+.3f}",
                    ha="center", fontsize=8.5,
                    color=RED if abs(d) < 1e-9 else (GREEN if d > 0 else RED),
                    fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}→{b}" for a, b in pairs], fontsize=9)
        ax.set_ylim(0, .88)
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("pooled macro F1")
    fig.suptitle("Figure L9 — The EVM data-routing defect is fixed\n"
                 "Left: in the old run every EVM/non-EVM pair was identical to four "
                 "decimals, proving both arms read the same tensors.\n"
                 "Right: in this run all six pairs differ, so the magnified tensor set "
                 "is genuinely being loaded and the EVM hypothesis is finally testable.",
                 fontsize=11.5, fontweight="bold", y=1.13)
    fig.tight_layout()
    save(fig, "figL9_evm_pairs.png")


# ───────────────────────────────────────────────────────────────── FIG L10
def fig_l10():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, c, t in [(axes[0], "C8", "C8 — proposed unified (all four components)"),
                     (axes[1], "C2", "C2 — transformer only (best macro F1)")]:
        rows = D["loso25_curves"][c]
        ep = [int(r["epoch"]) for r in rows]
        tl = [float(r["train_loss"]) for r in rows]
        vf = [float(r["val_f1"]) for r in rows]
        va = [float(r["val_acc"]) for r in rows]
        ax.plot(ep, tl, color=BLUE, lw=1.9, label="train loss")
        ax.set_ylabel("train loss", color=BLUE)
        ax.tick_params(axis="y", labelcolor=BLUE)
        ax.set_xlabel("epoch")
        ax2 = ax.twinx()
        ax2.plot(ep, va, color=GRAY, lw=1.2, alpha=.75, label="fold val accuracy")
        ax2.plot(ep, vf, color=AMBER, lw=1.9, label="fold val macro F1")
        ax2.set_ylabel("fold validation score", color=AMBER)
        ax2.tick_params(axis="y", labelcolor=AMBER)
        ax2.set_ylim(-.03, 1.05)
        ax2.grid(False)
        ax.set_title(t, fontsize=10)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="center right")
    fig.suptitle("Figure L10 — Training dynamics of the LAST LOSO fold (held-out "
                 "subject 26, 12 clips), 50 epochs\nTrain loss falls smoothly; the "
                 "validation curve is spiky because it is measured on a single "
                 "subject's handful of clips, not because training diverged.",
                 fontsize=11, fontweight="bold", y=1.08)
    fig.tight_layout()
    save(fig, "figL10_training_curves.png")


# ───────────────────────────────────────────────────────────────── FIG L11
def fig_l11():
    hwd = D["loso25_hw"]
    # Manual label offsets so the near-coincident EVM twins stay readable.
    OFF = {"C1": (0, -.020), "C4": (0, .011), "C2": (0, .012), "C12": (0, -.021),
           "C3": (0, -.021), "C13": (0, .011), "C5": (0, .011), "C16": (0, -.021),
           "C9": (0, -.021), "C7": (0, .011), "C6": (0, -.021), "C8": (0, .011)}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    for ax, key, xlab, scale, xmax in [
            (axes[0], "train_sec",
             "extrapolated cost of the whole 25-fold sweep (GPU hours)\n"
             "= measured per-fold training time x 25", 25 / 3600, 7.8),
            (axes[1], "vram_mb", "peak VRAM (GB)", 1 / 1024, 24.5)]:
        for c in IDS:
            xv = hwd[c][key] * scale
            yv = pooled_mf1(L[c])
            col = GREEN if TOG[c]["trans"] else RED
            ax.scatter(xv, yv, s=115, color=col, edgecolor="white", lw=1.4, zorder=3)
            dx, dy = OFF[c]
            ax.text(xv + dx, yv + dy, c, ha="center", fontsize=8.5, zorder=4)
        ax.set_xlabel(xlab)
        ax.set_ylabel("pooled macro F1")
        ax.set_ylim(.37, .79)
        ax.set_xlim(-xmax * .05, xmax)
    axes[0].legend(handles=[Patch(color=GREEN, label="Transformer ON"),
                            Patch(color=RED, label="Transformer OFF")],
                   frameon=False, loc="upper right")
    axes[0].annotate("same score band as C1,\n15x the GPU cost",
                     xy=(6.0, .432), xytext=(3.6, .500), fontsize=8.5, color=RED,
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    axes[0].annotate("best macro F1,\ncheapest to train",
                     xy=(0.55, .706), xytext=(1.5, .755), fontsize=8.5, color=GREEN,
                     arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1))
    fig.suptitle("Figure L11 — Cost versus benefit: the cheapest model is the best\n"
                 "C2 (transformer, no 3D-CNN) tops the macro-F1 table for ~0.5 GPU hours "
                 "and 0.17 GB of VRAM; the 3D-CNN configs cost 5.8-6.5 GPU hours and up "
                 "to 19.6 GB for no gain.",
                 fontsize=11, fontweight="bold", y=1.07)
    fig.tight_layout()
    save(fig, "figL11_cost_vs_performance.png")


# ───────────────────────────────────────────────────────────────── FIG L12
def fig_l12():
    lit = [(r["reference"], float(r["accuracy"]),
            float(r["macro_f1"]) if r["macro_f1"] else None)
           for r in D["literature"]]
    names = [l[0].split("(")[0].strip() for l in lit]
    accs = [l[1] for l in lit]
    f1s = [l[2] for l in lit]
    ours = [("C8 proposed unified\n(this project, full LOSO)",
             L["C8"]["metrics"]["micro_f1"], pooled_mf1(L["C8"])),
            ("C2 transformer only\n(this project, full LOSO)",
             L["C2"]["metrics"]["micro_f1"], pooled_mf1(L["C2"]))]
    labels = names + [o[0] for o in ours]
    A = accs + [o[1] for o in ours]
    F = f1s + [o[2] for o in ours]
    cols = [GRAY] * 3 + [PURPLE] + [BLUE, GREEN]
    x = np.arange(len(labels))
    w = .38
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    b1 = ax.bar(x - w / 2, A, w, color=cols, label="accuracy")
    b2 = ax.bar(x + w / 2, [f if f else 0 for f in F], w,
                color=cols, alpha=.5, hatch="///", label="macro F1")
    for i, (a, f) in enumerate(zip(A, F)):
        ax.text(x[i] - w / 2, a + .008, f"{a:.3f}", ha="center", fontsize=8.5)
        ax.text(x[i] + w / 2, (f if f else 0) + .008,
                f"{f:.3f}" if f else "n/r", ha="center", fontsize=8.5)
    ax.axhline(TARGET_ACC, color=BLUE, ls="--", lw=1.2)
    ax.axhline(TARGET_F1, color=AMBER, ls="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.2)
    ax.set_ylim(0, .88)
    ax.set_ylabel("score")
    ax.legend(handles=[Patch(color=GRAY, label="published literature (LOSO)"),
                       Patch(color=PURPLE, label="dissertation target"),
                       Patch(color=GREEN, label="this project — best macro F1"),
                       Patch(color=BLUE, label="this project — proposed model")],
              frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Figure L12 — This project's full-LOSO results against the literature "
                 "baselines and the dissertation target\n"
                 "Solid bars = accuracy, hatched bars = macro F1. 'n/r' = the source does "
                 "not report macro F1. All bars are LOSO on CASME-II, 3-class grouped.")
    save(fig, "figL12_literature.png")


# ───────────────────────────────────────────────────────────────── FIG L13
def fig_l13():
    prev, cur = D["loso20"], D["loso25"]
    order = sorted(IDS, key=lambda c: -pooled_mf1(cur[c]))
    x = np.arange(len(order)); w = .38
    fig, ax = plt.subplots(figsize=(11.5, 5))
    p = [pooled_mf1(prev[c]) for c in order]
    q = [pooled_mf1(cur[c]) for c in order]
    ax.bar(x - w / 2, p, w, color=AMBER, label="Previous LOSO baseline — 20/25 folds, N=139")
    ax.bar(x + w / 2, q, w, color=GREEN, label="This branch — full 25/25 folds, N=156")
    for i in range(len(order)):
        d = q[i] - p[i]
        ax.text(x[i], max(p[i], q[i]) + .012, f"{d:+.3f}", ha="center", fontsize=8,
                color=GREEN if d > 0 else RED, fontweight="bold")
    ax.axhline(TARGET_F1, color=RED, ls="--", lw=1.2)
    ax.text(len(order) - .4, TARGET_F1 + .012, "target 0.68", color=RED,
            fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("pooled macro F1")
    ax.set_ylim(0, .82)
    ax.legend(frameon=False, loc="upper right")
    up = sum(1 for i in range(len(order)) if q[i] > p[i])
    tgain = [q[i] - p[i] for i in range(len(order)) if TOG[order[i]]["trans"]]
    ax.set_title("Figure L13 — Full LOSO versus the immediately preceding 20-fold LOSO "
                 f"baseline\nAll six transformer configs gain heavily "
                 f"({min(tgain):+.3f} to {max(tgain):+.3f}); the four 3D-CNN-only configs "
                 f"lose ground. {up} of {len(order)} improve overall.")
    save(fig, "figL13_loso_vs_prev_loso.png")


# ───────────────────────────────────────────────────────────────── FIG L14
def fig_l14():
    raw = D["raw_emotion_counts"]
    grp = D["grouped_counts"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    ks = sorted(raw, key=lambda k: -raw[k])
    cols = [GRAY if k == "others" else BLUE for k in ks]
    b = ax.bar(ks, [raw[k] for k in ks], color=cols)
    for bb, k in zip(b, ks):
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 1,
                str(raw[k]), ha="center", fontsize=9)
    ax.set_ylabel("clips")
    ax.set_title("All 255 CASME-II clips by raw label\n(grey = 'others', excluded)")
    ax.tick_params(axis="x", rotation=30)
    ax = axes[1]
    ks = ["Negative", "Positive", "Surprise", "Others"]
    cols = [CLASS_COLORS.get(k, GRAY) for k in ks]
    b = ax.bar(ks, [grp[k] for k in ks], color=cols)
    for bb, k in zip(b, ks):
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 1,
                str(grp[k]), ha="center", fontsize=9)
    ax.set_title("Grouped 3-class pool actually used: 156 clips\n"
                 "ratio 99 : 32 : 25  ≈  4 : 1.3 : 1")
    fig.suptitle("Figure L14 — The dataset behind every number in this report",
                 fontsize=11.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "figL14_dataset.png")


for f in [fig_l1, fig_l2, fig_l3, fig_l4, fig_l5, fig_l6, fig_l7, fig_l8,
          fig_l9, fig_l10, fig_l11, fig_l12, fig_l13, fig_l14]:
    f()
print("\nAll figures written to", OUT)
