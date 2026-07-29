"""Generate the complete visual result set for ALL_RESULTS_LOSO.md."""
import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

SP = os.path.dirname(os.path.abspath(__file__))
ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
OUT = f"{ROOT}/report_figures_all_results"
os.makedirs(OUT, exist_ok=True)
D = json.load(open(f"{SP}/data.json"))
L, HW = D["loso25"], D["loso25_hw"]

BLUE, AMBER, GREEN, RED, PURPLE, GRAY = ("#4C78A8", "#F58518", "#54A24B",
                                         "#E45756", "#B279A2", "#8C8C8C")
CLASSES = ["Negative", "Positive", "Surprise"]
CCOL = {"Negative": BLUE, "Positive": AMBER, "Surprise": GREEN}
TRUE_N = [99, 32, 25]
TARGET_ACC, TARGET_F1 = 0.70, 0.68
CEIL = 0.6267

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 170, "savefig.bbox": "tight",
})

# numeric config order: 1,2,3,4,5,6,7,8,9,12,13,16
NUM = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C12", "C13", "C16"]
TOG = {o[0]: dict(evm=o[2], simam=o[3], cnn=o[4], trans=o[5]) for o in D["order"]}
FULL = {o[0]: o[1] for o in D["order"]}
PHASE = {o[0]: o[6] for o in D["order"]}
NICE = {"C1": "pure_base", "C2": "temporal_only", "C3": "spatial_only",
        "C4": "motion_amp_base", "C5": "attention_base", "C6": "full_stage2_noevm",
        "C7": "full_no_attention", "C8": "proposed_unified", "C9": "permutation",
        "C12": "permutation", "C13": "permutation", "C16": "permutation"}


def pmf1(c):
    return sum(L[c]["metrics"]["per_class_f1"]) / 3


def pacc(c):
    return L[c]["metrics"]["micro_f1"]


RANK = {c: r for r, c in enumerate(sorted(L, key=lambda x: -pmf1(x)), 1)}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}")
    plt.close(fig)
    print("wrote", name)


def cm_heat(ax, c, title=None, annot_pct=True, fs=10):
    cm = np.asarray(L[c]["metrics"]["confusion_matrix"], dtype=int)
    rown = cm.sum(1, keepdims=True)
    norm = cm / np.where(rown == 0, 1, rown)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            txt = f"{cm[i,j]}\n{norm[i,j]*100:.0f}%" if annot_pct else f"{cm[i,j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fs,
                    color="white" if norm[i, j] > .55 else "#1a1a1a",
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([c[:3] for c in CLASSES], fontsize=8.5)
    ax.set_yticklabels([c[:3] for c in CLASSES], fontsize=8.5)
    ax.set_xlabel("predicted", fontsize=8.5)
    ax.set_ylabel("true", fontsize=8.5)
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=9.5)


# ══════════════════════════════════ M1 — technology matrix
def m1():
    comps = ["EVM", "SimAM", "3D-CNN", "Transformer"]
    keys = ["evm", "simam", "cnn", "trans"]
    grid = np.array([[1 if TOG[c][k] else 0 for k in keys] for c in NUM])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(["#EFEFEF", GREEN]),
              vmin=0, vmax=1, aspect="auto")
    for i in range(len(NUM)):
        for j in range(4):
            ax.text(j, i, "ON" if grid[i, j] else "off", ha="center", va="center",
                    fontsize=9, color="white" if grid[i, j] else "#888",
                    fontweight="bold" if grid[i, j] else "normal")
    ax.set_xticks(range(4)); ax.set_xticklabels(comps, fontsize=10)
    ax.set_yticks(range(len(NUM)))
    ax.set_yticklabels([f"{c} · {NICE[c]}" for c in NUM], fontsize=9)
    ax.set_title("Which technology each configuration uses")
    ax.grid(False)
    for i in range(len(NUM) + 1):
        ax.axhline(i - .5, color="white", lw=2)
    for j in range(5):
        ax.axvline(j - .5, color="white", lw=2)
    ax = axes[1]
    y = np.arange(len(NUM))
    f1 = [pmf1(c) for c in NUM]
    cols = [GREEN if TOG[c]["trans"] else RED for c in NUM]
    ax.barh(y, f1, color=cols, height=.62)
    for i, c in enumerate(NUM):
        ax.text(f1[i] + .008, i, f"{f1[i]:.4f}   (rank {RANK[c]})",
                va="center", fontsize=8.8)
    ax.axvline(TARGET_F1, color="black", ls="--", lw=1.3)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_ylim(len(NUM) - .5, -.5)   # exactly match the imshow panel's row alignment
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("pooled macro F1 (25-fold LOSO, N=156)\n"
                  "dashed line = macro-F1 target 0.68")
    ax.set_title("Result for that configuration")
    ax.legend(handles=[Patch(color=GREEN, label="Transformer ON"),
                       Patch(color=RED, label="Transformer OFF")],
              frameon=False, loc="lower center", bbox_to_anchor=(.5, -.34),
              ncol=2, fontsize=9)
    fig.suptitle("Figure M1 — Technology matrix and outcome, all 12 configurations "
                 "(config_1 → config_16)", fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "figM1_technology_matrix.png")


# ══════════════════════════════════ M2 — all four metrics heatmap
def m2():
    cols = [("Pooled\naccuracy", lambda c: pacc(c), 1.0),
            ("Pooled\nmacro F1", lambda c: pmf1(c), 1.0),
            ("Mean-of-folds\naccuracy", lambda c: L[c]["metrics"]["accuracy"], 1.0),
            ("Mean-of-folds\nmacro F1", lambda c: L[c]["metrics"]["macro_f1"], CEIL)]
    M = np.array([[f(c) for _, f, _ in cols] for c in NUM])
    Mn = M / np.array([m for *_, m in cols])[None, :]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.imshow(Mn, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(len(NUM)):
        for j in range(4):
            ax.text(j, i, f"{M[i,j]:.4f}", ha="center", va="center", fontsize=9.5,
                    fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels([n for n, *_ in cols], fontsize=9.5)
    ax.set_yticks(range(len(NUM)))
    ax.set_yticklabels([f"{c} · {NICE[c]}" for c in NUM], fontsize=9)
    ax.grid(False)
    for i in range(len(NUM) + 1):
        ax.axhline(i - .5, color="white", lw=2)
    for j in range(5):
        ax.axvline(j - .5, color="white", lw=2)
    ax.set_title("Figure M2 — All four aggregate metrics for every configuration\n"
                 "Colour is each value as a fraction of ITS OWN ceiling — 1.00 for the "
                 "first three columns,\nbut 0.627 for mean-of-folds macro F1, which is "
                 "capped by the fold class composition.", fontsize=10.5)
    save(fig, "figM2_all_metrics_heatmap.png")


# ══════════════════════════════════ M3 — confusion-matrix contact sheet
def m3():
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
    for ax, c in zip(axes.ravel(), NUM):
        m = L[c]["metrics"]
        cm_heat(ax, c, fs=9.5)
        ax.set_title(f"{c} · {NICE[c]}   (rank {RANK[c]})\n"
                     f"acc {pacc(c):.4f} · macro F1 {pmf1(c):.4f} · "
                     f"{sum(m['confusion_matrix'][i][i] for i in range(3))}/156",
                     fontsize=9)
    fig.suptitle("Figure M3 — Every confusion matrix in one sheet (config_1 → config_16)\n"
                 "All 156 clips, pooled over the 25 LOSO folds. Each cell shows the clip "
                 "count and the row-normalised recall.",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout()
    save(fig, "figM3_all_confusion_matrices.png")


# ══════════════════════════════════ M4 — per-class P / R / F1 heatmaps
def m4():
    keys = [("per_class_precision", "Precision"), ("per_class_recall", "Recall"),
            ("per_class_f1", "F1")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, (k, lbl) in zip(axes, keys):
        M = np.array([L[c]["metrics"][k] for c in NUM])
        ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        for i in range(len(NUM)):
            for j in range(3):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES, fontsize=9.5)
        ax.set_yticks(range(len(NUM)))
        ax.set_yticklabels(NUM if ax is axes[0] else [], fontsize=9)
        ax.set_title(f"Per-class {lbl}")
        ax.grid(False)
        for i in range(len(NUM) + 1):
            ax.axhline(i - .5, color="white", lw=2)
        for j in range(4):
            ax.axvline(j - .5, color="white", lw=2)
    fig.suptitle("Figure M4 — Per-class precision, recall and F1 for every configuration "
                 "(pooled over 156 clips)\nRed = poor, green = good. Read a row across all "
                 "three panels to see HOW a configuration handles each class.",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "figM4_per_class_heatmaps.png")


# ══════════════════════════════════ M5 — precision vs recall per class
def m5():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (ci, cls) in zip(axes, enumerate(CLASSES)):
        pts = [(L[c]["metrics"]["per_class_recall"][ci],
                L[c]["metrics"]["per_class_precision"][ci], c) for c in NUM]
        for r, p, c in pts:
            col = GREEN if TOG[c]["trans"] else RED
            ax.scatter(r, p, s=115, color=col, edgecolor="white", lw=1.3, zorder=3)
        # nudge labels apart where points nearly coincide
        placed = []
        for r, p, c in sorted(pts, key=lambda t: (-t[1], t[0])):
            dy = .030
            while any(abs(r - pr) < .075 and abs(p + dy - pp) < .042
                      for pr, pp in placed):
                dy += .046
            ax.text(r, p + dy, c, ha="center", fontsize=8.5, zorder=4)
            placed.append((r, p + dy))
        for f in (.3, .5, .7):
            xs = np.linspace(.02, 1, 200)
            ys = f * xs / (2 * xs - f)
            ok = (ys > 0) & (ys <= 1.02)
            ax.plot(xs[ok], ys[ok], color=GRAY, ls=":", lw=.9, zorder=1)
            ax.text(1.0, f * 1.0 / (2 * 1.0 - f) if f < 1 else 1, f" F1={f}",
                    fontsize=7.5, color=GRAY, va="center")
        ax.set_xlim(0, 1.14); ax.set_ylim(0, 1.14)
        ax.set_xlabel("recall — of the real clips, how many were caught")
        if ax is axes[0]:
            ax.set_ylabel("precision — of the clips called this class,\nhow many were right")
        ax.set_title(f"{cls}  (n = {TRUE_N[ci]} clips)")
    axes[0].legend(handles=[Patch(color=GREEN, label="Transformer ON"),
                            Patch(color=RED, label="Transformer OFF")],
                   frameon=False, loc="lower left", fontsize=8.5)
    fig.suptitle("Figure M5 — Precision against recall, per class, all 12 configurations\n"
                 "Dotted curves are constant-F1 contours. Top-right is good. Points hugging "
                 "the top-left are 'cautious' models (rarely guess the class, right when "
                 "they do); bottom-right are 'trigger-happy' ones.",
                 fontsize=11.5, fontweight="bold", y=1.05)
    fig.tight_layout()
    save(fig, "figM5_precision_recall.png")


# ══════════════════════════════════ M6 — correct vs wrong per class
def m6():
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    x = np.arange(len(NUM))
    w = .26
    for ci, cls in enumerate(CLASSES):
        corr, wrong = [], []
        for c in NUM:
            cm = L[c]["metrics"]["confusion_matrix"]
            corr.append(cm[ci][ci]); wrong.append(TRUE_N[ci] - cm[ci][ci])
        off = (ci - 1) * w
        ax.bar(x + off, corr, w, color=CCOL[cls], label=f"{cls} — correct")
        ax.bar(x + off, wrong, w, bottom=corr, color=CCOL[cls], alpha=.28,
               hatch="///", label=f"{cls} — missed")
        for i in range(len(NUM)):
            ax.text(x[i] + off, TRUE_N[ci] + 1.2, str(corr[i]), ha="center",
                    fontsize=7.5, color=CCOL[cls], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{NICE[c][:14]}" for c in NUM], fontsize=8.2)
    ax.set_ylabel("clips")
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper center")
    ax.set_title("Figure M6 — How many clips of each class each configuration actually got "
                 "right\nSolid = correct, hatched = missed. Bar totals are fixed at the "
                 "true class sizes (99 / 32 / 25). Numbers above bars are the correct counts.")
    save(fig, "figM6_correct_per_class.png")


# ══════════════════════════════════ M7 — LOSO procedure schematic
def m7():
    folds = D["folds"]
    subs = [f["subject"] for f in folds]
    n = len(subs)
    sizes = [f["Negative"] + f["Positive"] + f["Surprise"] for f in folds]
    grid = np.zeros((n, n))
    for i in range(n):
        grid[i, i] = 1
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap([BLUE, AMBER]),
              vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"S{s}" for s in subs], fontsize=6.5, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"fold {i+1}" for i in range(n)], fontsize=6.5)
    ax.set_xlabel("the 25 subjects")
    ax.set_title("The 25-fold LOSO schedule\n"
                 "orange = held out for testing, blue = used for training", fontsize=10)
    ax.grid(False)
    for i in range(n + 1):
        ax.axhline(i - .5, color="white", lw=.6)
        ax.axvline(i - .5, color="white", lw=.6)
    ax = axes[1]
    y = np.arange(n)
    ax.barh(y, sizes, color=AMBER, height=.68)
    for i, s in enumerate(sizes):
        k = sum(1 for c in CLASSES if folds[i][c] > 0)
        ax.text(s + .4, i, f"{s} clip{'s' if s != 1 else ''}  ·  {k} class"
                           f"{'es' if k > 1 else ''}", va="center", fontsize=7.5,
                color=RED if k == 1 else ("#8a6d00" if k == 2 else GREEN))
    ax.set_yticks(y)
    ax.set_yticklabels([f"fold {i+1} — S{subs[i]}" for i in range(n)], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 52)
    ax.set_xlabel("clips tested in that fold")
    ax.set_title("What each fold actually tests\n"
                 "every clip is tested exactly once — 25 folds, 156 clips total",
                 fontsize=10)
    fig.suptitle("Figure M7 — The validation procedure itself: Leave-One-Subject-Out, "
                 "25 folds, 156 clips\nA fresh model is trained from scratch for every "
                 "row, then tested on the one subject it never saw.",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "figM7_loso_procedure.png")


# ══════════════════════════════════ M8 — pipeline schematic
def m8():
    fig, ax = plt.subplots(figsize=(14.5, 7.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, txt, fc, ec="#333", fs=8.6, tc="#111", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal", linespacing=1.45)

    def arrow(x1, y1, x2, y2, col="#333"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, color=col, lw=1.3))

    XS = [1, 20.5, 40, 59.5, 79]
    W = 16

    def chain(xs, y, texts, fcs):
        for x, t, fc in zip(xs, texts, fcs):
            box(x, y, W, 12, t, fc)
        for i in range(len(xs) - 1):
            arrow(xs[i] + W + .4, y + 6, xs[i + 1] - .6, y + 6)

    ax.text(50, 97, "STAGE 1  —  offline data preparation (run once per EVM setting)",
            ha="center", fontsize=10.5, fontweight="bold", color="#333")
    chain(XS, 79,
          ["CASME-II\n255 raw clips\n+ coding xlsx",
           "filter to 3 grouped\nclasses, drop 'others'\n→ 156 clips",
           "trim onset→offset\ninterpolate to\nT = 32 frames",
           "optical flow (u,v)\n+ optical strain\n3 channels",
           "tensor\n(3, 32, 224, 224)\nsaved to disk"],
          ["#EAF1F8"] * 5)
    box(40, 64, W, 9, "EVM switch (A)\nmagnify frames first?", "#FDEBD3", fs=8.4,
        bold=True)
    arrow(48, 73.6, 48, 78.4, AMBER)
    ax.text(58.5, 68.5, "ON → tensors/      OFF → tensors_raw/\n"
                        "a DATA-level switch: the network is unchanged",
            fontsize=8, color="#8a5a00", va="center")

    ax.text(50, 56, "STAGE 2  —  training + evaluation, repeated for 12 configs × 25 folds "
                    "= 300 trainings",
            ha="center", fontsize=10.5, fontweight="bold", color="#333")
    chain(XS, 38,
          ["load fold:\ntrain 24 subjects\ntest 1 subject",
           "balanced sampler\noversamples\nminority clips",
           "SPATIAL STEM\n3D-CNN (C)  or\n4×4 patch grid",
           "TEMPORAL\nTransformer (D)  or\nmean pooling",
           "classifier head\n→ Neg / Pos / Sur"],
          ["#EAF1F8", "#EAF1F8", "#E8F5E9", "#E8F5E9", "#EAF1F8"])
    box(40, 23, W, 9, "SimAM switch (B)\nre-weight CNN features", "#FDEBD3", fs=8.4,
        bold=True)
    arrow(48, 32.6, 48, 37.4, AMBER)
    ax.text(58.5, 27.5, "needs the 3D-CNN to be ON — this is why the\n"
                        "2⁴ = 16 grid reduces to 12 valid configurations",
            fontsize=8, color="#8a5a00", va="center")

    ax.text(50, 16, "AGGREGATION  —  what produces the numbers in this report",
            ha="center", fontsize=10.5, fontweight="bold", color="#333")
    box(2, 2, 21, 11, "25 folds × predictions\n= 156 predictions,\nevery clip exactly once",
        "#EAF1F8")
    box(29, 2, 21, 11, "sum the 25 fold\nconfusion matrices\n→ one 3×3 matrix", "#EAF1F8")
    box(56, 8, 24, 5.5, "POOLED ACCURACY\n= diagonal ÷ 156", "#DFF0D8", bold=True)
    box(56, 1.5, 24, 5.5, "POOLED MACRO F1\n= mean of 3 class F1s", "#DFF0D8", bold=True)
    arrow(23.4, 7.5, 28.4, 7.5)
    arrow(50.4, 7.5, 55.4, 10.7)
    arrow(50.4, 7.5, 55.4, 4.2)
    ax.text(81.5, 7.5, "◀ the two\n     headline\n     metrics", fontsize=8.8, color=GREEN,
            fontweight="bold", va="center")
    fig.suptitle("Figure M8 — The complete testing and validation procedure, end to end",
                 fontsize=12, fontweight="bold", y=.99)
    save(fig, "figM8_pipeline_procedure.png")


# ══════════════════════════════════ M9..M20 — per-config result cards
def cards():
    for c in NUM:
        m = L[c]["metrics"]
        cm = m["confusion_matrix"]
        corr = sum(cm[i][i] for i in range(3))
        fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.0),
                                 gridspec_kw={"width_ratios": [1, 1.25, 1.15]})
        cm_heat(axes[0], c, "Confusion matrix (156 clips)")
        # per-class bars
        ax = axes[1]
        y = np.arange(3)
        h = .25
        for k, (key, lbl, col) in enumerate([
                ("per_class_precision", "precision", BLUE),
                ("per_class_recall", "recall", AMBER),
                ("per_class_f1", "F1", GREEN)]):
            v = m[key]
            ax.barh(y + (1 - k) * h, v, h, color=col, label=lbl)
            for i in range(3):
                ax.text(v[i] + .012, y[i] + (1 - k) * h, f"{v[i]:.3f}",
                        va="center", fontsize=7.8)
        ax.set_yticks(y); ax.set_yticklabels(
            [f"{cl}\n(n={n})" for cl, n in zip(CLASSES, TRUE_N)], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.24)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center",
                  bbox_to_anchor=(.5, -.30))
        ax.set_title("Per-class scores", fontsize=9.5)
        # headline metrics
        ax = axes[2]
        vals = [pacc(c), pmf1(c), m["accuracy"], m["macro_f1"]]
        labs = ["Pooled\naccuracy", "Pooled\nmacro F1",
                "Mean-fold\naccuracy", "Mean-fold\nmacro F1"]
        cols = [BLUE, GREEN, "#BBD4E8", "#C3E0BF"]
        b = ax.bar(range(4), vals, .62, color=cols)
        for i, bb in enumerate(b):
            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + .015,
                    f"{vals[i]:.4f}", ha="center", fontsize=8.6,
                    fontweight="bold" if i < 2 else "normal")
        ax.axhline(TARGET_ACC, color=BLUE, ls="--", lw=1.1)
        ax.axhline(TARGET_F1, color=GREEN, ls=":", lw=1.1)
        ax.axhline(CEIL, color=RED, ls="-.", lw=1.1)
        ax.text(3.45, CEIL, " 0.627\n ceiling", fontsize=7, color=RED, va="center")
        ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=7.8)
        ax.set_ylim(0, 1.02)
        ax.set_title("Headline metrics (bold = quote these)", fontsize=9.5)
        tk = lambda b: "ON" if b else "off"
        t = TOG[c]
        fig.suptitle(
            f"{c} · {FULL[c]}   —   rank {RANK[c]} of 12   ·   phase {PHASE[c]}\n"
            f"EVM {tk(t['evm'])} · SimAM {tk(t['simam'])} · 3D-CNN {tk(t['cnn'])} · "
            f"Transformer {tk(t['trans'])}    |    "
            f"25/25 LOSO folds · N=156 · {corr}/156 correct · "
            f"{HW[c]['train_sec']*25/3600:.2f} GPU-h · {HW[c]['vram_mb']/1024:.2f} GB",
            fontsize=10.5, fontweight="bold", y=1.10)
        fig.tight_layout()
        save(fig, f"figM_card_{c}.png")


for f in [m1, m2, m3, m4, m5, m6, m7, m8, cards]:
    f()
print("\nAll figures written to", OUT)
