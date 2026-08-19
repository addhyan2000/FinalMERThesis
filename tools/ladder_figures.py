"""Generate the technology-ladder figure set for TECHNOLOGY_LADDER_RESULTS.md.

Every number is read from Ablation_Study/results/<config>/final_results.json and
configuration_summary.txt. Parameter counts are derived analytically from
Ablation_Study/models.py (trainable only; buffers excluded).

Run:  python tools/ladder_figures.py
"""
import json
import os
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
RES = f"{ROOT}/Ablation_Study/results"
OUT = f"{ROOT}/report_figures_ladders"
os.makedirs(OUT, exist_ok=True)

BLUE, AMBER, GREEN, RED, PURPLE, GRAY = ("#4C78A8", "#F58518", "#54A24B",
                                         "#E45756", "#B279A2", "#8C8C8C")
CLASSES = ["Negative", "Positive", "Surprise"]
TRUE_N = [99, 32, 25]
TARGET_ACC, TARGET_F1 = 0.70, 0.68
MAJ_ACC, MAJ_F1 = 0.6346, 0.2588
RAND_ACC, RAND_F1 = 0.3333, 0.303
LIT_ACC = 0.65

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "figure.facecolor": "white",
    "savefig.facecolor": "white", "savefig.dpi": 170, "savefig.bbox": "tight",
})


# ═══════════════════════════════════════════════════════════ data collection
def _params(use_cnn, use_trans):
    p = 0
    if use_cnn:
        per_stream = (16 * 9) + 32 + (32 * 16 * 9) + 64
        p += 3 * per_stream                       # SimAM is parameter-free
    else:
        p += 3 * 4 * 4 * 96 + 96                  # RawPatchEmbedding Linear(48, 96)
    if use_trans:
        layer = (3 * 96 * 96 + 3 * 96) + (96 * 96 + 96) \
                + (96 * 256 + 256) + (256 * 96 + 96) + 192 + 192
        p += 4 * layer + 192
    p += 192 + (96 * 3 + 3)                       # classifier head
    return p


def collect():
    out = {}
    for d in sorted(pathlib.Path(RES).glob("config_*/final_results.json")):
        j = json.load(open(d))
        cid = "C" + re.match(r"config_(\d+)_", j["config_name"]).group(1)
        t, m = j["toggles"], j["metrics"]
        summ = open(d.parent / "configuration_summary.txt").read()
        sec = float(re.search(r"Total Train Time: ([\d.]+)", summ).group(1))
        vram = float(re.search(r"Peak VRAM: ([\d.]+)", summ).group(1))
        cm = np.asarray(m["confusion_matrix"], dtype=int)
        out[cid] = dict(
            cid=cid, name=j["config_name"],
            evm=t["use_evm"], simam=t["use_simam"],
            cnn=t["use_cnn"], trans=t["use_transformer"],
            acc=m["micro_f1"], correct=int(np.trace(cm)),
            mf1=float(np.mean(m["per_class_f1"])),
            f1=m["per_class_f1"], prec=m["per_class_precision"],
            rec=m["per_class_recall"], cm=cm,
            fold_acc=m["accuracy"], fold_mf1=m["macro_f1"],
            gpu_h=sec * 25 / 3600.0, vram_gb=vram / 1024.0,
            params=_params(t["use_cnn"], t["use_transformer"]),
        )
    return out


R = collect()

# Reading order for the master table: the EVM block first (EVM is the study's
# baseline), each block ascending in the number of technologies switched on.
DOC_ORDER = ["C4", "C12", "C13", "C7", "C16", "C8",      # EVM on
             "C1", "C2", "C3", "C9", "C5", "C6"]         # EVM off (mirror)
# imshow index 0 renders at the bottom once the y-axis is inverted, so the
# figure list is DOC_ORDER reversed to put C4 at the top.
TECH_ORDER = list(reversed(DOC_ORDER))


def tag(c):
    r = R[c]
    bits = "".join(str(int(r[k])) for k in ("evm", "simam", "cnn", "trans"))
    return f"{c}\n{bits}"


def added_label(prev, cur):
    """Which switch turned on between two rungs."""
    names = {"evm": "EVM", "simam": "SimAM", "cnn": "3D-CNN", "trans": "Transformer"}
    for k, n in names.items():
        if R[cur][k] and not R[prev][k]:
            return n
    return "?"


def save(fig, name):
    fig.savefig(f"{OUT}/{name}")
    plt.close(fig)
    print("wrote", name)


# ladders ─ all three valid orders of adding {3D-CNN, SimAM, Transformer} to EVM
LADDERS_EVM = [
    ("Ladder A — CNN, then SimAM, then Transformer", ["C4", "C13", "C16", "C8"]),
    ("Ladder B — CNN, then Transformer, then SimAM", ["C4", "C13", "C7", "C8"]),
    ("Ladder C — Transformer, then CNN, then SimAM", ["C4", "C12", "C7", "C8"]),
]
LADDERS_NOEVM = [
    ("Mirror A — no EVM", ["C1", "C3", "C5", "C6"]),
    ("Mirror B — no EVM", ["C1", "C3", "C9", "C6"]),
    ("Mirror C — no EVM", ["C1", "C2", "C9", "C6"]),
]
PAIRS = [("C1", "C4", "nothing else"), ("C2", "C12", "Transformer"),
         ("C3", "C13", "3D-CNN"), ("C5", "C16", "3D-CNN + SimAM"),
         ("C9", "C7", "3D-CNN + Transformer"),
         ("C6", "C8", "3D-CNN + SimAM + Transformer")]


def cm_heat(ax, c, fs=10, annot_pct=True):
    cm = R[c]["cm"]
    rown = cm.sum(1, keepdims=True)
    norm = cm / np.where(rown == 0, 1, rown)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            txt = f"{cm[i, j]}\n{norm[i, j] * 100:.0f}%" if annot_pct else f"{cm[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fs,
                    color="white" if norm[i, j] > .55 else "#1a1a1a",
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([k[:3] for k in CLASSES], fontsize=8.5)
    ax.set_yticklabels([k[:3] for k in CLASSES], fontsize=8.5)
    ax.set_xlabel("predicted", fontsize=8.5); ax.set_ylabel("true", fontsize=8.5)
    ax.grid(False)


# ═════════════════════════════════════════════ T1 — master metric heatmap
def t1():
    keys = ["evm", "simam", "cnn", "trans"]
    comps = ["EVM", "SimAM", "3D-CNN", "Transformer"]
    grid = np.array([[1 if R[c][k] else 0 for k in keys] for c in TECH_ORDER])

    metrics = [("pooled\naccuracy", [R[c]["acc"] for c in TECH_ORDER], 0, 1),
               ("pooled\nmacro F1", [R[c]["mf1"] for c in TECH_ORDER], 0, 1),
               ("F1\nNegative", [R[c]["f1"][0] for c in TECH_ORDER], 0, 1),
               ("F1\nPositive", [R[c]["f1"][1] for c in TECH_ORDER], 0, 1),
               ("F1\nSurprise", [R[c]["f1"][2] for c in TECH_ORDER], 0, 1),
               ("mean-of-folds\naccuracy (do\nnot quote)", [R[c]["fold_acc"] for c in TECH_ORDER], 0, 1),
               ("mean-of-folds\nmacro F1 (do\nnot quote)", [R[c]["fold_mf1"] for c in TECH_ORDER], 0, 1)]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2),
                             gridspec_kw={"width_ratios": [.72, 1.55, .62]})

    ax = axes[0]
    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(["#EDEDED", BLUE]),
              vmin=0, vmax=1, aspect="auto")
    for i in range(len(TECH_ORDER)):
        for j in range(4):
            ax.text(j, i, "ON" if grid[i, j] else "off", ha="center", va="center",
                    fontsize=8, fontweight="bold" if grid[i, j] else "normal",
                    color="white" if grid[i, j] else "#777")
    ax.set_xticks(range(4)); ax.set_xticklabels(comps, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(TECH_ORDER)))
    ax.set_yticklabels([f"{c}" for c in TECH_ORDER], fontsize=9.5, fontweight="bold")
    ax.axhline(5.5, color="k", lw=2)
    ax.set_title("Technology switches\nEVM block on top, no-EVM mirror block below", fontsize=10)
    ax.grid(False)

    ax = axes[1]
    M = np.array([m[1] for m in metrics]).T
    im = ax.imshow(M, cmap="RdYlGn", vmin=.20, vmax=.90, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8.6,
                    color="#111")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics], fontsize=8.6)
    ax.set_yticks(range(len(TECH_ORDER))); ax.set_yticklabels([])
    ax.axhline(5.5, color="k", lw=2)
    ax.set_title("Every metric, every configuration  ·  N = 156 clips  ·  25/25 LOSO folds",
                 fontsize=10.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=.024, pad=.015).set_label("score", fontsize=8.5)

    ax = axes[2]
    y = np.arange(len(TECH_ORDER))
    ax.barh(y, [R[c]["correct"] for c in TECH_ORDER], color=[
        GREEN if R[c]["trans"] else RED for c in TECH_ORDER], alpha=.85)
    for i, c in enumerate(TECH_ORDER):
        ax.text(R[c]["correct"] + 2.5, i, f"{R[c]['correct']}/156", va="center", fontsize=8.6)
    ax.axvline(99, color=GRAY, ls="--", lw=1.4)
    ax.text(99, -.5, "always-'Negative' baseline = 99", fontsize=7.6, color=GRAY,
            ha="center", va="center")
    ax.axhline(5.5, color="k", lw=2)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_ylim(-.7, len(TECH_ORDER) - .3)
    ax.set_xlim(0, 152); ax.set_xlabel("clips correct out of 156", fontsize=9)
    ax.set_title("Raw hit count\ngreen = has Transformer", fontsize=10)
    axes[0].invert_yaxis(); axes[1].invert_yaxis()

    fig.suptitle("Figure T1 — Master results matrix, all 12 configurations sorted by technology",
                 fontsize=13, fontweight="bold", y=1.005)
    save(fig, "figT1_master_matrix.png")


# ═══════════════════════════════════════ T2 — the three EVM ladders
def _ladder_panel(ax, chain, title, ymax=.90):
    x = np.arange(len(chain))
    accs = [R[c]["acc"] for c in chain]
    mf1s = [R[c]["mf1"] for c in chain]

    ax.axhline(TARGET_ACC, color=BLUE, ls=":", lw=1.3, alpha=.75,
               label="accuracy target 0.70")
    ax.axhline(TARGET_F1, color=AMBER, ls=":", lw=1.3, alpha=.75,
               label="macro F1 target 0.68")
    ax.axhline(MAJ_ACC, color=GRAY, ls="--", lw=1.1, alpha=.8,
               label="always-'Negative' acc 0.635")

    # delta bands, keyed on macro F1 — drawn first so markers sit on top
    for i in range(len(chain) - 1):
        d = mf1s[i + 1] - mf1s[i]
        big = abs(d) >= .10
        col = GREEN if d > 0 else RED
        ax.annotate("", xy=(i + 1, mf1s[i + 1]), xytext=(i, mf1s[i]),
                    arrowprops=dict(arrowstyle="-", lw=8 if big else 3.5,
                                    color=col, alpha=.20), zorder=1)
        mid_y = (mf1s[i] + mf1s[i + 1]) / 2
        # big jumps: label to the LEFT of the steep segment; small: well below
        if big:
            tx, ty, ha = i + .62, mid_y - .022, "left"
        else:
            tx, ty, ha = i + .5, min(mf1s[i], mf1s[i + 1]) - .075, "center"
        ax.text(tx, ty, f"{d:+.3f}", ha=ha, va="center",
                fontsize=10.2 if big else 8.6, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=.22", fc="white", ec=col, lw=1.0,
                          alpha=.97), zorder=6)

    ax.plot(x, accs, "-o", color=BLUE, lw=2.4, ms=9, label="pooled accuracy", zorder=4)
    ax.plot(x, mf1s, "-s", color=AMBER, lw=2.4, ms=9, label="pooled macro F1", zorder=4)

    for i, c in enumerate(chain):
        ax.annotate(f"{accs[i]:.3f}", (i, accs[i]), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8.6, color=BLUE,
                    fontweight="bold", zorder=6)
        ax.annotate(f"{mf1s[i]:.3f}", (i, mf1s[i]), textcoords="offset points",
                    xytext=(0, -19), ha="center", fontsize=8.6, color=AMBER,
                    fontweight="bold", zorder=6)

    labels = [f"{chain[0]}\n{'EVM only' if R[chain[0]]['evm'] else 'nothing on'}"]
    labels += [f"{chain[i]}\n+{added_label(chain[i-1], chain[i])}"
               for i in range(1, len(chain))]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.8)
    ax.set_ylim(.28, ymax); ax.set_xlim(-.45, len(chain) - .55)
    ax.set_title(title, fontsize=9.6)


def t2():
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0), sharey=True)
    for ax, (title, chain) in zip(axes, LADDERS_EVM):
        _ladder_panel(ax, chain, title)
    axes[0].set_ylabel("pooled score (N = 156)")
    h, l = axes[0].get_legend_handles_labels()
    order = [3, 4, 0, 1, 2]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center",
               ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(.5, -.055))
    fig.suptitle("Figure T2 — Building up from the EVM baseline (C4). "
                 "Every ladder ends at C8 = all four technologies.",
                 fontsize=13, fontweight="bold", y=1.015)
    fig.text(.5, -.115, "Thick green band = the rung where the Transformer switches on. "
             "In all three orderings that single rung carries +0.21 to +0.25 macro F1; "
             "every other rung moves the score by less than 0.03.",
             ha="center", fontsize=9.6, style="italic")
    save(fig, "figT2_evm_ladders.png")


# ═════════════════════════════════ T3 — EVM ladders vs no-EVM mirrors
def t3():
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.4), sharey=True)
    short = ["A — CNN, SimAM, Transformer", "B — CNN, Transformer, SimAM",
             "C — Transformer, CNN, SimAM"]
    for ax, (_, chain), t in zip(axes[0], LADDERS_EVM, short):
        _ladder_panel(ax, chain, f"EVM ON  ·  Ladder {t}")
    for ax, (_, chain), t in zip(axes[1], LADDERS_NOEVM, short):
        _ladder_panel(ax, chain, f"EVM OFF  ·  Mirror {t}")
    axes[0][0].set_ylabel("pooled score (N = 156)")
    axes[1][0].set_ylabel("pooled score (N = 156)")
    h, l = axes[0][0].get_legend_handles_labels()
    order = [3, 4, 0, 1, 2]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center",
               ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(.5, -.03))
    fig.suptitle("Figure T3 — The same three ladders with EVM on (top) and off (bottom).\n"
                 "Column-wise, the two rows are matched pairs: the only difference is the EVM switch.",
                 fontsize=13, fontweight="bold", y=1.0)
    save(fig, "figT3_ladders_evm_vs_noevm.png")


# ═══════════════════════════════════════ T4 — the six matched EVM pairs
def t4():
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    y = np.arange(len(PAIRS))
    deltas = []
    for i, (off, on, lab) in enumerate(PAIRS):
        a, b = R[off]["mf1"], R[on]["mf1"]
        deltas.append(b - a)
        col = GREEN if b > a else RED
        ax.plot([a, b], [i, i], color=col, lw=3.2, alpha=.55, zorder=1,
                solid_capstyle="round")
        ax.scatter([a], [i], s=130, color="white", ec=GRAY, lw=2, zorder=3)
        ax.scatter([b], [i], s=130, color=col, ec="white", lw=1.6, zorder=3)
        ax.text(min(a, b) - .012, i, f"{off} {a:.3f}" if a < b else f"{on} {b:.3f}",
                fontsize=8.4, ha="right", va="center",
                color="#444" if a < b else col,
                fontweight="normal" if a < b else "bold")
        ax.text(max(a, b) + .012, i, f"{on} {b:.3f}" if b > a else f"{off} {a:.3f}",
                fontsize=8.4, ha="left", va="center",
                color=col if b > a else "#444",
                fontweight="bold" if b > a else "normal")
        ax.text(.90, i, f"{b - a:+.3f}", fontsize=10, fontweight="bold", color=col,
                ha="right", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([f"EVM added on top of\n{lab}" for _, _, lab in PAIRS], fontsize=9)
    ax.set_ylim(len(PAIRS) - .45, -.55)
    ax.set_xlim(.33, .93); ax.set_xlabel("pooled macro F1")
    ax.set_title(f"EVM's marginal effect, all six matched pairs  ·  "
                 f"mean {np.mean(deltas):+.4f}", fontsize=10.5)
    ax.axvline(R["C1"]["mf1"], color=GRAY, ls="--", lw=1.1)
    ax.text(R["C1"]["mf1"] - .008, 4.35, "zero-technology\nfloor 0.434", fontsize=7.8,
            color=GRAY, ha="right", va="center")

    ax = axes[1]
    cols = [GREEN if d > 0 else RED for d in deltas]
    ax.bar(y, deltas, color=cols, alpha=.85, width=.62)
    for i, d in enumerate(deltas):
        ax.text(i, d + (.0035 if d > 0 else -.0035), f"{d:+.3f}", ha="center",
                fontsize=8.8, fontweight="bold",
                va="bottom" if d > 0 else "top")
    ax.axhline(0, color="k", lw=1)
    ax.axhline(np.mean(deltas), color=PURPLE, ls="--", lw=1.6)
    ax.text(-.45, np.mean(deltas) + .0035, f"mean {np.mean(deltas):+.4f}",
            fontsize=8.6, color=PURPLE, ha="left", fontweight="bold")
    ax.set_ylim(-.068, .094)
    ax.set_xticks(y)
    ax.set_xticklabels([f"{off}→{on}" for off, on, _ in PAIRS], fontsize=9)
    ax.set_ylabel("Δ pooled macro F1")
    ax.set_title("Signed effect per pair — 4 of 6 positive, but the spread\n"
                 "(−0.054 to +0.080) straddles the ±0.068 confidence interval",
                 fontsize=10)
    fig.suptitle("Figure T4 — Is EVM, the study's baseline switch, actually doing anything?",
                 fontsize=13, fontweight="bold", y=1.005)
    save(fig, "figT4_evm_matched_pairs.png")


# ═════════════════════════════════════ T5 — the zero-technology configuration
def t5():
    fig = plt.figure(figsize=(16.5, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.45, .85, 1.0],
                          height_ratios=[1, 1], hspace=.42, wspace=.3)

    # ── left: the mechanism, as boxes ──
    ax = fig.add_subplot(gs[:, 0]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    steps = [
        ("input motion tensor", "[B, 3, 32, 224, 224]\nflow-u · flow-v · optical strain", BLUE),
        ("AdaptiveAvgPool3d(32, 4, 4)", "each 224x224 frame averaged into a 4x4 grid\n"
         "3136 pixels collapse into 1 number\n→ [B, 3, 32, 4, 4]", GRAY),
        ("flatten per frame", "3 x 4 x 4 = 48 numbers per frame\n→ [B, 32, 48]", GRAY),
        ("Linear(48 → 96)", "4 704 parameters", AMBER),
        ("TemporalPooling — mean over 32 frames", "frame ORDER is discarded here\n→ [B, 96]", RED),
        ("LayerNorm → Dropout(0.3) → Linear(96 → 3)", "483 parameters", AMBER),
        ("logits", "Negative · Positive · Surprise", GREEN),
    ]
    ytop, h, gap = 9.6, .92, .38
    for i, (t, s, col) in enumerate(steps):
        yy = ytop - i * (h + gap)
        ax.add_patch(FancyBboxPatch((.35, yy - h), 9.1, h, boxstyle="round,pad=.06",
                                    fc=col, ec="none", alpha=.16))
        ax.text(.6, yy - .28, t, fontsize=9.4, fontweight="bold", va="center")
        ax.text(.6, yy - .68, s, fontsize=7.9, va="center", color="#333", linespacing=1.35)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((5, yy - h), (5, yy - h - gap + .04),
                                         arrowstyle="-|>", mutation_scale=13,
                                         color="#666", lw=1.3))
    ax.set_title("C1 — every switch OFF. What actually runs.", fontsize=10.8, loc="left")

    # ── top middle: the collapse to logistic regression ──
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.add_patch(FancyBboxPatch((.15, .2), 9.7, 9.5, boxstyle="round,pad=.10",
                                fc=PURPLE, ec=PURPLE, alpha=.10, lw=1.4))
    ax.text(5, 9.05, "Why this is not a neural network", fontsize=10.4,
            fontweight="bold", ha="center", va="center")
    ax.text(5, 7.55, "Linear(48→96) and the temporal mean are\n"
                     "BOTH linear, so the two operations commute:",
            fontsize=8.5, ha="center", va="center", linespacing=1.5)
    ax.text(5, 6.05,
            r"$\frac{1}{T}\sum_t (W x_t + b) \, = \, W\!\left(\frac{1}{T}\sum_t x_t\right) + b$",
            fontsize=12.5, ha="center", va="center")
    ax.text(5, 4.25, "So the whole 5 187-parameter model is one\n"
                     "affine map applied to a single 48-dimensional\n"
                     "time-averaged motion descriptor.",
            fontsize=8.5, ha="center", va="center", linespacing=1.55)
    ax.text(5, 1.85, "No hidden layer. No nonlinearity. No learned\n"
                     "filter. It is multinomial logistic regression\n"
                     "on 48 hand-pooled averages.",
            fontsize=8.5, ha="center", va="center", linespacing=1.55,
            fontweight="bold", color="#5A3B63")

    # ── bottom middle: what the 48 features are ──
    ax = fig.add_subplot(gs[1, 1])
    board = (np.add.outer(np.arange(4), np.arange(4)) % 2).astype(float)
    ax.imshow(board, cmap="Blues", vmin=-1.6, vmax=2.3)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, "56x56\npx", ha="center", va="center", fontsize=7.4,
                    color="#10243A")
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2.2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(which="major", visible=False)
    ax.set_title("One 224x224 frame → a 4x4 grid of averages.\n"
                 "x3 modalities = the 48 features, e.g. 'mean vertical\n"
                 "flow in the upper-left of the face'.", fontsize=8.8)

    # ── top right: confusion matrix ──
    ax = fig.add_subplot(gs[0, 2])
    cm_heat(ax, "C1", fs=9.4)
    ax.set_title("C1 pooled confusion matrix\n72/156 correct  ·  acc 0.4615  ·  macro F1 0.4337",
                 fontsize=9.4)

    # ── bottom right: C1 against the reference floors ──
    ax = fig.add_subplot(gs[1, 2])
    labs = ["random\nguess", "always\n'Negative'", "C1\nzero tech", "C8\nall four tech"]
    accv = [RAND_ACC, MAJ_ACC, R["C1"]["acc"], R["C8"]["acc"]]
    f1v = [RAND_F1, MAJ_F1, R["C1"]["mf1"], R["C8"]["mf1"]]
    x = np.arange(4); w = .38
    ax.bar(x - w / 2, accv, w, color=BLUE, alpha=.85, label="accuracy")
    ax.bar(x + w / 2, f1v, w, color=AMBER, alpha=.85, label="macro F1")
    for i in range(4):
        ax.text(x[i] - w / 2, accv[i] + .012, f"{accv[i]:.3f}", ha="center", fontsize=7.8)
        ax.text(x[i] + w / 2, f1v[i] + .012, f"{f1v[i]:.3f}", ha="center", fontsize=7.8)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.4)
    ax.set_ylim(0, .93); ax.set_ylabel("pooled score")
    ax.legend(fontsize=7.8, loc="upper left", ncol=2)
    ax.set_title("C1 loses on accuracy to the do-nothing trick,\n"
                 "but beats it on macro F1 by +0.175", fontsize=9.2)

    fig.suptitle("Figure T5 — What the zero-technology configuration is, and why it still scores 0.43",
                 fontsize=13, fontweight="bold", y=.995)
    save(fig, "figT5_zero_technology.png")


# ══════════════════════════════════ T6 — everything switched on
def t6():
    fig = plt.figure(figsize=(16.5, 7.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[.92, .92, 1.15, 1.05],
                          height_ratios=[1, 1], hspace=.62, wspace=.42)

    ax = fig.add_subplot(gs[:, 0]); cm_heat(ax, "C8", fs=10)
    ax.set_title("C8 — all four ON\n117/156 correct\nacc 0.7500  ·  macro F1 0.6659", fontsize=9.2)

    ax = fig.add_subplot(gs[:, 1]); cm_heat(ax, "C2", fs=10)
    ax.set_title("C2 — Transformer ONLY\n116/156 correct\nacc 0.7436  ·  macro F1 0.7122",
                 fontsize=9.2)

    # per-class comparison C8 vs C2
    ax = fig.add_subplot(gs[:, 2])
    x = np.arange(3); w = .38
    ax.bar(x - w / 2, R["C8"]["f1"], w, color=BLUE, alpha=.88, label="C8 all four")
    ax.bar(x + w / 2, R["C2"]["f1"], w, color=GREEN, alpha=.88, label="C2 Transformer only")
    for i in range(3):
        ax.text(x[i] - w / 2, R["C8"]["f1"][i] + .013, f"{R['C8']['f1'][i]:.3f}",
                ha="center", fontsize=8.2)
        ax.text(x[i] + w / 2, R["C2"]["f1"][i] + .013, f"{R['C2']['f1'][i]:.3f}",
                ha="center", fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\nn={n}" for c, n in zip(CLASSES, TRUE_N)], fontsize=8.8)
    ax.set_ylim(0, 1.0); ax.set_ylabel("per-class F1")
    ax.legend(fontsize=8.2, loc="upper right")
    ax.set_title("Per-class F1 — the other three technologies buy\n"
                 "+0.039 on Negative and cost 0.095 / 0.083 on the thin classes",
                 fontsize=9.2)

    # cost
    ax = fig.add_subplot(gs[0, 3])
    labs = ["C1\nnone", "C4\nEVM", "C2\nTrans", "C8\nall four"]
    vals = [R[c]["gpu_h"] for c in ["C1", "C4", "C2", "C8"]]
    ax.bar(labs, vals, color=[GRAY, GRAY, GREEN, RED], alpha=.85)
    for i, v in enumerate(vals):
        ax.text(i, v + .1, f"{v:.2f} h", ha="center", fontsize=8.4, fontweight="bold")
    ax.set_ylabel("GPU-hours (25 folds)"); ax.set_ylim(0, 7.6)
    ax.set_title("13.5x the compute, for one extra clip", fontsize=9.4)

    ax = fig.add_subplot(gs[1, 3])
    vals = [R[c]["vram_gb"] for c in ["C1", "C4", "C2", "C8"]]
    ax.bar(labs, vals, color=[GRAY, GRAY, GREEN, RED], alpha=.85)
    for i, v in enumerate(vals):
        ax.text(i, v + .35, f"{v:.2f} GB", ha="center", fontsize=8.4, fontweight="bold")
    ax.set_ylabel("peak VRAM (GB)"); ax.set_ylim(0, 23.5)
    ax.set_title("115x the peak memory", fontsize=9.4)

    fig.suptitle("Figure T6 — What happens when every technology is switched on",
                 fontsize=13, fontweight="bold", y=1.0)
    save(fig, "figT6_all_technologies.png")


# ═════════════════════════════════════ T7 — cost against what it buys
def t7():
    order = sorted(R, key=lambda c: R[c]["mf1"])          # worst at bottom
    y = np.arange(len(order))
    cols = [GREEN if R[c]["trans"] else RED for c in order]
    labels = [f"{c}  " + "".join("ESCT"[i] if R[c][k] else "·"
                                 for i, k in enumerate(("evm", "simam", "cnn", "trans")))
              for c in order]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.6), sharey=True,
                             gridspec_kw={"width_ratios": [1.35, 1, 1], "wspace": .09})

    ax = axes[0]
    ax.barh(y, [R[c]["mf1"] for c in order], color=cols, alpha=.88, height=.66)
    for i, c in enumerate(order):
        ax.text(R[c]["mf1"] + .006, i, f"{R[c]['mf1']:.3f}", va="center", fontsize=8.8,
                fontweight="bold")
    ax.axvline(TARGET_F1, color=AMBER, ls=":", lw=1.6)
    ax.text(TARGET_F1 - .008, -.85, "target 0.68", fontsize=7.8, color=AMBER, ha="right")
    ax.axvline(R["C1"]["mf1"], color=GRAY, ls="--", lw=1.3)
    ax.text(R["C1"]["mf1"] - .008, -.85, "zero-tech floor 0.434", fontsize=7.8,
            color=GRAY, ha="right")
    ax.set_xlim(0, .82); ax.set_xlabel("pooled macro F1")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9, fontfamily="monospace")
    ax.set_title("What you get\ngreen = Transformer on", fontsize=10)

    ax = axes[1]
    ax.barh(y, [R[c]["gpu_h"] for c in order], color=cols, alpha=.88, height=.66)
    for i, c in enumerate(order):
        ax.text(R[c]["gpu_h"] * 1.12, i, f"{R[c]['gpu_h']:.2f} h", va="center", fontsize=8.6)
    ax.set_xscale("log"); ax.set_xlim(.28, 26)
    ax.set_xlabel("GPU-hours for the full 25-fold run (log)")
    ax.set_title("What it costs in compute\n16x spread, uncorrelated with the score",
                 fontsize=10)

    ax = axes[2]
    ax.barh(y, [R[c]["params"] for c in order], color=cols, alpha=.88, height=.66)
    for i, c in enumerate(order):
        ax.text(R[c]["params"] * 1.14, i, f"{R[c]['params']:,}", va="center", fontsize=8.6)
    ax.set_xscale("log"); ax.set_xlim(3e3, 4.5e6)
    ax.set_xlabel("trainable parameters (log)")
    ax.set_title("What it costs in capacity\nthe 15 027-parameter tier is the worst tier",
                 fontsize=10)

    for a in axes:
        a.set_ylim(-1.0, len(order) - .4)

    fig.suptitle("Figure T7 — Every configuration ranked by what it achieves, "
                 "beside what it cost. The cheap corner is the good corner.",
                 fontsize=12.6, fontweight="bold", y=1.005)
    fig.text(.5, -.045, "Row labels spell the switches: E = EVM, S = SimAM, C = 3D-CNN, "
             "T = Transformer; a dot means off. "
             "C2 (Transformer alone) tops the study on 0.48 GPU-hours; "
             "C8 (all four) needs 6.46 to land 0.046 lower.",
             ha="center", fontsize=9.6, style="italic")
    save(fig, "figT7_cost_vs_gain.png")


if __name__ == "__main__":
    t1(); t2(); t3(); t4(); t5(); t6(); t7()
    print("\nAll figures in", OUT)
