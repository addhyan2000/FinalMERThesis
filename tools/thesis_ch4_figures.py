"""Build the Chapter 4 figures and the single-panel Figure 5.5.

Every value plotted here is taken from the thesis text itself (Chapter 4
tables and the released CASME II coding counts of Section 2.2) or, for the
learning-rate schedule, from the PyTorch schedulers configured exactly as in
the training code. Figure 5.5 is re-read from the stored results so that it
matches Table 5.1.

Outputs (report_figures_thesis/):
    fig4_1_class_distribution.png   255 coded clips -> 156-clip working set
    fig4_2_ablation_grid.png        the 2^4 design: 12 trained, 4 excluded
    fig4_3_parameter_counts.png     parameter totals per architecture type
    fig4_4_lr_schedule.png          warmup + cosine learning-rate schedule
    fig5_5_cost_vs_performance.png  pooled macro F1 against GPU-hours only
"""
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

import thesis_fig_regen as R   # shared style, colours and result loaders

OUT = R.OUT
BLUE, AMBER, GREEN, RED, PURPLE, GRAY = R.BLUE, R.AMBER, R.GREEN, R.RED, R.PURPLE, R.GRAY


# --------------------------------------------------------------------------
def fig4_1():
    """Released seven-label coding (Section 2.2.4) grouped into three classes."""
    groups = ["Negative", "Positive", "Surprise", "Others"]
    parts = {
        "Negative": [("disgust", 63), ("repression", 27), ("sadness", 7), ("fear", 2)],
        "Positive": [("happiness", 32)],
        "Surprise": [("surprise", 25)],
        "Others":   [("others", 99)],
    }
    shades = {"Negative": ["#B2182B", "#D6604D", "#F4A582", "#FDDBC7"],
              "Positive": [GREEN], "Surprise": [BLUE], "Others": ["#BDBDBD"]}
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, g in enumerate(groups):
        bottom = 0
        for (lab, n), col in zip(parts[g], shades[g]):
            ax.bar(i, n, bottom=bottom, color=col, width=.62, edgecolor="white",
                   hatch="//" if g == "Others" else None)
            if n >= 7:
                ax.text(i, bottom + n / 2, f"{lab} {n}", ha="center", va="center", fontsize=8.5,
                        color="white" if col in ("#B2182B", "#D6604D", GREEN, BLUE) else "#222")
            bottom += n
        ax.text(i, bottom + 2.5, str(bottom), ha="center", fontsize=10, fontweight="bold")
    ax.annotate("fear 2", xy=(0.31, 98), xytext=(0.5, 106), fontsize=8,
                arrowprops=dict(arrowstyle="-", color="#555", lw=.8))
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Negative", "Positive", "Surprise", "Others\n(excluded)"])
    ax.set_ylabel("number of clips")
    ax.set_ylim(0, 118)
    ax.grid(axis="x", visible=False)
    ax.text(1.5, 75, "working set:\n99 + 32 + 25 = 156 clips", ha="center", fontsize=9.5, color="#333")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_1_class_distribution.png")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig4_2():
    """Rows: (EVM, SimAM); columns: (CNN, Transformer). Excluded: SimAM on, CNN off."""
    rows = [(0, 0), (0, 1), (1, 0), (1, 1)]
    cols = [(0, 0), (0, 1), (1, 0), (1, 1)]
    cfg = {(0, 0, 0, 0): 1, (0, 0, 0, 1): 2, (0, 0, 1, 0): 3, (1, 0, 0, 0): 4,
           (0, 1, 1, 0): 5, (0, 1, 1, 1): 6, (1, 0, 1, 1): 7, (1, 1, 1, 1): 8,
           (0, 0, 1, 1): 9, (0, 1, 0, 0): 10, (0, 1, 0, 1): 11, (1, 0, 0, 1): 12,
           (1, 0, 1, 0): 13, (1, 1, 0, 0): 14, (1, 1, 0, 1): 15, (1, 1, 1, 0): 16}
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for r, (evm, sim) in enumerate(rows):
        for c, (cnn, tr) in enumerate(cols):
            n = cfg[(evm, sim, cnn, tr)]
            excluded = sim == 1 and cnn == 0
            face = "#E0E0E0" if excluded else ("#DCEAF7" if tr else "#FFFFFF")
            ax.add_patch(Rectangle((c, 3 - r), 1, 1, facecolor=face, edgecolor="#555", lw=1))
            label = f"C{n}"
            sub = "excluded" if excluded else ("baseline" if n == 4 else ("all four" if n == 8 else ""))
            ax.text(c + .5, 3 - r + .58, label, ha="center", va="center", fontsize=12,
                    fontweight="bold", color="#9E9E9E" if excluded else "#1a1a1a")
            if sub:
                ax.text(c + .5, 3 - r + .27, sub, ha="center", va="center", fontsize=8.5,
                        color="#9E9E9E" if excluded else RED, style="italic")
            if excluded:
                ax.plot([c + .08, c + .92], [3 - r + .08, 3 - r + .92], color="#BDBDBD", lw=1)
    onoff = lambda v: "on" if v else "off"
    ax.set_xticks([.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels([f"CNN {onoff(a)}\nTransf. {onoff(b)}" for a, b in cols], fontsize=9)
    ax.set_yticks([3.5, 2.5, 1.5, .5])
    ax.set_yticklabels([f"EVM {onoff(a)}\nSimAM {onoff(b)}" for a, b in rows], fontsize=9)
    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.legend(handles=[Patch(facecolor="#FFFFFF", edgecolor="#555", label="trained, transformer off"),
                       Patch(facecolor="#DCEAF7", edgecolor="#555", label="trained, transformer on"),
                       Patch(facecolor="#E0E0E0", edgecolor="#555", label="excluded (SimAM without CNN)")],
              loc="upper center", bbox_to_anchor=(.5, -.02), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_2_ablation_grid.png")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig4_3():
    """Totals from Section 4.4 (component counts 14,544 / 348,736 / 4,704 / 483)."""
    kinds = ["no CNN,\nno transformer", "CNN,\nno transformer", "no CNN,\ntransformer", "CNN and\ntransformer"]
    totals = [5187, 15027, 353923, 363763]
    comp = ["fallback 4,704 + head 483", "CNN 14,544 + head 483",
            "transformer 348,736 + fallback 4,704\n+ head 483", "transformer 348,736 + CNN 14,544\n+ head 483"]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    cols = [GRAY, AMBER, BLUE, PURPLE]
    y = range(len(kinds))[::-1]
    ax.barh(list(y), totals, color=cols, height=.58)
    ax.set_xscale("log")
    for yi, t, c in zip(y, totals, comp):
        ax.text(t * 1.12, yi, f"{t:,}", va="center", fontsize=10, fontweight="bold")
        ax.text(t * 1.12, yi - .3, c, va="center", fontsize=7.8, color="#444")
    ax.set_yticks(list(y)); ax.set_yticklabels(kinds, fontsize=9)
    ax.set_xlim(2e3, 3e7)
    ax.set_xlabel("trainable parameters (log scale)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_3_parameter_counts.png")
    plt.close(fig)


# --------------------------------------------------------------------------
def lr_curve(epochs=50, warmup=5, base=1e-4, eta_min=1e-7):
    try:
        import torch
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=base, weight_decay=1e-4)
        w = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup)
        c = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup), eta_min=eta_min)
        s = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[w, c], milestones=[warmup])
        out = []
        for _ in range(epochs):
            out.append(opt.param_groups[0]["lr"])
            opt.step(); s.step()
        return out, "torch"
    except ImportError:
        out = [base * (0.1 + 0.9 * e / warmup) for e in range(warmup)]
        T = epochs - warmup
        out += [eta_min + (base - eta_min) * (1 + math.cos(math.pi * k / T)) / 2 for k in range(T)]
        return out, "closed form"


def fig4_4():
    lr, how = lr_curve()
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ep = list(range(1, len(lr) + 1))
    ax.plot(ep, [v * 1e4 for v in lr], "o-", color=BLUE, ms=3.2, lw=1.8)
    ax.axvspan(.5, 5.5, color=AMBER, alpha=.12)
    ax.text(3, 1.04, "linear warmup", ha="center", fontsize=9, color="#8a4b00")
    ax.text(28, 1.04, "cosine annealing", ha="center", fontsize=9, color="#1f4e79")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"learning rate ($\times 10^{-4}$)")
    ax.set_xlim(-2, 51); ax.set_ylim(0, 1.12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_4_lr_schedule.png")
    plt.close(fig)
    return how


# --------------------------------------------------------------------------
TIMES = {  # last-fold training time (s), Section 4.7
    "C1": 56.47, "C4": 58.12, "C12": 67.58, "C2": 68.96, "C3": 830.94, "C13": 831.79,
    "C7": 832.81, "C9": 832.81, "C5": 924.85, "C16": 925.61, "C6": 927.65, "C8": 930.63}
CNN_ON = {"C3", "C13", "C7", "C9", "C5", "C16", "C6", "C8"}
SIMAM_ON = {"C5", "C16", "C6", "C8"}


def fig4_5():
    names = list(TIMES)
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    cols = [(PURPLE if n in SIMAM_ON else AMBER) if n in CNN_ON else GRAY for n in names]
    ax.bar(range(len(names)), [TIMES[n] for n in names], color=cols, width=.66)
    for i, n in enumerate(names):
        ax.text(i, TIMES[n] + 14, f"{TIMES[n]:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel("last-fold training time (s)")
    ax.set_ylim(0, 1050)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=GRAY, label="no convolutional stem"),
                       Patch(color=AMBER, label="convolutional stem"),
                       Patch(color=PURPLE, label="convolutional stem + SimAM")],
              frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_5_training_time.png")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig5_5():
    """Pooled macro F1 against extrapolated GPU-hours (the VRAM panel is dropped)."""
    P, C, F = R.load(), R.load_cost(), R.load_flags()
    order = list(P)
    gpu = {c: C[c][0] * 25 / 3600 for c in order}
    fig, a1 = plt.subplots(figsize=(7.6, 5.0))
    for c in order:
        col = GREEN if F[c]["use_transformer"] else RED
        a1.scatter(gpu[c], P[c], s=95, color=col, zorder=3)
        below = c in ("C1", "C16", "C3")
        a1.annotate(c, (gpu[c], P[c]), textcoords="offset points", xytext=(0, -16 if below else 10),
                    ha="center", fontsize=9)
    a1.set_xlabel("extrapolated cost of the whole 25-fold sweep (GPU hours)\n"
                  "= last fold's training time × 25")
    a1.set_ylabel("pooled macro F1")
    cheapest = min(order, key=lambda c: gpu[c])
    a1.annotate(f"best macro F1, and {gpu['C8']/gpu['C2']:.0f}× cheaper than C8\n"
                f"(C1 at {gpu[cheapest]:.2f} GPU-h is the cheapest run)",
                xy=(gpu["C2"], P["C2"]), xytext=(gpu["C2"] + 1.2, P["C2"] + .02),
                fontsize=9, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
    a1.annotate(f"same score band as C1,\n{gpu['C13']/gpu['C1']:.0f}× the GPU cost",
                xy=(gpu["C13"], P["C13"]), xytext=(gpu["C13"] - 3.4, P["C13"] + .055),
                fontsize=9, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    a1.legend(handles=[Patch(color=GREEN, label="Transformer ON"),
                       Patch(color=RED, label="Transformer OFF")],
              frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(.5, .78))
    a1.set_ylim(.39, .76)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_5_cost_vs_performance.png")
    plt.close(fig)
    return P, gpu


# --------------------------------------------------------------------------
def fig5_10():
    """Predicted-class counts per configuration against the true class supports.

    Read from the pooled confusion matrix stored for each configuration
    (column sums = number of predictions per class).
    """
    import glob, json, os
    order = ["C2", "C8", "C7", "C12", "C6", "C9", "C13", "C4", "C1", "C5", "C3", "C16"]  # Table 5.1 rank
    pred = {}
    for d in glob.glob(f"{R.RESULTS}/config_*"):
        n = "C" + os.path.basename(d).split("_")[1]
        cm = json.load(open(f"{d}/final_results.json"))["metrics"]["confusion_matrix"]
        pred[n] = [sum(r[k] for r in cm) for k in range(3)]
    names, cols, true = ["Negative", "Positive", "Surprise"], ["#B2182B", GREEN, BLUE], [99, 32, 25]
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    w = .26
    for k in range(3):
        xs = [i + (k - 1) * w for i in range(len(order))]
        ax.bar(xs, [pred[c][k] for c in order], w, color=cols[k], label=f"predicted {names[k]}")
        ax.axhline(true[k], color=cols[k], ls="--", lw=1.1, alpha=.8)
        ax.text(11.55, true[k] + (2.5 if k != 2 else -6), f"true {names[k]} ({true[k]})", color=cols[k], fontsize=8, ha="left")
    ax.axvline(5.5, color="#555", lw=1, ls=":")
    ax.text(2.5, 118, "with temporal transformer", ha="center", fontsize=9, color="#333")
    ax.text(8.5, 118, "without temporal transformer", ha="center", fontsize=9, color="#333")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_xlim(-.6, 13.3)
    ax.set_ylabel("number of clips predicted")
    ax.set_ylim(0, 125)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(.5, -.1), ncol=3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_10_prediction_distribution.png")
    plt.close(fig)
    return pred


if __name__ == "__main__":
    fig4_1(); fig4_2(); fig4_3()
    how = fig4_4()
    P, gpu = fig5_5()
    print("lr schedule computed with", how)
    print({k: round(v, 4) for k, v in P.items()})
    print({k: round(v, 3) for k, v in gpu.items()})
