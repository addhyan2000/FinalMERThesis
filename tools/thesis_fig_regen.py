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
"""
import glob
import json
import os
import pathlib

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


if __name__ == "__main__":
    P = load()
    assert len(P) == 12, f"expected 12 configurations, found {len(P)}"
    fig5_4(P)
    fig5_8(P)
    print(f"C2 = {P['C2']:.4f}  C8 = {P['C8']:.4f}")
    print(f"wrote {OUT}/fig5_4_evm_pairs.png and {OUT}/fig5_8_literature.png")
