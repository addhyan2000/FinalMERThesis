"""
plot_ablation_results.py
========================
Generate comparison plots from ablation results (summary.csv + per-config JSON).

Outputs under Ablation_Study/results/plots/:
  - accuracy_macro_f1_bar.png
  - per_class_f1_grouped.png
  - key_configs_confusion_side_by_side.png (alias: confusion_matrices.png)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


KEY_CONFIGS = [
    "config_1_pure_base",
    "config_3_spatial_only",
    "config_7_full_no_attention",
    "config_8_proposed_unified",
]


def _configs_in_summary(df: pd.DataFrame) -> List[str]:
    """Return config folder names from the current summary.csv only."""
    return [str(name) for name in df["config_name"].tolist()]


def _short_name(folder_name: str) -> str:
    match = re.match(r"(config_\d+_\w+)", folder_name)
    return match.group(1) if match else folder_name[:24]


def load_summary(results_root: Path) -> pd.DataFrame:
    summary_path = results_root / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run ablation first.")
    df = pd.read_csv(summary_path)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["macro_f1"] = pd.to_numeric(df["macro_f1"], errors="coerce")
    df["short_name"] = df["config_name"].map(_short_name)
    return df


def load_per_class_f1(results_root: Path, config_folder: str) -> Optional[List[float]]:
    json_path = results_root / config_folder / "final_results.json"
    if not json_path.exists():
        return None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    return metrics.get("per_class_f1")


def load_class_names(results_root: Path) -> List[str]:
    for child in sorted(results_root.iterdir()):
        json_path = child / "final_results.json"
        if json_path.exists():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            names = payload.get("class_names")
            if names:
                return list(names)
    return []


def plot_accuracy_f1(df: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.55), 5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["accuracy"], width, label="Accuracy")
    ax.bar(x + width / 2, df["macro_f1"], width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["short_name"], rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Configurations — Accuracy vs Macro F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=0.28)
    out_path = out_dir / "accuracy_macro_f1_bar.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_per_class_f1(results_root: Path, df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    class_names = load_class_names(results_root)
    if not class_names:
        return None

    rows = []
    labels = []
    for _, row in df.iterrows():
        f1 = load_per_class_f1(results_root, row["config_name"])
        if f1 and len(f1) == len(class_names):
            rows.append(f1)
            labels.append(_short_name(row["config_name"]))

    if not rows:
        return None

    data = np.asarray(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
    x = np.arange(len(class_names))
    width = 0.8 / max(len(labels), 1)
    for i, (scores, label) in enumerate(zip(data, labels)):
        offset = (i - (len(labels) - 1) / 2) * width
        ax.bar(x + offset, scores, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Per-Class F1 by Configuration")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "per_class_f1_grouped.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion_side_by_side(
    results_root: Path,
    out_dir: Path,
    df: pd.DataFrame,
) -> Optional[Path]:
    """Plot confusion matrices for every config row in summary.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    panels = []
    titles = []
    class_names: List[str] = []
    for config_name in _configs_in_summary(df):
        cfg_dir = results_root / config_name
        if not cfg_dir.is_dir():
            matches = [
                p for p in results_root.iterdir()
                if p.is_dir() and p.name.startswith(config_name.split("__")[0] + "__")
            ]
            cfg_dir = matches[0] if matches else cfg_dir
        cm_path = cfg_dir / "confusion_matrix.npy"
        json_path = cfg_dir / "final_results.json"
        if not cm_path.exists():
            continue
        cm = np.load(cm_path)
        panels.append(cm)
        titles.append(_short_name(config_name))
        if json_path.exists() and not class_names:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            class_names = payload.get("class_names", [])

    if not panels:
        return None

    ncols = min(len(panels), 4)
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_list = np.atleast_1d(axes).ravel().tolist()
    for ax, cm, title in zip(axes_list, panels, titles):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title, fontsize=9)
        if class_names:
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(class_names, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes_list[len(panels):]:
        ax.axis("off")
    fig.suptitle("Confusion Matrices — Current Run", fontsize=10)
    fig.tight_layout()
    out_path = out_dir / "confusion_matrices.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_key_confusion_side_by_side(results_root: Path, out_dir: Path, df: pd.DataFrame) -> Optional[Path]:
    """Backward-compatible alias — plots only configs present in summary.csv."""
    return plot_confusion_side_by_side(results_root, out_dir, df)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Plot ablation study results.")
    parser.add_argument(
        "--results_root",
        type=Path,
        default=project_root / "Ablation_Study" / "results",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    out_dir = results_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_summary(results_root)
    except FileNotFoundError as err:
        print(err)
        return 1

    if df.empty:
        print("summary.csv is empty.")
        return 1

    paths = []
    paths.append(plot_accuracy_f1(df, out_dir))
    p2 = plot_per_class_f1(results_root, df, out_dir)
    if p2:
        paths.append(p2)
    p3 = plot_confusion_side_by_side(results_root, out_dir, df)
    if p3:
        paths.append(p3)
        legacy = out_dir / "key_configs_confusion_side_by_side.png"
        if legacy != p3 and p3.exists():
            import shutil
            shutil.copy2(p3, legacy)
            paths.append(legacy)

    print("Plots written:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
