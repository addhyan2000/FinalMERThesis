"""
run_holdout_then_loso_gpu.py
============================
Run holdout first, then LOSO sequentially across all 12 valid ablations.
Intended for unattended weekend sweeps.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run holdout then LOSO sequentially on all 12 ablations."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--label_mode",
        choices=["grouped", "raw", "individual"],
        default="grouped",
    )
    parser.add_argument("--include_others", action="store_true")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--loso_max_folds", type=int, default=5)
    parser.add_argument("--full_loso", action="store_true")
    parser.add_argument(
        "--output_base",
        type=Path,
        default=PROJECT_ROOT / "Ablation_Study" / "results_weekend",
        help="Base folder; holdout/ and loso/ subfolders are created.",
    )
    parser.add_argument(
        "--fresh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear each output folder before its run (default true).",
    )
    return parser.parse_args()


def _run_step(label: str, cmd: list[str], cwd: Path) -> int:
    print(f"\n[STEP] {label}")
    print("[CMD] ", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=cwd)
    return completed.returncode


def main() -> int:
    args = _parse_args()
    runner = PROJECT_ROOT / "Ablation_Study" / "run_ablation_experiments.py"
    holdout_root = args.output_base / "holdout"
    loso_root = args.output_base / "loso"

    from Ablation_Study.ablation_config import ALL_GPU_CONFIGS

    common = [
        sys.executable,
        str(runner),
        "--device",
        "cuda",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--label_mode",
        args.label_mode,
        "--configs",
        *ALL_GPU_CONFIGS,
    ]
    if args.include_others and args.label_mode == "grouped":
        common.append("--include_others")

    holdout_cmd = [
        *common,
        "--protocol",
        "holdout",
        "--val_fraction",
        str(args.val_fraction),
        "--output_root",
        str(holdout_root),
    ]
    if args.fresh:
        holdout_cmd.append("--fresh")
    else:
        holdout_cmd.append("--no-fresh")

    loso_cmd = [
        *common,
        "--protocol",
        "loso",
        "--output_root",
        str(loso_root),
    ]
    if args.full_loso:
        loso_cmd.append("--full_loso")
    else:
        loso_cmd.extend(["--loso_max_folds", str(args.loso_max_folds)])
    if args.fresh:
        loso_cmd.append("--fresh")
    else:
        loso_cmd.append("--no-fresh")

    rc = _run_step("Holdout sweep (12 configs)", holdout_cmd, PROJECT_ROOT)
    if rc != 0:
        print(f"[STOP] Holdout failed with exit code {rc}")
        return rc

    rc = _run_step("LOSO sweep (12 configs)", loso_cmd, PROJECT_ROOT)
    if rc != 0:
        print(f"[STOP] LOSO failed with exit code {rc}")
        return rc

    print("\n[DONE] Weekend chain complete.")
    print(f"  holdout results: {holdout_root}")
    print(f"  loso results:    {loso_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
