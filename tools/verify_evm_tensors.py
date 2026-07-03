"""
verify_evm_tensors.py — check that EVM and raw tensor sets differ.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Verify EVM vs raw tensors differ.")
    parser.add_argument(
        "--evm_dir",
        type=Path,
        default=root / "Processed_Data" / "tensors",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=root / "Processed_Data" / "tensors_raw",
    )
    parser.add_argument("--max_files", type=int, default=10)
    args = parser.parse_args()

    evm_files = sorted(args.evm_dir.glob("*.npy"))
    if not evm_files:
        print(f"No tensors in {args.evm_dir}. Run Step 2 with --use_evm first.")
        return 1

    checked = identical = 0
    for evm_path in evm_files[: args.max_files]:
        raw_path = args.raw_dir / evm_path.name
        if not raw_path.is_file():
            print(f"  missing raw pair: {raw_path.name}")
            continue
        evm = np.load(evm_path)
        raw = np.load(raw_path)
        diff = float(np.mean(np.abs(evm.astype(np.float64) - raw.astype(np.float64))))
        checked += 1
        if diff < 1e-6:
            identical += 1
            print(f"  IDENTICAL: {evm_path.name} (EVM not applied?)")
        else:
            print(f"  OK: {evm_path.name} mean |evm-raw| = {diff:.6f}")

    print(f"\nChecked {checked} pairs; identical={identical}")
    if checked == 0:
        return 1
    if identical == checked:
        print("\nFAILED: EVM tensors match raw — re-run Preprocess (Step 2 EVM must use --use_evm).")
        return 2
    print("\nPASSED: EVM and raw tensor sets differ as expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
