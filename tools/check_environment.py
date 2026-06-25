"""
check_environment.py
====================
Print Python/PyTorch/CUDA readiness for CPU or GPU mode.
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print("=" * 60)
    print("MER Client GPU - Environment Check")
    print("=" * 60)
    print(f"Python        : {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    print(f"Executable    : {sys.executable}")

    try:
        import torch
    except ImportError:
        print("PyTorch       : NOT INSTALLED")
        print("\nInstall dependencies: pip install -r requirements.txt")
        return 1

    print(f"PyTorch       : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")
    if cuda_ok:
        print(f"CUDA device   : {torch.cuda.get_device_name(0)}")
        print(f"CUDA version  : {torch.version.cuda}")
        print("\nGPU mode is READY.")
        print("Quick GPU test:")
        print("  python tools/run_ablation_gpu.py --epochs 1 --max_samples 16")
    else:
        print("\nGPU mode is NOT available on this machine.")
        print("Install CUDA PyTorch: pip install -r requirements.txt")
        print("Smoke tests still work on CPU:")
        print("  python tools/smoke_step1_cpu.py")

    print("\nSynthetic smoke tests (no dataset):")
    print("  python tools/smoke_step1_cpu.py")
    print("  python tools/smoke_step2_cpu.py")
    print("  python tools/smoke_ablation_cpu.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
