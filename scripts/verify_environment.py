#!/usr/bin/env python3
"""
Environment & Dependency Verification Utility.
Zetheta Algorithms - Bayesian Regime Detection Engine.

Validates host environment, Python version, dependencies, hardware accelerators, and R runtime.
"""

import sys
import platform
import shutil
import subprocess

def check_python_version() -> bool:
    v = sys.version_info
    print(f"[*] Python Version: {v.major}.{v.minor}.{v.micro} on {platform.system()} ({platform.machine()})")
    if v.major != 3 or v.minor < 10 or v.minor > 13:
        print("  [!] WARNING: Recommended Python version is 3.10, 3.11, or 3.12.")
        return True
    return True

def check_package(pkg_name: str, import_name: str = None) -> bool:
    if import_name is None:
        import_name = pkg_name
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  [+] {pkg_name:<20}: Installed (v{ver})")
        return True
    except ImportError:
        print(f"  [-] {pkg_name:<20}: NOT FOUND")
        return False

def check_r_environment() -> bool:
    r_path = shutil.which("R")
    if r_path:
        try:
            ver = subprocess.check_output([r_path, "--version"], stderr=subprocess.STDOUT).decode("utf-8").split("\n")[0]
            print(f"  [+] R Runtime           : Installed ({ver}) at {r_path}")
            return True
        except Exception:
            print("  [-] R Runtime           : Found binary but failed to query version")
            return False
    else:
        print("  [-] R Runtime           : NOT FOUND in PATH (Dual-language validation will run via Conda environment or Rscript container)")
        return False

def check_hardware_accelerator() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"  [+] PyTorch Accelerator : CUDA GPU available ({device_name})")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  [+] PyTorch Accelerator : Apple Silicon MPS available")
        else:
            print("  [*] PyTorch Accelerator : CPU fallback mode active")
    except ImportError:
        print("  [-] PyTorch not installed")

def main():
    print("=" * 70)
    print("ZETHETA ALGORITHMS - BAYESIAN REGIME ENGINE")
    print("Environment Verification Suite")
    print("=" * 70)

    check_python_version()

    print("\n[*] Checking Core Scientific Stack:")
    core_pkgs = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("pydantic", "pydantic"),
        ("pyyaml", "yaml"),
        ("statsmodels", "statsmodels"),
        ("hmmlearn", "hmmlearn"),
        ("torch", "torch"),
        ("pytest", "pytest"),
    ]
    all_core_ok = True
    for pkg, imp in core_pkgs:
        if not check_package(pkg, imp):
            all_core_ok = False

    print("\n[*] Checking Probabilistic & Deep Learning Packages:")
    prob_pkgs = [
        ("arviz", "arviz"),
        ("pymc", "pymc"),
        ("pyro-ppl", "pyro"),
        ("transformers", "transformers"),
    ]
    for pkg, imp in prob_pkgs:
        check_package(pkg, imp)

    print("\n[*] Checking Hardware & Dual-Language Runtime:")
    check_hardware_accelerator()
    check_r_environment()

    print("\n" + "=" * 70)
    if all_core_ok:
        print("[SUCCESS] Core dependencies are satisfied. Ready for architectural testing.")
    else:
        print("[NOTICE] Some dependencies missing. Install via: pip install -r requirements.txt")
    print("=" * 70)

if __name__ == "__main__":
    main()
