#!/usr/bin/env python3
"""
Standalone Data Leakage & Lookahead Verification Script.
Zetheta Algorithms - Bayesian Regime Detection Engine.

Executes strict adversarial leakage checks to ensure no feature or split contamination.
"""

import sys
import subprocess

def run_leakage_audit():
    print("=" * 70)
    print("EXECUTING TEMPORAL LEAKAGE & ANTI-CONTAMINATION AUDIT")
    print("=" * 70)

    cmd = [sys.executable, "-m", "pytest", "tests/leakage/", "-v", "--tb=short"]
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n[PASSED] Zero temporal lookahead or data contamination detected.")
    else:
        print("\n[FAILED] Temporal leakage violations identified. Immediate review required.")
        sys.exit(1)

if __name__ == "__main__":
    run_leakage_audit()
