#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) Generator.
Zetheta Algorithms - Bayesian Regime Detection Engine.

Analyzes the real Indian market dataset (2009-2024):
1. Log returns, mean, std, skewness, kurtosis
2. Price paths and cumulative return indices
3. Volatility clustering and India VIX dynamics
4. Large-cap vs Mid-cap relative performance and breadth spread
5. Multi-asset rolling correlations (Nifty vs VIX, USD/INR)
6. Exports publication-grade figures to reports/figures/ and tables to reports/tables/
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.data.snapshot import DataSnapshotEngine
from src.utils.config import get_project_root


def main():
    root = get_project_root()
    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ZETHETA ALGORITHMS — EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)

    # Load latest snapshot
    engine = DataSnapshotEngine()
    with open(engine.manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    snapshot_id = manifest["latest_snapshot"]
    print(f"[*] Loading snapshot: {snapshot_id}")

    data, meta = engine.load_snapshot(snapshot_id)

    # Extract price series
    nifty = data["NIFTY_50"]["close"].dropna()
    midcap = data["NIFTY_MIDCAP_50"]["close"].dropna()
    nifty500 = data["NIFTY_500"]["close"].dropna()
    vix = data["INDIA_VIX"]["close"].dropna()
    usdinr = data["USD_INR"]["close"].dropna()

    # Align to common trading days
    common_idx = nifty.index.intersection(midcap.index).intersection(vix.index).intersection(usdinr.index)
    common_idx = common_idx.sort_values()

    df_aligned = pd.DataFrame(
        {
            "NIFTY_50": nifty.loc[common_idx],
            "NIFTY_MIDCAP_50": midcap.loc[common_idx],
            "NIFTY_500": nifty500.loc[common_idx],
            "INDIA_VIX": vix.loc[common_idx],
            "USD_INR": usdinr.loc[common_idx],
        },
        index=common_idx,
    )

    # 1. Compute Return Statistics
    log_returns = np.log(df_aligned / df_aligned.shift(1)).dropna()

    stats_records = []
    for col in log_returns.columns:
        s = log_returns[col]
        mean_daily = float(s.mean())
        std_daily = float(s.std())
        annualized_return = mean_daily * 252.0
        annualized_vol = std_daily * np.sqrt(252.0)
        skew = float(stats.skew(s))
        kurt = float(stats.kurtosis(s))  # Excess kurtosis (Fisher: 0 for normal)
        jarque_bera_stat, jb_pvalue = stats.jarque_bera(s)

        stats_records.append(
            {
                "series": col,
                "observations": len(s),
                "mean_daily_return": round(mean_daily, 6),
                "std_daily_return": round(std_daily, 6),
                "annualized_return": round(annualized_return, 4),
                "annualized_volatility": round(annualized_vol, 4),
                "skewness": round(skew, 4),
                "excess_kurtosis": round(kurt, 4),
                "jarque_bera_stat": round(float(jarque_bera_stat), 2),
                "is_gaussian": bool(jb_pvalue > 0.05),
            }
        )

    stats_df = pd.DataFrame(stats_records)
    stats_csv_path = tables_dir / "eda_return_statistics.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"[+] Summary Statistics Table: {stats_csv_path.relative_to(root)}")
    print(stats_df[["series", "annualized_return", "annualized_volatility", "skewness", "excess_kurtosis"]])

    # 2. Plot Price Paths (Normalized to 100 at 2009)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 6))
    (df_aligned["NIFTY_50"] / df_aligned["NIFTY_50"].iloc[0] * 100).plot(ax=ax, label="Nifty 50", color="#1f77b4", lw=1.8)
    (df_aligned["NIFTY_MIDCAP_50"] / df_aligned["NIFTY_MIDCAP_50"].iloc[0] * 100).plot(ax=ax, label="Nifty Midcap 50", color="#ff7f0e", lw=1.5, alpha=0.9)
    (df_aligned["NIFTY_500"] / df_aligned["NIFTY_500"].iloc[0] * 100).plot(ax=ax, label="Nifty 500 Broad", color="#2ca02c", lw=1.5, alpha=0.8)
    (df_aligned["USD_INR"] / df_aligned["USD_INR"].iloc[0] * 100).plot(ax=ax, label="USD/INR", color="#d62728", lw=1.2, ls="--")
    ax.set_title("Normalized Price Paths (2009 - 2024, Base = 100)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Index Level (Rebased to 100)")
    ax.legend(frameon=True)
    plt.tight_layout()
    p1 = figures_dir / "01_price_paths.png"
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"[+] Figure Generated: {p1.relative_to(root)}")

    # 3. Plot Return Distributions & Fat Tails (KDE vs Gaussian)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, series_col, title in zip(
        axes,
        ["NIFTY_50", "NIFTY_MIDCAP_50", "USD_INR"],
        ["Nifty 50 Daily Returns", "Nifty Midcap 50 Daily Returns", "USD/INR Daily Returns"]
    ):
        rets = log_returns[series_col].values
        ax.hist(rets, bins=70, density=True, alpha=0.6, color="#1f77b4", label="Empirical Density")
        # Overlay Gaussian
        x = np.linspace(rets.min(), rets.max(), 200)
        ax.plot(x, stats.norm.pdf(x, rets.mean(), rets.std()), "r--", lw=2, label="Gaussian Fit")
        kurt_val = stats_df.loc[stats_df["series"] == series_col, "excess_kurtosis"].values[0]
        skew_val = stats_df.loc[stats_df["series"] == series_col, "skewness"].values[0]
        ax.set_title(f"{title}\n(Skew: {skew_val:.2f}, Excess Kurt: {kurt_val:.2f})", fontsize=11)
        ax.set_xlabel("Daily Log Return")
        ax.legend()
    plt.tight_layout()
    p2 = figures_dir / "02_return_distributions_fat_tails.png"
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"[+] Figure Generated: {p2.relative_to(root)}")

    # 4. India VIX Dynamics & Volatility Clusters
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    df_aligned["NIFTY_50"].plot(ax=ax1, color="#1f77b4", lw=1.5)
    ax1.set_title("Nifty 50 Index vs India VIX Regime Shifts (2009 - 2024)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Nifty 50 Index")
    ax1.grid(True, alpha=0.3)

    df_aligned["INDIA_VIX"].plot(ax=ax2, color="#d62728", lw=1.2)
    ax2.axhline(15.0, color="green", ls="--", label="Low Vol Threshold (VIX = 15)")
    ax2.axhline(22.0, color="orange", ls="--", label="Elevated Vol Threshold (VIX = 22)")
    ax2.axhline(35.0, color="red", ls=":", label="Crisis Spike Threshold (VIX = 35)")
    ax2.set_ylabel("India VIX Level")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = figures_dir / "03_india_vix_dynamics.png"
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"[+] Figure Generated: {p3.relative_to(root)}")

    # 5. Large-Cap vs Mid-Cap Relative Breadth Spread
    fig, ax = plt.subplots(figsize=(12, 5))
    relative_perf = (df_aligned["NIFTY_MIDCAP_50"] / df_aligned["NIFTY_MIDCAP_50"].iloc[0]) / (df_aligned["NIFTY_50"] / df_aligned["NIFTY_50"].iloc[0])
    relative_perf.plot(ax=ax, color="#9467bd", lw=1.8)
    ax.axhline(1.0, color="black", ls="--", alpha=0.7)
    ax.set_title("Midcap-to-Largecap Relative Performance Ratio (Breadth Leadership)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Nifty Midcap 50 / Nifty 50 (Normalized)")
    plt.tight_layout()
    p4 = figures_dir / "04_midcap_largecap_breadth_spread.png"
    plt.savefig(p4, dpi=200)
    plt.close()
    print(f"[+] Figure Generated: {p4.relative_to(root)}")

    # 6. Rolling 63-Day Correlation: Nifty 50 vs India VIX and USD/INR
    fig, ax = plt.subplots(figsize=(12, 5))
    roll_corr_vix = log_returns["NIFTY_50"].rolling(63).corr(log_returns["INDIA_VIX"])
    roll_corr_fx = log_returns["NIFTY_50"].rolling(63).corr(log_returns["USD_INR"])
    roll_corr_vix.plot(ax=ax, label="Rolling 63d Corr(Nifty, India VIX)", color="#d62728", lw=1.5)
    roll_corr_fx.plot(ax=ax, label="Rolling 63d Corr(Nifty, USD/INR)", color="#ff7f0e", lw=1.5)
    ax.axhline(0.0, color="black", ls="-", lw=0.8)
    ax.set_title("Rolling 63-Day Cross-Asset Correlations (Regime Transmission)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Pearson Correlation")
    ax.legend(frameon=True)
    plt.tight_layout()
    p5 = figures_dir / "05_rolling_cross_asset_correlations.png"
    plt.savefig(p5, dpi=200)
    plt.close()
    print(f"[+] Figure Generated: {p5.relative_to(root)}")

    print("\n[SUCCESS] All exploratory figures and statistical tables generated.")


if __name__ == "__main__":
    main()
