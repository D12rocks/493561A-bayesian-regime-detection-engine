"""
Execute Walk-Forward Portfolio Backtest, Conviction Tiers & Monte Carlo VaR/CVaR.

Runs the formal out-of-sample (2019-2024) backtest:
1. Translates calibrated ensemble probabilities into conviction-scaled tactical asset allocations.
2. Applies 15 bps transaction costs, no-trade hysteresis bands, and scheme constraints.
3. Computes CAGR, Volatility, Sharpe, Sortino, Calmar, Information Ratio, and Deflated Sharpe Ratio (DSR).
4. Generates performance attribution by Conviction Tier and Crisis Episodes.
5. Runs 10,000-path Monte Carlo forward fan chart.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import ConvictionAwareAllocationOverlay, WalkForwardBacktester
from src.simulation import RegimeConditionedSimulator, ScenarioEngine
from src.models.contracts import RegimeProbabilities


def main() -> None:
    preds_path = Path("data/processed/calibrated_ensemble_predictions.parquet")
    feats_path = Path("data/processed/features_matrix.parquet")

    if not preds_path.exists() or not feats_path.exists():
        raise FileNotFoundError("Prerequisite matrices not found.")

    preds_df = pd.read_parquet(preds_path)
    feats_df = pd.read_parquet(feats_path)

    # Align dates
    common_idx = preds_df.index.intersection(feats_df.index).sort_values()
    preds_df = preds_df.loc[common_idx]
    feats_df = feats_df.loc[common_idx]

    # Target 2019-2024 out-of-sample backtest window
    oos_mask = common_idx >= "2019-01-01"
    oos_idx = common_idx[oos_mask]
    print(f"Executing 2019-2024 Walk-Forward Backtest across {len(oos_idx)} trading days...")

    # Extract probabilities (N, 5)
    regimes = ["risk_on", "late_cycle", "transitional", "post_shock", "risk_off"]
    probs = preds_df.loc[oos_idx, [f"ensemble_prob_{r}" for r in regimes]].values

    # Extract returns: Nifty 50 return
    nifty_daily_ret = np.exp(feats_df.loc[oos_idx, "nifty_ret_1d"].values) - 1.0
    rf_daily = 0.065 / 252.0
    asset_returns = pd.DataFrame(
        {
            "equity_return": nifty_daily_ret,
            "cash_return": np.full(len(oos_idx), rf_daily),
        },
        index=oos_idx,
    )
    benchmark_returns = pd.Series(nifty_daily_ret, index=oos_idx)

    # 1. Conviction-Aware Allocation Overlay
    print("Computing conviction-aware equity weights with hysteresis...")
    overlay = ConvictionAwareAllocationOverlay(
        base_equity_weight=0.65,
        no_trade_band=0.04,
        max_daily_turnover=0.10,
        min_equity=0.20,
        max_equity=1.00,
    )
    weights_df = overlay.compute_weights(probs, oos_idx)

    # 2. Walk-Forward Backtest Simulation
    print("Running portfolio backtest with 15 bps friction...")
    backtester = WalkForwardBacktester(risk_free_rate=0.065, n_trials_tried=15)
    metrics, sim_df = backtester.run(weights_df, asset_returns, benchmark_returns, transaction_cost_bps=15.0)

    # Deflated Sharpe Ratio
    dsr = backtester._compute_deflated_sharpe(sim_df["strategy_return"].values, metrics.sharpe_ratio)

    print("\n================ BACKTEST PERFORMANCE SUMMARY (2019-2024) ================")
    print(f"Total Return:          {metrics.total_return:.2%} (Benchmark: {sim_df['benchmark_wealth'].iloc[-1]-1.0:.2%})")
    print(f"CAGR:                  {metrics.cagr:.2%}")
    print(f"Annualized Volatility: {metrics.annualized_volatility:.2%}")
    print(f"Sharpe Ratio (Rf=6.5%):{metrics.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:         {metrics.sortino_ratio:.2f}")
    print(f"Max Drawdown:          {metrics.max_drawdown:.2%} (Benchmark DD: {np.min((sim_df['benchmark_wealth'] - np.maximum.accumulate(sim_df['benchmark_wealth']))/np.maximum.accumulate(sim_df['benchmark_wealth'])):.2%})")
    print(f"Calmar Ratio:          {metrics.calmar_ratio:.2f}")
    print(f"Information Ratio:     {metrics.information_ratio:.2f}")
    print(f"Tracking Error:        {metrics.tracking_error:.2%}")
    print(f"Annual Turnover:       {metrics.turnover_annual:.2%}")
    print(f"Total Friction Paid:   {metrics.total_transaction_costs:.2%}")
    print(f"Monthly Win Rate:      {metrics.win_rate_monthly:.1%}")
    print(f"Deflated Sharpe (DSR): {dsr:.3f} (Statistically Significant > 0.95: {dsr >= 0.95})")

    # 3. Conviction Tiers Breakdown
    print("\n--- Evaluating Performance by Conviction Tiers ---")
    conviction_df = backtester.evaluate_conviction_tiers(sim_df, weights_df["conviction"])
    print(conviction_df.to_string())

    # 4. Crisis Episodes Breakdown
    print("\n--- Evaluating Performance During Indian Market Crisis Episodes ---")
    # For historical crisis episodes, run over full sample simulation
    full_weights = overlay.compute_weights(preds_df[[f"ensemble_prob_{r}" for r in regimes]].values, common_idx)
    full_ret_df = pd.DataFrame({
        "equity_return": np.exp(feats_df["nifty_ret_1d"].values) - 1.0,
        "cash_return": np.full(len(common_idx), rf_daily),
    }, index=common_idx)
    _, full_sim_df = backtester.run(full_weights, full_ret_df, pd.Series(full_ret_df["equity_return"], index=common_idx))
    crisis_df = backtester.evaluate_crisis_episodes(full_sim_df)
    print(crisis_df.to_string())

    # 5. Save Summary Tables
    tables_dir = Path("reports/tables")
    figures_dir = Path("reports/figures")

    # Metric summary table
    bench_dd = float(np.min((sim_df['benchmark_wealth'] - np.maximum.accumulate(sim_df['benchmark_wealth']))/np.maximum.accumulate(sim_df['benchmark_wealth'])))
    summary_table = pd.DataFrame([
        {"Metric": "CAGR", "Strategy (Regime-Aware)": f"{metrics.cagr:.2%}", "NIFTY 50 Benchmark": f"{(sim_df['benchmark_wealth'].iloc[-1])**(1/6.0)-1.0:.2%}"},
        {"Metric": "Annualized Volatility", "Strategy (Regime-Aware)": f"{metrics.annualized_volatility:.2%}", "NIFTY 50 Benchmark": f"{np.std(benchmark_returns)*np.sqrt(252):.2%}"},
        {"Metric": "Sharpe Ratio (Rf=6.5%)", "Strategy (Regime-Aware)": f"{metrics.sharpe_ratio:.2f}", "NIFTY 50 Benchmark": f"{(np.mean(benchmark_returns-rf_daily)*252)/(np.std(benchmark_returns)*np.sqrt(252)):.2f}"},
        {"Metric": "Max Drawdown", "Strategy (Regime-Aware)": f"{metrics.max_drawdown:.2%}", "NIFTY 50 Benchmark": f"{bench_dd:.2%}"},
        {"Metric": "Calmar Ratio", "Strategy (Regime-Aware)": f"{metrics.calmar_ratio:.2f}", "NIFTY 50 Benchmark": f"{(sim_df['benchmark_wealth'].iloc[-1]**(1/6)-1)/abs(bench_dd):.2f}"},
        {"Metric": "Information Ratio", "Strategy (Regime-Aware)": f"{metrics.information_ratio:.2f}", "NIFTY 50 Benchmark": "0.00"},
        {"Metric": "Annual Turnover", "Strategy (Regime-Aware)": f"{metrics.turnover_annual:.2%}", "NIFTY 50 Benchmark": "0.00%"},
        {"Metric": "Deflated Sharpe Ratio (DSR)", "Strategy (Regime-Aware)": f"{dsr:.3f}", "NIFTY 50 Benchmark": "N/A"},
    ])
    summary_table.to_csv(tables_dir / "backtest_performance_summary.csv", index=False)
    conviction_df.to_csv(tables_dir / "backtest_conviction_tiers.csv", index=False)
    crisis_df.to_csv(tables_dir / "backtest_crisis_episodes.csv", index=False)

    # 6. Plot 10: Equity Curves & Drawdown
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(oos_idx, sim_df["strategy_wealth"], label="Regime-Aware Conviction Allocation (Net of 15 bps)", color="#16a085", lw=2.0)
    axes[0].plot(oos_idx, sim_df["benchmark_wealth"], label="NIFTY 50 Buy-and-Hold Benchmark", color="#7f8c8d", lw=1.5, linestyle="--")
    axes[0].set_title("Walk-Forward Equity Curve (2019–2024 Out-of-Sample)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Normalized Wealth (₹1.00 Base)")
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(oos_idx, sim_df["drawdown"], color="#c0392b", lw=1.2, label="Strategy Drawdown")
    axes[1].plot(oos_idx, (sim_df["benchmark_wealth"] - np.maximum.accumulate(sim_df["benchmark_wealth"])) / np.maximum.accumulate(sim_df["benchmark_wealth"]), color="#95a5a6", lw=1.0, linestyle=":", label="Benchmark Drawdown")
    axes[1].set_title("Drawdown Profile (Capital Preservation Comparison)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Drawdown")
    axes[1].legend(loc="lower left", frameon=True)
    axes[1].grid(True, alpha=0.3)

    axes[2].fill_between(oos_idx, 0, weights_df["equity_weight"], color="#2980b9", alpha=0.6, label="Equity Allocation Weight")
    axes[2].plot(oos_idx, weights_df["conviction"], color="#f39c12", lw=1.2, label="Conviction Score C_t")
    axes[2].set_title("Dynamic Tactical Asset Allocation & Model Conviction", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Allocation Weight / Conviction")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(loc="lower left", frameon=True)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig10_path = figures_dir / "10_walk_forward_equity_curve.png"
    plt.savefig(fig10_path, dpi=200)
    plt.close()
    print(f"Saved figure: {fig10_path}")

    # 7. Plot 11: Monte Carlo Tail Risk Fan Chart
    print("\n--- Running 10,000-Path Monte Carlo Simulation ---")
    sim = RegimeConditionedSimulator()
    latest_probs = RegimeProbabilities.from_array(probs[-1])
    mc_paths = sim.simulate_paths(latest_probs, horizon_days=21, n_paths=10000, random_seed=42)
    mc_wealth = np.cumprod(1.0 + mc_paths, axis=1)
    tail_metrics = sim.compute_tail_risk(mc_paths, horizon_days=21)

    fig, ax = plt.subplots(figsize=(10, 6))
    days = np.arange(1, 22)
    # Percentile cones
    p5 = np.percentile(mc_wealth, 5, axis=0)
    p25 = np.percentile(mc_wealth, 25, axis=0)
    p50 = np.percentile(mc_wealth, 50, axis=0)
    p75 = np.percentile(mc_wealth, 75, axis=0)
    p95 = np.percentile(mc_wealth, 95, axis=0)
    p1 = np.percentile(mc_wealth, 1, axis=0)

    ax.fill_between(days, p1, p95, color="#3498db", alpha=0.15, label="1st - 95th Percentile")
    ax.fill_between(days, p5, p95, color="#3498db", alpha=0.25, label="5th - 95th Percentile")
    ax.fill_between(days, p25, p75, color="#2980b9", alpha=0.40, label="25th - 75th Percentile (IQR)")
    ax.plot(days, p50, color="#1a5276", lw=2.0, label="Median Path (50th)")
    ax.axhline(1.0, color="gray", linestyle="--", lw=1.0)

    ax.set_title(f"Regime-Conditioned Monte Carlo Forward Simulation (10,000 Paths, 21 Days)\n21-Day 99% VaR = {tail_metrics.var_99:.1%}, 99% CVaR = {tail_metrics.cvar_99:.1%}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Horizon (Trading Days Ahead)")
    ax.set_ylabel("Simulated Wealth Multiple")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig11_path = figures_dir / "11_monte_carlo_var_cvar_fan_chart.png"
    plt.savefig(fig11_path, dpi=200)
    plt.close()
    print(f"Saved figure: {fig11_path}")


if __name__ == "__main__":
    main()
