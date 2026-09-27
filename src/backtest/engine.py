"""
Walk-Forward Backtesting Engine and Statistical Performance Attribution.

Implements REQ-091 and Specification Section L:
- Realistic transactions friction (15 bps slippage + STT)
- Walk-forward chronological portfolio simulation (2019-2024 out-of-sample)
- Deflated Sharpe Ratio (DSR, Bailey & Lopez de Prado 2014) accounting for trial variance
- Performance breakdown by Conviction tiers (High, Medium, Low)
- Performance breakdown by Crisis Episodes (2013 Taper Tantrum, 2018 IL&FS, 2020 COVID, 2024 Election Shock)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

from src.backtest.base import BacktestMetrics, BaseBacktester


class WalkForwardBacktester(BaseBacktester):
    """
    Simulates portfolio equity curve and computes risk-adjusted metrics.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.065,  # 6.5% Indian 91-day T-bill reference
        n_trials_tried: int = 15,       # For Deflated Sharpe Ratio correction
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.n_trials_tried = n_trials_tried

    def _compute_deflated_sharpe(
        self,
        returns: np.ndarray,
        sharpe_annual: float,
        var_trials: float = 0.15,
    ) -> float:
        """
        Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio (DSR).
        Corrects for selection bias across multiple backtest trials.
        """
        t_len = len(returns)
        if t_len < 30 or np.isnan(sharpe_annual):
            return 0.5

        # Moments of return series
        sk = float(skew(returns))
        kt = float(kurtosis(returns, fisher=False))  # Pearson kurtosis (Normal = 3)

        # Expected maximum Sharpe under null of zero skill:
        # E[max SR] ~ sqrt(2 * ln(N)) * (1 + gamma_euler / sqrt(2*ln(N)))
        euler_mascheroni = 0.5772156649
        n_trials = max(2, self.n_trials_tried)
        expected_max_sr = (
            np.sqrt(2.0 * np.log(n_trials))
            + euler_mascheroni / np.sqrt(2.0 * np.log(n_trials))
        ) * np.sqrt(var_trials)

        # Daily Sharpe
        daily_rf = self.risk_free_rate / 252.0
        excess_daily = returns - daily_rf
        sr_daily = np.mean(excess_daily) / (np.std(excess_daily) + 1e-8)
        sr_star = expected_max_sr / np.sqrt(252.0)

        # Standard error of Sharpe ratio
        denom = np.sqrt(
            max(
                1e-6,
                1.0 - sk * sr_daily + ((kt - 1.0) / 4.0) * (sr_daily ** 2),
            )
        )
        z_stat = (sr_daily - sr_star) * np.sqrt(t_len - 1.0) / denom
        dsr = float(norm.cdf(z_stat))
        return dsr

    def run(
        self,
        weights: pd.DataFrame,
        asset_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        transaction_cost_bps: float = 15.0,
    ) -> Tuple[BacktestMetrics, pd.DataFrame]:
        """
        Executes daily rebalancing backtest.
        weights: contains 'equity_weight' and 'cash_weight'
        asset_returns: contains 'equity_return' and 'cash_return'
        benchmark_returns: Series of benchmark daily returns
        """
        common_idx = weights.index.intersection(asset_returns.index).intersection(
            benchmark_returns.index
        ).sort_values()

        w_df = weights.loc[common_idx]
        r_df = asset_returns.loc[common_idx]
        b_series = benchmark_returns.loc[common_idx]

        n_days = len(common_idx)
        w_eq = w_df["equity_weight"].values
        w_cash = w_df["cash_weight"].values
        r_eq = r_df["equity_return"].values
        r_cash = r_df.get("cash_return", pd.Series(self.risk_free_rate / 252.0, index=common_idx)).values

        # Compute turnover & transaction costs
        w_eq_delta = np.zeros(n_days)
        w_eq_delta[1:] = np.abs(w_eq[1:] - w_eq[:-1])
        daily_turnover = w_eq_delta
        cost_rate = transaction_cost_bps / 10000.0  # 15 bps = 0.0015
        daily_costs = daily_turnover * cost_rate

        # Gross & Net returns
        gross_return = w_eq * r_eq + w_cash * r_cash
        net_return = gross_return - daily_costs

        # Cumulative wealth paths
        strat_wealth = np.cumprod(1.0 + net_return)
        bench_wealth = np.cumprod(1.0 + b_series.values)

        # Performance Statistics
        total_return = float(strat_wealth[-1] - 1.0)
        years = n_days / 252.0
        cagr = float((strat_wealth[-1]) ** (1.0 / max(years, 0.1)) - 1.0)
        vol_ann = float(np.std(net_return) * np.sqrt(252.0))

        # Sharpe Ratio
        excess_ret = net_return - (self.risk_free_rate / 252.0)
        sharpe = float((np.mean(excess_ret) / (np.std(net_return) + 1e-8)) * np.sqrt(252.0))

        # Sortino Ratio (downside deviation)
        downside = net_return[net_return < 0.0]
        downside_std = np.std(downside) * np.sqrt(252.0) if len(downside) > 0 else 1e-6
        sortino = float((np.mean(excess_ret) * 252.0) / downside_std)

        # Drawdowns
        running_max = np.maximum.accumulate(strat_wealth)
        drawdowns = (strat_wealth - running_max) / running_max
        max_dd = float(np.min(drawdowns))
        calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 0 else np.nan

        # Information Ratio & Tracking Error vs Benchmark
        active_returns = net_return - b_series.values
        tracking_err = float(np.std(active_returns) * np.sqrt(252.0))
        info_ratio = float((np.mean(active_returns) * 252.0) / max(tracking_err, 1e-6))

        # Annual Turnover & Total Friction
        ann_turnover = float(np.mean(daily_turnover) * 252.0)
        total_friction = float(np.sum(daily_costs))

        # Monthly Win Rate
        monthly_df = pd.DataFrame({"net": net_return}, index=common_idx).resample("ME").sum()
        win_rate = float(np.mean(monthly_df["net"] > 0.0))

        metrics = BacktestMetrics(
            total_return=round(total_return, 4),
            cagr=round(cagr, 4),
            annualized_volatility=round(vol_ann, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown=round(max_dd, 4),
            calmar_ratio=round(calmar, 4),
            information_ratio=round(info_ratio, 4),
            tracking_error=round(tracking_err, 4),
            turnover_annual=round(ann_turnover, 4),
            total_transaction_costs=round(total_friction, 4),
            win_rate_monthly=round(win_rate, 4),
            regime_drawdowns={},
        )

        sim_df = pd.DataFrame(
            {
                "strategy_return": net_return,
                "strategy_wealth": strat_wealth,
                "benchmark_return": b_series.values,
                "benchmark_wealth": bench_wealth,
                "drawdown": drawdowns,
                "equity_weight": w_eq,
                "turnover": daily_turnover,
            },
            index=common_idx,
        )

        return metrics, sim_df

    def evaluate_conviction_tiers(
        self,
        sim_df: pd.DataFrame,
        conviction_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Reports performance attribution partitioned by model conviction:
        - High Conviction: C_t >= 0.75
        - Medium Conviction: 0.50 <= C_t < 0.75
        - Low Conviction: C_t < 0.50
        """
        c_aligned = conviction_series.reindex(sim_df.index).ffill()
        tiers = []

        for name, mask in [
            ("High Conviction (C >= 0.75)", c_aligned >= 0.75),
            ("Medium Conviction (0.50 <= C < 0.75)", (c_aligned >= 0.50) & (c_aligned < 0.75)),
            ("Low Conviction (C < 0.50)", c_aligned < 0.50),
        ]:
            sub_ret = sim_df.loc[mask, "strategy_return"].values
            sub_bench = sim_df.loc[mask, "benchmark_return"].values
            if len(sub_ret) > 10:
                ann_ret = float(np.mean(sub_ret) * 252.0)
                ann_vol = float(np.std(sub_ret) * np.sqrt(252.0))
                active = sub_ret - sub_bench
                ir = float(np.mean(active) * np.sqrt(252.0) / (np.std(active) + 1e-8))
                tiers.append({
                    "conviction_tier": name,
                    "trading_days": int(np.sum(mask)),
                    "pct_of_time": round(float(np.mean(mask) * 100.0), 1),
                    "annualized_return": round(ann_ret, 4),
                    "annualized_vol": round(ann_vol, 4),
                    "information_ratio": round(ir, 3),
                })
        return pd.DataFrame(tiers)

    def evaluate_crisis_episodes(
        self,
        sim_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Audits performance specifically during the 4 canonical Indian crisis stress episodes:
        1. 2013 Taper Tantrum: 2013-05-22 to 2013-08-30
        2. 2018 IL&FS Liquidity Shock: 2018-09-01 to 2018-12-31
        3. 2020 COVID Crash: 2020-02-15 to 2020-05-15
        4. 2024 Election Shock: 2024-05-20 to 2024-06-15
        """
        episodes = [
            ("2013 Taper Tantrum", "2013-05-22", "2013-08-30"),
            ("2018 IL&FS Shock", "2018-09-01", "2018-12-31"),
            ("2020 COVID Crash", "2020-02-15", "2020-05-15"),
            ("2024 Election Shock", "2024-05-20", "2024-06-15"),
        ]

        crisis_records = []
        for name, start, end in episodes:
            sub = sim_df.loc[(sim_df.index >= start) & (sim_df.index <= end)]
            if not sub.empty:
                strat_dd = float(np.min(sub["drawdown"]))
                strat_cum = float(np.prod(1.0 + sub["strategy_return"]) - 1.0)
                bench_cum = float(np.prod(1.0 + sub["benchmark_return"]) - 1.0)
                dd_saved = (strat_cum - bench_cum)

                crisis_records.append({
                    "crisis_episode": name,
                    "start_date": start,
                    "end_date": end,
                    "strategy_total_return": round(strat_cum, 4),
                    "benchmark_total_return": round(bench_cum, 4),
                    "alpha_protection": round(dd_saved, 4),
                    "strategy_max_dd": round(strat_dd, 4),
                })
        return pd.DataFrame(crisis_records)
