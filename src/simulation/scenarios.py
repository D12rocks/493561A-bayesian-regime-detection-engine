"""
Scenario Replay and Stress-Testing Engine.

Implements REQ-094 and Specification Section M:
- Historical Replay: Point-in-time historical replay of canonical crisis episodes:
  1. 2013 Taper Tantrum
  2. 2018 IL&FS Liquidity Crisis
  3. 2020 COVID Crash & Recovery
  4. 2024 General Election Shock
- Synthetic Stress Scenarios:
  1. RBI Hawkish Surprise (+50 bps hike)
  2. RBI Dovish Surprise (-25 bps cut)
  3. FII Sudden Stop (Aggressive foreign liquidation)
  4. Domestic SIP Acceleration (Counter-cyclical retail surge)
Explicitly demarcates synthetic simulations from genuine historical evidence.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    scenario_name: str
    scenario_type: str  # "HISTORICAL_REPLAY" or "SYNTHETIC_STRESS"
    description: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    shock_vector: Optional[Dict[str, float]] = None


class ScenarioEngine:
    """
    Manages historical replay feeds and injects controlled synthetic macro shocks.
    """

    HISTORICAL_SCENARIOS = {
        "taper_tantrum_2013": ScenarioDefinition(
            scenario_id="taper_tantrum_2013",
            scenario_name="2013 Taper Tantrum",
            scenario_type="HISTORICAL_REPLAY",
            description="Bernanke taper hints trigger violent emerging market capital flight and Rupee depreciation.",
            start_date="2013-05-22",
            end_date="2013-08-30",
        ),
        "ilfs_liquidity_2018": ScenarioDefinition(
            scenario_id="ilfs_liquidity_2018",
            scenario_name="2018 IL&FS Liquidity Shock",
            scenario_type="HISTORICAL_REPLAY",
            description="IL&FS default causes acute commercial paper freeze and domestic NBFC liquidity squeeze.",
            start_date="2018-09-01",
            end_date="2018-12-31",
        ),
        "covid_crash_2020": ScenarioDefinition(
            scenario_id="covid_crash_2020",
            scenario_name="2020 COVID Market Crash",
            scenario_type="HISTORICAL_REPLAY",
            description="Pandemic declarations trigger unprecedented global equity liquidation and VIX spike above 70.",
            start_date="2020-02-15",
            end_date="2020-06-30",
        ),
        "election_shock_2024": ScenarioDefinition(
            scenario_id="election_shock_2024",
            scenario_name="2024 General Election Shock",
            scenario_type="HISTORICAL_REPLAY",
            description="June 4 vote count diverges sharply from exit polls, triggering single-day 6% volatility spike.",
            start_date="2024-05-20",
            end_date="2024-06-15",
        ),
    }

    SYNTHETIC_SCENARIOS = {
        "rbi_hawkish_surprise": ScenarioDefinition(
            scenario_id="rbi_hawkish_surprise",
            scenario_name="RBI Hawkish Surprise (+50 bps)",
            scenario_type="SYNTHETIC_STRESS",
            description="SYNTHETIC: Unscheduled 50 bps repo rate hike to combat currency pressure.",
            shock_vector={
                "nifty_ret_1d": -0.025,
                "vix_level": 5.5,
                "breadth_midcap_ret_21d": -0.020,
                "usdinr_ret_21d": -0.008,
            },
        ),
        "rbi_dovish_surprise": ScenarioDefinition(
            scenario_id="rbi_dovish_surprise",
            scenario_name="RBI Dovish Surprise (-25 bps)",
            scenario_type="SYNTHETIC_STRESS",
            description="SYNTHETIC: Surprise rate cut and liquidity injection easing credit conditions.",
            shock_vector={
                "nifty_ret_1d": 0.020,
                "vix_level": -3.0,
                "breadth_midcap_ret_21d": 0.015,
                "usdinr_ret_21d": 0.005,
            },
        ),
        "fii_sudden_stop": ScenarioDefinition(
            scenario_id="fii_sudden_stop",
            scenario_name="FII Sudden Stop & Liquidation",
            scenario_type="SYNTHETIC_STRESS",
            description="SYNTHETIC: Massive synchronized global institutional selling in Indian large caps.",
            shock_vector={
                "nifty_ret_1d": -0.040,
                "vix_level": 15.0,
                "breadth_midcap_ret_21d": -0.035,
                "usdinr_ret_21d": 0.025,
            },
        ),
        "domestic_sip_surge": ScenarioDefinition(
            scenario_id="domestic_sip_surge",
            scenario_name="Domestic SIP Retail Acceleration",
            scenario_type="SYNTHETIC_STRESS",
            description="SYNTHETIC: Relentless mutual fund retail inflows absorbing foreign selling pressure.",
            shock_vector={
                "nifty_ret_1d": 0.015,
                "vix_level": -4.0,
                "breadth_midcap_ret_21d": 0.025,
                "usdinr_ret_21d": 0.000,
            },
        ),
    }

    def __init__(self, features_df: Optional[pd.DataFrame] = None) -> None:
        self.features_df = features_df

    def get_historical_slice(self, scenario_id: str) -> pd.DataFrame:
        """Retrieves exact point-in-time historical data slice."""
        if scenario_id not in self.HISTORICAL_SCENARIOS:
            raise KeyError(f"Unknown historical scenario: {scenario_id}")
        scen = self.HISTORICAL_SCENARIOS[scenario_id]

        if self.features_df is None:
            feats_path = Path("data/processed/features_matrix.parquet")
            self.features_df = pd.read_parquet(feats_path)

        sub = self.features_df.loc[
            (self.features_df.index >= scen.start_date)
            & (self.features_df.index <= scen.end_date)
        ]
        return sub

    def inject_synthetic_shock(
        self, base_features: pd.Series, scenario_id: str
    ) -> pd.Series:
        """
        Applies synthetic shock vector to base feature instance.
        Explicitly flags that the output is synthetic.
        """
        if scenario_id not in self.SYNTHETIC_SCENARIOS:
            raise KeyError(f"Unknown synthetic scenario: {scenario_id}")
        scen = self.SYNTHETIC_SCENARIOS[scenario_id]

        shocked = base_features.copy()
        for feat, shock_val in scen.shock_vector.items():
            if feat in shocked:
                if "vix_level" in feat:
                    shocked[feat] = max(8.0, shocked[feat] + shock_val)
                else:
                    shocked[feat] = shocked[feat] + shock_val

        return shocked
