# Data Contract & Lineage Specification

> **CONFIDENTIAL** — Zetheta Algorithms Private Limited  
> **Project:** Bayesian Regime Detection Engine for Equity Direction Forecasting

---

## 1. Overview & Data Integrity Axioms

Financial time series research is vulnerable to subtle forms of data contamination that produce illusory strategy performance. This project enforces **zero-tolerance data integrity rules**:

1. **Never fabricate financial data**: Missing values must never be filled with synthetic random walks or ungrounded heuristics.
2. **Never silently fill missing observations**: Every missing value handling step must be explicitly logged, flagged, and auditable.
3. **Never leak future information**: Feature transformations (such as moving averages, z-scores, rolling quantiles, or PCA) must use strictly backward-looking windows.
4. **Never randomly shuffle financial time series**: All train, calibration, and test splits must maintain temporal chronology.
5. **Mandatory Snapshot Lineage**: Every dataset utilized by the engine is assigned an immutable SHA-256 checksum and cryptographic snapshot identifier.

---

## 2. Indian Market Data Universe Specification

The historical data universe spans approximately **15 years (2009-01-01 to 2024-12-31)**, encompassing equities, volatility, currencies, sovereign debt, credit spreads, institutional flows, and domestic retail flows:

| Variable Identifier | Asset / Series Name | Source | Frequency | Native Units | Transformation / Scale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `NIFTY_50` | Nifty 50 Index (OHLCV) | NSE / Yahoo | Daily | INR Index Points | Log returns, EWMA vol, 200 DMA ratio |
| `NIFTY_MIDCAP_100` | Nifty Midcap 100 Index | NSE / Yahoo | Daily | INR Index Points | Log returns, relative strength vs Nifty 50 |
| `NIFTY_SMALLCAP_100`| Nifty Smallcap 100 Index | NSE / Yahoo | Daily | INR Index Points | Log returns, small/large breadth spread |
| `INDIA_VIX` | India Volatility Index | NSE / Yahoo | Daily | Volatility % (Annualized) | Level, 1st difference, rolling quantile |
| `USD_INR` | US Dollar to Indian Rupee | RBI / Yahoo | Daily | INR per 1 USD | Log returns, 20-day realized volatility |
| `GILT_10Y` | India 10-Year Benchmark Gilt Yield | RBI / CCIL | Daily | Annualized Yield (%) | Basis point changes, yield curve slope |
| `AAA_GILT_SPREAD` | AAA Corporate Bond - 10Y Gilt Spread | CCIL / CRISIL | Daily | Basis Points (bps) | Spread level, 30-day z-score |
| `FII_EQUITY_FLOW` | Foreign Institutional Net Equity Flow | SEBI / NSDL | Daily | INR Crores | Cumulative 5d/20d flow, z-score |
| `DII_EQUITY_FLOW` | Domestic Institutional Net Equity Flow | SEBI / NSE | Daily | INR Crores | Cumulative 5d/20d flow, z-score |
| `SIP_MONTHLY_TOTAL` | Monthly Mutual Fund SIP Inflows | AMFI | Monthly | INR Crores | Month-over-month growth, trend deviation |

---

## 3. Schema Definitions

### 3.1 Raw Ingestion Schema (`MarketObservation`)
```json
{
  "timestamp": "2024-03-15T15:30:00+05:30",
  "symbol": "NIFTY_50",
  "open": 22020.30,
  "high": 22150.75,
  "low": 21980.10,
  "close": 22120.50,
  "volume": 345020100,
  "source": "NSE",
  "ingestion_timestamp": "2024-03-15T18:02:11Z",
  "quality_flags": 0
}
```

### 3.2 Engineered Feature Matrix Schema (`FeatureSnapshot`)
Every feature matrix fed to models must adhere to strict typing and index requirements:
- **Index**: `pd.DatetimeIndex` monotonically increasing, normalized to Indian Standard Time (`Asia/Kolkata`, UTC+05:30), business days only.
- **Columns**: Strictly named feature tags (e.g., `ret_nifty_1d`, `vol_nifty_20d`, `vix_level`, `flow_fii_norm_20d`).
- **Metadata**:
  - `feature_snapshot_id`: `feat_v1.0_<sha256[:12]>`
  - `data_snapshot_id`: `data_v1.0_<sha256[:12]>`
  - `start_date`: `YYYY-MM-DD`
  - `end_date`: `YYYY-MM-DD`
  - `lookback_buffer_days`: `252`

---

## 4. Central Output Contract: `RegimePrediction`

Every model, ensemble, and online filtering module must emit predictions convertible into this central, typed dataclass / Pydantic contract:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

@dataclass(frozen=True)
class RegimeProbabilities:
    risk_on: float
    late_cycle: float
    transitional: float
    post_shock: float
    risk_off: float

    def to_array(self) -> List[float]:
        return [self.risk_on, self.late_cycle, self.transitional, self.post_shock, self.risk_off]

    def validate(self, tol: float = 1e-5) -> bool:
        s = sum(self.to_array())
        return abs(s - 1.0) <= tol and all(p >= -tol for p in self.to_array())

@dataclass(frozen=True)
class UncertaintyMetrics:
    predictive_uncertainty: float    # Total Shannon entropy H(p)
    epistemic_uncertainty: float     # Parameter lack of knowledge (mutual info / ensemble variance)
    aleatoric_uncertainty: float     # Irreducible stochasticity (expected data entropy)

@dataclass(frozen=True)
class ModelLineage:
    model_name: str
    model_version: str
    feature_snapshot_id: str
    data_snapshot_id: str
    training_period: str
    random_seed: int

@dataclass(frozen=True)
class RegimePrediction:
    timestamp: datetime
    regime_probabilities: RegimeProbabilities
    predicted_regime: str            # "Risk-On" | "Risk-Off" | "Transitional" | "Late-Cycle" | "Post-Shock"
    uncertainty: UncertaintyMetrics
    conformal_prediction_set: List[str]  # e.g. ["Risk-On", "Transitional"]
    changepoint_probability: float  # [0.0, 1.0] from BOCPD
    lineage: ModelLineage
    explanation_metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## 5. Missing Data Treatment & Sanity Checks

1. **Trading Holiday Alignment**: Indian market holidays (NSE calendar) must be uniformly aligned. If asset prices are absent due to an official market holiday, the date is excluded across all asset equations rather than zero-filled.
2. **Macro Variable Forward-Filling**: Monthly SIP and macro series are released with reporting lags. They may only be forward-filled from their **official publication date**, not their observation reference date, to eliminate look-ahead leakage.
3. **Outlier and Anomaly Filters**:
   - Price jumps $> 20\%$ within single daily sessions trigger anomaly investigation.
   - VIX quotes $< 5.0$ or $> 90.0$ require raw source audit.
   - Flow data sign reversals exceeding 5 standard deviations trigger source re-query.
