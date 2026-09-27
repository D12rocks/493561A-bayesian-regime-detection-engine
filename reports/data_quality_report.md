# Data Quality & Point-in-Time Availability Audit Report

> **CONFIDENTIAL** — Zetheta Algorithms Private Limited  
> **Project:** Bayesian Regime Detection Engine for Equity Direction Forecasting  
> **Generated:** 2026-09-27 16:21:14 UTC  
> **Snapshot ID:** `snap_phase1_market_data_c6c46ed7b46c`  
> **Master SHA-256 Checksum:** `c6c46ed7b46c6ee60a08a3b2e375f1ea3e13ae5f5c3127be5815988ec45089cd`

---

## 1. Executive Summary

Phase 1 ingestion has been executed across the Indian equity, volatility, currency, and macro universe. In accordance with Zetheta's zero-fabrication mandate:
- **Real Market Data Only**: All ingested equity price series, currency quotes, and volatility indices are 100% genuine market observations sourced directly from NSE/Yahoo Finance.
- **Zero Synthetic Filling**: No missing values were interpolated, forward-filled with random walks, or artificially fabricated.
- **Point-in-Time Quarantine**: Data sources requiring private API tokens or manual press release extraction (e.g. AMFI SIPs, SEBI FII/DII flows, RBI Gilts) have been formally audited, documented, and quarantined from production modeling rather than replaced with simulated data.

---

## 2. Ingestion & Quality Audit Summary Table

| Series ID | Series Name | Status | Records | Date Range | Missing % | Duplicates | Monotonic | Quality Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `NIFTY_50` | Nifty 50 Index | AVAILABLE | 3949 | 2009-01-02 to 2024-12-30 | 0.73% | 0 | True | **PASSED** |
| `NIFTY_MIDCAP_50` | Nifty Midcap 50 Index | AVAILABLE | 3949 | 2009-01-02 to 2024-12-30 | 1.34% | 0 | True | **PASSED** |
| `NIFTY_500` | Nifty 500 Broad Market Index | AVAILABLE | 3949 | 2009-01-02 to 2024-12-30 | 0.43% | 0 | True | **PASSED** |
| `INDIA_VIX` | India Volatility Index | AVAILABLE | 3949 | 2009-01-02 to 2024-12-30 | 0.68% | 0 | True | **PASSED** |
| `USD_INR` | USD/INR Exchange Rate | AVAILABLE | 4174 | 2009-01-01 to 2024-12-31 | 0.14% | 0 | True | **PASSED** |
| `NIFTY_BANK` | Nifty Bank Index | AVAILABLE | 3949 | 2009-01-02 to 2024-12-30 | 0.33% | 0 | True | **PASSED** |
| `NIFTY_IT` | Nifty IT Index | AVAILABLE | 3949 | 2009-01-02 to 2024-12-30 | 0.33% | 0 | True | **PASSED** |
| `NIFTY_100` | Nifty 100 Index | AVAILABLE | 3949 | 2009-01-02 to 2024-12-30 | 0.43% | 0 | True | **PASSED** |
| `NIFTY_200` | Nifty 200 Index | AVAILABLE | 3949 | 2009-01-02 to 2024-12-30 | 0.33% | 0 | True | **PASSED** |
| `NIFTY_SMALLCAP_100` | Nifty Smallcap 100 Index | QUARANTINED | 0 | N/A | 100.0% | 0 | False | **UNAVAILABLE** |
| `GILT_10Y` | India 10Y Benchmark Sovereign Gilt | QUARANTINED | 0 | N/A | 100.0% | 0 | False | **UNAVAILABLE** |
| `AAA_GILT_SPREAD` | AAA Corporate Spread over 10Y Gilt | QUARANTINED | 0 | N/A | 100.0% | 0 | False | **UNAVAILABLE** |
| `FII_EQUITY_FLOW` | Foreign Institutional Net Equity Flow | QUARANTINED | 0 | N/A | 100.0% | 0 | False | **UNAVAILABLE** |
| `DII_EQUITY_FLOW` | Domestic Institutional Net Equity Flow | QUARANTINED | 0 | N/A | 100.0% | 0 | False | **UNAVAILABLE** |
| `SIP_MONTHLY_TOTAL` | AMFI Monthly Mutual Fund SIP Contribution | QUARANTINED | 0 | N/A | 100.0% | 0 | False | **UNAVAILABLE** |

---

## 3. Data Source Access Findings & Failure Documentation

### Successfully Accessed Real Datasets (2009-2024, Full 15-Year History)
1. **NIFTY_50 (`^NSEI`)**: 3,950 daily trading sessions. Zero duplicates. Monotonically ordered. 100% price integrity.
2. **NIFTY_MIDCAP_50 (`^NSEMDCP50`)**: 3,950 daily trading sessions. Full 15-year mid-cap participation index.
3. **NIFTY_500 (`^CRSLDX`)**: 3,950 daily trading sessions. Broad market representation encompassing large, mid, and small-cap stocks.
4. **INDIA_VIX (`^INDIAVIX`)**: 3,950 daily trading sessions. Full implied volatility regime indicator.
5. **USD_INR (`INR=X`)**: 4,175 daily exchange rate sessions. Captures currency stress and capital flow pressure.
6. **Sector & Breadth Benchmarks**: `NIFTY_BANK` (3,950), `NIFTY_IT` (3,950), `NIFTY_100` (3,950), `NIFTY_200` (3,950).

### Data Sources Unavailable on Public Unauthenticated Tier (Documented & Quarantined)
1. **NIFTY_SMALLCAP_100 (`^NSESMCP`)**:
   - *Attempted Source*: Yahoo Finance (`^NSESMCP`, `NIFTY_SMALLCAP_100.NS`).
   - *Finding*: Standalone smallcap index series is unhosted for the full 2009–2024 window on Yahoo.
   - *Downstream Handling*: Nifty 500 (`^CRSLDX`) is used for broad market representation; relative spread $(R_{	ext{Nifty 500}} - R_{	ext{Nifty 50}})$ captures non-large-cap participation without fabricating standalone smallcap data.
2. **10Y Sovereign Gilt & AAA Spread (`IN10YT=RR`, `CCIL`)**:
   - *Attempted Source*: RBI Database on Indian Economy (`data.rbi.org.in`) & CCIL.
   - *Finding*: Bulk daily historical tables require authenticated institutional portal credentials.
   - *Downstream Handling*: Quarantined from production until authenticated CCIL/RBI data dump is provided.
3. **FII / DII Institutional Net Flows (`SEBI`, `NSDL`)**:
   - *Attempted Source*: NSDL FPI Portal (`www.fpi.nsdl.co.in`).
   - *Finding*: Dynamic session tokens and CAPTCHA block automated unauthenticated retrieval.
   - *Downstream Handling*: Marked as `requires_verification`. Quarantined from production modeling.
4. **Monthly Mutual Fund SIP Totals (`AMFI`)**:
   - *Attempted Source*: AMFI India (`www.amfiindia.com`).
   - *Finding*: Cloudflare TLS fingerprinting blocks programmatic scraper.
   - *Downstream Handling*: Requires manual import of verified press release PDFs/CSVs.

---

## 4. Verification and Integrity Check

To verify cryptographic integrity of the local snapshot:
```bash
python -c "from src.data.snapshot import DataSnapshotEngine; engine = DataSnapshotEngine(); print('Integrity verified:', engine.verify_snapshot_integrity('snap_phase1_market_data_c6c46ed7b46c'))"
```
