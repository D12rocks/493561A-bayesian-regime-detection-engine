# CLAUDE.md - Assistant Guidelines & Repository Protocol

## Project Information
- **Project Title:** Bayesian Regime Detection Engine for Equity Direction Forecasting
- **Organization:** Zetheta Algorithms Private Limited
- **Repository:** `bayesian-regime-detection-engine`
- **Classification:** Strictly Private and Confidential

---

## 1. Absolute Confidentiality Rules
- **NEVER** publish, distribute, or publicly expose repository code, architecture, data, figures, or documentation.
- **NEVER** push code to public repositories or external portfolios.
- **NEVER** upload data or credentials to third-party endpoints.
- Do not create public GitHub issues, discussions, or wikis.
- Preserve all proprietary confidentiality notices in headers and files.

---

## 2. Core Philosophy & Mandates
1. **Direction over price**: Predict discrete regime probabilities rather than noisy continuous price forecasts.
2. **Probability over point forecast**: All models must output a full 5-state categorical distribution over `[Risk-On, Risk-Off, Transitional, Late-Cycle, Post-Shock]`.
3. **Calibrated uncertainty over false precision**: Distinguish epistemic uncertainty (model parameter distribution variance) from aleatoric uncertainty (entropy / irreducible market noise).
4. **Documented ensemble over a single black box**: Combine model outputs via Bayesian Model Averaging (BMA) or constrained stacking. No single unvalidated model should dictate output.
5. **No Data Fabrication**: Never synthesize fake financial returns or fake test metrics. If a model or data source is unavailable, document the exact limitation honestly.

---

## 3. Mandatory Development & Branching Rules
- **No Direct Commits to Main**: All development must occur on dedicated feature branches.
  - Branch naming convention:
    - `development` (integration branch)
    - `feature/data-pipeline`
    - `feature/hmm`
    - `feature/bayesian-hmm`
    - `feature/rsvar`
    - `feature/bayesian-dl`
    - `feature/foundation-models`
    - `feature/ensemble`
    - `feature/online-inference`
    - `feature/calibration`
    - `feature/backtest`
    - `feature/r-layer`
- **Git Safety**:
  - Always run `git status` and verify staged changes before committing.
  - Never stage `.parquet`, `.csv`, `.h5`, `.pt`, `.safetensors`, `.env`, or `.DS_Store`.
  - Use informative commit messages prefixed by module (e.g., `feat(contracts): add typed RegimePrediction schema`).

---

## 4. Environment & Execution Commands
- **Python Setup**: Python 3.11/3.12 within virtualenv or Conda (`environment.yml`).
- **Run Tests**:
  - `pytest tests/unit/ -v` (Fast unit tests)
  - `pytest tests/statistical/ -v` (Probability and stochastic invariants)
  - `pytest tests/leakage/ -v` (Temporal leakage verification)
  - `pytest tests/ -v` (Full suite)
- **Code Quality**:
  - `ruff check src/ tests/` (Fast linting)
  - `black --check src/ tests/` (Formatting)
  - `mypy src/` (Strict static type checking)
- **Environment Diagnostics**:
  - `python scripts/verify_environment.py`

---

## 5. Architectural Contracts
- Every model class must inherit from `src.models.base.BaseRegimeModel` and implement:
  - `fit(X, y=None)`
  - `predict_proba(X) -> np.ndarray` (shape: `(N, 5)`)
  - `diagnostics() -> Dict[str, Any]`
  - `get_lineage() -> ModelLineage`
- The central prediction output must strictly conform to `src.models.contracts.RegimePrediction`.
- Foundation models must implement the common adapter `src.models.foundation.base.FoundationModelAdapter`.

---

## 6. Validation & Anti-Leakage Protocol
- All financial splits must be strictly chronological: `Train` $\to$ `Calibration` $\to$ `Test`.
- No standard random K-Fold cross validation on time series.
- All rolling transformations (e.g., z-scores, EWMA volatility) must use historical expanding or backward rolling windows with no lookahead.
- Any model entering the production ensemble must have:
  1. Validated out-of-sample metrics.
  2. Brier score / Expected Calibration Error (ECE) audit.
  3. MCMC convergence verification ($\hat{R} < 1.05$, zero divergences) for Bayesian components.
