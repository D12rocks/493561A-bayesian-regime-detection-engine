# R Model Implementations & Cross-Language Reconciliation

**Organization:** Zetheta Algorithms Private Limited  
**Corporate Identity Number (CIN):** U62012MH2023PTC410415  
**Document Status:** Production Reference Specifications  

---

## 1. Environment & Execution Status
- **Host Status:** `BLOCKED BY ENVIRONMENT` (Host system lacks native `R` and `Rscript` runtime).
- **Compliance Policy:** In accordance with the Zero-Fabrication Directive, cross-language benchmarks are NOT marked as complete on the host. Instead, fully compliant R reference implementations, test adapters, and reconciliation scripts are provided.
- **Docker / Production Reproduction:** All scripts can be executed seamlessly in any containerized environment providing R >= 4.2.0 (e.g. `rocker/r-ver:4.3.0`).

---

## 2. Directory Structure & Script Manifest
| Subsystem | Script Path | Description | Required R Packages |
|:---|:---|:---|:---|
| **depmixS4 HMM** | [`R/models/hmm_depmixs4.R`](file:///Users/dhruvarora/bayesian-regime-detection-engine/R/models/hmm_depmixs4.R) | 5-State Hidden Markov Model with Gaussian emissions fitted via EM. | `depmixS4`, `jsonlite` |
| **MSwM / MS-VAR** | [`R/models/ms_var.R`](file:///Users/dhruvarora/bayesian-regime-detection-engine/R/models/ms_var.R) | Markov-Switching Vector Autoregression with filtered and smoothed probabilities. | `MSwM`, `jsonlite` |
| **Bayesian HMM Spec** | [`R/models/bayesian_hmm.R`](file:///Users/dhruvarora/bayesian-regime-detection-engine/R/models/bayesian_hmm.R) | Bayesian regime specification via Gibbs sampling / MCMC. | `MCMCpack`, `jsonlite` |
| **Changepoint (PELT)**| [`R/models/changepoint.R`](file:///Users/dhruvarora/bayesian-regime-detection-engine/R/models/changepoint.R) | Pruned Exact Linear Time (PELT) variance and mean/variance regime shifts. | `changepoint`, `jsonlite` |
| **Conformal Prediction** | [`R/validation/conformal.R`](file:///Users/dhruvarora/bayesian-regime-detection-engine/R/validation/conformal.R) | Split Conformal Inference with finite-sample marginal coverage verification. | `jsonlite` |
| **Reconciliation Engine** | [`R/reconciliation/reconcile_python_r.R`](file:///Users/dhruvarora/bayesian-regime-detection-engine/R/reconciliation/reconcile_python_r.R) | Evaluates state sequence alignment between Python and R implementations. | `jsonlite` |

---

## 3. Step-by-Step Instructions for Running on an R-Enabled Environment

### Option A: Local R Runtime
```bash
# 1. Install required packages
Rscript -e "install.packages(c('depmixS4', 'MSwM', 'MCMCpack', 'changepoint', 'jsonlite'), repos='https://cloud.r-project.org')"

# 2. Run depmixS4 5-state HMM
Rscript R/models/hmm_depmixs4.R data/processed/clean_features.csv artifacts/depmix_results.json

# 3. Run Markov-Switching Model
Rscript R/models/ms_var.R data/processed/clean_features.csv artifacts/mswm_results.json

# 4. Run Bayesian HMM
Rscript R/models/bayesian_hmm.R data/processed/clean_features.csv artifacts/bayesian_r_results.json

# 5. Run Changepoint Analysis
Rscript R/models/changepoint.R data/processed/clean_features.csv artifacts/changepoint_results.json

# 6. Run Conformal Prediction Validation
Rscript R/validation/conformal.R artifacts/test_probs.csv artifacts/test_labels.csv artifacts/conformal_r_results.json

# 7. Execute Cross-Language State Reconciliation
Rscript R/reconciliation/reconcile_python_r.R artifacts/depmix_results.json artifacts/python_hmm_results.json
```

### Option B: Docker Container (One-Command Execution)
```bash
docker run --rm -v $(pwd):/workspace -w /workspace rocker/r-ver:4.3.0 bash -c "
  Rscript -e \"install.packages(c('depmixS4', 'MSwM', 'MCMCpack', 'changepoint', 'jsonlite'), repos='https://cloud.r-project.org')\" && \
  Rscript R/models/hmm_depmixs4.R data/processed/clean_features.csv artifacts/depmix_results.json && \
  Rscript R/reconciliation/reconcile_python_r.R artifacts/depmix_results.json artifacts/python_hmm_results.json
"
```
