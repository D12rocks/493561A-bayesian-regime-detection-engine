# ==============================================================================
# Cross-Language Numerical Reconciliation Script
# Zetheta Algorithms - Bayesian Regime Detection Engine
# Compares Python vs R HMM outputs on benchmark vector
# ==============================================================================

library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
py_output_file <- ifelse(length(args) >= 1, args[1], "artifacts/reconciliation/python_hmm_output.json")
r_output_file <- ifelse(length(args) >= 2, args[2], "artifacts/reconciliation/r_hmm_output.json")
report_file <- ifelse(length(args) >= 3, args[3], "artifacts/reconciliation/reconciliation_report.json")

message("Starting Python vs R Numerical Reconciliation Audit...")

if (!file.exists(py_output_file) || !file.exists(r_output_file)) {
  message("Benchmark run outputs not yet present. Skipping live comparison.")
  quit(status = 0)
}

py_data <- fromJSON(py_output_file)
r_data <- fromJSON(r_output_file)

# 1. Compare Transition Matrices
A_py <- as.matrix(py_data$transition_matrix)
A_r <- as.matrix(r_data$transition_matrix)
frobenius_diff <- sqrt(sum((A_py - A_r)^2))

# 2. Compare State Posterior Probabilities
P_py <- as.matrix(py_data$regime_probabilities)
P_r <- as.matrix(r_data$regime_probabilities)
mean_abs_prob_diff <- mean(abs(P_py - P_r))

# 3. Determine Parity Pass
pass_parity <- (frobenius_diff < 0.05) && (mean_abs_prob_diff < 0.02)

report <- list(
  timestamp = Sys.time(),
  frobenius_diff = frobenius_diff,
  mean_abs_prob_diff = mean_abs_prob_diff,
  pass_parity = pass_parity,
  max_allowed_frobenius = 0.05,
  max_allowed_prob_diff = 0.02
)

write_json(report, report_file, auto_unbox = TRUE, pretty = TRUE)
message(paste("Reconciliation complete. Parity status:", pass_parity))
