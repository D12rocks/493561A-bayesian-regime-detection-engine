# ==============================================================================
# Dual-Language Reconciliation Engine (Python vs R)
# Zetheta Algorithms Private Limited
# ==============================================================================

suppressPackageStartupMessages({
  if (!require("jsonlite", quietly = TRUE)) {
    install.packages("jsonlite", repos = "https://cloud.r-project.org")
    library("jsonlite")
  }
})

reconcile_models <- function(python_results_json, r_results_json) {
  py <- jsonlite::fromJSON(python_results_json)
  r_res <- jsonlite::fromJSON(r_results_json)
  
  # Frobenius distance between transition matrices
  py_trans <- as.matrix(py$transition_matrix)
  r_trans <- as.matrix(r_res$transition_matrix)
  
  frob_dist <- sqrt(sum((py_trans - r_trans)^2))
  
  # Log likelihood difference
  ll_diff <- abs(py$log_likelihood - r_res$logLik)
  
  is_reconciled <- frob_dist < 0.15 && ll_diff < 50.0
  
  audit <- list(
    frobenius_distance = frob_dist,
    log_likelihood_diff = ll_diff,
    is_reconciled = is_reconciled,
    tolerance = 0.15,
    timestamp = Sys.time()
  )
  
  print(audit)
  return(audit)
}
