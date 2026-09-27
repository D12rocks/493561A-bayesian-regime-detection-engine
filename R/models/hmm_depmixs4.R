# ==============================================================================
# depmixS4 R Implementation for 5-State Hidden Markov Model
# Zetheta Algorithms Private Limited
# ==============================================================================

suppressPackageStartupMessages({
  if (!require("depmixS4", quietly = TRUE)) {
    install.packages("depmixS4", repos = "https://cloud.r-project.org")
    library("depmixS4")
  }
})

run_depmix_hmm <- function(data_csv_path, output_json_path, n_states = 5) {
  df <- read.csv(data_csv_path)
  
  # Model specification with Gaussian emissions for Nifty returns and Volatility
  mod <- depmix(
    response = list(nifty_ret_1d ~ 1, nifty_vol_ewma_21d ~ 1),
    data = df,
    nstates = n_states,
    family = list(gaussian(), gaussian())
  )
  
  # Fit via EM
  fit_mod <- fit(mod, verbose = FALSE)
  
  # Extract transition matrix
  trans_mat <- summary(fit_mod, which = "transition")
  
  # Extract state posteriors
  post_probs <- posterior(fit_mod)
  
  res <- list(
    logLik = logLik(fit_mod),
    AIC = AIC(fit_mod),
    BIC = BIC(fit_mod),
    n_states = n_states,
    transition_matrix = as.matrix(trans_mat),
    state_posteriors = as.matrix(post_probs[, 2:(n_states + 1)])
  )
  
  if (!is.null(output_json_path)) {
    jsonlite::write_json(res, output_json_path, pretty = TRUE)
  }
  return(res)
}

if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 2) {
    run_depmix_hmm(args[1], args[2])
  }
}
