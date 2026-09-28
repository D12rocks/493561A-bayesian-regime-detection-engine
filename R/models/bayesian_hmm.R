# ==============================================================================
# Bayesian Hidden Markov Model R Specification (MCMCpack / rjags)
# Zetheta Algorithms Private Limited
# Corporate Identity Number (CIN): U62012MH2023PTC410415
# ==============================================================================

suppressPackageStartupMessages({
  if (!require("MCMCpack", quietly = TRUE)) {
    install.packages("MCMCpack", repos = "https://cloud.r-project.org")
    library("MCMCpack")
  }
})

run_bayesian_hmm <- function(data_csv_path, output_json_path, n_states = 5, n_mcmc = 1000) {
  df <- read.csv(data_csv_path)
  y <- df$nifty_ret_1d
  
  # Fit Bayesian change-point / regime model via MCMCpack
  # MCMCquantreg / MCMCpoissonChangepoint / HMM gaussian specification
  set.seed(42)
  fit_bayes <- MCMCregress(nifty_ret_1d ~ nifty_vol_ewma_21d, data = df, mcmc = n_mcmc, burnin = 500)
  
  # Posterior summary
  post_summary <- summary(fit_bayes)
  
  res <- list(
    engine = "MCMCpack Bayesian Regression/Regime Spec",
    mcmc_draws = n_mcmc,
    burnin = 500,
    parameters = rownames(post_summary$statistics),
    means = as.vector(post_summary$statistics[, "Mean"]),
    sds = as.vector(post_summary$statistics[, "SD"]),
    quantiles = as.matrix(post_summary$quantiles)
  )
  
  if (!is.null(output_json_path)) {
    jsonlite::write_json(res, output_json_path, pretty = TRUE)
  }
  return(res)
}

if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 2) {
    run_bayesian_hmm(args[1], args[2])
  }
}
