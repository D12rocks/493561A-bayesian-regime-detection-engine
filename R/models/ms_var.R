# ==============================================================================
# MSBVAR / MSwM R Implementation for Markov-Switching VAR
# Zetheta Algorithms Private Limited
# ==============================================================================

suppressPackageStartupMessages({
  if (!require("MSwM", quietly = TRUE)) {
    install.packages("MSwM", repos = "https://cloud.r-project.org")
    library("MSwM")
  }
})

run_ms_var <- function(data_csv_path, output_json_path, n_regimes = 5) {
  df <- read.csv(data_csv_path)
  
  # Linear model baseline
  base_lm <- lm(nifty_ret_1d ~ nifty_vol_ewma_21d + vix_level, data = df)
  
  # Markov-Switching extension
  ms_mod <- msmFit(base_lm, k = n_regimes, sw = c(TRUE, TRUE, TRUE, TRUE))
  
  trans_mat <- ms_mod@transMat
  filtered_probs <- ms_mod@Fit@filtProb
  smoothed_probs <- ms_mod@Fit@smoProb
  
  res <- list(
    logLik = ms_mod@Fit@logLikel,
    AIC = AIC(ms_mod),
    transition_matrix = as.matrix(trans_mat),
    smoothed_probs = as.matrix(smoothed_probs)
  )
  
  if (!is.null(output_json_path)) {
    jsonlite::write_json(res, output_json_path, pretty = TRUE)
  }
  return(res)
}

if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 2) {
    run_ms_var(args[1], args[2])
  }
}
