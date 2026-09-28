# ==============================================================================
# Changepoint Detection in R (PELT Algorithm via 'changepoint' package)
# Zetheta Algorithms Private Limited
# Corporate Identity Number (CIN): U62012MH2023PTC410415
# ==============================================================================

suppressPackageStartupMessages({
  if (!require("changepoint", quietly = TRUE)) {
    install.packages("changepoint", repos = "https://cloud.r-project.org")
    library("changepoint")
  }
})

run_changepoint_detection <- function(data_csv_path, output_json_path, penalty = "MBIC") {
  df <- read.csv(data_csv_path)
  ret <- df$nifty_ret_1d
  
  # Detect changepoints in variance (volatility regimes)
  cpt_var <- cpt.var(ret, method = "PELT", penalty = penalty)
  cpts_variance <- cpts(cpt_var)
  
  # Detect changepoints in mean and variance
  cpt_meanvar <- cpt.meanvar(ret, method = "PELT", penalty = penalty)
  cpts_mv <- cpts(cpt_meanvar)
  
  res <- list(
    algorithm = "Pruned Exact Linear Time (PELT)",
    penalty = penalty,
    n_changepoints_var = length(cpts_variance),
    changepoints_var = cpts_variance,
    n_changepoints_meanvar = length(cpts_mv),
    changepoints_meanvar = cpts_mv
  )
  
  if (!is.null(output_json_path)) {
    jsonlite::write_json(res, output_json_path, pretty = TRUE)
  }
  return(res)
}

if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 2) {
    run_changepoint_detection(args[1], args[2])
  }
}
