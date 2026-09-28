# ==============================================================================
# Conformal Prediction in R (Split Conformal Classification Sets)
# Zetheta Algorithms Private Limited
# Corporate Identity Number (CIN): U62012MH2023PTC410415
# ==============================================================================

suppressPackageStartupMessages({
  if (!require("jsonlite", quietly = TRUE)) {
    install.packages("jsonlite", repos = "https://cloud.r-project.org")
    library("jsonlite")
  }
})

run_conformal_validation <- function(probs_csv_path, labels_csv_path, output_json_path, alpha = 0.10) {
  # probs_csv: N x K matrix of predicted probabilities
  # labels_csv: N x 1 true class labels (0 to K-1)
  probs <- as.matrix(read.csv(probs_csv_path))
  labels <- as.integer(read.csv(labels_csv_path)[, 1])
  
  n <- nrow(probs)
  n_cal <- floor(n * 0.5)
  cal_idx <- 1:n_cal
  test_idx <- (n_cal + 1):n
  
  # Non-conformity scores on calibration set: s_i = 1 - p(y_i)
  cal_scores <- numeric(n_cal)
  for (i in 1:n_cal) {
    y_i <- labels[cal_idx[i]] + 1 # 1-indexed for R
    cal_scores[i] <- 1 - probs[cal_idx[i], y_i]
  }
  
  # Conformal quantile: ceil((n_cal + 1) * (1 - alpha)) / n_cal
  p_level <- min(1.0, ceiling((n_cal + 1) * (1 - alpha)) / n_cal)
  q_hat <- quantile(cal_scores, probs = p_level, type = 1)
  
  # Construct prediction sets on test set
  n_test <- length(test_idx)
  covered <- 0
  set_sizes <- numeric(n_test)
  
  for (i in 1:n_test) {
    idx <- test_idx[i]
    p_vec <- probs[idx, ]
    # Prediction set contains classes where 1 - p_k <= q_hat (i.e. p_k >= 1 - q_hat)
    pred_set <- which((1 - p_vec) <= q_hat) - 1 # 0-indexed
    set_sizes[i] <- length(pred_set)
    if (labels[idx] %in% pred_set) {
      covered <- covered + 1
    }
  }
  
  empirical_coverage <- covered / n_test
  mean_set_size <- mean(set_sizes)
  
  res <- list(
    nominal_confidence = 1 - alpha,
    q_hat = as.numeric(q_hat),
    n_calibration = n_cal,
    n_test = n_test,
    empirical_coverage = empirical_coverage,
    marginal_validity = (empirical_coverage >= (1 - alpha)),
    mean_set_size = mean_set_size
  )
  
  if (!is.null(output_json_path)) {
    write_json(res, output_json_path, pretty = TRUE)
  }
  return(res)
}

if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) >= 3) {
    run_conformal_validation(args[1], args[2], args[3])
  }
}
