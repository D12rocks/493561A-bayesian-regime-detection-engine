# ==============================================================================
# Dual-Language R Dependency Setup
# Zetheta Algorithms - Bayesian Regime Detection Engine
# ==============================================================================

required_packages <- c(
  "depmixS4",       # Hidden Markov Models with multi-modal emissions
  "MSwM",           # Markov-Switching Autoregressive Models
  "changepoint",    # Changepoint detection
  "tidyverse",      # Data wrangling (dplyr, readr, ggplot2)
  "jsonlite",       # JSON contract reading and writing
  "yaml"            # Configuration loading
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(paste("Installing R package:", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    message(paste("Package already installed:", pkg))
  }
}

for (pkg in required_packages) {
  install_if_missing(pkg)
}

message("R dependencies verified.")
