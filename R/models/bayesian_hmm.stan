// ==============================================================================
// 5-State Bayesian Hidden Markov Model in Stan
// Zetheta Algorithms Private Limited
// Corporate Identity Number (CIN): U62012MH2023PTC410415
// ==============================================================================

data {
  int<lower=1> T;                  // Number of time steps
  int<lower=1> K;                  // Number of regimes (5)
  int<lower=1> D;                  // Observation dimension (returns, vol)
  matrix[T, D] y;                  // Observed feature series
  vector<lower=0>[K] alpha_prior;  // Dirichlet prior hyperparameter on transitions
  real<lower=0> kappa;             // Sticky prior pseudo-count on self-transitions
}

parameters {
  simplex[K] pi_init;              // Initial state distribution
  simplex[K] A[K];                 // Transition matrix rows
  vector[D] mu[K];                 // State-dependent emission means
  cholesky_factor_corr[D] L_Omega[K]; // Emission correlation cholesky
  vector<lower=0>[D] sigma[K];     // Emission standard deviations
}

transformed parameters {
  matrix[D, D] L_Sigma[K];
  for (k in 1:K) {
    L_Sigma[k] = diag_pre_multiply(sigma[k], L_Omega[k]);
  }
}

model {
  // Priors
  pi_init ~ dirichlet(rep_vector(1.0, K));
  
  for (k in 1:K) {
    vector[K] dir_hyper = alpha_prior;
    dir_hyper[k] += kappa;         // Sticky prior: persistence on diagonal
    A[k] ~ dirichlet(dir_hyper);
    
    mu[k] ~ normal(0, 0.05);
    sigma[k] ~ normal(0, 0.05);
    L_Omega[k] ~ lkj_corr_cholesky(2.0);
  }

  // Forward Algorithm for Marginal Likelihood in Stan
  {
    matrix[T, K] forward_lp;
    vector[K] acc;
    
    // Initial step
    for (k in 1:K) {
      forward_lp[1, k] = log(pi_init[k]) + multi_normal_cholesky_lpdf(y[1] | mu[k], L_Sigma[k]);
    }
    
    // Dynamic programming recursion
    for (t in 2:T) {
      for (k in 1:K) {
        for (j in 1:K) {
          acc[j] = forward_lp[t - 1, j] + log(A[j, k]);
        }
        forward_lp[t, k] = log_sum_exp(acc) + multi_normal_cholesky_lpdf(y[t] | mu[k], L_Sigma[k]);
      }
    }
    
    // Accumulate total log-likelihood
    target += log_sum_exp(forward_lp[T]);
  }
}
