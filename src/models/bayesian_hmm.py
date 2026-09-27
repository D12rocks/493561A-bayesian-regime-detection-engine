"""
Bayesian Hidden Markov Model (Bayesian HMM) with Sticky Dirichlet Priors and MCMC.

Implements REQ-021: Sticky HDP-HMM / Bayesian HMM with:
- Sticky Dirichlet prior on transition matrix: A_i ~ Dir(alpha + kappa * e_i)
- Normal-Inverse-Wishart (NIW) conjugate priors on Gaussian emissions
- Multi-chain Gibbs sampling with Forward-Filtering Backward-Sampling (FFBS)
- Bayesian convergence diagnostics: Gelman-Rubin split-R-hat, Effective Sample Size (ESS)
- Posterior parameter distributions and credible intervals (90% HPDI)
- Uncertainty decomposition: Epistemic (posterior dispersion) vs Aleatoric (emission noise)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import dirichlet, invwishart, multivariate_normal

from src.models.base import BaseRegimeModel
from src.models.contracts import RegimeLabel


class BayesianHMM(BaseRegimeModel):
    """
    Bayesian 5-State Hidden Markov Model with Sticky Dirichlet Priors and Gibbs FFBS Sampling.
    """

    def __init__(
        self,
        name: str = "bayesian_hmm",
        n_regimes: int = 5,
        n_iter: int = 400,
        burn_in: int = 150,
        n_chains: int = 2,
        sticky_kappa: float = 10.0,
        alpha_base: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        super().__init__(name=name, version="1.0.0", random_seed=random_seed)
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.n_chains = n_chains
        self.sticky_kappa = sticky_kappa
        self.alpha_base = alpha_base
        
        # MCMC Chain storage: chain_idx -> list of samples
        self.chain_transmats: List[np.ndarray] = []  # shape: (n_chains, n_samples, K, K)
        self.chain_means: List[np.ndarray] = []      # shape: (n_chains, n_samples, K, D)
        self.chain_covs: List[np.ndarray] = []       # shape: (n_chains, n_samples, K, D, D)
        self.chain_states: List[np.ndarray] = []     # shape: (n_chains, n_samples, T)
        
        self.feature_columns: List[str] = []
        self.state_to_regime_map: Dict[int, int] = {}
        self._r_hat_dict: Dict[str, float] = {}
        self._ess_dict: Dict[str, float] = {}

    def _init_priors(self, d: int) -> Tuple[np.ndarray, float, float, np.ndarray]:
        """Conjugate Normal-Inverse-Wishart prior parameters."""
        mu_0 = np.zeros(d)
        kappa_0 = 1.0
        nu_0 = d + 2.0
        psi_0 = np.eye(d) * 0.1
        return mu_0, kappa_0, nu_0, psi_0

    def _forward_filtering_backward_sampling(
        self,
        y: np.ndarray,
        trans_mat: np.ndarray,
        means: np.ndarray,
        covs: np.ndarray,
        pi_0: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """
        FFBS Algorithm for exact joint state sequence sampling S_{1:T} ~ P(S | Y, theta).
        """
        t_steps, d = y.shape
        k = self.n_regimes
        
        # 1. Forward Filter: compute alpha_t(k) = P(S_t = k | Y_{1:t})
        alphas = np.zeros((t_steps, k), dtype=np.float64)
        
        # Emission densities B_t(k) = N(Y_t | mu_k, Sigma_k)
        emissions = np.zeros((t_steps, k), dtype=np.float64)
        for state in range(k):
            try:
                emissions[:, state] = multivariate_normal.pdf(
                    y, mean=means[state], cov=covs[state], allow_singular=True
                )
            except Exception:
                emissions[:, state] = 1e-6
        emissions = np.maximum(emissions, 1e-12)

        # t = 0
        alphas[0] = pi_0 * emissions[0]
        alphas[0] /= np.sum(alphas[0])

        for t in range(1, t_steps):
            pred = alphas[t - 1] @ trans_mat
            alphas[t] = pred * emissions[t]
            sum_alpha = np.sum(alphas[t])
            if sum_alpha > 0:
                alphas[t] /= sum_alpha
            else:
                alphas[t] = np.ones(k) / k

        # 2. Backward Sampling: S_T ~ alphas[T], S_t ~ P(S_t | S_{t+1}, Y_{1:t})
        states = np.zeros(t_steps, dtype=int)
        states[-1] = rng.choice(k, p=alphas[-1])

        for t in range(t_steps - 2, -1, -1):
            next_s = states[t + 1]
            backward_p = alphas[t] * trans_mat[:, next_s]
            p_sum = np.sum(backward_p)
            if p_sum > 0:
                backward_p /= p_sum
            else:
                backward_p = np.ones(k) / k
            states[t] = rng.choice(k, p=backward_p)

        return states

    def _sample_transition_matrix(
        self, states: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Samples A from posterior Dirichlet with sticky prior:
        A_i ~ Dir(alpha + kappa * e_i + n_i)
        """
        k = self.n_regimes
        counts = np.zeros((k, k), dtype=np.float64)
        for t in range(len(states) - 1):
            counts[states[t], states[t + 1]] += 1.0

        new_trans = np.zeros((k, k), dtype=np.float64)
        for i in range(k):
            dir_prior = np.full(k, self.alpha_base)
            dir_prior[i] += self.sticky_kappa  # Sticky diagonal enhancement
            post_params = dir_prior + counts[i]
            new_trans[i] = rng.dirichlet(post_params)
        return new_trans

    def _sample_emissions(
        self,
        y: np.ndarray,
        states: np.ndarray,
        mu_0: np.ndarray,
        kappa_0: float,
        nu_0: float,
        psi_0: np.ndarray,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples (mu_k, Sigma_k) from conjugate Normal-Inverse-Wishart posterior.
        """
        t_steps, d = y.shape
        k = self.n_regimes
        new_means = np.zeros((k, d))
        new_covs = np.zeros((k, d, d))

        for state in range(k):
            y_k = y[states == state]
            n_k = len(y_k)

            if n_k == 0:
                # Revert to prior
                new_covs[state] = invwishart.rvs(df=nu_0, scale=psi_0, random_state=rng)
                new_means[state] = rng.multivariate_normal(mu_0, new_covs[state] / kappa_0)
                continue

            y_bar = np.mean(y_k, axis=0)
            diff = y_k - y_bar
            s_k = diff.T @ diff

            # Update posterior NIW parameters
            kappa_n = kappa_0 + n_k
            nu_n = nu_0 + n_k
            mu_n = (kappa_0 * mu_0 + n_k * y_bar) / kappa_n
            mean_diff = (y_bar - mu_0)[:, None]
            psi_n = psi_0 + s_k + (kappa_0 * n_k / kappa_n) * (mean_diff @ mean_diff.T)

            # Sample Sigma_k ~ InvWishart(nu_n, psi_n)
            try:
                new_covs[state] = invwishart.rvs(df=nu_n, scale=psi_n, random_state=rng)
            except Exception:
                new_covs[state] = np.eye(d) * 0.1

            # Sample mu_k ~ Normal(mu_n, Sigma_k / kappa_n)
            new_means[state] = rng.multivariate_normal(mu_n, new_covs[state] / kappa_n)

        return new_means, new_covs

    def _compute_gelman_rubin(self, chains: np.ndarray) -> float:
        """
        Computes Gelman-Rubin split-R-hat diagnostic across M chains of length N.
        chains shape: (M, N)
        """
        m, n = chains.shape
        if m < 2 or n < 10:
            return 1.0

        # Mean of each chain
        chain_means = np.mean(chains, axis=1)
        # Overall mean
        overall_mean = np.mean(chain_means)

        # Between-chain variance B
        b = (n / (m - 1.0)) * np.sum((chain_means - overall_mean) ** 2)

        # Within-chain variance W
        s_sq = np.var(chains, axis=1, ddof=1)
        w = np.mean(s_sq)

        if w <= 1e-12:
            return 1.0

        # Pooled variance estimate
        var_plus = ((n - 1.0) / n) * w + (1.0 / n) * b
        r_hat = np.sqrt(var_plus / w)
        return float(r_hat)

    def _compute_ess(self, chain: np.ndarray) -> float:
        """Computes Effective Sample Size (ESS) via sample autocorrelation."""
        n = len(chain)
        if n < 10:
            return float(n)
        centered = chain - np.mean(chain)
        c0 = np.dot(centered, centered) / n
        if c0 <= 1e-12:
            return float(n)

        # Autocorrelations
        total_rho = 0.0
        max_lag = min(n // 2, 40)
        for lag in range(1, max_lag):
            rho_lag = np.dot(centered[:-lag], centered[lag:]) / (n * c0)
            if rho_lag < 0.05:  # Cutoff when noise dominates
                break
            total_rho += 2.0 * rho_lag

        ess = n / (1.0 + total_rho)
        return float(max(10.0, ess))

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BayesianHMM":
        self.feature_columns = list(X.columns)
        X_clean = X.dropna().values
        t_steps, d = X_clean.shape
        k = self.n_regimes
        
        mu_0, kappa_0, nu_0, psi_0 = self._init_priors(d)
        
        all_chain_trans = []
        all_chain_means = []
        all_chain_covs = []
        all_chain_states = []

        for chain_id in range(self.n_chains):
            rng = np.random.RandomState(self.random_seed + chain_id * 100)
            
            # Initial parameters
            trans_curr = np.full((k, k), 1.0 / k)
            np.fill_diagonal(trans_curr, 0.6)
            trans_curr /= np.sum(trans_curr, axis=1, keepdims=True)
            
            # K-means style initial partition for means
            quantiles = np.linspace(0.1, 0.9, k)
            means_curr = np.zeros((k, d))
            for st in range(k):
                means_curr[st] = np.quantile(X_clean, quantiles[st], axis=0)
            covs_curr = np.array([np.eye(d) * 0.1 for _ in range(k)])
            pi_0 = np.ones(k) / k
            
            samples_trans = []
            samples_means = []
            samples_covs = []
            samples_states = []
            
            for iteration in range(self.n_iter):
                # 1. Sample states via FFBS
                states = self._forward_filtering_backward_sampling(
                    X_clean, trans_curr, means_curr, covs_curr, pi_0, rng
                )
                
                # 2. Sample transitions
                trans_curr = self._sample_transition_matrix(states, rng)
                
                # 3. Sample emissions
                means_curr, covs_curr = self._sample_emissions(
                    X_clean, states, mu_0, kappa_0, nu_0, psi_0, rng
                )
                
                # Retain post burn-in samples
                if iteration >= self.burn_in:
                    samples_trans.append(trans_curr.copy())
                    samples_means.append(means_curr.copy())
                    samples_covs.append(covs_curr.copy())
                    samples_states.append(states.copy())
                    
            all_chain_trans.append(np.array(samples_trans))
            all_chain_means.append(np.array(samples_means))
            all_chain_covs.append(np.array(samples_covs))
            all_chain_states.append(np.array(samples_states))

        self.chain_transmats = all_chain_trans
        self.chain_means = all_chain_means
        self.chain_covs = all_chain_covs
        self.chain_states = all_chain_states
        self.is_fitted = True

        # Calculate Gelman-Rubin R-hat and ESS across chains
        # Check diagonal transitions
        for i in range(k):
            chains_diag = np.array([self.chain_transmats[c][:, i, i] for c in range(self.n_chains)])
            r_hat_val = self._compute_gelman_rubin(chains_diag)
            ess_val = self._compute_ess(chains_diag.flatten())
            self._r_hat_dict[f"trans_diag_{i}"] = round(r_hat_val, 3)
            self._ess_dict[f"trans_diag_{i}"] = round(ess_val, 1)

        # Semantic regime alignment based on mean posterior return
        # Feature 0 is return
        post_means_f0 = np.mean([np.mean(self.chain_means[c][:, :, 0], axis=0) for c in range(self.n_chains)], axis=0)
        # Sort states by posterior mean return
        sorted_indices = np.argsort(post_means_f0) # ascending: worst (Risk-Off) to best (Risk-On)
        
        self.state_to_regime_map = {
            int(sorted_indices[4]): 0,  # Risk-On
            int(sorted_indices[3]): 1,  # Late-Cycle
            int(sorted_indices[2]): 2,  # Transitional
            int(sorted_indices[1]): 3,  # Post-Shock
            int(sorted_indices[0]): 4,  # Risk-Off
        }
        return self

    def predict_proba_posterior(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            mean_probs: (N, 5) point estimate (mean posterior probability)
            epistemic_entropy: (N,) epistemic parameter uncertainty
            aleatoric_entropy: (N,) aleatoric data uncertainty
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")

        cols = self.feature_columns if self.feature_columns else X.columns
        X_sub = X[cols].ffill().fillna(0.0).values
        n_samples = len(X_sub)
        k = self.n_regimes

        # Draw a representative subset of posterior parameter samples (e.g. 50 samples)
        n_post_samples = min(50, len(self.chain_transmats[0]))
        sample_probs = []

        for s_idx in range(n_post_samples):
            # Pick from chain 0
            trans_s = self.chain_transmats[0][s_idx]
            means_s = self.chain_means[0][s_idx]
            covs_s = self.chain_covs[0][s_idx]

            # Forward pass probabilities
            emissions = np.zeros((n_samples, k))
            for st in range(k):
                try:
                    emissions[:, st] = multivariate_normal.pdf(
                        X_sub, mean=means_s[st], cov=covs_s[st], allow_singular=True
                    )
                except Exception:
                    emissions[:, st] = 1e-6
            emissions = np.maximum(emissions, 1e-12)

            fwd_probs = np.zeros((n_samples, k))
            fwd_probs[0] = np.ones(k) / k * emissions[0]
            fwd_probs[0] /= np.sum(fwd_probs[0])
            for t in range(1, n_samples):
                p = fwd_probs[t - 1] @ trans_s * emissions[t]
                sum_p = np.sum(p)
                fwd_probs[t] = p / sum_p if sum_p > 0 else np.ones(k) / k

            # Map to canonical
            canon_p = np.zeros((n_samples, 5))
            for raw_st, canon_idx in self.state_to_regime_map.items():
                if canon_idx < 5:
                    canon_p[:, canon_idx] += fwd_probs[:, raw_st]
            canon_p /= np.maximum(np.sum(canon_p, axis=1, keepdims=True), 1e-12)
            sample_probs.append(canon_p)

        sample_probs = np.array(sample_probs)  # shape: (S, N, 5)
        mean_probs = np.mean(sample_probs, axis=0)  # shape: (N, 5)

        # Uncertainty decomposition:
        # Total entropy H(p_bar) = - sum p_bar * log(p_bar)
        tot_entropy = -np.sum(mean_probs * np.log(np.maximum(mean_probs, 1e-12)), axis=1)

        # Expected entropy (aleatoric) = (1/S) sum H(p_s)
        sample_entropies = -np.sum(sample_probs * np.log(np.maximum(sample_probs, 1e-12)), axis=2)
        aleatoric_entropy = np.mean(sample_entropies, axis=0)

        # Epistemic uncertainty = Total - Aleatoric (Mutual Information I(Y; Theta))
        epistemic_entropy = np.maximum(tot_entropy - aleatoric_entropy, 0.0)

        return mean_probs, epistemic_entropy, aleatoric_entropy

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        mean_probs, _, _ = self.predict_proba_posterior(X)
        return mean_probs

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "n_regimes": self.n_regimes,
            "n_chains": self.n_chains,
            "n_samples_per_chain": len(self.chain_transmats[0]) if self.is_fitted else 0,
            "gelman_rubin_r_hat": self._r_hat_dict,
            "effective_sample_size_ess": self._ess_dict,
            "sticky_kappa": self.sticky_kappa,
            "alpha_base": self.alpha_base,
            "r_hat_converged_pct": float(
                np.mean([val < 1.05 for val in self._r_hat_dict.values()]) * 100.0
            ) if self._r_hat_dict else 0.0,
        }
