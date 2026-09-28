"""
Probabilistic Scoring Benchmarks and Reference Baselines.

Implements REQ-040 and Specification Section G:
- Reference Baselines:
  1. Climatology Baseline (unconditional historical marginal frequencies)
  2. Persistence Baseline (S_t = S_{t-1})
- Strictly Proper Scoring Rules:
  - Multi-class Logarithmic Loss (Predictive Cross-Entropy)
  - Ranked Probability Score (RPS) for ordered financial regimes
  - Brier Score
  - Expected Calibration Error (ECE)
- Skill Scores relative to Climatology and Persistence:
  Skill = 1 - (Score_model / Score_reference)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class ClimatologyBaseline:
    """
    Unconditional climatological reference baseline:
    Emits the historical empirical marginal frequency vector for every time step.
    """

    def __init__(self) -> None:
        self.climatology_probs: np.ndarray = np.ones(5) / 5.0
        self.is_fitted: bool = False

    def fit(self, y: np.ndarray) -> "ClimatologyBaseline":
        counts = np.bincount(y.astype(int), minlength=5)
        self.climatology_probs = counts / np.sum(counts)
        self.is_fitted = True
        return self

    def predict_proba(self, n_samples: int) -> np.ndarray:
        return np.tile(self.climatology_probs, (n_samples, 1))


class PersistenceBaseline:
    """
    Persistence reference baseline:
    Predicts that yesterday's regime persists today:
    P(S_t = k | S_{t-1} = j) = 1.0 if k == j else 0.0 (with slight smoothing).
    """

    def __init__(self, smoothing: float = 0.02) -> None:
        self.smoothing = smoothing

    def predict_proba(self, y_lagged: np.ndarray) -> np.ndarray:
        n_samples = len(y_lagged)
        probs = np.full((n_samples, 5), self.smoothing / 4.0)
        for i, prev_state in enumerate(y_lagged):
            prev_int = int(prev_state)
            if 0 <= prev_int < 5:
                probs[i, prev_int] = 1.0 - self.smoothing
        return probs


def compute_log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    """
    Multi-class logarithmic loss (predictive cross-entropy):
    L = - (1 / N) * sum_i log(p_{i, y_i})
    """
    n_samples = len(y)
    eps = 1e-12
    p_true = np.array([probs[i, int(y[i])] for i in range(n_samples)])
    return float(-np.mean(np.log(np.maximum(p_true, eps))))


def compute_ranked_probability_score(probs: np.ndarray, y: np.ndarray) -> float:
    """
    Ranked Probability Score (RPS) for ordered regime ontology:
    RPS = (1 / (K - 1)) * sum_{k=1}^{K-1} (CDF_pred(k) - CDF_true(k))^2
    """
    n_samples, k_dim = probs.shape
    rps_list = []

    # One-hot true matrix
    y_onehot = np.zeros((n_samples, k_dim))
    for i, label in enumerate(y):
        y_onehot[i, int(label)] = 1.0

    cdf_pred = np.cumsum(probs, axis=1)
    cdf_true = np.cumsum(y_onehot, axis=1)

    # Sum squared differences across K-1 boundaries
    diff_sq = (cdf_pred[:, :-1] - cdf_true[:, :-1]) ** 2
    rps = np.mean(np.sum(diff_sq, axis=1) / (k_dim - 1.0))
    return float(rps)


def compute_brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score: (1/N) * sum_i sum_k (p_{ik} - y_{ik})^2"""
    n_samples, k_dim = probs.shape
    y_onehot = np.zeros((n_samples, k_dim))
    for i, label in enumerate(y):
        y_onehot[i, int(label)] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


class BenchmarkTournament:
    """
    Evaluates competitive regime models against Climatology and Persistence baselines.
    A model must outperform Persistence on proper scores to demonstrate regime skill.
    """

    @staticmethod
    def evaluate_models(
        model_predictions: Dict[str, np.ndarray],
        ground_truth: np.ndarray,
    ) -> pd.DataFrame:
        """
        Args:
            model_predictions: Mapping of model_name -> probability matrix (N, 5)
            ground_truth: True regime sequence (N,)
        
        Returns:
            DataFrame with proper scores and skill scores relative to baselines.
        """
        n_samples = len(ground_truth)

        # Build reference baselines
        climatology = ClimatologyBaseline().fit(ground_truth)
        p_clim = climatology.predict_proba(n_samples)

        # Lagged true states for persistence
        y_lagged = np.roll(ground_truth, 1)
        y_lagged[0] = ground_truth[0]
        persistence = PersistenceBaseline()
        p_persist = persistence.predict_proba(y_lagged)

        all_candidates = {
            "Climatology (Baseline 1)": p_clim,
            "Persistence (Baseline 2)": p_persist,
        }
        all_candidates.update(model_predictions)

        # Baseline scores for skill score normalization
        clim_ll = compute_log_loss(p_clim, ground_truth)
        persist_ll = compute_log_loss(p_persist, ground_truth)
        clim_brier = compute_brier_score(p_clim, ground_truth)
        persist_brier = compute_brier_score(p_persist, ground_truth)
        clim_rps = compute_ranked_probability_score(p_clim, ground_truth)
        persist_rps = compute_ranked_probability_score(p_persist, ground_truth)

        results = []

        for name, probs in all_candidates.items():
            ll = compute_log_loss(probs, ground_truth)
            bs = compute_brier_score(probs, ground_truth)
            rps = compute_ranked_probability_score(probs, ground_truth)

            # Skill scores: 1 - Score_model / Score_baseline (strictly negative if model score > baseline score)
            skill_clim_ll = 1.0 - (ll / max(clim_ll, 1e-6))
            skill_persist_ll = 1.0 - (ll / max(persist_ll, 1e-6))

            skill_clim_brier = 1.0 - (bs / max(clim_brier, 1e-6))
            skill_persist_brier = 1.0 - (bs / max(persist_brier, 1e-6))

            skill_clim_rps = 1.0 - (rps / max(clim_rps, 1e-6))
            skill_persist_rps = 1.0 - (rps / max(persist_rps, 1e-6))

            # Specific metric dominance flags
            beats_persist_ll = bool(ll < persist_ll)
            beats_persist_brier = bool(bs < persist_brier)
            beats_persist_rps = bool(rps < persist_rps)

            results.append({
                "model_name": name,
                "log_loss": round(ll, 4),
                "brier_score": round(bs, 4),
                "ranked_prob_score_rps": round(rps, 4),
                "skill_vs_climatology_log_loss": round(skill_clim_ll, 4),
                "skill_vs_persistence_log_loss": round(skill_persist_ll, 4),
                "skill_vs_climatology_brier": round(skill_clim_brier, 4),
                "skill_vs_persistence_brier": round(skill_persist_brier, 4),
                "skill_vs_climatology_rps": round(skill_clim_rps, 4),
                "skill_vs_persistence_rps": round(skill_persist_rps, 4),
                "beats_persistence_log_loss": beats_persist_ll,
                "beats_persistence_brier": beats_persist_brier,
                "beats_persistence_rps": beats_persist_rps,
            })

        df_res = pd.DataFrame(results).sort_values("log_loss")
        return df_res
