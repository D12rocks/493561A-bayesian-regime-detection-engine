"""
Quantitative Evaluation & Calibration Metrics.

Implements Expected Calibration Error (ECE), Multi-Class Brier Score,
Directional Scoring, and Conformal Prediction evaluation metrics.
"""

from typing import Dict, List, Tuple
import numpy as np


def compute_expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) across multi-class predictions.
    probs: ndarray of shape (N, K)
    labels: 1D array of ground truth class indices (0 to K-1)
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)


def compute_multi_class_brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute Multi-Class Brier Score:
    BS = (1 / N) * sum_i sum_k (p_{ik} - y_{ik})^2
    """
    N, K = probs.shape
    one_hot = np.zeros((N, K), dtype=np.float64)
    for i, label in enumerate(labels):
        one_hot[i, label] = 1.0
        
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(brier)


def compute_directional_accuracy(
    regime_probs: np.ndarray,
    forward_returns: np.ndarray,
) -> float:
    """
    Directional conviction score:
    Evaluates whether Risk-On/Late-Cycle probability aligns with positive forward returns,
    and Risk-Off probability aligns with negative forward returns.
    """
    # Risk-On (col 0) + Late-Cycle (col 1) vs Risk-Off (col 4)
    bullish_conviction = regime_probs[:, 0] + 0.5 * regime_probs[:, 1]
    bearish_conviction = regime_probs[:, 4]
    
    predicted_up = bullish_conviction > bearish_conviction
    actual_up = forward_returns > 0.0
    
    return float(np.mean(predicted_up == actual_up))


def compute_conformal_efficiency(
    prediction_sets: List[List[str]],
    true_regimes: List[str],
) -> Dict[str, float]:
    """
    Evaluates empirical marginal coverage and average prediction set cardinality.
    """
    n = len(true_regimes)
    covered = sum(1 for true_reg, pset in zip(true_regimes, prediction_sets) if true_reg in pset)
    set_sizes = [len(pset) for pset in prediction_sets]
    
    return {
        "empirical_coverage": float(covered / n) if n > 0 else 0.0,
        "average_set_size": float(np.mean(set_sizes)) if set_sizes else 0.0,
        "single_regime_rate": float(np.mean([s == 1 for s in set_sizes])) if set_sizes else 0.0,
        "empty_set_rate": float(np.mean([s == 0 for s in set_sizes])) if set_sizes else 0.0,
    }
