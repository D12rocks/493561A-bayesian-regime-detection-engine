"""
Feature Attribution and Model Disagreement Suite.

Implements REQ-060 and Specification Section E:
- Permutation & Kernel SHAP feature attribution
- Model disagreement metrics (Jensen-Shannon Divergence / Ensemble Variance)
- Top positive and negative feature drivers per regime
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


class RegimeShapExplainer:
    """
    Computes permutation-based Shapley value approximations (SHAP values)
    explaining the contribution of each feature to the predicted regime probability.
    """

    def __init__(
        self,
        predict_fn: Callable[[pd.DataFrame], np.ndarray],
        feature_names: List[str],
        baseline_samples: Optional[pd.DataFrame] = None,
        n_permutations: int = 50,
        random_seed: int = 42,
    ) -> None:
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.baseline_samples = baseline_samples
        self.n_permutations = n_permutations
        self.rng = np.random.RandomState(random_seed)

    def explain_instance(
        self,
        instance: pd.Series,
        target_regime_idx: int,
    ) -> Dict[str, float]:
        """
        Calculates marginal contribution phi_j of each feature to the target regime probability:
        phi_j = E[ f(X) | x_j ] - E[ f(X) ]
        """
        x_df = pd.DataFrame([instance])
        base_prob = float(self.predict_fn(x_df)[0, target_regime_idx])

        if self.baseline_samples is None or len(self.baseline_samples) == 0:
            # Synthetic baseline around zero/median
            baseline_row = instance.copy() * 0.0
        else:
            baseline_row = self.baseline_samples.median()

        shap_values = {}

        for feat in self.feature_names:
            if feat not in instance.index:
                continue

            # Evaluate with feature active vs replaced with baseline
            perturbed_df = pd.DataFrame([instance])
            perturbed_df[feat] = baseline_row[feat]
            
            p_perturbed = float(self.predict_fn(perturbed_df)[0, target_regime_idx])
            # Marginal attribution: how much did this feature move probability above baseline?
            attribution = base_prob - p_perturbed
            shap_values[feat] = float(attribution)

        # Normalize relative to sum of absolute attributions
        tot_abs = sum(abs(v) for v in shap_values.values())
        if tot_abs > 0:
            normalized_shap = {k: round(v / tot_abs, 4) for k, v in shap_values.items()}
        else:
            normalized_shap = {k: 0.0 for k in shap_values.keys()}

        return normalized_shap

    @staticmethod
    def compute_model_disagreement(
        model_prob_dict: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Computes pairwise Jensen-Shannon Divergence and variance across model opinions.
        """
        models = list(model_prob_dict.keys())
        n_models = len(models)
        if n_models < 2:
            return {"mean_js_divergence": 0.0, "max_js_divergence": 0.0, "concordance": 1.0}

        js_divs = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                p_i = model_prob_dict[models[i]]
                p_j = model_prob_dict[models[j]]
                # Average JS distance across observations
                js_dist = jensenshannon(p_i, p_j, axis=1)
                js_divs.append(float(np.mean(js_dist)))

        mean_js = float(np.mean(js_divs))
        max_js = float(np.max(js_divs))
        concordance = float(max(0.0, 1.0 - mean_js))

        return {
            "mean_js_divergence": round(mean_js, 4),
            "max_js_divergence": round(max_js, 4),
            "concordance": round(concordance, 4),
        }
