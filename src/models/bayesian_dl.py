"""
Bayesian Deep Learning: Monte Carlo Dropout and Deep Ensemble.

Implements REQ-023:
- Multi-layer Temporal Neural Network with Monte Carlo Dropout (p=0.2)
- Inference-time stochastic forward passes (M=100)
- Epistemic uncertainty (Mutual Information) vs Aleatoric uncertainty (Expected Entropy)
- Deep Ensemble (5 independently seeded models)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.models.base import BaseRegimeModel
from src.models.contracts import RegimeLabel


class MCDropoutNetwork:
    """
    Feedforward Neural Network with exact Monte Carlo Dropout layers.
    W0: (D, H1), W1: (H1, H2), W2: (H2, K)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 32,
        hidden_dim2: int = 16,
        output_dim: int = 5,
        dropout_rate: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # He initialization
        self.w0 = self.rng.randn(input_dim, hidden_dim1) * np.sqrt(2.0 / input_dim)
        self.b0 = np.zeros(hidden_dim1)
        self.w1 = self.rng.randn(hidden_dim1, hidden_dim2) * np.sqrt(2.0 / hidden_dim1)
        self.b1 = np.zeros(hidden_dim2)
        self.w2 = self.rng.randn(hidden_dim2, output_dim) * np.sqrt(2.0 / hidden_dim2)
        self.b2 = np.zeros(output_dim)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, x: np.ndarray, apply_dropout: bool = True) -> np.ndarray:
        """Forward pass with stochastic dropout scaling."""
        # Layer 1: ReLU
        z0 = x @ self.w0 + self.b0
        a0 = np.maximum(0.0, z0)
        if apply_dropout and self.dropout_rate > 0.0:
            mask0 = (self.rng.rand(*a0.shape) >= self.dropout_rate) / (1.0 - self.dropout_rate)
            a0 = a0 * mask0

        # Layer 2: ReLU
        z1 = a0 @ self.w1 + self.b1
        a1 = np.maximum(0.0, z1)
        if apply_dropout and self.dropout_rate > 0.0:
            mask1 = (self.rng.rand(*a1.shape) >= self.dropout_rate) / (1.0 - self.dropout_rate)
            a1 = a1 * mask1

        # Output: Softmax
        z2 = a1 @ self.w2 + self.b2
        probs = self._softmax(z2)
        return probs

    def train_adam(
        self,
        x: np.ndarray,
        y_targets: np.ndarray,
        epochs: int = 120,
        lr: float = 0.005,
        batch_size: int = 64,
    ) -> List[float]:
        """Trains network with Adam optimizer and Cross-Entropy loss."""
        n_samples = len(x)
        losses = []

        # Adam moment buffers
        m_w0, v_w0 = np.zeros_like(self.w0), np.zeros_like(self.w0)
        m_b0, v_b0 = np.zeros_like(self.b0), np.zeros_like(self.b0)
        m_w1, v_w1 = np.zeros_like(self.w1), np.zeros_like(self.w1)
        m_b1, v_b1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        m_w2, v_w2 = np.zeros_like(self.w2), np.zeros_like(self.w2)
        m_b2, v_b2 = np.zeros_like(self.b2), np.zeros_like(self.b2)

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t_step = 0

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            self.rng.shuffle(indices)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                b_idx = indices[start:end]
                bx, by = x[b_idx], y_targets[b_idx]
                bsize = len(bx)

                # Forward
                z0 = bx @ self.w0 + self.b0
                a0 = np.maximum(0.0, z0)
                mask0 = (self.rng.rand(*a0.shape) >= self.dropout_rate) / (1.0 - self.dropout_rate)
                a0_drop = a0 * mask0

                z1 = a0_drop @ self.w1 + self.b1
                a1 = np.maximum(0.0, z1)
                mask1 = (self.rng.rand(*a1.shape) >= self.dropout_rate) / (1.0 - self.dropout_rate)
                a1_drop = a1 * mask1

                z2 = a1_drop @ self.w2 + self.b2
                probs = self._softmax(z2)

                # Cross-entropy loss: - (1/N) sum y * log(p)
                loss = -np.mean(np.sum(by * np.log(np.maximum(probs, 1e-12)), axis=1))
                epoch_loss += loss
                n_batches += 1

                # Backward
                dz2 = (probs - by) / bsize
                dw2 = a1_drop.T @ dz2
                db2 = np.sum(dz2, axis=0)

                da1_drop = dz2 @ self.w2.T
                da1 = da1_drop * mask1
                dz1 = da1 * (z1 > 0.0)
                dw1 = a0_drop.T @ dz1
                db1 = np.sum(dz1, axis=0)

                da0_drop = dz1 @ self.w1.T
                da0 = da0_drop * mask0
                dz0 = da0 * (z0 > 0.0)
                dw0 = bx.T @ dz0
                db0 = np.sum(dz0, axis=0)

                # Adam updates
                t_step += 1
                for param, grad, m, v in [
                    (self.w0, dw0, m_w0, v_w0),
                    (self.b0, db0, m_b0, v_b0),
                    (self.w1, dw1, m_w1, v_w1),
                    (self.b1, db1, m_b1, v_b1),
                    (self.w2, dw2, m_w2, v_w2),
                    (self.b2, db2, m_b2, v_b2),
                ]:
                    m[:] = beta1 * m + (1.0 - beta1) * grad
                    v[:] = beta2 * v + (1.0 - beta2) * (grad ** 2)
                    m_hat = m / (1.0 - beta1 ** t_step)
                    v_hat = v / (1.0 - beta2 ** t_step)
                    param -= lr * m_hat / (np.sqrt(v_hat) + eps)

            losses.append(epoch_loss / max(1, n_batches))
        return losses


class BayesianDeepLearningModel(BaseRegimeModel):
    """
    Bayesian Deep Learning model combining MC Dropout and Deep Ensembling.
    """

    def __init__(
        self,
        name: str = "bayesian_dl",
        n_regimes: int = 5,
        n_mc_samples: int = 100,
        n_ensemble: int = 5,
        dropout_rate: float = 0.2,
        epochs: int = 100,
        random_seed: int = 42,
    ) -> None:
        super().__init__(name=name, version="1.0.0", random_seed=random_seed)
        self.n_regimes = n_regimes
        self.n_mc_samples = n_mc_samples
        self.n_ensemble = n_ensemble
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.ensemble_models: List[MCDropoutNetwork] = []
        self.feature_columns: List[str] = []
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self._training_losses: List[float] = []

    def _generate_pseudo_targets(self, X_clean: np.ndarray) -> np.ndarray:
        """
        Creates smooth prior target distribution using return and volatility quantiles:
        Index 0: Risk-On (High return, low vol)
        Index 1: Late-Cycle (Positive return, higher vol)
        Index 2: Transitional (Neutral return, moderate vol)
        Index 3: Post-Shock (Bouncing return, high vol)
        Index 4: Risk-Off (Negative return, high vol)
        """
        n_samples = len(X_clean)
        targets = np.zeros((n_samples, 5))
        
        ret = X_clean[:, 0]
        vol = X_clean[:, 1] if X_clean.shape[1] > 1 else np.abs(ret)

        ret_z = (ret - np.mean(ret)) / (np.std(ret) + 1e-8)
        vol_z = (vol - np.mean(vol)) / (np.std(vol) + 1e-8)

        # Soft assignment
        for i in range(n_samples):
            rz, vz = ret_z[i], vol_z[i]
            # Risk-On: rz > 0, vz < 0
            s_on = np.exp(1.2 * rz - 0.8 * vz)
            # Late-Cycle: rz > 0, vz > 0
            s_late = np.exp(0.8 * rz + 0.8 * vz)
            # Transitional: -0.5 < rz < 0.5
            s_trans = np.exp(-1.5 * (rz ** 2))
            # Post-Shock: rz > 0, vz > 1.5
            s_post = np.exp(0.5 * rz + 1.5 * vz)
            # Risk-Off: rz < 0, vz > 0
            s_off = np.exp(-1.5 * rz + 1.0 * vz)

            row = np.array([s_on, s_late, s_trans, s_post, s_off])
            targets[i] = row / np.sum(row)
        return targets

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BayesianDeepLearningModel":
        self.feature_columns = list(X.columns)
        X_clean = X.dropna().values
        n_samples, n_features = X_clean.shape

        # Standardize features
        self.scaler_mean = np.mean(X_clean, axis=0)
        self.scaler_std = np.std(X_clean, axis=0) + 1e-8
        X_norm = (X_clean - self.scaler_mean) / self.scaler_std

        # Targets
        if y is None:
            y_targets = self._generate_pseudo_targets(X_clean)
        else:
            # One-hot encode if discrete labels provided
            y_targets = pd.get_dummies(y).values

        self.ensemble_models = []
        for i in range(self.n_ensemble):
            net = MCDropoutNetwork(
                input_dim=n_features,
                hidden_dim1=32,
                hidden_dim2=16,
                output_dim=5,
                dropout_rate=self.dropout_rate,
                seed=self.random_seed + i * 17,
            )
            losses = net.train_adam(X_norm, y_targets, epochs=self.epochs, lr=0.005)
            self.ensemble_models.append(net)
            if i == 0:
                self._training_losses = losses

        self.is_fitted = True
        return self

    def predict_proba_with_uncertainty(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs MC Dropout inference across ensemble members.
        
        Returns:
            mean_probs: (N, 5) probability distribution
            epistemic_uncertainty: (N,) mutual information / model disagreement
            aleatoric_uncertainty: (N,) expected data entropy
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted.")

        cols = self.feature_columns if self.feature_columns else X.columns
        X_sub = X[cols].ffill().fillna(0.0).values
        X_norm = (X_sub - self.scaler_mean) / self.scaler_std
        n_samples = len(X_norm)

        # Draw MC forward passes: n_mc_samples distributed across ensemble
        mc_passes = []
        per_model_passes = max(5, self.n_mc_samples // self.n_ensemble)

        for net in self.ensemble_models:
            for _ in range(per_model_passes):
                p = net.forward(X_norm, apply_dropout=True)
                mc_passes.append(p)

        mc_passes = np.array(mc_passes)  # shape: (M, N, 5)
        mean_probs = np.mean(mc_passes, axis=0)  # shape: (N, 5)
        mean_probs /= np.sum(mean_probs, axis=1, keepdims=True)

        # Predictive entropy
        tot_entropy = -np.sum(mean_probs * np.log(np.maximum(mean_probs, 1e-12)), axis=1)

        # Expected entropy
        pass_entropies = -np.sum(mc_passes * np.log(np.maximum(mc_passes, 1e-12)), axis=2)
        aleatoric_entropy = np.mean(pass_entropies, axis=0)

        # Mutual information = Epistemic uncertainty
        epistemic_entropy = np.maximum(tot_entropy - aleatoric_entropy, 0.0)

        return mean_probs, epistemic_entropy, aleatoric_entropy

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        mean_probs, _, _ = self.predict_proba_with_uncertainty(X)
        return mean_probs

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "n_ensemble": self.n_ensemble,
            "n_mc_samples": self.n_mc_samples,
            "dropout_rate": self.dropout_rate,
            "final_loss": self._training_losses[-1] if self._training_losses else np.nan,
            "n_features": len(self.feature_columns),
        }
