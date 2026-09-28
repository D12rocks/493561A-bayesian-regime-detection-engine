"""
Deep Ensemble for Regime Uncertainty Estimation (Lakshminarayanan et al. 2017).

Implements REQ-023:
- Ensemble of M=5 independently initialized deterministic neural networks
- Non-parametric epistemic uncertainty via ensemble divergence
- Predictive mean: mixture of member predictive distributions
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseRegimeModel


class MLPClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 32, n_classes: int = 5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 16)
        self.out = nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.out(h)


class DeepEnsembleModel(BaseRegimeModel):
    """
    Deep Ensemble consisting of M independently trained neural networks.
    """

    def __init__(
        self,
        name: str = "deep_ensemble",
        n_regimes: int = 5,
        n_members: int = 5,
        hidden_dim: int = 32,
        epochs: int = 60,
        lr: float = 0.01,
        random_seed: int = 42,
    ) -> None:
        super().__init__(name=name, version="1.0.0", random_seed=random_seed)
        self.n_regimes = n_regimes
        self.n_members = n_members
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.members: List[MLPClassifier] = []
        self.feature_columns: List[str] = []
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "DeepEnsembleModel":
        self.feature_columns = list(X.columns)
        X_clean = X.dropna().values
        n_obs, n_features = X_clean.shape

        self.scaler_mean = np.mean(X_clean, axis=0)
        self.scaler_std = np.std(X_clean, axis=0) + 1e-8
        X_norm = (X_clean - self.scaler_mean) / self.scaler_std

        if y is None:
            ret = X_norm[:, 0]
            vol = X_norm[:, 1] if n_features > 1 else np.abs(ret)
            y_soft = np.zeros((n_obs, 5))
            for i in range(n_obs):
                r, v = ret[i], vol[i]
                scores = np.array([
                    np.exp(1.2 * r - 0.8 * v),
                    np.exp(0.8 * r + 0.8 * v),
                    np.exp(-1.5 * (r ** 2)),
                    np.exp(0.5 * r + 1.5 * v),
                    np.exp(-1.5 * r + 1.0 * v),
                ])
                y_soft[i] = scores / np.sum(scores)
            targets_tensor = torch.tensor(y_soft, dtype=torch.float32)
        else:
            if y.ndim == 1:
                targets_tensor = torch.tensor(y, dtype=torch.long)
            else:
                targets_tensor = torch.tensor(y, dtype=torch.float32)

        x_tensor = torch.tensor(X_norm, dtype=torch.float32)
        self.members = []

        for m_idx in range(self.n_members):
            torch.manual_seed(self.random_seed + m_idx * 101)
            net = MLPClassifier(in_features=n_features, hidden_dim=self.hidden_dim, n_classes=self.n_regimes)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

            net.train()
            for epoch in range(self.epochs):
                # Bootstrap sampling / data shuffling
                perm = torch.randperm(n_obs)
                bx = x_tensor[perm]
                by = targets_tensor[perm]

                optimizer.zero_grad()
                logits = net(bx)
                if by.ndim == 1:
                    loss = F.cross_entropy(logits, by)
                else:
                    loss = -torch.mean(torch.sum(by * F.log_softmax(logits, dim=1), dim=1))
                loss.backward()
                optimizer.step()

            self.members.append(net)

        self.is_fitted = True
        return self

    def predict_proba_with_uncertainty(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted.")

        cols = self.feature_columns if self.feature_columns else X.columns
        X_sub = X[cols].ffill().fillna(0.0).values
        X_norm = (X_sub - self.scaler_mean) / self.scaler_std
        x_tensor = torch.tensor(X_norm, dtype=torch.float32)

        member_probs = []
        for net in self.members:
            net.eval()
            with torch.no_grad():
                logits = net(x_tensor)
                p = F.softmax(logits, dim=1).cpu().numpy()
                member_probs.append(p)

        member_probs = np.array(member_probs)  # (M, N, 5)
        mean_probs = np.mean(member_probs, axis=0)
        mean_probs /= np.sum(mean_probs, axis=1, keepdims=True)

        tot_entropy = -np.sum(mean_probs * np.log(np.maximum(mean_probs, 1e-12)), axis=1)
        pass_entropies = -np.sum(member_probs * np.log(np.maximum(member_probs, 1e-12)), axis=2)
        aleatoric = np.mean(pass_entropies, axis=0)
        epistemic = np.maximum(tot_entropy - aleatoric, 0.0)

        return mean_probs, epistemic, aleatoric

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p, _, _ = self.predict_proba_with_uncertainty(X)
        return p

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "model_type": "Deep Ensemble (Lakshminarayanan et al. 2017)",
            "n_members": self.n_members,
            "hidden_dim": self.hidden_dim,
            "features": len(self.feature_columns),
        }
