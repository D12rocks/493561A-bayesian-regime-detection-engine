"""
Variational Bayesian Neural Network (Bayes by Backprop).

Implements REQ-023:
- Mean-field Gaussian variational approximation: q(w | mu, rho)
- Re-parameterization trick: w = mu + log(1 + exp(rho)) * eps
- Loss: Negative Evidence Lower Bound (ELBO) = CrossEntropy + beta * KL(q(w) || p(w))
- Uncertainty decomposition: Epistemic (weight variance) vs Aleatoric (softmax entropy)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseRegimeModel


class VariationalLinear(nn.Module):
    """
    Linear layer with Mean-Field Gaussian weights: w ~ N(mu, sigma^2),
    where sigma = softplus(rho).
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Variational parameters
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))
        self.w_rho = nn.Parameter(torch.empty(out_features, in_features).uniform_(-4.0, -3.0))

        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(torch.empty(out_features).uniform_(-4.0, -3.0))

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        w_sigma = torch.log1p(torch.exp(self.w_rho))
        b_sigma = torch.log1p(torch.exp(self.b_rho))

        if sample:
            eps_w = torch.randn_like(w_sigma)
            eps_b = torch.randn_like(b_sigma)
            w = self.w_mu + w_sigma * eps_w
            b = self.b_mu + b_sigma * eps_b
        else:
            w = self.w_mu
            b = self.b_mu

        return F.linear(x, w, b)

    def kl_divergence(self) -> torch.Tensor:
        """Analytical KL divergence against isotropic Gaussian prior N(0, prior_sigma^2)."""
        w_sigma = torch.log1p(torch.exp(self.w_rho))
        b_sigma = torch.log1p(torch.exp(self.b_rho))
        p_var = self.prior_sigma ** 2

        kl_w = torch.sum(
            torch.log(self.prior_sigma / w_sigma) + (w_sigma ** 2 + self.w_mu ** 2) / (2.0 * p_var) - 0.5
        )
        kl_b = torch.sum(
            torch.log(self.prior_sigma / b_sigma) + (b_sigma ** 2 + self.b_mu ** 2) / (2.0 * p_var) - 0.5
        )
        return kl_w + kl_b


class VariationalBNNModule(nn.Module):
    """2-layer Variational Bayesian MLP with ELBO loss."""

    def __init__(self, in_features: int, hidden_dim: int = 32, n_classes: int = 5) -> None:
        super().__init__()
        self.layer1 = VariationalLinear(in_features, hidden_dim)
        self.layer2 = VariationalLinear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        h = F.relu(self.layer1(x, sample=sample))
        logits = self.layer2(h, sample=sample)
        return logits

    def kl_divergence(self) -> torch.Tensor:
        return self.layer1.kl_divergence() + self.layer2.kl_divergence()


class VariationalBNNModel(BaseRegimeModel):
    """
    Variational Bayesian Neural Network (Bayes by Backprop) for regime classification.
    """

    def __init__(
        self,
        name: str = "variational_bnn",
        n_regimes: int = 5,
        hidden_dim: int = 32,
        n_mc_samples: int = 50,
        epochs: int = 80,
        lr: float = 0.01,
        random_seed: int = 42,
    ) -> None:
        super().__init__(name=name, version="1.0.0", random_seed=random_seed)
        self.n_regimes = n_regimes
        self.hidden_dim = hidden_dim
        self.n_mc_samples = n_mc_samples
        self.epochs = epochs
        self.lr = lr
        self.model: Optional[VariationalBNNModule] = None
        self.feature_columns: List[str] = []
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.final_loss: float = 0.0

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "VariationalBNNModel":
        torch.manual_seed(self.random_seed)
        self.feature_columns = list(X.columns)
        X_clean = X.dropna().values
        n_obs, n_features = X_clean.shape

        self.scaler_mean = np.mean(X_clean, axis=0)
        self.scaler_std = np.std(X_clean, axis=0) + 1e-8
        X_norm = (X_clean - self.scaler_mean) / self.scaler_std

        if y is None:
            # Soft targets based on standardized return and vol
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
        self.model = VariationalBNNModule(in_features=n_features, hidden_dim=self.hidden_dim, n_classes=self.n_regimes)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model(x_tensor, sample=True)
            if targets_tensor.ndim == 1:
                nll = F.cross_entropy(logits, targets_tensor)
            else:
                log_probs = F.log_softmax(logits, dim=1)
                nll = -torch.mean(torch.sum(targets_tensor * log_probs, dim=1))

            kl = self.model.kl_divergence() / n_obs
            loss = nll + 0.01 * kl
            loss.backward()
            optimizer.step()
            self.final_loss = float(loss.item())

        self.is_fitted = True
        return self

    def predict_proba_with_uncertainty(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted.")

        cols = self.feature_columns if self.feature_columns else X.columns
        X_sub = X[cols].ffill().fillna(0.0).values
        X_norm = (X_sub - self.scaler_mean) / self.scaler_std
        x_tensor = torch.tensor(X_norm, dtype=torch.float32)

        self.model.eval()
        mc_probs = []
        with torch.no_grad():
            for _ in range(self.n_mc_samples):
                logits = self.model(x_tensor, sample=True)
                p = F.softmax(logits, dim=1).cpu().numpy()
                mc_probs.append(p)

        mc_probs = np.array(mc_probs)  # (S, N, 5)
        mean_probs = np.mean(mc_probs, axis=0)
        mean_probs /= np.sum(mean_probs, axis=1, keepdims=True)

        tot_entropy = -np.sum(mean_probs * np.log(np.maximum(mean_probs, 1e-12)), axis=1)
        pass_entropies = -np.sum(mc_probs * np.log(np.maximum(mc_probs, 1e-12)), axis=2)
        aleatoric = np.mean(pass_entropies, axis=0)
        epistemic = np.maximum(tot_entropy - aleatoric, 0.0)

        return mean_probs, epistemic, aleatoric

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p, _, _ = self.predict_proba_with_uncertainty(X)
        return p

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "model_type": "Variational BNN (Bayes by Backprop)",
            "hidden_dim": self.hidden_dim,
            "n_mc_samples": self.n_mc_samples,
            "final_loss": self.final_loss,
            "features": len(self.feature_columns),
        }
