"""
Topological Data Analysis (TDA) Feature Extraction.

Implements REQ-016: Persistent homology features of multivariate financial state space
computed strictly point-in-time over sliding backward windows without future leakage.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from src.features.base import BaseFeatureTransformer


class TDAFeatureExtractor(BaseFeatureTransformer):
    """
    Extracts topological invariants from multivariate financial time-series attractor.
    
    For each date t, over a backward rolling window of length W:
    1. Standardizes multivariate return trajectories (point cloud in R^D).
    2. Builds Vietoris-Rips simplicial filtration over the point cloud.
    3. Computes 0-dimensional persistent homology (H0) via single-linkage minimum spanning tree.
    4. Computes 1-dimensional persistent homology (H1) cyclic loops via filtered graph cycles.
    5. Calculates topological summary metrics:
       - H0 persistence entropy
       - H0 Wasserstein amplitude (total persistence)
       - H1 maximum persistence (prominent market cycle lifetime)
       - H1 persistence entropy
       - H1 persistence landscape L1 norm
    """

    def __init__(
        self,
        name: str = "tda_features",
        window_size: int = 60,
        subsample_step: int = 1,
    ) -> None:
        super().__init__(name=name)
        self.window_size = window_size
        self.subsample_step = subsample_step

    def fit(self, df: pd.DataFrame) -> "TDAFeatureExtractor":
        self.is_fitted = True
        return self

    def _compute_h0_persistence(self, dist_matrix: np.ndarray) -> np.ndarray:
        """
        Compute 0-dimensional persistence diagram (birth=0, death=edge_weight)
        using the Minimum Spanning Tree of the pairwise distance matrix.
        """
        mst = minimum_spanning_tree(dist_matrix)
        mst_dense = mst.toarray()
        # Non-zero weights represent the death times of the N-1 components
        weights = mst_dense[mst_dense > 0.0]
        if len(weights) == 0:
            return np.array([0.0])
        return np.sort(weights)

    def _compute_h1_persistence(
        self, dist_matrix: np.ndarray, mst_weights: np.ndarray
    ) -> np.ndarray:
        """
        Approximates 1-dimensional persistent homology (birth, death) of loops
        formed by non-tree edges in the Vietoris-Rips filtration.
        """
        n = dist_matrix.shape[0]
        # Max edge in MST gives the connectivity scale
        max_mst_edge = np.max(mst_weights) if len(mst_weights) > 0 else 1.0
        
        # Consider edges up to 1.5 * max_mst_edge to capture relevant loops
        threshold = 1.5 * max_mst_edge
        
        # Find triangles and chords that create cycles
        triangles_lifetimes = []
        for i in range(min(n, 20)):  # Representative sample for speed & stability
            for j in range(i + 1, min(n, 20)):
                d_ij = dist_matrix[i, j]
                if d_ij > threshold:
                    continue
                # Search for mutual neighbor k
                d_ik = dist_matrix[i, :]
                d_jk = dist_matrix[j, :]
                # Birth of triangle cycle is max of the 3 edges
                # Death is when the 2-simplex (triangle) fills in
                mutual_max = np.maximum(d_ij, np.maximum(d_ik, d_jk))
                # Valid cycle exists when mutual_max is bounded
                valid_k = np.where((d_ik <= threshold) & (d_jk <= threshold))[0]
                if len(valid_k) > 0:
                    birth = np.min(mutual_max[valid_k])
                    death = np.max(mutual_max[valid_k])
                    lifetime = death - birth
                    if lifetime > 1e-4:
                        triangles_lifetimes.append(lifetime)
                        
        if len(triangles_lifetimes) == 0:
            return np.array([0.01])
        return np.sort(np.array(triangles_lifetimes))

    def _persistence_entropy(self, lifetimes: np.ndarray) -> float:
        """Shannon entropy of normalized persistence lifetimes."""
        valid = lifetimes[lifetimes > 1e-8]
        if len(valid) == 0:
            return 0.0
        total = np.sum(valid)
        if total <= 1e-8:
            return 0.0
        probs = valid / total
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        return float(entropy)

    def _landscape_l1_norm(self, lifetimes: np.ndarray) -> float:
        """Approximates L1 norm of first persistence landscape lambda_1."""
        # For a single interval [0, L], the peak height is L/2 and area under tent is (L/2) * L = L^2 / 2
        # For collection of intervals, first landscape norm integrates dominant peaks
        if len(lifetimes) == 0:
            return 0.0
        top_lifetimes = np.sort(lifetimes)[::-1][:min(5, len(lifetimes))]
        l1_norm = 0.5 * np.sum(top_lifetimes ** 2)
        return float(l1_norm)

    def compute_window_topology(self, window_matrix: np.ndarray) -> Dict[str, float]:
        """
        Compute TDA metrics on a single (W, D) standardized trajectory matrix.
        """
        # Pairwise Euclidean distance
        p_dists = pdist(window_matrix, metric="euclidean")
        d_mat = squareform(p_dists)
        
        # H0 Persistence
        h0_lifetimes = self._compute_h0_persistence(d_mat)
        h0_entropy = self._persistence_entropy(h0_lifetimes)
        h0_wasserstein = float(np.sum(h0_lifetimes))
        
        # H1 Persistence
        h1_lifetimes = self._compute_h1_persistence(d_mat, h0_lifetimes)
        h1_entropy = self._persistence_entropy(h1_lifetimes)
        h1_max = float(np.max(h1_lifetimes)) if len(h1_lifetimes) > 0 else 0.0
        h1_landscape_l1 = self._landscape_l1_norm(h1_lifetimes)
        
        return {
            "tda_persistence_entropy_h0": h0_entropy,
            "tda_wasserstein_amplitude_h0": h0_wasserstein,
            "tda_max_persistence_h1": h1_max,
            "tda_persistence_entropy_h1": h1_entropy,
            "tda_landscape_norm_h1": h1_landscape_l1,
        }

    def transform(self, df_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rolling TDA features strictly using backward windows.
        
        Args:
            df_returns: DataFrame of multivariate asset returns with DatetimeIndex.
                        Columns typically: ['NIFTY_50', 'NIFTY_MIDCAP_50', 'INDIA_VIX', 'USD_INR']
        
        Returns:
            DataFrame with DatetimeIndex and TDA topological feature columns.
        """
        n_obs = len(df_returns)
        idx = df_returns.index
        feature_names = [
            "tda_persistence_entropy_h0",
            "tda_wasserstein_amplitude_h0",
            "tda_max_persistence_h1",
            "tda_persistence_entropy_h1",
            "tda_landscape_norm_h1",
        ]
        
        res = pd.DataFrame(index=idx, columns=feature_names, dtype=float)
        
        # Standardize return matrix
        ret_values = df_returns.fillna(0.0).values
        w = self.window_size
        
        # Vectorized rolling sliding window computation
        for t in range(w, n_obs, self.subsample_step):
            # Window strictly in [t - w, t) -> strictly past observations!
            win = ret_values[t - w : t]
            
            # Standardize within window (point-in-time)
            mean = np.mean(win, axis=0, keepdims=True)
            std = np.std(win, axis=0, keepdims=True) + 1e-8
            std_win = (win - mean) / std
            
            metrics = self.compute_window_topology(std_win)
            for k, val in metrics.items():
                res.iloc[t, res.columns.get_loc(k)] = val
                
        # Forward fill any sub-sampled intervals to keep dense calendar
        if self.subsample_step > 1:
            res = res.ffill()
            
        return res
