"""
Sector Correlation Graph & Graph Neural Network (GNN) Feature Extraction.

Implements REQ-017: Sector correlation graph topology, spectral graph invariants,
and message-passing graph neural network embeddings computed strictly point-in-time
without future temporal leakage.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.features.base import BaseFeatureTransformer


class SectorGraphFeatureExtractor(BaseFeatureTransformer):
    """
    Extracts topological and spectral invariants from rolling sector correlation graphs,
    along with GCN (Graph Convolutional Network) relational embeddings.
    
    Nodes in Sector Graph:
    - NIFTY_BANK (Financials bellwether)
    - NIFTY_IT (Technology export hedge)
    - NIFTY_MIDCAP_50 (Domestic growth & risk appetite)
    - NIFTY_500 (Broad market participation)
    - NIFTY_50 (Benchmark large cap)
    
    Topological Invariants:
    1. Spectral Radius: Maximum eigenvalue of adjacency matrix lambda_max(A)
    2. Algebraic Connectivity (Fiedler value): Second-smallest eigenvalue of Laplacian lambda_2(L)
    3. Von Neumann Graph Entropy: Entropy of normalized Laplacian spectrum
    4. Sector Centrality: Banking and IT eigenvector centrality
    5. GCN Graph Embedding: 2-layer Graph Convolutional readout representation
    """

    def __init__(
        self,
        name: str = "gnn_features",
        window_size: int = 60,
        embedding_dim: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__(name=name)
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.seed = seed
        
        # Deterministic random projection weights for GCN message passing
        rng = np.random.RandomState(self.seed)
        # Input node features: [return, volatility, momentum] (dimension 3)
        self.w0 = rng.normal(0.0, 0.5, size=(3, 8))
        self.w1 = rng.normal(0.0, 0.5, size=(8, self.embedding_dim))

    def fit(self, df: pd.DataFrame) -> "SectorGraphFeatureExtractor":
        self.is_fitted = True
        return self

    def _compute_graph_spectral_metrics(
        self, adj_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes spectral radius, Fiedler value, von Neumann entropy, and node centralities.
        """
        k = adj_matrix.shape[0]
        # Degree matrix
        degrees = np.sum(adj_matrix, axis=1)
        d_mat = np.diag(degrees)
        laplacian = d_mat - adj_matrix
        
        # Adjacency eigenvalues
        adj_eigvals, adj_eigvecs = np.linalg.eigh(adj_matrix)
        spectral_radius = float(np.max(np.abs(adj_eigvals)))
        
        # Eigenvector centrality (principal eigenvector of adjacency)
        principal_idx = np.argmax(adj_eigvals)
        centrality = np.abs(adj_eigvecs[:, principal_idx])
        centrality_sum = np.sum(centrality)
        if centrality_sum > 1e-8:
            centrality = centrality / centrality_sum
        else:
            centrality = np.ones(k) / k
            
        # Laplacian eigenvalues (sorted ascending: 0 = lambda_1 <= lambda_2 <= ...)
        lap_eigvals = np.sort(np.linalg.eigvalsh(laplacian))
        # Fiedler value is lambda_2
        fiedler_val = float(lap_eigvals[1]) if k > 1 else 0.0
        
        # Von Neumann Graph Entropy
        trace_l = np.sum(lap_eigvals)
        if trace_l > 1e-8:
            rho = lap_eigvals / trace_l
            pos_rho = rho[rho > 1e-10]
            vn_entropy = -float(np.sum(pos_rho * np.log(pos_rho)))
        else:
            vn_entropy = 0.0
            
        return {
            "gnn_spectral_radius": spectral_radius,
            "gnn_algebraic_connectivity": fiedler_val,
            "gnn_graph_entropy": vn_entropy,
            "gnn_bank_centrality": float(centrality[0]),  # Bank is index 0
            "gnn_it_centrality": float(centrality[1]),    # IT is index 1
        }

    def _gcn_forward(
        self, adj_matrix: np.ndarray, node_features: np.ndarray
    ) -> np.ndarray:
        """
        2-layer Graph Convolutional Network message passing:
        H^(1) = ReLU( D_hat^(-1/2) A_tilde D_hat^(-1/2) H^(0) W0 )
        H^(2) = D_hat^(-1/2) A_tilde D_hat^(-1/2) H^(1) W1
        Readout = MeanPool(H^(2))
        """
        k = adj_matrix.shape[0]
        # Renormalization trick: A_tilde = A + I_k
        a_tilde = adj_matrix + np.eye(k)
        d_tilde_diag = np.sum(a_tilde, axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(d_tilde_diag, 1e-6)))
        
        # Normalized adjacency
        norm_adj = d_inv_sqrt @ a_tilde @ d_inv_sqrt
        
        # Layer 1
        h1 = np.maximum(0.0, norm_adj @ node_features @ self.w0)
        
        # Layer 2
        h2 = norm_adj @ h1 @ self.w1
        
        # Readout pooling across all nodes
        graph_embedding = np.mean(h2, axis=0)
        return graph_embedding

    def extract_from_sector_prices(
        self, sector_dfs: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Extract sector graph features from price histories strictly using backward rolling windows.
        
        Required symbols in sector_dfs:
        ['NIFTY_BANK', 'NIFTY_IT', 'NIFTY_MIDCAP_50', 'NIFTY_500', 'NIFTY_50']
        """
        symbols = ["NIFTY_BANK", "NIFTY_IT", "NIFTY_MIDCAP_50", "NIFTY_500", "NIFTY_50"]
        available_syms = [s for s in symbols if s in sector_dfs]
        if len(available_syms) < 3:
            raise ValueError(f"Need at least 3 sector series for graph modeling. Found: {available_syms}")
            
        # Align on common index
        common_idx = sector_dfs[available_syms[0]].index
        for s in available_syms[1:]:
            common_idx = common_idx.intersection(sector_dfs[s].index)
        common_idx = common_idx.sort_values()
        
        # Build price and return panels
        ret_panel = pd.DataFrame(index=common_idx)
        for s in available_syms:
            close_s = sector_dfs[s]["close"].reindex(common_idx)
            ret_panel[s] = np.log(close_s / close_s.shift(1))
            
        n_obs = len(common_idx)
        w = self.window_size
        k = len(available_syms)
        
        feature_names = [
            "gnn_spectral_radius",
            "gnn_algebraic_connectivity",
            "gnn_graph_entropy",
            "gnn_bank_centrality",
            "gnn_it_centrality",
        ] + [f"gnn_embed_{d}" for d in range(self.embedding_dim)]
        
        res = pd.DataFrame(index=common_idx, columns=feature_names, dtype=float)
        
        ret_vals = ret_panel.fillna(0.0).values
        
        for t in range(w, n_obs):
            # Window strictly in [t - w, t) -> backward only!
            win = ret_vals[t - w : t]
            
            # Correlation matrix
            # Covariance
            mean_adj = win - np.mean(win, axis=0, keepdims=True)
            cov = (mean_adj.T @ mean_adj) / (w - 1.0)
            stds = np.sqrt(np.diag(cov)) + 1e-8
            corr = cov / np.outer(stds, stds)
            corr = np.nan_to_num(corr, nan=0.0)
            
            # Adjacency: positive off-diagonal correlations
            adj = np.maximum(corr, 0.0)
            np.fill_diagonal(adj, 0.0)
            
            # Spectral metrics
            spec_metrics = self._compute_graph_spectral_metrics(adj)
            for m_key, m_val in spec_metrics.items():
                res.iloc[t, res.columns.get_loc(m_key)] = m_val
                
            # Node features: [mean return, volatility, momentum (last return)]
            node_mean = np.mean(win, axis=0) * 252.0
            node_vol = stds * np.sqrt(252.0)
            node_mom = win[-1, :]
            node_features = np.column_stack([node_mean, node_vol, node_mom])
            
            # GCN Forward Embedding
            gcn_embed = self._gcn_forward(adj, node_features)
            for d in range(self.embedding_dim):
                col_name = f"gnn_embed_{d}"
                res.iloc[t, res.columns.get_loc(col_name)] = gcn_embed[d]
                
        return res

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback if passed dataframe with sector columns directly."""
        dict_dfs = {col: pd.DataFrame({"close": df[col]}, index=df.index) for col in df.columns}
        return self.extract_from_sector_prices(dict_dfs)
