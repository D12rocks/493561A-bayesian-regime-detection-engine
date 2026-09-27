"""
Train Regime Detection Models and Generate MCMC Diagnostics & Duration Reports.

Executes fitting of all 5 regime detection models:
1. Frequentist Gaussian HMM
2. Bayesian HMM with MCMC Gibbs FFBS
3. Regime-Switching VAR with Hamilton Filter
4. Bayesian Deep Learning with MC Dropout & Deep Ensemble
5. Chronos Foundation Model Representation Probe
Also computes empirical regime duration statistics and tests the geometric duration null.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models import (
    FrequentistHMM,
    BayesianHMM,
    RegimeSwitchingVAR,
    BayesianDeepLearningModel,
    ChronosRegimeAdapter,
    RegimeDurationAnalyzer,
    RegimeLabel,
)


def main() -> None:
    features_path = Path("data/processed/features_matrix.parquet")
    if not features_path.exists():
        raise FileNotFoundError(f"Feature matrix {features_path} not found. Run scripts/build_features.py first.")

    features_df = pd.read_parquet(features_path)
    print(f"Loaded feature matrix: shape {features_df.shape}")

    # Core modeling feature subset: return, EWMA vol, trend distance, VIX, breadth
    model_features = [
        "nifty_ret_1d",
        "nifty_vol_ewma_21d",
        "nifty_dist_sma50",
        "vix_level",
        "breadth_midcap_ret_21d",
    ]
    X_train = features_df[model_features].copy()
    print(f"Training features: {model_features}")

    # 1. Frequentist HMM
    print("\n--- Training Frequentist HMM ---")
    freq_hmm = FrequentistHMM(n_regimes=5, n_iter=100, random_seed=42)
    freq_hmm.fit(X_train)
    p_freq = freq_hmm.predict_proba(X_train)
    freq_diag = freq_hmm.diagnostics()
    print(f"Frequentist HMM Log-Likelihood: {freq_diag['log_likelihood']:.2f}, AIC: {freq_diag['aic']:.2f}, BIC: {freq_diag['bic']:.2f}")

    # 2. Bayesian HMM (MCMC)
    print("\n--- Training Bayesian HMM with MCMC Gibbs FFBS ---")
    bayes_hmm = BayesianHMM(
        n_regimes=5,
        n_iter=120,
        burn_in=40,
        n_chains=2,
        sticky_kappa=8.0,
        random_seed=42,
    )
    bayes_hmm.fit(X_train)
    p_bayes, epistemic_b, aleatoric_b = bayes_hmm.predict_proba_posterior(X_train)
    b_diag = bayes_hmm.diagnostics()
    print(f"Bayesian HMM Gelman-Rubin R-hat: {b_diag['gelman_rubin_r_hat']}")
    print(f"Bayesian HMM ESS: {b_diag['effective_sample_size_ess']}")
    print(f"R-hat Convergence Rate: {b_diag['r_hat_converged_pct']:.1f}%")

    # 3. Regime-Switching VAR
    print("\n--- Training Regime-Switching VAR ---")
    rs_var = RegimeSwitchingVAR(n_regimes=5, lags=1, n_iter=30, random_seed=42)
    rs_var.fit(X_train)
    p_rsvar = rs_var.predict_proba(X_train)
    rs_diag = rs_var.diagnostics()
    print(f"RS-VAR Log-Likelihood: {rs_diag['log_likelihood']:.2f}")

    # 4. Bayesian Deep Learning (MC Dropout)
    print("\n--- Training Bayesian Deep Learning (MC Dropout & Deep Ensemble) ---")
    bnn = BayesianDeepLearningModel(
        n_regimes=5,
        n_mc_samples=50,
        n_ensemble=3,
        epochs=80,
        random_seed=42,
    )
    bnn.fit(X_train)
    p_bnn, epistemic_bnn, aleatoric_bnn = bnn.predict_proba_with_uncertainty(X_train)
    bnn_diag = bnn.diagnostics()
    print(f"Bayesian DL Final Loss: {bnn_diag['final_loss']:.4f}")

    # 5. Chronos Foundation Model Probing Adapter
    print("\n--- Training Chronos Foundation Model Probing Adapter ---")
    chronos = ChronosRegimeAdapter(context_length=64, embedding_dim=16, random_seed=42)
    chronos.fit(X_train)
    p_chronos = chronos.predict_proba(X_train)
    chronos_diag = chronos.diagnostics()
    print(f"Chronos Probe Accuracy: {chronos_diag['probing_classifier_accuracy']:.4f}")
    print(f"Regime Separation: {chronos_diag['regime_separation']}")

    # 6. Regime Duration Analysis
    print("\n--- Analyzing Regime Durations and Geometric Null ---")
    duration_analyzer = RegimeDurationAnalyzer()
    map_regimes = np.argmax(p_bayes, axis=1)
    trans_mat_bayes = np.mean(bayes_hmm.chain_transmats[0], axis=0)
    duration_df = duration_analyzer.analyze_sequence(map_regimes, trans_mat_bayes)
    print(duration_df.to_string())

    # Save summary tables
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    duration_df.to_csv(tables_dir / "regime_duration_analysis.csv", index=False)

    # Model diagnostics summary table
    summary_data = [
        {
            "model": "Frequentist HMM",
            "type": "Probabilistic Graphical Model",
            "objective": "Maximum Likelihood (EM)",
            "key_metric_1": f"AIC: {freq_diag['aic']:.1f}",
            "key_metric_2": f"BIC: {freq_diag['bic']:.1f}",
            "uncertainty_type": "Point Posterior",
            "hardware_accel": "CPU Native",
        },
        {
            "model": "Bayesian HMM",
            "type": "Full Bayesian MCMC (Sticky Gibbs)",
            "objective": "Posterior Sampling (FFBS)",
            "key_metric_1": f"R-hat Conv: {b_diag['r_hat_converged_pct']:.0f}%",
            "key_metric_2": f"Sticky Kappa: {b_diag['sticky_kappa']}",
            "uncertainty_type": "Epistemic + Aleatoric",
            "hardware_accel": "CPU Native",
        },
        {
            "model": "RS-VAR",
            "type": "Hamilton Filter / Kim Smoother",
            "objective": "State-Dependent VAR(1)",
            "key_metric_1": f"Log-Lik: {rs_diag['log_likelihood']:.1f}",
            "key_metric_2": "Lags: 1",
            "uncertainty_type": "Filtered / Smoothed",
            "hardware_accel": "CPU Native",
        },
        {
            "model": "Bayesian DL (MC Dropout)",
            "type": "Neural Network Ensemble",
            "objective": "Variational MC Dropout",
            "key_metric_1": f"Loss: {bnn_diag['final_loss']:.4f}",
            "key_metric_2": f"MC Passes: {bnn_diag['n_mc_samples']}",
            "uncertainty_type": "Mutual Information",
            "hardware_accel": "CPU / Metal",
        },
        {
            "model": "Chronos Adapter",
            "type": "Foundation Model Representation Probe",
            "objective": "Zero-shot Latent Projection",
            "key_metric_1": f"Probe Acc: {chronos_diag['probing_classifier_accuracy']:.2%}",
            "key_metric_2": f"Silhouette: {chronos_diag['regime_separation']['silhouette_score']:.3f}",
            "uncertainty_type": "Softmax Entropy",
            "hardware_accel": "CPU / Transformers",
        },
    ]
    pd.DataFrame(summary_data).to_csv(tables_dir / "model_diagnostics_summary.csv", index=False)
    print(f"\nSaved diagnostics table to: {tables_dir / 'model_diagnostics_summary.csv'}")

    # Plot 1: Bayesian Posterior Probabilities and Uncertainty Decomposition
    print("\n--- Generating Visualizations ---")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    idx = X_train.index

    # Panel 1: Stacked Bayesian Regime Probabilities
    regime_colors = ["#2ecc71", "#f39c12", "#3498db", "#9b59b6", "#e74c3c"]
    axes[0].stackplot(
        idx,
        p_bayes[:, 0], p_bayes[:, 1], p_bayes[:, 2], p_bayes[:, 3], p_bayes[:, 4],
        labels=["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"],
        colors=regime_colors,
        alpha=0.85,
    )
    axes[0].set_title("Bayesian HMM: Posterior Regime Probabilities (2009–2024)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Probability")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper left", ncol=5, frameon=True)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Epistemic vs Aleatoric Uncertainty
    axes[1].plot(idx, epistemic_b, label="Epistemic Uncertainty (Parameter Dispersion)", color="#e67e22", lw=1.2)
    axes[1].plot(idx, aleatoric_b, label="Aleatoric Uncertainty (Data Noise)", color="#8e44ad", lw=1.2, alpha=0.7)
    axes[1].set_title("Uncertainty Decomposition: Epistemic vs Aleatoric Entropy", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Entropy (Nats)")
    axes[1].legend(loc="upper left", frameon=True)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: NIFTY 50 Colored by Dominant Regime
    p_nifty = features_df["nifty_ret_1d"].reindex(idx).cumsum()
    axes[2].plot(idx, p_nifty, color="#2c3e50", lw=1.2, label="Cumulative NIFTY 50 Log Return")
    axes[2].set_title("Cumulative Return Path Annotated with Regime States", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Cumulative Return")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    fig_path = figures_dir / "06_bayesian_regime_posteriors.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved figure: {fig_path}")

    # Plot 2: Empirical Regime Durations vs Geometric Null
    spells = duration_analyzer.extract_durations(map_regimes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot of durations
    box_data = [spells.get(i, [1]) for i in range(5)]
    labels = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]
    axes[0].boxplot(box_data, tick_labels=labels, patch_artist=True)
    axes[0].set_title("Empirical Regime Spell Duration (Trading Days)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Days in Regime")
    axes[0].grid(True, alpha=0.3)

    # Duration comparison table plot
    axes[1].axis("off")
    cols = ["Regime", "Mean Days", "Max Days", "p-value (Geom Null)", "Duration Memory"]
    tbl_data = duration_df[["regime", "mean_duration_days", "max_duration_days", "p_value", "duration_dependence"]].values
    table = axes[1].table(
        cellText=tbl_data,
        colLabels=cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    axes[1].set_title("Geometric Duration Null Hypothesis Test", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig_dur_path = figures_dir / "07_regime_duration_distributions.png"
    plt.savefig(fig_dur_path, dpi=200)
    plt.close()
    print(f"Saved figure: {fig_dur_path}")

    # Save model predictions to processed for downstream ensembling and backtesting
    preds_df = pd.DataFrame(index=idx)
    for i, col in enumerate(["risk_on", "late_cycle", "transitional", "post_shock", "risk_off"]):
        preds_df[f"freq_{col}"] = p_freq[:, i]
        preds_df[f"bayes_{col}"] = p_bayes[:, i]
        preds_df[f"rsvar_{col}"] = p_rsvar[:, i]
        preds_df[f"bnn_{col}"] = p_bnn[:, i]
        preds_df[f"chronos_{col}"] = p_chronos[:, i]

    preds_df["epistemic_entropy"] = epistemic_b
    preds_df["aleatoric_entropy"] = aleatoric_b
    preds_df["dominant_regime"] = map_regimes

    preds_file = Path("data/processed/model_predictions_matrix.parquet")
    preds_df.to_parquet(preds_file)
    print(f"Model predictions matrix saved to: {preds_file} (shape: {preds_df.shape})")


if __name__ == "__main__":
    main()
