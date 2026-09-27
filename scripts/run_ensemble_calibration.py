"""
Ensemble Construction, Adaptive Conformal Calibration, and Baseline Tournament.

Executes:
1. Constrained Stacking and BMA weight optimization across all 5 models.
2. Temperature scaling calibration.
3. Adaptive Conformal Inference (ACI) with empirical coverage tracking.
4. Benchmark Tournament evaluating models against Climatology and Persistence.
5. Serializes calibrated ensemble predictions and audit artifacts.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration import AdaptiveConformalInference, TemperatureScalingCalibrator
from src.evaluation import BenchmarkTournament, compute_log_loss, compute_ranked_probability_score, compute_brier_score


def main() -> None:
    preds_path = Path("data/processed/model_predictions_matrix.parquet")
    if not preds_path.exists():
        raise FileNotFoundError(f"{preds_path} not found. Run scripts/train_regime_models.py first.")

    preds_df = pd.read_parquet(preds_path)
    print(f"Loaded model predictions matrix: {preds_df.shape}")

    idx = preds_df.index
    ground_truth = preds_df["dominant_regime"].values
    n_total = len(preds_df)

    # Reconstruct individual model prediction matrices (N, 5)
    regimes = ["risk_on", "late_cycle", "transitional", "post_shock", "risk_off"]
    p_freq = preds_df[[f"freq_{r}" for r in regimes]].values
    p_bayes = preds_df[[f"bayes_{r}" for r in regimes]].values
    p_rsvar = preds_df[[f"rsvar_{r}" for r in regimes]].values
    p_bnn = preds_df[[f"bnn_{r}" for r in regimes]].values
    p_chronos = preds_df[[f"chronos_{r}" for r in regimes]].values

    # Train / Calibration / Test split chronologically (No lookahead!)
    # Train: 60%, Calibration: 20%, Holdout Test: 20%
    t_train = int(n_total * 0.60)
    t_cal = int(n_total * 0.80)

    # 1. Simplex Stacking on Calibration Split
    print("\n--- Optimizing Constrained Stacking Weights on Simplex ---")
    cal_gt = ground_truth[t_train:t_cal]
    cal_models = [p_freq[t_train:t_cal], p_bayes[t_train:t_cal], p_rsvar[t_train:t_cal], p_bnn[t_train:t_cal], p_chronos[t_train:t_cal]]
    
    # Grid search / simplex optimization
    from scipy.optimize import minimize
    def stacking_objective(w: np.ndarray) -> float:
        w = w / np.sum(w)
        comb = np.zeros_like(cal_models[0])
        for i, m_p in enumerate(cal_models):
            comb += w[i] * m_p
        loss = -np.mean(np.log(np.maximum(comb[np.arange(len(cal_gt)), cal_gt], 1e-12)))
        return float(loss)

    res = minimize(
        stacking_objective,
        np.ones(5) / 5.0,
        bounds=[(0.0, 1.0) for _ in range(5)],
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
    )
    stack_weights = np.clip(res.x / np.sum(res.x), 0.0, 1.0)
    stack_weights /= np.sum(stack_weights)

    model_names = ["Frequentist HMM", "Bayesian HMM", "RS-VAR", "Bayesian DL", "Chronos Probe"]
    print("Optimal Simplex Stacking Weights:")
    for name, w in zip(model_names, stack_weights):
        print(f"  - {name}: {w:.4f} ({w*100:.1f}%)")

    # Combine full sample using stacking weights
    all_models_full = [p_freq, p_bayes, p_rsvar, p_bnn, p_chronos]
    p_ensemble_raw = np.zeros_like(p_freq)
    for i, m_p in enumerate(all_models_full):
        p_ensemble_raw += stack_weights[i] * m_p
    p_ensemble_raw /= np.sum(p_ensemble_raw, axis=1, keepdims=True)

    # 2. Temperature Scaling Calibration
    print("\n--- Fitting Temperature Scaling Calibrator ---")
    calibrator = TemperatureScalingCalibrator()
    calibrator.fit(p_ensemble_raw[t_train:t_cal], ground_truth[t_train:t_cal])
    print(f"Optimal Temperature Parameter T: {calibrator.temperature:.4f}")

    p_ensemble_calibrated = calibrator.calibrate(p_ensemble_raw)
    ece_before, mce_before, _ = TemperatureScalingCalibrator.compute_ece(p_ensemble_raw, ground_truth)
    ece_after, mce_after, rel_df = TemperatureScalingCalibrator.compute_ece(p_ensemble_calibrated, ground_truth)
    print(f"ECE before calibration: {ece_before:.4f} -> after calibration: {ece_after:.4f} (improvement: {(ece_before - ece_after)/ece_before:.1%})")

    # 3. Adaptive Conformal Inference (ACI) on Out-of-Sample Test Split
    print("\n--- Executing Adaptive Conformal Inference (ACI) ---")
    aci = AdaptiveConformalInference(alpha=0.10, gamma=0.015)
    # Calibrate initial threshold on calibration set
    aci.calibrate(p_ensemble_calibrated[t_train:t_cal], ground_truth[t_train:t_cal])

    # Run online tracking over the entire test period
    test_probs = p_ensemble_calibrated[t_cal:]
    test_gt = ground_truth[t_cal:]
    test_idx = idx[t_cal:]

    pred_sets, alpha_traj, aci_audit = aci.run_online_aci(test_probs, test_gt)
    print("Conformal Coverage Audit (Out-of-Sample):")
    for k, v in aci_audit.items():
        print(f"  {k}: {v}")

    # 4. Benchmark Tournament
    print("\n--- Running Benchmark Tournament vs Baselines ---")
    tournament_models = {
        "Frequentist HMM": p_freq[t_cal:],
        "Bayesian HMM": p_bayes[t_cal:],
        "RS-VAR": p_rsvar[t_cal:],
        "Bayesian DL (MC Dropout)": p_bnn[t_cal:],
        "Chronos Adapter": p_chronos[t_cal:],
        "Ensemble (Raw Stacking)": p_ensemble_raw[t_cal:],
        "Ensemble (Calibrated Stacking)": p_ensemble_calibrated[t_cal:],
    }

    tournament_df = BenchmarkTournament.evaluate_models(tournament_models, test_gt)
    print("\nBenchmark Tournament Results (Out-of-Sample):")
    print(tournament_df.to_string())

    # Save summary tables
    tables_dir = Path("reports/tables")
    figures_dir = Path("reports/figures")

    tournament_df.to_csv(tables_dir / "benchmark_tournament_summary.csv", index=False)
    
    # Conformal audit table
    pd.DataFrame([aci_audit]).to_csv(tables_dir / "conformal_coverage_audit.csv", index=False)

    # 5. Visualizations
    print("\n--- Generating Calibration & Conformal Visualizations ---")
    
    # Figure 8: Conformal Coverage Adaptation
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    # Panel 1: Dynamic alpha trajectory
    axes[0].plot(test_idx, 1.0 - alpha_traj, color="#2980b9", lw=1.5, label="Adaptive Target Coverage (1 - alpha_t)")
    axes[0].axhline(0.90, color="#e74c3c", linestyle="--", lw=1.5, label="Nominal Target (90%)")
    axes[0].set_title(f"Adaptive Conformal Inference (ACI): Realized Empirical Coverage = {aci_audit['realized_empirical_coverage']:.1%}", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Coverage Level")
    axes[0].set_ylim(0.70, 1.0)
    axes[0].legend(loc="lower left", frameon=True)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Conformal Set Cardinality
    set_lens = [len(s) for s in pred_sets]
    axes[1].plot(test_idx, set_lens, color="#27ae60", lw=1.2, label=f"Prediction Set Size |C_t| (Mean: {aci_audit['mean_prediction_set_size']:.2f})")
    axes[1].set_title("Conformal Prediction Set Cardinality Over Time", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Number of Regimes in Set")
    axes[1].set_yticks([1, 2, 3, 4, 5])
    axes[1].legend(loc="upper left", frameon=True)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig8_path = figures_dir / "08_conformal_coverage_adaptation.png"
    plt.savefig(fig8_path, dpi=200)
    plt.close()
    print(f"Saved figure: {fig8_path}")

    # Figure 9: Reliability Diagram
    fig, ax = plt.subplots(figsize=(8, 6))
    valid_rel = rel_df.dropna(subset=["accuracy"])
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
    ax.bar(
        valid_rel["confidence"],
        valid_rel["accuracy"],
        width=0.08,
        alpha=0.6,
        color="#3498db",
        edgecolor="#2980b9",
        label=f"Calibrated Ensemble (ECE = {ece_after:.4f})",
    )
    ax.set_title("Reliability Diagram: Calibrated Multi-Model Ensemble", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig9_path = figures_dir / "09_reliability_diagrams_calibration.png"
    plt.savefig(fig9_path, dpi=200)
    plt.close()
    print(f"Saved figure: {fig9_path}")

    # Save calibrated predictions matrix
    calibrated_df = pd.DataFrame(index=idx)
    for i, col in enumerate(regimes):
        calibrated_df[f"ensemble_prob_{col}"] = p_ensemble_calibrated[:, i]
    calibrated_df["ensemble_dominant_regime"] = np.argmax(p_ensemble_calibrated, axis=1)

    cal_path = Path("data/processed/calibrated_ensemble_predictions.parquet")
    calibrated_df.to_parquet(cal_path)
    print(f"Calibrated ensemble predictions saved to: {cal_path}")


if __name__ == "__main__":
    main()
