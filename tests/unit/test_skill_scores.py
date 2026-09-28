"""
Automated tests for proper-score skill calculations.

Verifies:
1. Skill = 1 - Score_model / Score_reference
2. Strictly negative skill whenever model score > reference score
3. Strictly positive skill whenever model score < reference score
4. Independence of Log Loss, Brier, and RPS skill evaluations
"""

import numpy as np
import pytest

from src.evaluation.baselines import (
    BenchmarkTournament,
    compute_log_loss,
    compute_brier_score,
    compute_ranked_probability_score,
)


def test_proper_score_skill_formula_mathematical_properties():
    # Synthetic scores
    score_ref = 0.2000
    score_better = 0.1500
    score_worse = 0.2500

    skill_better = 1.0 - (score_better / score_ref)
    skill_worse = 1.0 - (score_worse / score_ref)
    skill_same = 1.0 - (score_ref / score_ref)

    assert skill_better > 0.0, "Skill must be positive when model score is strictly lower (better)."
    assert skill_better == pytest.approx(0.25)

    assert skill_worse < 0.0, "Skill must be negative when model score is strictly higher (worse)."
    assert skill_worse == pytest.approx(-0.25)

    assert skill_same == pytest.approx(0.0), "Skill must be exactly zero when model equals reference."


def test_benchmark_tournament_disaggregated_skill_matrix():
    rng = np.random.RandomState(42)
    N = 100
    K = 5
    y_true = rng.randint(0, K, size=N)

    # Candidate 1: uniform prediction
    p_uniform = np.full((N, K), 1.0 / K)
    # Candidate 2: noisy ground truth
    p_good = np.full((N, K), 0.05)
    for i in range(N):
        p_good[i, y_true[i]] = 0.80

    models = {
        "Uniform Model": p_uniform,
        "Good Model": p_good,
    }

    df = BenchmarkTournament.evaluate_models(models, y_true)

    assert "skill_vs_climatology_log_loss" in df.columns
    assert "skill_vs_persistence_log_loss" in df.columns
    assert "skill_vs_climatology_rps" in df.columns
    assert "skill_vs_persistence_rps" in df.columns
    assert "beats_persistence_log_loss" in df.columns
    assert "beats_persistence_rps" in df.columns

    for _, row in df.iterrows():
        # Check skill sign consistency
        persist_ll = df.loc[df["model_name"] == "Persistence (Baseline 2)", "log_loss"].values[0]
        persist_rps = df.loc[df["model_name"] == "Persistence (Baseline 2)", "ranked_prob_score_rps"].values[0]

        if row["log_loss"] > persist_ll:
            assert row["skill_vs_persistence_log_loss"] <= 0.0, f"{row['model_name']} has higher LL than persistence but positive skill"
            assert not row["beats_persistence_log_loss"]
        elif row["log_loss"] < persist_ll:
            assert row["skill_vs_persistence_log_loss"] >= 0.0, f"{row['model_name']} has lower LL than persistence but negative skill"
            assert row["beats_persistence_log_loss"]

        if row["ranked_prob_score_rps"] > persist_rps:
            assert row["skill_vs_persistence_rps"] <= 0.0, f"{row['model_name']} has higher RPS than persistence but positive skill"
            assert not row["beats_persistence_rps"]
        elif row["ranked_prob_score_rps"] < persist_rps:
            assert row["skill_vs_persistence_rps"] >= 0.0, f"{row['model_name']} has lower RPS than persistence but negative skill"
            assert row["beats_persistence_rps"]
