"""Fit and score one regressor candidate on a train/test fold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from ml.training_fold import TrainingFold


def make_training_fold(
    df_arch: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> TrainingFold:
    """Split, scale, and package matrices for a regression target column."""
    features = df_arch[feature_cols]
    target = df_arch[target_col]
    feat_train, feat_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )
    scaler = StandardScaler()
    feat_train_scaled = scaler.fit_transform(feat_train)
    feat_test_scaled = scaler.transform(feat_test)
    return TrainingFold(feat_train_scaled, y_train, feat_test_scaled, y_test, scaler)


def evaluate_regressor_candidate(
    model_name: str,
    model: Any,
    fold: TrainingFold,
) -> dict[str, Any] | None:
    """Fit one regressor and return metrics or None on failure."""
    try:
        model.fit(fold.feat_train_scaled, fold.y_train)
        y_pred = model.predict(fold.feat_test_scaled)
        y_pred = np.clip(y_pred, 0, None)
        mae = mean_absolute_error(fold.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(fold.y_test, y_pred))
        r2 = r2_score(fold.y_test, y_pred)
        cv_scores = cross_val_score(
            model, fold.feat_train_scaled, fold.y_train, cv=5, scoring="r2"
        )
        return {
            "model": model_name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "cv_r2": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "n_test": len(fold.feat_test_scaled),
            "model_obj": model,
            "scaler": fold.scaler,
        }
    except (ValueError, ArithmeticError, MemoryError, TypeError):
        return None


def collect_architecture_results(
    models: dict[str, Any],
    fold: TrainingFold,
) -> list[dict[str, Any]]:
    """Fit each candidate model and collect non-failed results."""
    arch_results = []
    for model_name, model in models.items():
        row = evaluate_regressor_candidate(model_name, model, fold)
        if row:
            arch_results.append(row)
    return arch_results
