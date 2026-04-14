"""Shared sklearn training utilities for architecture-specific regressors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR


class BaseSklearnArchitecturePredictor(ABC):
    """Common CSV loading, feature prep, and model zoo for per-architecture training."""

    base_features = ["steps", "res", "frames", "params", "duration", "fps"]

    @abstractmethod
    def train_architecture(self, arch_name: str) -> None:
        """Fit and score all candidate models for one architecture label."""

    @abstractmethod
    def save_models(self) -> None:
        """Persist chosen models and scalers for each architecture."""

    def __init__(self, data_file: str = "prepared_data.csv") -> None:
        self.data_file = data_file
        self.feature_cols: list[str] | None = None
        self.results: dict[str, Any] = {}
        self.best_models: dict[str, Any] = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot input type and align feature columns."""
        df_prep = df.copy()
        if "Input type" in df_prep.columns:
            input_dummies = pd.get_dummies(df_prep["Input type"], prefix="input")
            df_prep = pd.concat([df_prep, input_dummies], axis=1)
            input_cols = sorted(list(input_dummies.columns))
            if self.feature_cols is None:
                self.feature_cols = self.base_features + input_cols
            for col in input_cols:
                if col not in df_prep.columns:
                    df_prep[col] = 0
        return df_prep

    def get_models(self) -> dict[str, Any]:
        """Return dict of regression models to test."""
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "SVR_rbf": SVR(kernel="rbf", C=100, epsilon=0.1),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            ),
        }

    def train_all_architectures(self) -> None:
        """Train all models for all architectures."""
        df = pd.read_csv(self.data_file)
        df = self.prepare_features(df)
        for arch in sorted(df["architecture"].unique()):
            self.train_architecture(arch)
        self.save_models()

    def _require_feature_columns(self) -> list[str]:
        """Return feature column names after ``prepare_features`` has run."""
        cols = self.feature_cols
        if cols is None:
            raise RuntimeError("feature_cols is unset; call prepare_features first")
        return cols

    def _arch_dataframe(self, arch_name: str) -> pd.DataFrame | None:
        """Rows for one architecture, or None if fewer than five samples."""
        df = pd.read_csv(self.data_file)
        df = self.prepare_features(df)
        df_arch = df[df["architecture"] == arch_name].copy()
        if len(df_arch) < 5:
            return None
        if arch_name == "hybrid":
            df_arch = df_arch.copy()
            df_arch["frames"] = np.ceil(df_arch["frames"] / 49)
        return df_arch
