"""
Video Generation run_time Predictor
Tests multiple regression models for each architecture (dit, hybrid, unet)
Predicts: run_time (seconds)
Models: LinearRegression, Ridge, SVR, ExtraTrees, RandomForest, GradientBoosting
"""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from ml.base_predictor import BaseSklearnArchitecturePredictor
from ml.prediction_params import FEATURE_ORDER, MlPredictionParams, feature_value_list
from ml.regressor_eval import collect_architecture_results, make_training_fold


def _pick_best_runtime_model(
    arch_results: list[dict[str, Any]], n_samples: int
) -> dict[str, Any] | None:
    """Select best runtime regressor using R2 / stability rules."""
    if not arch_results:
        return None
    if n_samples >= 100:
        tree_models = [
            r
            for r in arch_results
            if r["model"] in ("ExtraTrees", "RandomForest", "GradientBoosting")
        ]
        if tree_models:
            return sorted(tree_models, key=lambda x: x["r2"], reverse=True)[0]
        return sorted(arch_results, key=lambda x: x["r2"], reverse=True)[0]
    stable_models = [r for r in arch_results if (r["r2"] - r["cv_r2"]) <= 0.08]
    if stable_models:
        return sorted(stable_models, key=lambda x: x["mae"])[0]
    return sorted(arch_results, key=lambda x: (x["r2"] - x["cv_r2"], x["mae"]))[0]


class VideoRuntimePredictor(BaseSklearnArchitecturePredictor):
    """Trains and selects regressors for runtime (seconds) per architecture."""

    def _load_runtime_arch_frame(self, arch_name: str) -> pd.DataFrame | None:
        """Like ``_arch_dataframe`` but drop NaNs for runtime targets."""
        raw = self._arch_dataframe(arch_name)
        if raw is None:
            return None
        return raw.dropna().reset_index(drop=True)

    def train_architecture(self, arch_name: str) -> None:
        """Train all models on one architecture."""
        df_arch = self._load_runtime_arch_frame(arch_name)
        if df_arch is None:
            self.results[arch_name] = {}
            return

        n_samples = len(df_arch)
        fold = make_training_fold(df_arch, self._require_feature_columns(), "run_time")
        models = self.get_models()
        arch_results = collect_architecture_results(models, fold)

        self.results[arch_name] = arch_results
        best = _pick_best_runtime_model(arch_results, n_samples)
        if best:
            self.best_models[arch_name] = best

    def save_models(self) -> None:
        """Persist per-architecture best model and scaler."""
        for arch, best in self.best_models.items():
            model_path = f"./ml/model/best_model_run_time_{arch}.joblib"
            scaler_path = f"./ml/model/scaler_run_time_{arch}.joblib"
            joblib.dump(best["model_obj"], model_path)
            joblib.dump(best["scaler"], scaler_path)

    def predict(self, params_in: MlPredictionParams) -> dict[str, Any]:
        """Predict runtime (seconds) with uncertainty for one architecture."""
        arch = params_in.arch
        try:
            model = joblib.load(f"./ml/model/best_model_run_time_{arch}.joblib")
            scaler = joblib.load(f"./ml/model/scaler_run_time_{arch}.joblib")
            best = self.best_models[arch]

            vec = feature_value_list(params_in, params_in.res)
            row = pd.DataFrame([vec], columns=list(FEATURE_ORDER))
            scaled = scaler.transform(row)
            pred = float(model.predict(scaled)[0])
            pred = max(0.0, pred)

            return {
                "run_time_s": round(pred, 2),
                "run_time_min": round(pred / 60, 2),
                "uncertainty_s": round(best["mae"], 2),
                "margin_95_s": round(1.96 * best["rmse"], 2),
                "r2_score": round(best["r2"], 3),
                "model": best["model"],
            }
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
            return {"error": str(exc)}
