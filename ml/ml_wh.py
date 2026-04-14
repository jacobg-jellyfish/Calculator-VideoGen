"""
Video Generation Energy (Wh) Predictor
Tests multiple regression models for each architecture (dit, hybrid, unet)
Models: LinearRegression, Ridge, SVR, ExtraTrees, RandomForest, GradientBoosting
"""

from __future__ import annotations

import logging
from typing import Any

import joblib
import pandas as pd

from ml.base_predictor import BaseSklearnArchitecturePredictor
from ml.prediction_params import FEATURE_ORDER, MlPredictionParams, feature_value_list
from ml.regressor_eval import collect_architecture_results, make_training_fold

logger = logging.getLogger(__name__)

RES_FACTOR_HYBRID = 0.000045
RES_FACTOR_UNET = 0.000012
REFS_RES = {"hybrid": 345600, "unet": 589824}


class VideoEnergyPredictor(BaseSklearnArchitecturePredictor):
    """Trains and selects regressors for energy (Wh) per architecture."""

    def train_architecture(self, arch_name: str) -> None:
        """Train all models on one architecture."""
        df_arch = self._arch_dataframe(arch_name)
        if df_arch is None:
            self.results[arch_name] = {}
            return

        n_samples = len(df_arch)
        fold = make_training_fold(df_arch, self._require_feature_columns(), "Wh")
        models = self.get_models()
        arch_results = collect_architecture_results(models, fold)

        self.results[arch_name] = arch_results

        if n_samples < 70:
            best = next(r for r in arch_results if r["model"] == "Ridge")
        else:
            best = sorted(arch_results, key=lambda x: x["r2"], reverse=True)[0]
        self.best_models[arch_name] = best

    def save_models(self) -> None:
        """Save best models."""
        for arch, best in self.best_models.items():
            model_path = f"./ml/model/best_model_wh_{arch}.joblib"
            scaler_path = f"./ml/model/scaler_wh_{arch}.joblib"
            joblib.dump(best["model_obj"], model_path)
            joblib.dump(best["scaler"], scaler_path)

    def predict(self, params_in: MlPredictionParams) -> dict[str, Any]:
        """Predict energy (Wh) with uncertainty for one architecture."""
        arch = params_in.arch
        res = params_in.res
        try:
            model = joblib.load(f"./ml/model/best_model_wh_{arch}.joblib")
            scaler = joblib.load(f"./ml/model/scaler_wh_{arch}.joblib")
            best = self.best_models[arch]
            res_arch = res if arch == "dit" else REFS_RES[arch]
            vec = feature_value_list(params_in, res_arch)
            row = pd.DataFrame([vec], columns=list(FEATURE_ORDER))
            scaled = scaler.transform(row)
            pred = float(model.predict(scaled)[0])
            pred = max(0.0, pred)
            adj = _resolution_energy_adjustment(arch, res, res_arch, params_in.params)
            logger.debug("resolution energy adjustment = %s", adj)
            pred += adj
            return {
                "energy_wh": round(pred, 2),
                "uncertainty_wh": round(best["mae"], 2),
                "margin_95_wh": round(1.96 * best["rmse"], 2),
                "r2_score": round(best["r2"], 3),
                "model": best["model"],
            }
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
            return {"error": str(exc)}


def _resolution_energy_adjustment(
    arch: str, res: float, res_arch: float, params: float
) -> float:
    """Extra energy term for hybrid/unet resolution deltas."""
    if arch == "hybrid":
        return (res - res_arch) * RES_FACTOR_HYBRID
    if arch == "unet":
        return (res - res_arch) * RES_FACTOR_UNET * (params / 1.5)
    return 0.0
