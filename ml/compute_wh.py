"""Orchestrate energy/runtime ML predictors and downstream carbon and water metrics."""

from __future__ import annotations

import math
import os
from typing import Any

import joblib
import pandas as pd

from ml.ml_runtime import VideoRuntimePredictor
from ml.ml_wh import VideoEnergyPredictor
from ml.prediction_params import MlPredictionParams, RunMlInputs

MIN_WH = 2.0
MIN_RUN_TIME = 4.0


def emission_factor(
    country: str, wh: float, run_time: float
) -> tuple[float, float, float]:
    """Return embodied carbon (g), electricity carbon (g), and water (L)."""
    emission_factor_csv = pd.read_csv("./ml/data/carbone_kwh_country.csv", header=0)
    pue = 1.56
    water_usage = 0.35

    wh_w_pue = wh * pue
    try:
        country_factor = emission_factor_csv.loc[
            emission_factor_csv["country"] == country
        ]["Emission factor"].values[0]
    except (IndexError, KeyError):
        country_factor = 220.0
    gpu_embodied_co2 = 143.0
    gpu_lifetime_years = 3.0
    gpu_utilization = 0.75
    carbon_electricity = country_factor * (wh_w_pue / 1000)
    water_used = wh_w_pue / 1000 * water_usage

    seconds_in_3_years = 60 * 60 * 24 * 365.25 * gpu_lifetime_years
    carbon_embodied = (
        (run_time / seconds_in_3_years) / gpu_utilization * gpu_embodied_co2
    ) * 1000
    return carbon_embodied, carbon_electricity, water_used


def prepare_frames(frames: int, arch: str) -> int:
    """Normalize frame count for hybrid architectures."""
    if arch == "hybrid":
        return math.ceil(frames / 49)
    return frames


def _invalid_run_inputs(inp: RunMlInputs) -> bool:
    return (
        inp.steps <= 0
        or inp.res <= 0
        or inp.frames <= 0
        or inp.params <= 0
        or inp.fps <= 0
        or inp.duration <= 0
    )


def _format_invalid_msg(inp: RunMlInputs) -> str:
    return (
        f"Invalid input: steps={inp.steps}, res={inp.res}, frames={inp.frames}, "
        f"params={inp.params}. All must be > 0"
    )


def _ensure_predictors(
    arch: str,
) -> tuple[VideoEnergyPredictor, VideoRuntimePredictor]:
    wh_path = f"./ml/model/best_models_wh_{arch}.joblib"
    rt_path = f"./ml/model/best_models_run_time_{arch}.joblib"
    energy_predictor = VideoEnergyPredictor(data_file="./ml/data/prepared_data.csv")
    run_time_predictor = VideoRuntimePredictor(data_file="./ml/data/prepared_data.csv")
    if os.path.exists(wh_path):
        energy_predictor.best_models = joblib.load(wh_path)
    else:
        energy_predictor.train_all_architectures()
        joblib.dump(energy_predictor.best_models, wh_path)
    if os.path.exists(rt_path):
        run_time_predictor.best_models = joblib.load(rt_path)
    else:
        run_time_predictor.train_all_architectures()
        joblib.dump(run_time_predictor.best_models, rt_path)
    return energy_predictor, run_time_predictor


def _build_success_payload(
    country: str,
    pred_wh: dict[str, Any],
    pred_run_time: dict[str, Any],
) -> dict[str, Any]:
    total_emb, total_el, total_water = emission_factor(
        country, pred_wh["energy_wh"], pred_run_time["run_time_s"]
    )
    best_emb, best_el, best_water = emission_factor(
        country,
        max(0, pred_wh["energy_wh"] - pred_wh["margin_95_wh"]),
        max(0, pred_run_time["run_time_s"] - pred_run_time["margin_95_s"]),
    )
    worst_emb, worst_el, worst_water = emission_factor(
        country,
        pred_wh["energy_wh"] + pred_wh["margin_95_wh"],
        pred_run_time["run_time_s"] + pred_run_time["margin_95_s"],
    )
    return {
        "energy": {
            "value_wh": max(MIN_WH, round(pred_wh["energy_wh"], 2)),
            "uncertainty_wh": pred_wh["uncertainty_wh"],
            "margin_95_wh": pred_wh["margin_95_wh"],
            "best_case_wh": round(
                max(MIN_WH, pred_wh["energy_wh"] - pred_wh["margin_95_wh"]), 2
            ),
            "worst_case_wh": round(
                max(MIN_WH, pred_wh["energy_wh"] + pred_wh["margin_95_wh"]), 2
            ),
            "model": pred_wh["model"],
            "r2": pred_wh["r2_score"],
        },
        "run_time": {
            "value_s": max(MIN_RUN_TIME, round(pred_run_time["run_time_s"], 2)),
            "value_min": max(
                MIN_RUN_TIME / 60, round(pred_run_time["run_time_min"], 2)
            ),
            "uncertainty_s": pred_run_time["uncertainty_s"],
            "margin_95_s": pred_run_time["margin_95_s"],
            "best_case_s": round(
                max(
                    MIN_RUN_TIME,
                    pred_run_time["run_time_s"] - pred_run_time["margin_95_s"],
                ),
                2,
            ),
            "worst_case_s": round(
                max(
                    MIN_RUN_TIME,
                    pred_run_time["run_time_s"] + pred_run_time["margin_95_s"],
                ),
                2,
            ),
            "model": pred_run_time["model"],
            "r2": pred_run_time["r2_score"],
        },
        "carbon": {
            "value_gco2e": round(max(0.01, total_emb + total_el), 2),
            "best_case_gco2e": round(max(0.01, best_emb + best_el), 2),
            "worst_case_gco2e": round(max(0.01, worst_emb + worst_el), 2),
            "g_co2_embodied": round(max(0.01, total_emb), 2),
            "g_co2_electricity": round(max(0.01, total_el), 2),
        },
        "water_used": {
            "value_water_used": round(max(0.01, total_water), 2),
            "best_case_water_used": round(max(0.01, best_water), 2),
            "worst_case_water_used": round(max(0.01, worst_water), 2),
        },
    }


def run_ml(inp: RunMlInputs) -> dict[str, Any]:
    """Predict energy and run_time with uncertainties and derived sustainability metrics."""
    if _invalid_run_inputs(inp):
        return {"error": _format_invalid_msg(inp)}

    energy_predictor, run_time_predictor = _ensure_predictors(inp.arch)
    frames = prepare_frames(int(inp.frames), inp.arch)
    bundle = MlPredictionParams(
        inp.arch,
        inp.steps,
        inp.res,
        float(frames),
        inp.fps,
        inp.duration,
        inp.params,
        inp.input_type,
    )
    pred_wh = energy_predictor.predict(bundle)
    pred_run_time = run_time_predictor.predict(bundle)

    if "error" in pred_wh:
        return {"error": f"Energy prediction failed: {pred_wh['error']}"}
    if "error" in pred_run_time:
        return {"error": f"run_time prediction failed: {pred_run_time['error']}"}

    return _build_success_payload(inp.country, pred_wh, pred_run_time)
