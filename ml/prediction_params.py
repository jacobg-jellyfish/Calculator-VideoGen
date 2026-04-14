"""Shared parameter bundle for ML predictors."""

from __future__ import annotations

from typing import NamedTuple


class MlPredictionParams(NamedTuple):
    """Inputs passed to energy and runtime regression predictors."""

    arch: str
    steps: float
    res: float
    frames: float
    fps: int
    duration: int
    params: float
    input_type: str


FEATURE_ORDER = (
    "steps",
    "res",
    "frames",
    "params",
    "duration",
    "fps",
    "input_image",
    "input_text",
)


def input_modality_bits(input_type: str) -> tuple[int, int]:
    """Return ``(input_image, input_text)`` flags for the given modality string."""
    if input_type.lower() == "image":
        return 1, 0
    return 0, 1


def feature_value_list(params_in: MlPredictionParams, res_column: float) -> list[float]:
    """Eight regression inputs; ``res_column`` is reference resolution for the architecture."""
    input_image, input_text = input_modality_bits(params_in.input_type)
    return [
        float(params_in.steps),
        float(res_column),
        float(params_in.frames),
        float(params_in.params),
        float(params_in.duration),
        float(params_in.fps),
        float(input_image),
        float(input_text),
    ]


class RunMlInputs(NamedTuple):
    """Full inputs for the compute_wh orchestration pipeline."""

    steps: float
    res: float
    frames: float
    fps: int
    duration: int
    params: float
    arch: str
    input_type: str
    country: str
