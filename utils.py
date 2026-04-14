"""Shared helpers for YAML, CSV export, and configuration validation."""

from __future__ import annotations

import os
from typing import Any

import polars as pl
import yaml


def load_yaml(file: str) -> dict[str, Any]:
    """Load a YAML file and return a dict (empty dict if file is empty)."""
    with open(file, encoding="utf-8") as file_content:
        yaml_data = yaml.safe_load(file_content)
    if yaml_data is None:
        return {}
    if not isinstance(yaml_data, dict):
        raise ValueError("YAML root must be a mapping")
    return yaml_data


def dump_yaml(data: object, filename: str) -> None:
    """Write a Python object to a YAML file."""
    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(data, file)


def dump_csv_polars(data: pl.DataFrame, filename: str) -> None:
    """Write a Polars DataFrame to CSV, creating parent directories if needed."""
    foldername = "/".join(filename.split("/")[:-1])
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    data.write_csv(filename)


REQUIRED_CONFIG_KEYS = (
    "model",
    "duration",
    "fps",
    "resolution_height",
    "resolution_witdh",
    "country",
    "denoising_steps",
)


def validate_config(cfg: dict[str, Any]) -> None:
    """Ensure all required keys are present after YAML + CLI merge."""
    for key in REQUIRED_CONFIG_KEYS:
        if key not in cfg or cfg[key] is None:
            raise ValueError(f"Missing required configuration: {key}")
