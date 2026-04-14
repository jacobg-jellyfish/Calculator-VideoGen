"""CLI entry for the video generation environmental impact calculator."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ml.compute_wh import run_ml
from ml.prediction_params import RunMlInputs
from utils import load_yaml, validate_config


def get_model_archi(model: str) -> dict[str, Any]:
    """Returns architecture and parameters for a given model."""

    model_configs = {
        # --- UNet models ---
        "AnimateDiff": {"arch": "unet", "params": 0.9},
        "Stable Video Diffusion": {"arch": "unet", "params": 1.5},
        "Pika 1.0": {"arch": "unet", "params": 1.5},
        "ModelScopeT2V": {"arch": "unet", "params": 1.7},
        "Lumiere": {"arch": "unet", "params": 5.0},
        "MagicVideo-V2": {"arch": "unet", "params": 1.5},
        # --- DiT models ---
        "Sora": {"arch": "dit", "params": 10.0},
        "WAN2.1-T2V-1.3B": {"arch": "dit", "params": 1.3},
        "WAN2.1-T2V-14B": {"arch": "dit", "params": 14.0},
        "Mochi 1": {"arch": "dit", "params": 10.0},
        "ContentV": {"arch": "dit", "params": 8.0},
        "VEO": {"arch": "dit", "params": 10.0},
        "Latte-XL": {"arch": "dit", "params": 0.67},
        # --- Hybrid (Transformer + 3D VAE) ---
        "CogVideoX-5B": {"arch": "hybrid", "params": 5.0},
        "CogVideoX-2B": {"arch": "hybrid", "params": 2.0},
    }

    if model in model_configs:
        return model_configs[model]
    return {"arch": "error", "params": 0}


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    """Register config path and output mode arguments."""
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="YAML config; CLI flags override file values",
    )
    parser.add_argument(
        "--output",
        choices=("human", "json"),
        default="human",
        help="human: readable text; json: one minified JSON line on stdout",
    )


def _add_param_args(parser: argparse.ArgumentParser) -> None:
    """Register model and run-parameter CLI options."""
    parser.add_argument("--model", default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument(
        "--resolution-height", type=int, dest="resolution_height", default=None
    )
    parser.add_argument(
        "--resolution-witdh", type=int, dest="resolution_witdh", default=None
    )
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument(
        "--denoising-steps", type=int, dest="denoising_steps", default=None
    )
    parser.add_argument(
        "--input-type", choices=("text", "image"), dest="input_type", default=None
    )
    parser.add_argument("--country", default=None)


def parse_args() -> argparse.Namespace:
    """Parse argv and return the argparse namespace."""
    parser = argparse.ArgumentParser(
        description="Video generation environmental impact calculator."
    )
    _add_io_args(parser)
    _add_param_args(parser)
    return parser.parse_args()


def merge_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Merge optional YAML config with CLI overrides."""
    cfg: dict[str, Any] = {}
    if args.config:
        cfg.update(load_yaml(args.config))
    overrides = (
        ("model", args.model),
        ("duration", args.duration),
        ("resolution_height", args.resolution_height),
        ("resolution_witdh", args.resolution_witdh),
        ("fps", args.fps),
        ("denoising_steps", args.denoising_steps),
        ("input_type", args.input_type),
        ("country", args.country),
    )
    for key, val in overrides:
        if val is not None:
            cfg[key] = val
    if cfg.get("input_type") is None:
        cfg["input_type"] = "text"
    return cfg


def build_result_dict(
    cfg: dict[str, Any],
    steps: int,
    total_frames: int,
    predictions: dict[str, Any],
) -> dict[str, Any]:
    """Shape API output with inputs and prediction blocks."""
    return {
        "inputs": {
            "model": cfg["model"],
            "steps": steps,
            "resolution": f"{cfg['resolution_height']}x{cfg['resolution_witdh']}",
            "frames": total_frames,
        },
        "predictions": {
            "energy": predictions["energy"],
            "run_time": predictions["run_time"],
            "carbon": predictions["carbon"],
            "water_used": predictions["water_used"],
        },
    }


def _print_human_inputs(
    cfg: dict[str, Any],
    steps: int,
    total_frames: int,
    model_type: str,
    params: float,
) -> None:
    """Print the human-readable inputs section."""
    print("\n📥 INPUTS:")
    print(f"  Model: {cfg['model']} ({model_type}, {params}B params)")
    print(f"  Steps: {steps}")
    print(f"  Resolution: {cfg['resolution_height']}x{cfg['resolution_witdh']}")
    print(f"  Frames: {total_frames}\n")


def _print_energy_and_runtime(predictions: dict[str, Any]) -> None:
    """Print energy and runtime lines for human output."""
    print("📊 RESULTS:")
    en = predictions["energy"]
    rt = predictions["run_time"]
    print(f"  Energy: {en['value_wh']:.2f} ± {en['uncertainty_wh']:.2f} Wh")
    print(f"    Model: {en['model']} (R²={en['r2']})")
    print(f"    95% interval: {en['best_case_wh']:.2f} - {en['worst_case_wh']:.2f} Wh")
    print(
        f"\n  run_time: {rt['value_s']:.2f} ± {rt['uncertainty_s']:.2f} s "
        f"({rt['value_min']:.2f} min)"
    )
    print(f"    Model: {rt['model']} (R²={rt['r2']})")
    print(f"    95% interval: {rt['best_case_s']:.2f} - {rt['worst_case_s']:.2f} s")


def _print_carbon_and_water(predictions: dict[str, Any]) -> None:
    """Print carbon and water lines for human output."""
    cb = predictions["carbon"]
    wu = predictions["water_used"]
    print(f"\n  Carbon emissions: {cb['value_gco2e']:.2f} gCO2e")
    print(f"    Embodied: {cb['g_co2_embodied']:.2f} gCO2e")
    print(f"    Electricity: {cb['g_co2_electricity']:.2f} gCO2e")
    print(
        f"    95% interval: {cb['best_case_gco2e']:.2f} - {cb['worst_case_gco2e']:.2f} gCO2e"
    )
    print(f"\n  Water used: {wu['value_water_used']:.2f} L")
    print(
        f"    95% interval: {wu['best_case_water_used']:.2f} - "
        f"{wu['worst_case_water_used']:.2f} L"
    )


def _emit_json_line(payload: dict[str, Any]) -> None:
    """Write one minified JSON object to stdout."""
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _emit_error_payload(message: str, output_mode: str) -> tuple[None, int]:
    """Emit error in the selected output mode and return exit code 1."""
    err = {"error": message}
    if output_mode == "json":
        _emit_json_line(err)
    else:
        print(f"❌ Error: {message}")
    return None, 1


def _invoke_run_ml(
    cfg: dict[str, Any],
    model_type: str,
    params: float,
    steps: int,
    total_frames: int,
) -> dict[str, Any]:
    """Call compute pipeline with resolved configuration."""
    inp = RunMlInputs(
        steps=float(steps),
        res=int(cfg["resolution_height"]) * int(cfg["resolution_witdh"]),
        frames=float(total_frames),
        fps=int(cfg["fps"]),
        duration=int(cfg["duration"]),
        params=params,
        arch=model_type,
        input_type=str(cfg["input_type"]),
        country=str(cfg["country"]),
    )
    return run_ml(inp)


def run_with_config(
    cfg: dict[str, Any], output_mode: str
) -> tuple[dict[str, Any] | None, int]:
    """Validate config, run ML, and print results. Returns payload and exit code."""
    validate_config(cfg)
    model_config = get_model_archi(str(cfg["model"]))
    if model_config["arch"] == "error":
        return _emit_error_payload(
            "Model can't be handled or is badly written", output_mode
        )

    model_type = str(model_config["arch"])
    params = float(model_config["params"])
    steps = int(cfg["denoising_steps"])
    total_frames = int(cfg["duration"]) * int(cfg["fps"])
    predictions = _invoke_run_ml(cfg, model_type, params, steps, total_frames)

    if "error" in predictions:
        return _emit_error_payload(str(predictions["error"]), output_mode)

    result = build_result_dict(cfg, steps, total_frames, predictions)
    if output_mode == "json":
        _emit_json_line(result)
    else:
        _print_human_inputs(cfg, steps, total_frames, model_type, params)
        _print_energy_and_runtime(predictions)
        _print_carbon_and_water(predictions)
    return result, 0


def main() -> int:
    """CLI main: parse args, merge config, run, return process exit code."""
    args = parse_args()
    try:
        cfg = merge_config_from_args(args)
        _, code = run_with_config(cfg, args.output)
        return code
    except (ValueError, OSError) as exc:
        if args.output == "json":
            _emit_json_line({"error": str(exc)})
        else:
            print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
