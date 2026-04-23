"""Sweep survey height and Curie depth, then plot spectral-depth recovery.

This script reuses the existing forward model and spectral estimator without
modifying either class. It runs two separate 1D sweeps:

1. Survey height sweep with Curie depth held fixed.
2. Curie depth sweep with survey height held fixed.

For each case, it records the recovered z_top, z_centroid, and z_bottom,
then generates csv files with the results.
"""

from __future__ import annotations

import argparse
import copy
import csv
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import yaml

from forward_simulation import ForwardSimulation
from spectral_methods import SpectralMethods


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "spectral_analysis_forward_model.yml"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "plots"


@dataclass
class SweepResult:
    sweep_name: str
    sweep_value: float
    survey_height: float
    true_curie_depth: float
    pred_z_top: float
    pred_z_centroid: float
    pred_z_bottom: float
    is_physical: bool


def parse_float_list(text: str) -> list[float]:
    values = [piece.strip() for piece in text.split(",") if piece.strip()]
    return [float(value) for value in values]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def update_config(
    base_config: dict,
    *,
    survey_height: float | None = None,
    curie_depth: float | None = None,
) -> dict:
    config = copy.deepcopy(base_config)
    if survey_height is not None:
        config.setdefault("survey", {})["z_height"] = float(survey_height)
    if curie_depth is not None:
        config.setdefault("model", {})["curie_depth"] = float(curie_depth)
    return config


def write_temp_config(config: dict, directory: Path, stem: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.yml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def run_case(
    base_config: dict,
    *,
    survey_height: float,
    curie_depth: float,
    temp_dir: Path,
    use_paper_ranges: bool,
) -> SweepResult:
    case_config = update_config(
        base_config, survey_height=survey_height, curie_depth=curie_depth
    )
    config_path = write_temp_config(
        case_config,
        temp_dir,
        stem=f"survey_{int(round(survey_height))}_curie_{int(round(curie_depth))}",
    )

    simulation = ForwardSimulation(config_path, randomize_model=True)
    estimator = SpectralMethods(simulation)
    curie_estimate = estimator.estimate_curie_depth(
        remove_mean=True,
        apply_window=True,
        use_paper_ranges=use_paper_ranges,
        reject_non_physical=False,
    )

    return SweepResult(
        sweep_name="",
        sweep_value=float(survey_height),
        survey_height=float(survey_height),
        true_curie_depth=float(curie_depth),
        pred_z_top=float(curie_estimate["z_top"]),
        pred_z_centroid=float(curie_estimate["z_centroid"]),
        pred_z_bottom=float(curie_estimate["z_bottom"]),
        is_physical=bool(curie_estimate["is_physical"]),
    )


def run_sweep(
    base_config: dict,
    *,
    sweep_name: str,
    sweep_values: Iterable[float],
    survey_height: float,
    curie_depth: float,
    temp_dir: Path,
    use_paper_ranges: bool,
) -> list[SweepResult]:
    results: list[SweepResult] = []

    for value in sweep_values:
        if sweep_name == "survey_height":
            case = run_case(
                base_config,
                survey_height=value,
                curie_depth=curie_depth,
                temp_dir=temp_dir,
                use_paper_ranges=use_paper_ranges,
            )
            case.sweep_name = sweep_name
            case.sweep_value = float(value)
        elif sweep_name == "curie_depth":
            case = run_case(
                base_config,
                survey_height=survey_height,
                curie_depth=value,
                temp_dir=temp_dir,
                use_paper_ranges=use_paper_ranges,
            )
            case.sweep_name = sweep_name
            case.sweep_value = float(value)
        else:
            raise ValueError(f"Unsupported sweep_name: {sweep_name}")

        results.append(case)

    return results


def results_to_rows(results: list[SweepResult]) -> list[dict[str, object]]:
    return [asdict(result) for result in results]


def save_results_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep survey height and Curie depth, then plot spectral recovery."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Baseline YAML config to sweep from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for plots and CSV output.",
    )
    parser.add_argument(
        "--survey-heights",
        type=parse_float_list,
        default=parse_float_list("200, 400, 600, 800, 1000"),
        help="Comma-separated survey heights in meters.",
    )
    parser.add_argument(
        "--curie-depths",
        type=parse_float_list,
        default=parse_float_list("1000, 3000, 5000, 7000, 9000, 11000"),
        help="Comma-separated true Curie depths in meters.",
    )
    parser.add_argument(
        "--survey-height-fixed",
        type=float,
        default=300.0,
        help="Fixed survey height for the Curie-depth sweep.",
    )
    parser.add_argument(
        "--curie-depth-fixed",
        type=float,
        default=5000.0,
        help="Fixed Curie depth for the survey-height sweep.",
    )
    parser.add_argument(
        "--use-paper-ranges",
        action="store_true",
        default=True,
        help="Use the fixed paper wavenumber ranges in spectral_methods.py.",
    )
    parser.add_argument(
        "--adaptive-ranges",
        action="store_false",
        dest="use_paper_ranges",
        help="Use caller-supplied ranges instead of the paper defaults.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_config = load_config(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="curie_sweep_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        survey_results = run_sweep(
            base_config,
            sweep_name="survey_height",
            sweep_values=args.survey_heights,
            survey_height=args.survey_height_fixed,
            curie_depth=args.curie_depth_fixed,
            temp_dir=temp_dir,
            use_paper_ranges=args.use_paper_ranges,
        )
        curie_results = run_sweep(
            base_config,
            sweep_name="curie_depth",
            sweep_values=args.curie_depths,
            survey_height=args.survey_height_fixed,
            curie_depth=args.curie_depth_fixed,
            temp_dir=temp_dir,
            use_paper_ranges=args.use_paper_ranges,
        )

    survey_csv = output_dir / "survey_height_sweep_results.csv"
    curie_csv = output_dir / "curie_depth_sweep_results.csv"
    save_results_csv(results_to_rows(survey_results), survey_csv)
    save_results_csv(results_to_rows(curie_results), curie_csv)

    print(f"Wrote {survey_csv}")
    print(f"Wrote {curie_csv}")


if __name__ == "__main__":
    main()
