"""Plot sweep figures directly from CSV results files.

This reproduces both sweep visuals from sweep_curie_depths.py, but without
rerunning the forward model. It reads saved sweep CSV files and writes:

1. A single-panel curie-depth sweep plot with z_top, z_centroid, z_bottom
   and a 1:1 reference line.
2. A single-panel survey-height sweep plot with z_top, z_centroid, z_bottom
    and true Curie depth reference.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CURIE_CSV = PROJECT_DIR / "plots" / "curie_depth_sweep_results.csv"
DEFAULT_SURVEY_CSV = PROJECT_DIR / "plots" / "survey_height_sweep_results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "plots"


@dataclass
class SweepRow:
    sweep_name: str
    sweep_value: float
    survey_height: float
    true_curie_depth: float
    pred_z_top: float
    pred_z_centroid: float
    pred_z_bottom: float
    is_physical: bool


def parse_bool(text: str) -> bool:
    return text.strip().lower() in {"true", "1", "yes", "y"}


def load_rows(csv_path: Path, sweep_name: str) -> list[SweepRow]:
    rows: list[SweepRow] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                SweepRow(
                    sweep_name=raw["sweep_name"],
                    sweep_value=float(raw["sweep_value"]),
                    survey_height=float(raw["survey_height"]),
                    true_curie_depth=float(raw["true_curie_depth"]),
                    pred_z_top=float(raw["pred_z_top"]),
                    pred_z_centroid=float(raw["pred_z_centroid"]),
                    pred_z_bottom=float(raw["pred_z_bottom"]),
                    is_physical=parse_bool(raw["is_physical"]),
                )
            )

    rows = [row for row in rows if row.sweep_name == sweep_name]
    if sweep_name == "curie_depth":
        rows.sort(key=lambda item: item.true_curie_depth)
    elif sweep_name == "survey_height":
        rows.sort(key=lambda item: item.survey_height)
    return rows


def curie_plotting_arrays(
    rows: list[SweepRow],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_true_depth = np.array([row.true_curie_depth for row in rows], dtype=float)
    survey_height = np.array([row.survey_height for row in rows], dtype=float)
    z_top = np.array([row.pred_z_top for row in rows], dtype=float) - survey_height
    z_centroid = (
        np.array([row.pred_z_centroid for row in rows], dtype=float) - survey_height
    )
    z_bottom = (
        np.array([row.pred_z_bottom for row in rows], dtype=float) - survey_height
    )
    return x_true_depth, z_top, z_centroid, z_bottom


def survey_plotting_arrays(
    rows: list[SweepRow],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_survey_height = np.array([row.survey_height for row in rows], dtype=float)
    true_depth = np.array([row.true_curie_depth for row in rows], dtype=float)
    z_top = np.array([row.pred_z_top for row in rows], dtype=float) - x_survey_height
    z_centroid = (
        np.array([row.pred_z_centroid for row in rows], dtype=float) - x_survey_height
    )
    z_bottom = (
        np.array([row.pred_z_bottom for row in rows], dtype=float) - x_survey_height
    )
    return x_survey_height, true_depth, z_top, z_centroid, z_bottom


def plot_single_panel(
    rows: list[SweepRow],
    output_path: Path,
) -> None:
    x_true_depth, z_top, z_centroid, z_bottom = curie_plotting_arrays(rows)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(
        x_true_depth,
        z_top,
        marker="o",
        linewidth=1.8,
        color="#1f77b4",
        label="Predicted z_top (adjusted)",
    )
    ax.plot(
        x_true_depth,
        z_centroid,
        marker="s",
        linewidth=1.8,
        color="#ff7f0e",
        label="Predicted z_centroid (adjusted)",
    )
    ax.plot(
        x_true_depth,
        z_bottom,
        marker="^",
        linewidth=1.8,
        color="#2ca02c",
        label="Predicted z_bottom (adjusted)",
    )

    ref_min = float(
        min(
            np.nanmin(x_true_depth),
            np.nanmin(z_top),
            np.nanmin(z_centroid),
            np.nanmin(z_bottom),
        )
    )
    ref_max = float(
        max(
            np.nanmax(x_true_depth),
            np.nanmax(z_top),
            np.nanmax(z_centroid),
            np.nanmax(z_bottom),
        )
    )
    ax.plot(
        [ref_min, ref_max],
        [ref_min, ref_max],
        color="#444444",
        linestyle="--",
        linewidth=1.5,
        label="1:1 reference",
    )

    ax.set_title(f"Curie depth sweep, survey height = {rows[0].survey_height:.1f} m")
    ax.set_xlabel("Adjusted true Curie depth (m)")
    ax.set_ylabel("Adjusted recovered depth (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_survey_single_panel(rows: list[SweepRow], output_path: Path) -> None:
    x_survey_height, true_depth, z_top, z_centroid, z_bottom = survey_plotting_arrays(
        rows
    )

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(
        x_survey_height,
        z_top,
        marker="o",
        linewidth=1.8,
        color="#1f77b4",
        label="Predicted z_top (adjusted)",
    )
    ax.plot(
        x_survey_height,
        z_centroid,
        marker="s",
        linewidth=1.8,
        color="#ff7f0e",
        label="Predicted z_centroid (adjusted)",
    )
    ax.plot(
        x_survey_height,
        z_bottom,
        marker="^",
        linewidth=1.8,
        color="#2ca02c",
        label="Predicted z_bottom (adjusted)",
    )
    ax.plot(
        x_survey_height,
        true_depth,
        color="#444444",
        linestyle="--",
        linewidth=1.5,
        label="True Curie depth",
    )

    ax.set_title("Survey height sweep")
    ax.set_xlabel("Survey height (m)")
    ax.set_ylabel("Adjusted recovered depth (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot sweep figures directly from CSV files."
    )
    parser.add_argument(
        "--curie-csv",
        type=Path,
        default=DEFAULT_CURIE_CSV,
        help="CSV file created by sweep_curie_depths.py (curie depth rows).",
    )
    parser.add_argument(
        "--survey-csv",
        type=Path,
        default=DEFAULT_SURVEY_CSV,
        help="CSV file created by sweep_curie_depths.py (survey height rows).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where output figures are written.",
    )
    parser.add_argument(
        "--single-panel-name",
        type=str,
        default="curie_depth_sweep_from_csv.png",
        help="Filename for the single-panel figure.",
    )
    parser.add_argument(
        "--survey-single-panel-name",
        type=str,
        default="survey_height_sweep_from_csv.png",
        help="Filename for the survey-height single-panel figure.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    curie_rows = load_rows(args.curie_csv, "curie_depth")
    if not curie_rows:
        raise ValueError(f"No curie_depth rows found in {args.curie_csv}")
    survey_rows = load_rows(args.survey_csv, "survey_height")
    if not survey_rows:
        raise ValueError(f"No survey_height rows found in {args.survey_csv}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    single_panel_path = output_dir / args.single_panel_name
    survey_single_panel_path = output_dir / args.survey_single_panel_name
    plot_single_panel(curie_rows, single_panel_path)
    plot_survey_single_panel(survey_rows, survey_single_panel_path)

    print(f"Wrote {single_panel_path}")
    print(f"Wrote {survey_single_panel_path}")


if __name__ == "__main__":
    main()
