from __future__ import annotations

import argparse
from pathlib import Path

from .plotting import (
    plot_boundary,
    plot_es_budget_boundary,
    plot_es_budget_noise,
    plot_es_trace,
    plot_generalization_bounds,
    plot_noise,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate PU-P2L plots from existing CSV results.")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument(
        "--kind",
        type=str,
        required=True,
        choices=["boundary", "noise", "es_trace", "es_budget_boundary", "es_budget_noise", "generalization_bounds"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_path = results_dir / "results.csv"
    plots_dir = results_dir / "plots"
    if args.kind == "boundary":
        plot_boundary(results_path, plots_dir)
    elif args.kind == "es_trace":
        plot_es_trace(results_path, plots_dir)
    elif args.kind == "es_budget_boundary":
        plot_es_budget_boundary(results_path, plots_dir)
    elif args.kind == "es_budget_noise":
        plot_es_budget_noise(results_path, plots_dir)
    elif args.kind == "generalization_bounds":
        plot_generalization_bounds(results_path, plots_dir)
    else:
        plot_noise(results_path, plots_dir)


if __name__ == "__main__":
    main()
