from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import pac_bayes_enabled_for_dataset
from .io_utils import MARGINAL_METHODS, read_csv, to_float


COLORS = {
    "MaxLoss": "#274753",
    "Marginal": "#297270",
    "EL2N": "#f3a361",
    "GraNdLast": "#e66d50",
    "RHO-PretrainRef": "#8ab07c",
    "PU-R": "#299d8f",
    "PU-R-Vol": "#7f8c8d",
    "PU-R-Manifold": "#e7c66b",
    "GREATS": "#9467bd",
    "ClippedLoss": "#f3a361",
    "ResidualOnly": "#7f8c8d",
    "RedundancyOnly": "#9467bd",
    "Loss+Residual": "#bcbd22",
    "Loss-Redundancy": "#d62728",
    "PU-C-style": "#ff9896",
    "Marginal+Residual": "#17becf",
    "Marginal-Redundancy": "#aec7e8",
    "Marginal+Residual-Redundancy": "#ffbb78",
}
METHOD_ORDER = [
    "MaxLoss",
    "Marginal",
    "EL2N",
    "GraNdLast",
    "RHO-PretrainRef",
    "PU-R",
    "PU-R-Vol",
    "PU-R-Manifold",
    "ClippedLoss",
    "ResidualOnly",
    "RedundancyOnly",
    "Loss+Residual",
    "Loss-Redundancy",
    "PU-C-style",
    "Marginal+Residual",
    "Marginal-Redundancy",
    "Marginal+Residual-Redundancy",
    "GREATS",
]
CERTIFIED_METHOD_ORDER = [method for method in METHOD_ORDER if method != "GREATS"]
BOUND_METHOD_ORDER = METHOD_ORDER
GENERALIZATION_BOUND_CURVES = [
    ("test_inappropriate_risk", "risk", "-", 1.8, 0.16),
    ("certified_bound", "P2L", "--", 1.7, 0.10),
    ("pac_bayes_bound", "PAC-Bayes", ":", 1.8, 0.08),
    ("self_selected_bound", "self-selected", "-.", 1.5, 0.08),
    ("ada_bound", "ADA growing", (0, (3, 1, 1, 1, 1, 1)), 1.5, 0.08),
]
SELECTION_DIAGNOSTIC_METRICS = [
    ("noise_hit_rate", "Noise-hit rate"),
    ("duplicate_hit_rate", "Duplicate-hit rate"),
    ("pairwise_feature_cosine", "Pairwise feature cosine"),
    ("strong_redundancy_hit_rate", "Strong redundancy-hit rate"),
    ("group_revisit_rate", "Group revisit rate"),
    ("unique_group_fraction", "Unique group fraction"),
    ("mean_selected_residual_novelty", "Mean selected residual novelty"),
    ("mode_entropy", "Mode/rotation entropy"),
]


def should_plot_pac_bayes(rows: list[dict[str, str]]) -> bool:
    dataset_names = {row.get("dataset", "").strip() for row in rows if row.get("dataset", "").strip()}
    if dataset_names and not any(pac_bayes_enabled_for_dataset(name) for name in dataset_names):
        return False
    return any(not np.isnan(to_float(row, "pac_bayes_bound")) for row in rows)


def format_float_for_filename(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def method_views(methods: list[str]) -> list[tuple[str, list[str]]]:
    with_marginal = methods
    without_marginal = [method for method in methods if method not in MARGINAL_METHODS]
    views = [("with_marginal", with_marginal)]
    if without_marginal != with_marginal:
        views.append(("without_marginal", without_marginal))
    else:
        views.append(("without_marginal", without_marginal))
    return [(name, view_methods) for name, view_methods in views if view_methods]


def annotate_minimum_step_bound(
    ax: plt.Axes,
    xs: list[float],
    means: list[float],
    method: str,
    method_index: int,
) -> None:
    if not xs:
        return
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(means, dtype=np.float64)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(finite):
        return
    finite_positions = np.flatnonzero(finite)
    best_pos = int(finite_positions[int(np.argmin(y_arr[finite]))])
    best_x = float(x_arr[best_pos])
    best_y = float(y_arr[best_pos])
    color = COLORS.get(method, "#333333")
    ax.scatter(
        [best_x],
        [best_y],
        s=58,
        marker="o",
        facecolor=color,
        edgecolor="white",
        linewidth=1.2,
        zorder=8,
    )
    offsets = [(8, 14), (8, -20), (-76, 14), (-76, -20), (8, 30), (-76, 30)]
    offset = offsets[method_index % len(offsets)]
    ax.annotate(
        f"{method}\nstep={best_x:g}, bound={best_y:.3f}",
        xy=(best_x, best_y),
        xytext=offset,
        textcoords="offset points",
        fontsize=7,
        color=color,
        arrowprops={"arrowstyle": "->", "color": color, "lw": 0.8, "alpha": 0.8},
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": color, "alpha": 0.82, "lw": 0.8},
        zorder=9,
    )


def grouped_curve(
    rows: list[dict[str, str]],
    method: str,
    noise_rate: float,
    metric: str,
) -> tuple[list[float], list[float], list[float]]:
    groups: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != method:
            continue
        if abs(to_float(row, "noise_rate") - noise_rate) > 1e-12:
            continue
        x = to_float(row, "pretrain_fraction")
        y = to_float(row, metric)
        if not np.isnan(x) and not np.isnan(y):
            groups[x].append(y)
    xs = sorted(groups)
    means = [float(np.mean(groups[x])) for x in xs]
    ses = [
        float(np.std(groups[x], ddof=1) / np.sqrt(len(groups[x]))) if len(groups[x]) > 1 else 0.0
        for x in xs
    ]
    return xs, means, ses


def grouped_noise_curve(
    rows: list[dict[str, str]],
    method: str,
    metric: str,
) -> tuple[list[float], list[float], list[float]]:
    groups: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != method:
            continue
        x = to_float(row, "noise_rate")
        y = to_float(row, metric)
        if not np.isnan(x) and not np.isnan(y):
            groups[x].append(y)
    xs = sorted(groups)
    means = [float(np.mean(groups[x])) for x in xs]
    ses = [
        float(np.std(groups[x], ddof=1) / np.sqrt(len(groups[x]))) if len(groups[x]) > 1 else 0.0
        for x in xs
    ]
    return xs, means, ses


def grouped_step_curve(
    rows: list[dict[str, str]],
    method: str,
    noise_rate: float,
    pretrain_fraction: float,
    metric: str,
    max_step: float | None = None,
) -> tuple[list[float], list[float], list[float]]:
    groups: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != method:
            continue
        if abs(to_float(row, "noise_rate") - noise_rate) > 1e-12:
            continue
        if abs(to_float(row, "pretrain_fraction") - pretrain_fraction) > 1e-12:
            continue
        x = to_float(row, "step")
        y = to_float(row, metric)
        if max_step is not None and x > max_step:
            continue
        if not np.isnan(x) and not np.isnan(y):
            groups[x].append(y)
    xs = sorted(groups)
    means = [float(np.mean(groups[x])) for x in xs]
    ses = [
        float(np.std(groups[x], ddof=1) / np.sqrt(len(groups[x]))) if len(groups[x]) > 1 else 0.0
        for x in xs
    ]
    return xs, means, ses


def has_step_metric(
    rows: list[dict[str, str]],
    methods: list[str],
    noise_rate: float,
    pretrain_fraction: float,
    metric: str,
    max_step: float | None = None,
) -> bool:
    for method in methods:
        xs, _, _ = grouped_step_curve(rows, method, noise_rate, pretrain_fraction, metric, max_step=max_step)
        if xs:
            return True
    return False


def grouped_budget_pretrain_curve(
    rows: list[dict[str, str]],
    method: str,
    noise_rate: float,
    es_budget: int,
    metric: str,
) -> tuple[list[float], list[float], list[float]]:
    groups: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != method:
            continue
        if abs(to_float(row, "noise_rate") - noise_rate) > 1e-12:
            continue
        if int(to_float(row, "es_budget")) != es_budget:
            continue
        x = to_float(row, "pretrain_fraction")
        y = to_float(row, metric)
        if not np.isnan(x) and not np.isnan(y):
            groups[x].append(y)
    xs = sorted(groups)
    means = [float(np.mean(groups[x])) for x in xs]
    ses = [
        float(np.std(groups[x], ddof=1) / np.sqrt(len(groups[x]))) if len(groups[x]) > 1 else 0.0
        for x in xs
    ]
    return xs, means, ses


def grouped_budget_noise_curve(
    rows: list[dict[str, str]],
    method: str,
    es_budget: int,
    pretrain_fraction: float,
    metric: str,
) -> tuple[list[float], list[float], list[float]]:
    groups: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != method:
            continue
        if int(to_float(row, "es_budget")) != es_budget:
            continue
        if abs(to_float(row, "pretrain_fraction") - pretrain_fraction) > 1e-12:
            continue
        x = to_float(row, "noise_rate")
        y = to_float(row, metric)
        if not np.isnan(x) and not np.isnan(y):
            groups[x].append(y)
    xs = sorted(groups)
    means = [float(np.mean(groups[x])) for x in xs]
    ses = [
        float(np.std(groups[x], ddof=1) / np.sqrt(len(groups[x]))) if len(groups[x]) > 1 else 0.0
        for x in xs
    ]
    return xs, means, ses


def grouped_time_matched_noise_curve(
    rows: list[dict[str, str]],
    method: str,
    pretrain_fraction: float,
    reference_es_budget: int,
    metric: str,
) -> tuple[list[float], list[float], list[float]]:
    groups: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != method:
            continue
        budget_value = to_float(row, "reference_es_budget")
        if np.isnan(budget_value):
            budget_value = to_float(row, "es_budget")
        if np.isnan(budget_value) or int(budget_value) != reference_es_budget:
            continue
        if abs(to_float(row, "pretrain_fraction") - pretrain_fraction) > 1e-12:
            continue
        x = to_float(row, "noise_rate")
        y = to_float(row, metric)
        if not np.isnan(x) and not np.isnan(y):
            groups[x].append(y)
    xs = sorted(groups)
    means = [float(np.mean(groups[x])) for x in xs]
    ses = [
        float(np.std(groups[x], ddof=1) / np.sqrt(len(groups[x]))) if len(groups[x]) > 1 else 0.0
        for x in xs
    ]
    return xs, means, ses


def plot_boundary(results_path: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(results_path)
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in rows)]
    noise_rates = sorted({to_float(row, "noise_rate") for row in rows})

    for view_name, view_methods in method_views(methods):
        view_dir = plots_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        for noise_rate in noise_rates:
            suffix = format_float_for_filename(noise_rate)
            plot_certified_bound_and_risk(
                rows,
                view_methods,
                noise_rate,
                view_dir / f"certified_bound_and_risk_vs_pretrain_noise_{suffix}.png",
            )
            plot_metric(
                rows,
                view_methods,
                noise_rate,
                "effective_compression_size",
                "Effective compression size",
                "Effective Compression Size vs Pretrain Fraction",
                view_dir / f"effective_compression_vs_pretrain_noise_{suffix}.png",
            )
            plot_metric(
                rows,
                view_methods,
                noise_rate,
                "runtime_sec",
                "Runtime seconds",
                "Runtime vs Pretrain Fraction",
                view_dir / f"runtime_vs_pretrain_noise_{suffix}.png",
            )
            plot_metric(
                rows,
                view_methods,
                noise_rate,
                "test_inappropriate_risk",
                "Test inappropriate risk",
                "Test Inappropriate Risk vs Pretrain Fraction",
                view_dir / f"test_inappropriate_risk_vs_pretrain_noise_{suffix}.png",
            )
            plot_metric(
                rows,
                view_methods,
                noise_rate,
                "test_error",
                "Test top-1 error",
                "Test Top-1 Error vs Pretrain Fraction",
                view_dir / f"test_error_vs_pretrain_noise_{suffix}.png",
            )
            plot_pretrain_selection_diagnostics(
                rows,
                view_methods,
                noise_rate,
                view_dir / f"selection_diagnostics_vs_pretrain_noise_{suffix}.png",
            )


def plot_generalization_bounds(results_path: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(results_path)
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in rows)]
    noise_rates = sorted({to_float(row, "noise_rate") for row in rows})

    for view_name, view_methods in method_views(methods):
        view_dir = plots_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        for noise_rate in noise_rates:
            suffix = format_float_for_filename(noise_rate)
            plot_generalization_bound_and_risk(
                rows,
                view_methods,
                noise_rate,
                view_dir / f"generalization_bounds_vs_pretrain_noise_{suffix}.png",
            )


def plot_generalization_bound_and_risk(
    rows: list[dict[str, str]],
    methods: list[str],
    noise_rate: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    for method in methods:
        for metric, label, linestyle, linewidth, alpha in GENERALIZATION_BOUND_CURVES:
            xs, means, ses = grouped_curve(rows, method, noise_rate, metric)
            if not xs:
                continue
            plot_mean_band(
                ax,
                xs,
                means,
                ses,
                method,
                label=f"{method} {label}",
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
            )
    ax.set_xlabel("Pretrain fraction")
    ax.set_ylabel("Test inappropriate risk / generalization bound")
    ax.set_title(f"MNIST Generalization Bounds vs Pretrain Fraction (noise={noise_rate:g})")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_certified_bound_and_risk(
    rows: list[dict[str, str]],
    methods: list[str],
    noise_rate: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    show_pac_bayes = should_plot_pac_bayes(rows)
    for method in methods:
        xs, means, ses = grouped_curve(rows, method, noise_rate, "test_inappropriate_risk")
        if xs:
            plot_mean_band(
                ax,
                xs,
                means,
                ses,
                method,
                label=f"{method} risk",
                linestyle="-",
                alpha=0.16,
            )

        xs, means, ses = grouped_curve(rows, method, noise_rate, "certified_bound")
        if xs:
            plot_mean_band(
                ax,
                xs,
                means,
                ses,
                method,
                label=f"{method} certif.",
                linestyle="--",
                alpha=0.10,
            )

        xs, means, ses = grouped_curve(rows, method, noise_rate, "pac_bayes_bound")
        if show_pac_bayes and xs:
            plot_mean_band(
                ax,
                xs,
                means,
                ses,
                method,
                label=f"{method} PAC-Bayes",
                linestyle=":",
                alpha=0.08,
            )
    ax.set_xlabel("Pretrain fraction")
    ax.set_ylabel(
        "Test inappropriate risk / generalization bound"
        if show_pac_bayes
        else "Test inappropriate risk / P2L bound"
    )
    title = (
        "P2L/PAC-Bayes Bounds and Test Inappropriate Risk"
        if show_pac_bayes
        else "P2L Bound and Test Inappropriate Risk"
    )
    ax.set_title(f"{title} vs Pretrain Fraction (noise={noise_rate:g})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metric(
    rows: list[dict[str, str]],
    methods: list[str],
    noise_rate: float,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for method in methods:
        xs, means, ses = grouped_curve(rows, method, noise_rate, metric)
        if not xs:
            continue
        plot_mean_band(ax, xs, means, ses, method)
    ax.set_xlabel("Pretrain fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (noise={noise_rate:g})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pretrain_selection_diagnostics(
    rows: list[dict[str, str]],
    methods: list[str],
    noise_rate: float,
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(SELECTION_DIAGNOSTIC_METRICS),
        1,
        figsize=(9, 3.1 * len(SELECTION_DIAGNOSTIC_METRICS)),
        sharex=True,
    )
    for ax, (metric, ylabel) in zip(axes, SELECTION_DIAGNOSTIC_METRICS):
        for method in methods:
            xs, means, ses = grouped_curve(rows, method, noise_rate, metric)
            if not xs:
                continue
            plot_mean_band(ax, xs, means, ses, method)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Pretrain fraction")
    axes[0].set_title(f"Selected-Set Diagnostics vs Pretrain Fraction (noise={noise_rate:g})")
    axes[0].legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_es_budget_boundary(results_path: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(results_path)
    methods = [method for method in BOUND_METHOD_ORDER if any(row["method"] == method for row in rows)]
    noise_rates = sorted({to_float(row, "noise_rate") for row in rows})
    budgets = sorted({int(to_float(row, "es_budget")) for row in rows})

    for view_name, view_methods in method_views(methods):
        view_dir = plots_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        for noise_rate in noise_rates:
            noise_suffix = format_float_for_filename(noise_rate)
            for budget in budgets:
                plot_es_budget_bound_and_risk_vs_pretrain(
                    rows,
                    view_methods,
                    noise_rate,
                    budget,
                    view_dir / f"es_budget_bound_and_risk_vs_pretrain_noise_{noise_suffix}_budget_{budget}.png",
                )
                plot_es_budget_pretrain_selection_diagnostics(
                    rows,
                    view_methods,
                    noise_rate,
                    budget,
                    view_dir
                    / f"es_budget_selection_diagnostics_vs_pretrain_noise_{noise_suffix}_budget_{budget}.png",
                )


def plot_es_budget_bound_and_risk_vs_pretrain(
    rows: list[dict[str, str]],
    methods: list[str],
    noise_rate: float,
    es_budget: int,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    show_pac_bayes = should_plot_pac_bayes(rows)
    for method in methods:
        xs, means, ses = grouped_budget_pretrain_curve(rows, method, noise_rate, es_budget, "test_inappropriate_risk")
        if xs:
            plot_mean_band(
                ax,
                xs,
                means,
                ses,
                method,
                label=f"{method} risk",
                linestyle="-",
                alpha=0.16,
            )

        xs, means, ses = grouped_budget_pretrain_curve(rows, method, noise_rate, es_budget, "certified_bound")
        if xs:
            plot_mean_band(
                ax,
                xs,
                means,
                ses,
                method,
                label=f"{method} certif.",
                linestyle="--",
                alpha=0.10,
            )

        xs, means, ses = grouped_budget_pretrain_curve(rows, method, noise_rate, es_budget, "pac_bayes_bound")
        if show_pac_bayes and xs:
            plot_mean_band(
                ax,
                xs,
                means,
                ses,
                method,
                label=f"{method} PAC-Bayes",
                linestyle=":",
                alpha=0.08,
            )
    ax.set_xlabel("Pretrain fraction")
    ax.set_ylabel(
        "Test inappropriate risk / ES generalization bound"
        if show_pac_bayes
        else "Test inappropriate risk / ES P2L bound"
    )
    title = (
        "ES P2L/PAC-Bayes Bounds and Test Inappropriate Risk"
        if show_pac_bayes
        else "ES P2L Bound and Test Inappropriate Risk"
    )
    ax.set_title(
        f"{title} vs Pretrain Fraction (noise={noise_rate:g}, ES={es_budget})"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_es_budget_pretrain_selection_diagnostics(
    rows: list[dict[str, str]],
    methods: list[str],
    noise_rate: float,
    es_budget: int,
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(SELECTION_DIAGNOSTIC_METRICS),
        1,
        figsize=(9, 3.1 * len(SELECTION_DIAGNOSTIC_METRICS)),
        sharex=True,
    )
    for ax, (metric, ylabel) in zip(axes, SELECTION_DIAGNOSTIC_METRICS):
        for method in methods:
            xs, means, ses = grouped_budget_pretrain_curve(rows, method, noise_rate, es_budget, metric)
            if not xs:
                continue
            plot_mean_band(ax, xs, means, ses, method)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Pretrain fraction")
    axes[0].set_title(
        f"Selected-Set Diagnostics vs Pretrain Fraction (noise={noise_rate:g}, ES={es_budget})"
    )
    axes[0].legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_es_trace(results_path: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(results_path)
    methods = [method for method in BOUND_METHOD_ORDER if any(row["method"] == method for row in rows)]
    noise_rates = sorted({to_float(row, "noise_rate") for row in rows})
    pretrain_fractions = sorted({to_float(row, "pretrain_fraction") for row in rows})

    for view_name, view_methods in method_views(methods):
        view_dir = plots_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        for noise_rate in noise_rates:
            noise_suffix = format_float_for_filename(noise_rate)
            for pretrain_fraction in pretrain_fractions:
                pretrain_suffix = format_float_for_filename(pretrain_fraction)
                plot_step_bound_and_risk(
                    rows,
                    view_methods,
                    noise_rate,
                    pretrain_fraction,
                    view_dir / f"es_bound_vs_step_noise_{noise_suffix}_pretrain_{pretrain_suffix}.png",
                    boundary_only=True,
                )
                plot_step_bound_and_risk(
                    rows,
                    view_methods,
                    noise_rate,
                    pretrain_fraction,
                    view_dir
                    / f"es_bound_and_risk_vs_step_noise_{noise_suffix}_pretrain_{pretrain_suffix}.png",
                    boundary_only=True,
                )
                plot_step_bound_and_risk(
                    rows,
                    view_methods,
                    noise_rate,
                    pretrain_fraction,
                    view_dir / f"es_bound_vs_step_first_100_noise_{noise_suffix}_pretrain_{pretrain_suffix}.png",
                    max_step=100,
                    boundary_only=True,
                )
                if has_step_metric(
                    rows,
                    view_methods,
                    noise_rate,
                    pretrain_fraction,
                    "test_inappropriate_risk",
                    max_step=100,
                ):
                    plot_step_bound_and_risk(
                        rows,
                        view_methods,
                        noise_rate,
                        pretrain_fraction,
                        view_dir
                        / f"es_bound_and_risk_vs_step_first_100_noise_{noise_suffix}_pretrain_{pretrain_suffix}.png",
                        max_step=100,
                    )
                plot_step_metric(
                    rows,
                    view_methods,
                    noise_rate,
                    pretrain_fraction,
                    "remaining_bad",
                    "Inappropriate points left",
                    "Inappropriate Points Left vs Step",
                    view_dir / f"remaining_bad_vs_step_noise_{noise_suffix}_pretrain_{pretrain_suffix}.png",
                )
                plot_step_metric(
                    rows,
                    view_methods,
                    noise_rate,
                    pretrain_fraction,
                    "remaining_bad",
                    "Inappropriate points left",
                    "Inappropriate Points Left vs Step",
                    view_dir / f"remaining_bad_vs_step_first_100_noise_{noise_suffix}_pretrain_{pretrain_suffix}.png",
                    max_step=100,
                )
                plot_step_metric(
                    rows,
                    view_methods,
                    noise_rate,
                    pretrain_fraction,
                    "effective_compression_size",
                    "ES effective compression size",
                    "ES Effective Compression Size vs Step",
                    view_dir
                    / f"effective_compression_vs_step_noise_{noise_suffix}_pretrain_{pretrain_suffix}.png",
                )
                plot_step_metric(
                    rows,
                    view_methods,
                    noise_rate,
                    pretrain_fraction,
                    "effective_compression_size",
                    "ES effective compression size",
                    "ES Effective Compression Size vs Step",
                    view_dir
                    / f"effective_compression_vs_step_first_100_noise_{noise_suffix}_pretrain_{pretrain_suffix}.png",
                    max_step=100,
                )
                if has_step_metric(rows, view_methods, noise_rate, pretrain_fraction, "spectral_entropy"):
                    plot_step_metric(
                        rows,
                        view_methods,
                        noise_rate,
                        pretrain_fraction,
                        "spectral_entropy",
                        "Support spectral entropy",
                        "Support Spectral Entropy vs Step",
                        view_dir / f"spectral_entropy_vs_step_noise_{noise_suffix}_pretrain_{pretrain_suffix}.png",
                    )
                if has_step_metric(rows, view_methods, noise_rate, pretrain_fraction, "dynamic_mu"):
                    plot_step_metric(
                        rows,
                        view_methods,
                        noise_rate,
                        pretrain_fraction,
                        "dynamic_mu",
                        "Dynamic novelty weight",
                        "Dynamic Novelty Weight vs Step",
                        view_dir / f"dynamic_mu_vs_step_noise_{noise_suffix}_pretrain_{pretrain_suffix}.png",
                    )


def plot_step_bound_and_risk(
    rows: list[dict[str, str]],
    methods: list[str],
    noise_rate: float,
    pretrain_fraction: float,
    path: Path,
    max_step: float | None = None,
    boundary_only: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    show_pac_bayes = (not boundary_only) and should_plot_pac_bayes(rows)
    for method_index, method in enumerate(methods):
        if not boundary_only:
            xs, means, ses = grouped_step_curve(
                rows, method, noise_rate, pretrain_fraction, "test_inappropriate_risk", max_step=max_step
            )
            if xs:
                plot_mean_band(
                    ax,
                    xs,
                    means,
                    ses,
                    method,
                    label=f"{method} risk",
                    linestyle="-",
                    alpha=0.16,
                )

        xs, means, ses = grouped_step_curve(
            rows, method, noise_rate, pretrain_fraction, "certified_bound", max_step=max_step
        )
        if xs:
            plot_mean_band(
                ax,
                xs,
                means,
                ses,
                method,
                label=f"{method} certif.",
                linestyle="--",
                alpha=0.10,
            )
            annotate_minimum_step_bound(ax, xs, means, method, method_index)

        if not boundary_only:
            xs, means, ses = grouped_step_curve(
                rows, method, noise_rate, pretrain_fraction, "pac_bayes_bound", max_step=max_step
            )
            if show_pac_bayes and xs:
                plot_mean_band(
                    ax,
                    xs,
                    means,
                    ses,
                    method,
                    label=f"{method} PAC-Bayes",
                    linestyle=":",
                    alpha=0.08,
                )
    if max_step is not None:
        ax.set_xlim(left=0, right=max_step)
    ax.set_xlabel("Selection step")
    if boundary_only:
        ax.set_ylabel("ES P2L bound")
    else:
        ax.set_ylabel(
            "Test inappropriate risk / ES generalization bound"
            if show_pac_bayes
            else "Test inappropriate risk / ES P2L bound"
        )
    window_label = f", first {max_step:g} steps" if max_step is not None else ""
    if boundary_only:
        title = "ES P2L Bound"
    else:
        title = (
            "ES P2L/PAC-Bayes Bounds and Test Inappropriate Risk"
            if show_pac_bayes
            else "ES P2L Bound and Test Inappropriate Risk"
        )
    ax.set_title(
        f"{title} vs Step{window_label} (noise={noise_rate:g}, pretrain={pretrain_fraction:g})"
    )
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_step_metric(
    rows: list[dict[str, str]],
    methods: list[str],
    noise_rate: float,
    pretrain_fraction: float,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    max_step: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for method in methods:
        xs, means, ses = grouped_step_curve(
            rows, method, noise_rate, pretrain_fraction, metric, max_step=max_step
        )
        if not xs:
            continue
        plot_mean_band(ax, xs, means, ses, method)
    if max_step is not None:
        ax.set_xlim(left=0, right=max_step)
    ax.set_xlabel("Selection step")
    ax.set_ylabel(ylabel)
    window_label = f", first {max_step:g} steps" if max_step is not None else ""
    ax.set_title(
        f"{title}{window_label} (noise={noise_rate:g}, pretrain={pretrain_fraction:g})"
    )
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_es_budget_noise(results_path: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(results_path)
    methods = [method for method in BOUND_METHOD_ORDER if any(row["method"] == method for row in rows)]
    budgets = sorted({int(to_float(row, "es_budget")) for row in rows})
    pretrain_fractions = sorted({to_float(row, "pretrain_fraction") for row in rows})

    for view_name, view_methods in method_views(methods):
        view_dir = plots_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        for pretrain_fraction in pretrain_fractions:
            pretrain_suffix = format_float_for_filename(pretrain_fraction)
            for budget in budgets:
                plot_es_budget_noise_metric(
                    rows,
                    view_methods,
                    budget,
                    pretrain_fraction,
                    "effective_compression_size",
                    "ES effective compression size",
                    f"ES Effective Compression Size vs Label-Noise Rate (ES={budget}, pretrain={pretrain_fraction:g})",
                    view_dir / f"es_budget_effective_compression_vs_noise_budget_{budget}_pretrain_{pretrain_suffix}.png",
                )
                plot_es_budget_noise_metric(
                    rows,
                    view_methods,
                    budget,
                    pretrain_fraction,
                    "test_inappropriate_risk",
                    "Test inappropriate risk",
                    f"Test Inappropriate Risk vs Label-Noise Rate (ES={budget}, pretrain={pretrain_fraction:g})",
                    view_dir / f"es_budget_test_risk_vs_noise_budget_{budget}_pretrain_{pretrain_suffix}.png",
                )
                plot_es_budget_bounds_vs_noise(
                    rows,
                    view_methods,
                    budget,
                    pretrain_fraction,
                    view_dir / f"es_budget_bounds_vs_noise_budget_{budget}_pretrain_{pretrain_suffix}.png",
                )


def plot_es_budget_bounds_vs_noise(
    rows: list[dict[str, str]],
    methods: list[str],
    es_budget: int,
    pretrain_fraction: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    show_pac_bayes = should_plot_pac_bayes(rows)
    for method in methods:
        xs, means, ses = grouped_budget_noise_curve(rows, method, es_budget, pretrain_fraction, "test_inappropriate_risk")
        if xs:
            plot_mean_band(ax, xs, means, ses, method, label=f"{method} risk", linestyle="-", alpha=0.16)

        xs, means, ses = grouped_budget_noise_curve(rows, method, es_budget, pretrain_fraction, "certified_bound")
        if xs:
            plot_mean_band(ax, xs, means, ses, method, label=f"{method} P2L", linestyle="--", alpha=0.10)

        xs, means, ses = grouped_budget_noise_curve(rows, method, es_budget, pretrain_fraction, "pac_bayes_bound")
        if show_pac_bayes and xs:
            plot_mean_band(ax, xs, means, ses, method, label=f"{method} PAC-Bayes", linestyle=":", alpha=0.08)
    ax.set_xlabel("Label-noise rate")
    ax.set_ylabel(
        "Test inappropriate risk / ES generalization bound"
        if show_pac_bayes
        else "Test inappropriate risk / ES P2L bound"
    )
    title = (
        "ES P2L/PAC-Bayes Bounds and Test Inappropriate Risk"
        if show_pac_bayes
        else "ES P2L Bound and Test Inappropriate Risk"
    )
    ax.set_title(f"{title} vs Label-Noise Rate (ES={es_budget}, pretrain={pretrain_fraction:g})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_es_budget_noise_metric(
    rows: list[dict[str, str]],
    methods: list[str],
    es_budget: int,
    pretrain_fraction: float,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for method in methods:
        xs, means, ses = grouped_budget_noise_curve(rows, method, es_budget, pretrain_fraction, metric)
        if not xs:
            continue
        plot_mean_band(ax, xs, means, ses, method)
    ax.set_xlabel("Label-noise rate")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_time_matched_noise(results_path: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(results_path)
    methods = [method for method in BOUND_METHOD_ORDER if any(row["method"] == method for row in rows)]
    pretrain_fractions = sorted({to_float(row, "pretrain_fraction") for row in rows})
    reference_budgets = sorted(
        {
            int(value)
            for row in rows
            for value in [to_float(row, "reference_es_budget", to_float(row, "es_budget"))]
            if not np.isnan(value)
        }
    )

    for view_name, view_methods in method_views(methods):
        view_dir = plots_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        for reference_budget in reference_budgets:
            for pretrain_fraction in pretrain_fractions:
                pretrain_suffix = format_float_for_filename(pretrain_fraction)
                plot_time_matched_bounds_vs_noise(
                    rows,
                    view_methods,
                    reference_budget,
                    pretrain_fraction,
                    view_dir
                    / f"time_matched_bounds_vs_noise_pur_es{reference_budget}_pretrain_{pretrain_suffix}.png",
                )
                plot_time_matched_noise_metric(
                    rows,
                    view_methods,
                    reference_budget,
                    pretrain_fraction,
                    "test_error",
                    "Clean test error",
                    (
                        "Time-Matched Clean Test Error vs Label-Noise Rate "
                        f"(PU-R ES={reference_budget}, pretrain={pretrain_fraction:g})"
                    ),
                    view_dir
                    / f"time_matched_clean_test_error_vs_noise_pur_es{reference_budget}_pretrain_{pretrain_suffix}.png",
                )
                plot_time_matched_noise_metric(
                    rows,
                    view_methods,
                    reference_budget,
                    pretrain_fraction,
                    "effective_compression_size",
                    "Effective compression size",
                    (
                        "Time-Matched Effective Compression Size vs Label-Noise Rate "
                        f"(PU-R ES={reference_budget}, pretrain={pretrain_fraction:g})"
                    ),
                    view_dir
                    / f"time_matched_effective_compression_vs_noise_pur_es{reference_budget}_pretrain_{pretrain_suffix}.png",
                )
                plot_time_matched_noise_metric(
                    rows,
                    view_methods,
                    reference_budget,
                    pretrain_fraction,
                    "step",
                    "Selected steps reached",
                    (
                        "Selected Steps Reached Under PU-R Time Budget vs Label-Noise Rate "
                        f"(PU-R ES={reference_budget}, pretrain={pretrain_fraction:g})"
                    ),
                    view_dir / f"time_matched_steps_vs_noise_pur_es{reference_budget}_pretrain_{pretrain_suffix}.png",
                )
                plot_time_matched_noise_metric(
                    rows,
                    view_methods,
                    reference_budget,
                    pretrain_fraction,
                    "selection_runtime_sec",
                    "Selection-loop runtime seconds",
                    (
                        "Selection Runtime vs Label-Noise Rate "
                        f"(PU-R ES={reference_budget}, pretrain={pretrain_fraction:g})"
                    ),
                    view_dir
                    / f"time_matched_selection_runtime_vs_noise_pur_es{reference_budget}_pretrain_{pretrain_suffix}.png",
                )


def plot_time_matched_bounds_vs_noise(
    rows: list[dict[str, str]],
    methods: list[str],
    reference_es_budget: int,
    pretrain_fraction: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    show_pac_bayes = should_plot_pac_bayes(rows)
    for method in methods:
        xs, means, ses = grouped_time_matched_noise_curve(
            rows, method, pretrain_fraction, reference_es_budget, "test_inappropriate_risk"
        )
        if xs:
            plot_mean_band(ax, xs, means, ses, method, label=f"{method} risk", linestyle="-", alpha=0.16)

        xs, means, ses = grouped_time_matched_noise_curve(
            rows, method, pretrain_fraction, reference_es_budget, "certified_bound"
        )
        if xs:
            plot_mean_band(ax, xs, means, ses, method, label=f"{method} P2L", linestyle="--", alpha=0.10)

        xs, means, ses = grouped_time_matched_noise_curve(
            rows, method, pretrain_fraction, reference_es_budget, "pac_bayes_bound"
        )
        if show_pac_bayes and xs:
            plot_mean_band(ax, xs, means, ses, method, label=f"{method} PAC-Bayes", linestyle=":", alpha=0.08)

    ax.set_xlabel("Label-noise rate")
    ax.set_ylabel(
        "Test inappropriate risk / time-matched generalization bound"
        if show_pac_bayes
        else "Test inappropriate risk / time-matched P2L bound"
    )
    title = "Time-Matched P2L/PAC-Bayes Bounds" if show_pac_bayes else "Time-Matched P2L Bound"
    ax.set_title(f"{title} vs Label-Noise Rate (PU-R ES={reference_es_budget}, pretrain={pretrain_fraction:g})")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_time_matched_noise_metric(
    rows: list[dict[str, str]],
    methods: list[str],
    reference_es_budget: int,
    pretrain_fraction: float,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for method in methods:
        xs, means, ses = grouped_time_matched_noise_curve(
            rows, method, pretrain_fraction, reference_es_budget, metric
        )
        if not xs:
            continue
        plot_mean_band(ax, xs, means, ses, method)
    ax.set_xlabel("Label-noise rate")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_noise(results_path: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(results_path)
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in rows)]

    for view_name, view_methods in method_views(methods):
        view_dir = plots_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        plot_noise_metric(
            rows,
            view_methods,
            "test_error",
            "Clean test error",
            "Clean Test Error vs Label-Noise Rate",
            view_dir / "test_error_vs_noise.png",
        )
        plot_noise_metric(
            rows,
            view_methods,
            "test_inappropriate_risk",
            "Test inappropriate risk",
            "Test Inappropriate Risk vs Label-Noise Rate",
            view_dir / "test_inappropriate_risk_vs_noise.png",
        )
        plot_noise_metric(
            rows,
            view_methods,
            "compression_size",
            "Compression set size",
            "Compression Set Size vs Label-Noise Rate",
            view_dir / "compression_size_vs_noise.png",
        )
        plot_noise_bounds_vs_noise(rows, view_methods, view_dir / "bounds_vs_noise.png")
        plot_redundancy_diagnostics(rows, view_methods, view_dir / "redundancy_diagnostics_vs_noise.png")


def plot_noise_bounds_vs_noise(
    rows: list[dict[str, str]],
    methods: list[str],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    show_pac_bayes = should_plot_pac_bayes(rows)
    for method in methods:
        xs, means, ses = grouped_noise_curve(rows, method, "test_inappropriate_risk")
        if xs:
            plot_mean_band(ax, xs, means, ses, method, label=f"{method} risk", linestyle="-", alpha=0.16)

        xs, means, ses = grouped_noise_curve(rows, method, "certified_bound")
        if xs:
            plot_mean_band(ax, xs, means, ses, method, label=f"{method} P2L", linestyle="--", alpha=0.10)

        xs, means, ses = grouped_noise_curve(rows, method, "pac_bayes_bound")
        if show_pac_bayes and xs:
            plot_mean_band(ax, xs, means, ses, method, label=f"{method} PAC-Bayes", linestyle=":", alpha=0.08)
    ax.set_xlabel("Label-noise rate")
    ax.set_ylabel(
        "Test inappropriate risk / generalization bound"
        if show_pac_bayes
        else "Test inappropriate risk / P2L bound"
    )
    title = (
        "P2L/PAC-Bayes Bounds and Test Inappropriate Risk"
        if show_pac_bayes
        else "P2L Bound and Test Inappropriate Risk"
    )
    ax.set_title(f"{title} vs Label-Noise Rate")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_noise_metric(
    rows: list[dict[str, str]],
    methods: list[str],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for method in methods:
        xs, means, ses = grouped_noise_curve(rows, method, metric)
        if not xs:
            continue
        plot_mean_band(ax, xs, means, ses, method)
    ax.set_xlabel("Label-noise rate")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_redundancy_diagnostics(
    rows: list[dict[str, str]],
    methods: list[str],
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(SELECTION_DIAGNOSTIC_METRICS),
        1,
        figsize=(9, 3.1 * len(SELECTION_DIAGNOSTIC_METRICS)),
        sharex=True,
    )
    for ax, (metric, ylabel) in zip(axes, SELECTION_DIAGNOSTIC_METRICS):
        for method in methods:
            xs, means, ses = grouped_noise_curve(rows, method, metric)
            if not xs:
                continue
            plot_mean_band(ax, xs, means, ses, method)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Label-noise rate")
    axes[0].set_title("Selected-Set Redundancy Diagnostics vs Label-Noise Rate")
    axes[0].legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_mean_band(
    ax: plt.Axes,
    xs: list[float],
    means: list[float],
    ses: list[float],
    method: str,
    label: str | None = None,
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha: float = 0.18,
) -> None:
    x_arr = np.asarray(xs, dtype=np.float64)
    mean_arr = np.asarray(means, dtype=np.float64)
    se_arr = np.asarray(ses, dtype=np.float64)
    color = COLORS.get(method)
    ax.plot(
        x_arr,
        mean_arr,
        marker="o",
        markersize=4,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        label=label or method,
    )
    ax.fill_between(
        x_arr,
        mean_arr - se_arr,
        mean_arr + se_arr,
        color=color,
        alpha=alpha,
        linewidth=0,
    )
