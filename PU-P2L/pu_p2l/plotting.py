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
from .io_utils import read_csv, to_float


COLORS = {
    "MaxLoss": "#274753",
    "PU-C": "#297270",
    "PU-R": "#299d8f",
    "Marginal": "#8ab07c",
    "PU-F": "#e7c66b",
    "PU-G": "#f3a361",
    "GREATS": "#e66d50",
}
METHOD_ORDER = ["MaxLoss", "PU-R", "PU-C", "Marginal", "PU-F", "PU-G", "GREATS"]
CERTIFIED_METHOD_ORDER = ["MaxLoss", "PU-R", "PU-C", "Marginal", "PU-F", "PU-G"]
BOUND_METHOD_ORDER = ["MaxLoss", "PU-R", "PU-C", "Marginal", "PU-F", "PU-G", "GREATS"]
GENERALIZATION_BOUND_CURVES = [
    ("test_error", "risk", "-", 1.8, 0.16),
    ("certified_bound", "P2L", "--", 1.7, 0.10),
    ("pac_bayes_bound", "PAC-Bayes", ":", 1.8, 0.08),
    ("self_selected_bound", "self-selected", "-.", 1.5, 0.08),
    ("ada_bound", "ADA growing", (0, (3, 1, 1, 1, 1, 1)), 1.5, 0.08),
]


def should_plot_pac_bayes(rows: list[dict[str, str]]) -> bool:
    dataset_names = {row.get("dataset", "").strip() for row in rows if row.get("dataset", "").strip()}
    if dataset_names and not any(pac_bayes_enabled_for_dataset(name) for name in dataset_names):
        return False
    return any(not np.isnan(to_float(row, "pac_bayes_bound")) for row in rows)


def format_float_for_filename(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


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


def plot_boundary(results_path: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(results_path)
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in rows)]
    noise_rates = sorted({to_float(row, "noise_rate") for row in rows})

    for noise_rate in noise_rates:
        suffix = format_float_for_filename(noise_rate)
        plot_certified_bound_and_risk(
            rows,
            methods,
            noise_rate,
            plots_dir / f"certified_bound_and_risk_vs_pretrain_noise_{suffix}.png",
        )
        plot_metric(
            rows,
            methods,
            noise_rate,
            "effective_compression_size",
            "Effective compression size",
            "Effective Compression Size vs Pretrain Fraction",
            plots_dir / f"effective_compression_vs_pretrain_noise_{suffix}.png",
        )
        plot_metric(
            rows,
            methods,
            noise_rate,
            "runtime_sec",
            "Runtime seconds",
            "Runtime vs Pretrain Fraction",
            plots_dir / f"runtime_vs_pretrain_noise_{suffix}.png",
        )


def plot_generalization_bounds(results_path: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(results_path)
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in rows)]
    noise_rates = sorted({to_float(row, "noise_rate") for row in rows})

    for noise_rate in noise_rates:
        suffix = format_float_for_filename(noise_rate)
        plot_generalization_bound_and_risk(
            rows,
            methods,
            noise_rate,
            plots_dir / f"generalization_bounds_vs_pretrain_noise_{suffix}.png",
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
    ax.set_ylabel("Clean test risk / generalization bound")
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
        xs, means, ses = grouped_curve(rows, method, noise_rate, "test_error")
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
    ax.set_ylabel("Clean test risk / generalization bound" if show_pac_bayes else "Clean test risk / P2L bound")
    title = (
        "P2L/PAC-Bayes Bounds and Clean Test Risk"
        if show_pac_bayes
        else "P2L Bound and Clean Test Risk"
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


def plot_es_budget_boundary(results_path: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(results_path)
    methods = [method for method in BOUND_METHOD_ORDER if any(row["method"] == method for row in rows)]
    noise_rates = sorted({to_float(row, "noise_rate") for row in rows})
    budgets = sorted({int(to_float(row, "es_budget")) for row in rows})

    for noise_rate in noise_rates:
        noise_suffix = format_float_for_filename(noise_rate)
        for budget in budgets:
            plot_es_budget_bound_and_risk_vs_pretrain(
                rows,
                methods,
                noise_rate,
                budget,
                plots_dir / f"es_budget_bound_and_risk_vs_pretrain_noise_{noise_suffix}_budget_{budget}.png",
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
        xs, means, ses = grouped_budget_pretrain_curve(rows, method, noise_rate, es_budget, "test_error")
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
        "Clean test risk / ES generalization bound" if show_pac_bayes else "Clean test risk / ES P2L bound"
    )
    title = (
        "ES P2L/PAC-Bayes Bounds and Clean Test Risk"
        if show_pac_bayes
        else "ES P2L Bound and Clean Test Risk"
    )
    ax.set_title(
        f"{title} vs Pretrain Fraction (noise={noise_rate:g}, ES={es_budget})"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_es_trace(results_path: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(results_path)
    methods = [method for method in BOUND_METHOD_ORDER if any(row["method"] == method for row in rows)]
    noise_rates = sorted({to_float(row, "noise_rate") for row in rows})
    pretrain_fractions = sorted({to_float(row, "pretrain_fraction") for row in rows})

    for noise_rate in noise_rates:
        noise_suffix = format_float_for_filename(noise_rate)
        for pretrain_fraction in pretrain_fractions:
            pretrain_suffix = format_float_for_filename(pretrain_fraction)
            plot_step_bound_and_risk(
                rows,
                methods,
                noise_rate,
                pretrain_fraction,
                plots_dir
                / f"es_bound_and_risk_vs_step_noise_{noise_suffix}_pretrain_{pretrain_suffix}.png",
            )
            plot_step_bound_and_risk(
                rows,
                methods,
                noise_rate,
                pretrain_fraction,
                plots_dir
                / f"es_bound_and_risk_vs_step_first_100_noise_{noise_suffix}_pretrain_{pretrain_suffix}.png",
                max_step=100,
            )
            plot_step_metric(
                rows,
                methods,
                noise_rate,
                pretrain_fraction,
                "remaining_bad",
                "Inappropriate points left",
                "Inappropriate Points Left vs Step",
                plots_dir / f"remaining_bad_vs_step_noise_{noise_suffix}_pretrain_{pretrain_suffix}.png",
            )
            plot_step_metric(
                rows,
                methods,
                noise_rate,
                pretrain_fraction,
                "remaining_bad",
                "Inappropriate points left",
                "Inappropriate Points Left vs Step",
                plots_dir / f"remaining_bad_vs_step_first_100_noise_{noise_suffix}_pretrain_{pretrain_suffix}.png",
                max_step=100,
            )


def plot_step_bound_and_risk(
    rows: list[dict[str, str]],
    methods: list[str],
    noise_rate: float,
    pretrain_fraction: float,
    path: Path,
    max_step: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    show_pac_bayes = should_plot_pac_bayes(rows)
    for method in methods:
        xs, means, ses = grouped_step_curve(
            rows, method, noise_rate, pretrain_fraction, "test_error", max_step=max_step
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
    ax.set_ylabel(
        "Clean test risk / ES generalization bound" if show_pac_bayes else "Clean test risk / ES P2L bound"
    )
    window_label = f", first {max_step:g} steps" if max_step is not None else ""
    title = (
        "ES P2L/PAC-Bayes Bounds and Clean Test Risk"
        if show_pac_bayes
        else "ES P2L Bound and Clean Test Risk"
    )
    ax.set_title(
        f"{title} vs Step{window_label} (noise={noise_rate:g}, pretrain={pretrain_fraction:g})"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
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
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_es_budget_noise(results_path: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(results_path)
    methods = [method for method in BOUND_METHOD_ORDER if any(row["method"] == method for row in rows)]
    budgets = sorted({int(to_float(row, "es_budget")) for row in rows})
    pretrain_fractions = sorted({to_float(row, "pretrain_fraction") for row in rows})

    for pretrain_fraction in pretrain_fractions:
        pretrain_suffix = format_float_for_filename(pretrain_fraction)
        for budget in budgets:
            plot_es_budget_noise_metric(
                rows,
                methods,
                budget,
                pretrain_fraction,
                "effective_compression_size",
                "ES effective compression size",
                f"ES Effective Compression Size vs Label-Noise Rate (ES={budget}, pretrain={pretrain_fraction:g})",
                plots_dir / f"es_budget_effective_compression_vs_noise_budget_{budget}_pretrain_{pretrain_suffix}.png",
            )
            plot_es_budget_noise_metric(
                rows,
                methods,
                budget,
                pretrain_fraction,
                "test_error",
                "Clean test risk",
                f"Clean Test Risk vs Label-Noise Rate (ES={budget}, pretrain={pretrain_fraction:g})",
                plots_dir / f"es_budget_test_risk_vs_noise_budget_{budget}_pretrain_{pretrain_suffix}.png",
            )
            plot_es_budget_bounds_vs_noise(
                rows,
                methods,
                budget,
                pretrain_fraction,
                plots_dir / f"es_budget_bounds_vs_noise_budget_{budget}_pretrain_{pretrain_suffix}.png",
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
        xs, means, ses = grouped_budget_noise_curve(rows, method, es_budget, pretrain_fraction, "test_error")
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
        "Clean test risk / ES generalization bound" if show_pac_bayes else "Clean test risk / ES P2L bound"
    )
    title = "ES P2L/PAC-Bayes Bounds" if show_pac_bayes else "ES P2L Bound and Clean Test Risk"
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


def plot_noise(results_path: Path, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(results_path)
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in rows)]

    plot_noise_metric(
        rows,
        methods,
        "test_error",
        "Clean test error",
        "Clean Test Error vs Label-Noise Rate",
        plots_dir / "test_error_vs_noise.png",
    )
    plot_noise_metric(
        rows,
        methods,
        "compression_size",
        "Compression set size",
        "Compression Set Size vs Label-Noise Rate",
        plots_dir / "compression_size_vs_noise.png",
    )
    plot_noise_bounds_vs_noise(rows, methods, plots_dir / "bounds_vs_noise.png")
    plot_redundancy_diagnostics(rows, methods, plots_dir / "redundancy_diagnostics_vs_noise.png")


def plot_noise_bounds_vs_noise(
    rows: list[dict[str, str]],
    methods: list[str],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    show_pac_bayes = should_plot_pac_bayes(rows)
    for method in methods:
        xs, means, ses = grouped_noise_curve(rows, method, "test_error")
        if xs:
            plot_mean_band(ax, xs, means, ses, method, label=f"{method} risk", linestyle="-", alpha=0.16)

        xs, means, ses = grouped_noise_curve(rows, method, "certified_bound")
        if xs:
            plot_mean_band(ax, xs, means, ses, method, label=f"{method} P2L", linestyle="--", alpha=0.10)

        xs, means, ses = grouped_noise_curve(rows, method, "pac_bayes_bound")
        if show_pac_bayes and xs:
            plot_mean_band(ax, xs, means, ses, method, label=f"{method} PAC-Bayes", linestyle=":", alpha=0.08)
    ax.set_xlabel("Label-noise rate")
    ax.set_ylabel("Clean test risk / generalization bound" if show_pac_bayes else "Clean test risk / P2L bound")
    title = (
        "P2L/PAC-Bayes Bounds and Clean Test Risk"
        if show_pac_bayes
        else "P2L Bound and Clean Test Risk"
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
    metrics = [
        ("noise_hit_rate", "Noise-hit rate"),
        ("duplicate_hit_rate", "Duplicate-hit rate"),
        ("pairwise_feature_cosine", "Pairwise feature cosine"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10.5), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
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
