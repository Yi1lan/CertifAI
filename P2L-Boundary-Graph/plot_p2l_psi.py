from __future__ import annotations

import argparse
import math
from functools import lru_cache
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_N_VALUES = (500, 200, 1000)
DEFAULT_DELTAS = (1e-6, 1e-4, 1e-2)
LINE_STYLES = ("-", "--", "-.")
EXPERIMENT_COLORS = (
    "#274753",
    "#297270",
    "#299d8f",
    "#8ab07c",
    "#e7c66b",
    "#f3a361",
    "#e66d50",
)
REFERENCE_COLOR = "0.45"


@lru_cache(maxsize=None)
def p2l_psi(k: int, n: int, delta: float) -> float:
    """Return the P2L sample-compression Psi bound for compression size k."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1).")

    k = int(max(0, k))
    if k >= n:
        return 1.0

    log_m_choose_k = np.array(
        [
            math.lgamma(m + 1) - math.lgamma(m - k + 1) - math.lgamma(k + 1)
            for m in range(k, n)
        ],
        dtype=np.float64,
    )
    log_n_choose_k = math.lgamma(n + 1) - math.lgamma(n - k + 1) - math.lgamma(k + 1)
    coeffs = log_m_choose_k - log_n_choose_k
    m_vec = np.arange(k, n, dtype=np.float64)

    t_low = 0.0
    t_high = 1.0
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        for _ in range(90):
            t = 0.5 * (t_low + t_high)
            log_t = math.log(max(t, 1e-300))
            terms = np.exp(coeffs - (n - m_vec) * log_t)
            value = 1.0 - (delta / n) * float(np.sum(terms))
            if value > 0.0:
                t_high = t
            else:
                t_low = t

    return float(1.0 - t_low)


def delta_label(delta: float) -> str:
    exponent = math.log10(delta)
    if abs(exponent - round(exponent)) < 1e-12:
        return rf"$\delta=10^{{{int(round(exponent))}}}$"
    return rf"$\delta={delta:g}$"


def plot_single_n(ax: plt.Axes, n: int, deltas: tuple[float, ...]) -> None:
    ks = np.arange(n + 1)

    for delta, color, line_style in zip(deltas, cycle(EXPERIMENT_COLORS), cycle(LINE_STYLES)):
        psi_values = np.array([p2l_psi(int(k), n, float(delta)) for k in ks])
        ax.plot(
            ks,
            psi_values,
            color=color,
            linestyle=line_style,
            linewidth=1.3,
            label=delta_label(delta),
        )

    ax.plot(
        ks,
        ks / n,
        color=REFERENCE_COLOR,
        linestyle=":",
        linewidth=1.0,
        label=r"$k/N$",
    )

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$k$")
    ax.set_title(rf"$N={n}$")
    ax.grid(True, color="0.85", linewidth=0.6)


def plot_p2l_psi(n_values: tuple[int, ...], deltas: tuple[float, ...], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    fig_width = 3.4 if len(n_values) == 1 else 3.2 * len(n_values)
    fig, axes = plt.subplots(
        1,
        len(n_values),
        figsize=(fig_width, 2.6),
        sharey=True,
        squeeze=False,
    )

    for ax, n in zip(axes[0], n_values):
        plot_single_n(ax, n, deltas)

    axes[0][0].set_ylabel(r"$\Psi_N(k,\delta)$")
    axes[0][-1].legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout(pad=0.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_p2l_psi_separately(
    n_values: tuple[int, ...],
    deltas: tuple[float, ...],
    output_dir: Path,
) -> None:
    for n in n_values:
        output = output_dir / f"p2l_psi_N{n}.png"
        plot_p2l_psi(n_values=(n,), deltas=deltas, output=output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the P2L Psi/compression-bound function against k."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Optional single N value to plot separately.",
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=None,
        help="Optional N values to plot separately. Default: 500 200 1000.",
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=DEFAULT_DELTAS,
        help="Delta values to plot. Default: 1e-6 1e-4 1e-2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for separate output images. Default: current directory.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=None,
        help="Optional path for one combined multi-panel image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_values is not None:
        n_values = tuple(args.n_values)
    elif args.n is not None:
        n_values = (args.n,)
    else:
        n_values = DEFAULT_N_VALUES

    deltas = tuple(float(delta) for delta in args.deltas)
    plot_p2l_psi_separately(n_values=n_values, deltas=deltas, output_dir=args.output_dir)
    if args.combined_output is not None:
        plot_p2l_psi(n_values=n_values, deltas=deltas, output=args.combined_output)


if __name__ == "__main__":
    main()
