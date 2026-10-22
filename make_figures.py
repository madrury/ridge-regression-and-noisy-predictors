"""Draw every figure in the post "Marginalizing Multiplicative Noise Applied to
Regression Predictors is Ridge Regression".

All figures share one simulated data set, and the random seed is fixed, so
running this script reproduces them exactly.

Usage:

    uv run make_figures.py [output-directory]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plots import (
    LEAST_SQUARES_COLOR,
    NOISE_AUGMENTED_COLOR,
    RIDGE_COLOR,
    label_xy_axes,
    plot_regression_line,
    plot_slope_convergence,
)
from ridge import (
    RidgeRegression,
    SimpleRegressionData,
    add_multiplicative_noise,
    equivalent_ridge_penalty,
    fit_noisy_copies,
    fit_noisy_regression,
    make_regression_data,
)

SEED = 154
NOISE_SDS = [0.01, 0.1, 0.5, 1.0, 5.0]
N_BUNDLE_LINES = 500
N_STACKED_COPIES = 200
# Only some of the stacked copies are drawn, so the points stay legible.
N_DRAWN_COPIES = 20


def save(fig: plt.Figure, output_dir: Path, name: str):
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=100)
    plt.close(fig)


def draw_regression_data(data: SimpleRegressionData, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(data.x, data.y, color="grey")
    label_xy_axes(ax, "Regression Data")
    save(fig, output_dir, "regression-data")


def draw_regression_data_with_line(
    data: SimpleRegressionData,
    output_dir: Path,
):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(data.x, data.y, color="grey")
    regression = RidgeRegression(lam=0.0).fit(data.x, data.y)
    plot_regression_line(
        ax,
        regression,
        color=LEAST_SQUARES_COLOR,
        linewidth=3,
    )
    label_xy_axes(ax, "Regression Data with Regression Line")
    save(fig, output_dir, "regression-data-w-line")


def draw_regression_data_with_ridge_lines(
    data: SimpleRegressionData,
    output_dir: Path,
):
    lams = np.logspace(-3, 3, num=25)
    colors = plt.cm.winter(np.linspace(0, 1, len(lams)))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(data.x, data.y, color="grey")
    for lam, color in zip(lams, colors):
        regression = RidgeRegression(lam=lam).fit(data.x, data.y)
        plot_regression_line(ax, regression, color=color, alpha=0.5)
    label_xy_axes(ax, "Ridge Regression for Various $\\lambda$")
    save(fig, output_dir, "regression-data-w-ridge-lines")


def draw_dispersed_data(
    rng: np.random.Generator,
    data: SimpleRegressionData,
    output_dir: Path,
):
    # The same noisy copy appears in both figures, so the shaded tails can be
    # compared point by point.
    x_noisy = add_multiplicative_noise(rng, data.x, σ=1.0)
    tail_colors = np.where(data.tail_mask(p=0.1), "green", "grey")
    # All three figures share an x range wide enough for every noisy point,
    # so they can be compared at the same scale.
    x_max = 1.05 * np.max(np.abs(x_noisy))
    x_lim = (-x_max, x_max)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(x_noisy, data.y, color="grey")
    ax.set_xlim(x_lim)
    label_xy_axes(ax, "Regression Data With Multiplicative Noise")
    save(fig, output_dir, "dispersed-data")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(data.x, data.y, color=tail_colors)
    ax.set_xlim(x_lim)
    label_xy_axes(ax, "Regression Data With Tails Highlighted")
    save(fig, output_dir, "regression-data-shaded-tails")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(x_noisy, data.y, color=tail_colors)
    ax.set_xlim(x_lim)
    label_xy_axes(
        ax,
        "Regression Data With Multiplicative Noise and Tails Highlighted",
    )
    save(fig, output_dir, "dispersed-data-shaded-tails")


def draw_dispersed_regression_lines(
    rng: np.random.Generator,
    data: SimpleRegressionData,
    output_dir: Path,
):
    fig, axs = plt.subplots(
        len(NOISE_SDS),
        1,
        figsize=(10, 10),
        sharey=True,
    )
    for ax, σ in zip(axs, NOISE_SDS):
        x_noisy = add_multiplicative_noise(rng, data.x, σ)
        ax.scatter(x_noisy, data.y, color="grey", alpha=0.5)
        ax.scatter(data.x, data.y, color="black")
        regression = RidgeRegression(lam=0.0).fit(x_noisy, data.y)
        plot_regression_line(
            ax,
            regression,
            color=LEAST_SQUARES_COLOR,
            linewidth=2,
        )
        ax.set_title(f"Multiplicative Noise $\\sigma = ${σ:2.2f}")
        ax.set_xlim(-3, 3)
    save(fig, output_dir, "dispersed-regression-lines")


def draw_regression_line_bundles(
    rng: np.random.Generator,
    data: SimpleRegressionData,
    output_dir: Path,
    with_ridge: bool,
):
    fig, axs = plt.subplots(
        len(NOISE_SDS),
        1,
        figsize=(10, 10),
        sharey=True,
    )
    for ax, σ in zip(axs, NOISE_SDS):
        ax.scatter(data.x, data.y, color="black")
        for _ in range(N_BUNDLE_LINES):
            regression = fit_noisy_regression(rng, data, σ)
            plot_regression_line(ax, regression, color="grey", alpha=0.05)
        ax.set_xlim(-3, 3)
        if with_ridge:
            lam = equivalent_ridge_penalty(len(data), σ)
            ridge = RidgeRegression(lam=lam).fit(data.x, data.y)
            plot_regression_line(ax, ridge, color=RIDGE_COLOR, linewidth=2)
            ax.set_title(
                f"Ridge $\\lambda = ${lam:4.2f}, Noise $\\sigma = ${σ:2.2f}"
            )
        else:
            ax.set_title(f"Multiplicative Noise $\\sigma = ${σ:2.2f}")
    if with_ridge:
        # The bundle lines are too faint to serve as their own legend swatch.
        handles = [
            Line2D([], [], color="grey", label="Fits to single noisy copies"),
            Line2D(
                [],
                [],
                color=RIDGE_COLOR,
                linewidth=2,
                label="Ridge, $\\lambda = N\\sigma^2$",
            ),
        ]
        axs[0].legend(handles=handles, loc="upper left")
        save(fig, output_dir, "regression-comparisons")
    else:
        save(fig, output_dir, "dispersed-regression-lines-bundle")


def draw_noise_augmented_regression_lines(
    rng: np.random.Generator,
    data: SimpleRegressionData,
    output_dir: Path,
    with_ridge: bool,
):
    fig, axs = plt.subplots(
        len(NOISE_SDS),
        1,
        figsize=(10, 10),
        sharey=True,
    )
    for ax, σ in zip(axs, NOISE_SDS):
        x_stacked = add_multiplicative_noise(
            rng,
            np.tile(data.x, N_STACKED_COPIES),
            σ,
        )
        y_stacked = np.tile(data.y, N_STACKED_COPIES)
        n_drawn = N_DRAWN_COPIES * len(data)
        ax.scatter(
            x_stacked[:n_drawn],
            y_stacked[:n_drawn],
            color="grey",
            alpha=0.2,
            s=10,
            label=f"Noisy copies ({N_DRAWN_COPIES} of {N_STACKED_COPIES} drawn)",
        )
        ax.scatter(data.x, data.y, color="black", label="Original data")
        regression = RidgeRegression(lam=0.0).fit(x_stacked, y_stacked)
        ax.set_xlim(-3, 3)
        if with_ridge:
            lam = equivalent_ridge_penalty(len(data), σ)
            ridge = RidgeRegression(lam=lam).fit(data.x, data.y)
            # Ridge is drawn wide underneath, so the dashed line on top of it
            # shows the two coinciding.
            plot_regression_line(
                ax,
                ridge,
                color=RIDGE_COLOR,
                linewidth=5,
                label="Ridge, $\\lambda = N\\sigma^2$",
            )
            plot_regression_line(
                ax,
                regression,
                color=NOISE_AUGMENTED_COLOR,
                linewidth=2,
                linestyle="--",
                label="Noise-augmented",
            )
            ax.set_title(
                f"Ridge $\\lambda = ${lam:4.2f}, Noise $\\sigma = ${σ:2.2f}"
            )
        else:
            plot_regression_line(
                ax,
                regression,
                color=NOISE_AUGMENTED_COLOR,
                linewidth=2,
                label="Noise-augmented",
            )
            ax.set_title(
                f"{N_STACKED_COPIES} Stacked Noisy Copies, "
                f"Noise $\\sigma = ${σ:2.2f}"
            )
    axs[0].legend(loc="upper left")
    if with_ridge:
        save(fig, output_dir, "noise-augmented-regression-comparisons")
    else:
        save(fig, output_dir, "noise-augmented-regression-lines")


def draw_stacked_convergence(
    rng: np.random.Generator,
    data: SimpleRegressionData,
    output_dir: Path,
):
    Ks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    n_replicates = 50
    n_single_fits = 20_000
    σs = [0.5, 1.0]

    fig, axs = plt.subplots(1, len(σs), figsize=(14, 4), sharey=False)
    for ax, σ in zip(axs, σs):
        fits = [
            [fit_noisy_copies(rng, data, σ, K) for _ in range(n_replicates)]
            for K in Ks
        ]
        noise_augmented_slopes = np.array([
            [fit.noise_augmented for fit in fits_at_K]
            for fits_at_K in fits
        ])
        averaged_slopes = np.array([
            [fit.averaged for fit in fits_at_K]
            for fits_at_K in fits
        ])
        averaged_limit = np.mean([
            fit_noisy_regression(rng, data, σ).slope
            for _ in range(n_single_fits)
        ])
        lam = equivalent_ridge_penalty(len(data), σ)
        ridge_slope = RidgeRegression(lam=lam).fit(data.x, data.y).slope
        plot_slope_convergence(
            ax,
            Ks,
            noise_augmented_slopes,
            averaged_slopes,
            ridge_slope=ridge_slope,
            averaged_limit=averaged_limit,
            σ=σ,
        )
    axs[0].legend(loc="upper right")
    save(fig, output_dir, "stacked-regression-convergence")


def main():
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("img")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("ggplot")
    rng = np.random.default_rng(SEED)
    data = make_regression_data(rng)

    draw_regression_data(data, output_dir)
    draw_regression_data_with_line(data, output_dir)
    draw_regression_data_with_ridge_lines(data, output_dir)
    draw_dispersed_data(rng, data, output_dir)
    draw_dispersed_regression_lines(rng, data, output_dir)
    draw_regression_line_bundles(rng, data, output_dir, with_ridge=False)
    draw_regression_line_bundles(rng, data, output_dir, with_ridge=True)
    draw_stacked_convergence(rng, data, output_dir)
    # Drawn last, so the random draws behind the earlier figures are unchanged.
    draw_noise_augmented_regression_lines(
        rng,
        data,
        output_dir,
        with_ridge=False,
    )
    draw_noise_augmented_regression_lines(
        rng,
        data,
        output_dir,
        with_ridge=True,
    )


if __name__ == "__main__":
    main()
