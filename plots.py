"""Plotting functions for the figures in the post.

Each function draws onto an Axes passed in by the caller, and returns it.
"""
import numpy as np
from matplotlib.axes import Axes

from ridge import RidgeRegression

# Each kind of regression line has one color across every figure in the post.
RIDGE_COLOR = "green"
NOISE_AUGMENTED_COLOR = "#348ABD"
LEAST_SQUARES_COLOR = "#555555"
AVERAGED_COLOR = "#988ED5"


def plot_regression_line(
    ax: Axes,
    regression: RidgeRegression,
    x_lim: tuple[float, float] = (-3.5, 3.5),
    **kwargs,
) -> Axes:
    """Draw the fitted line of a regression.

    Parameters
    ----------
    ax : Axes
        The axes to draw on.
    regression : RidgeRegression
        A fit regression.
    x_lim : tuple[float, float]
        The range of the predictor over which to draw the line.
    **kwargs
        Passed to `ax.plot`.

    Returns
    -------
    Axes
        The axes drawn on.
    """
    t = np.linspace(x_lim[0], x_lim[1], num=100)
    ax.plot(t, regression.predict(t), **kwargs)
    return ax


def label_xy_axes(ax: Axes, title: str) -> Axes:
    """Set the title, and label the axes of a scatterplot of y against x.

    Parameters
    ----------
    ax : Axes
        The axes to label.
    title : str
        The title of the plot.

    Returns
    -------
    Axes
        The axes labeled.
    """
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    return ax


def plot_slope_convergence(
    ax: Axes,
    Ks: np.ndarray,
    noise_augmented_slopes: np.ndarray,
    averaged_slopes: np.ndarray,
    ridge_slope: float,
    averaged_limit: float,
    σ: float,
) -> Axes:
    """Show noise-augmented and averaged slopes, each fit to K noisy copies
    of the data, converging to their limits as K grows.

    Parameters
    ----------
    ax : Axes
        The axes to draw on.
    Ks : np.ndarray
        The numbers of noisy copies, shape (n_K,).
    noise_augmented_slopes : np.ndarray
        Slopes of repeated noise-augmented fits, shape (n_K, n_replicates).
    averaged_slopes : np.ndarray
        Slopes of repeated averaged fits, shape (n_K, n_replicates).
    ridge_slope : float
        Slope of the ridge regression with λ = Nσ^2, the limit of the
        noise-augmented slopes.
    averaged_limit : float
        The limit of the averaged slopes as K grows.
    σ : float
        Standard deviation of the multiplicative noise, for the title.

    Returns
    -------
    Axes
        The axes drawn on.
    """
    # On the log scale, multiplicative offsets put the two clouds at equal
    # distances either side of each K.
    clouds = [
        (noise_augmented_slopes, 0.85, NOISE_AUGMENTED_COLOR),
        (averaged_slopes, 1.15, AVERAGED_COLOR),
    ]
    for slopes_by_K, offset, color in clouds:
        for K, slopes in zip(Ks, slopes_by_K):
            ax.scatter(
                np.full(len(slopes), offset * K),
                slopes,
                color=color,
                alpha=0.25,
                s=12,
            )
    # Each cloud is drawn once per K, so each gets a single proxy entry in
    # the legend.
    ax.scatter(
        [],
        [],
        color=NOISE_AUGMENTED_COLOR,
        s=12,
        label="Noise-augmented, from $K$ copies",
    )
    ax.scatter(
        [],
        [],
        color=AVERAGED_COLOR,
        s=12,
        label="Averaged, from $K$ copies",
    )
    ax.axhline(
        ridge_slope,
        color=RIDGE_COLOR,
        linewidth=2,
        label="Ridge, $\\lambda = N\\sigma^2$",
    )
    ax.axhline(
        averaged_limit,
        color=AVERAGED_COLOR,
        linewidth=2,
        linestyle="--",
        label="Averaged, $K \\to \\infty$",
    )
    ax.set_xscale("log")
    ax.set_title(f"Noise $\\sigma = ${σ:2.2f}")
    ax.set_xlabel("Number of noisy copies $K$")
    ax.set_ylabel("Fitted slope")
    return ax
