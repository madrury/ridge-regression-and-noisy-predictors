"""Simple ridge regression, and regression on data with multiplicative noise.

Everything here works with a single predictor, which is enough to draw the
pictures in the post. The intercept is never penalized.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SimpleRegressionData:
    """A regression data set with a single predictor.

    Attributes
    ----------
    x : np.ndarray
        The predictor, standardized to mean zero and variance one.
    y : np.ndarray
        The response.
    """
    x: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.x)

    def tail_mask(self, p: float = 0.1) -> np.ndarray:
        """Flag the points with predictor in the lower or upper p-quantile.

        Parameters
        ----------
        p : float
            The quantile defining the tails.

        Returns
        -------
        np.ndarray
            Boolean array, True for points in either tail.
        """
        lower = np.quantile(self.x, p)
        upper = np.quantile(self.x, 1 - p)
        return (self.x <= lower) | (self.x >= upper)


class RidgeRegression:
    """Ridge regression of y on a single predictor, with an unpenalized
    intercept.

    The slope minimizes

        |y - β0 - β1 x|^2 + λ β1^2

    With λ = 0 this is ordinary least squares.

    Parameters
    ----------
    lam : float
        The ridge penalty λ.

    Attributes
    ----------
    lam : float
        The ridge penalty λ.
    intercept : float
        The fitted intercept, available after calling `fit`.
    slope : float
        The fitted slope, available after calling `fit`.
    """

    def __init__(self, lam: float = 0.0):
        self.lam = lam

        self.intercept: float
        self.slope: float

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """Fit the regression.

        Parameters
        ----------
        x : np.ndarray
            The predictor.
        y : np.ndarray
            The response.

        Returns
        -------
        RidgeRegression
            The fit regression, self.
        """
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        self.slope = (
            np.dot(x_centered, y_centered)
            / (np.dot(x_centered, x_centered) + self.lam)
        )
        self.intercept = np.mean(y) - self.slope * np.mean(x)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the response at new values of the predictor.

        Parameters
        ----------
        x : np.ndarray
            The predictor.

        Returns
        -------
        np.ndarray
            The predicted response.
        """
        return self.intercept + self.slope * x


def make_regression_data(
    rng: np.random.Generator,
    intercept: float = 0.0,
    slope: float = 1.0,
    N: int = 100,
    sd: float = 0.6,
) -> SimpleRegressionData:
    """Simulate a regression data set with occasional outlying responses.

    The predictor is standardized after simulation, as is conventional before
    fitting a ridge regression.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    intercept : float
        Intercept of the true regression line.
    slope : float
        Slope of the true regression line.
    N : int
        Number of observations.
    sd : float
        Standard deviation of the bulk of the noise in the response.

    Returns
    -------
    SimpleRegressionData
        The simulated data.
    """
    x = rng.normal(0.0, 1.0, size=N)
    x = (x - np.mean(x)) / np.std(x)
    # Two independent sources of occasional large errors fatten the tails.
    outliers_1 = (
        rng.choice([0, 1], size=N, p=[0.85, 0.15])
        * rng.normal(0.0, 2*sd, size=N)
    )
    outliers_2 = (
        rng.choice([0, 1], size=N, p=[0.85, 0.15])
        * rng.normal(0.0, 2*sd, size=N)
    )
    y = (
        intercept
        + slope * x
        + rng.normal(0.0, sd, size=N)
        + outliers_1
        + outliers_2
    )
    return SimpleRegressionData(x=x, y=y)


def add_multiplicative_noise(
    rng: np.random.Generator,
    x: np.ndarray,
    σ: float,
) -> np.ndarray:
    """Multiply each entry of x by an independent N(1, σ^2) random number.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    x : np.ndarray
        The data to perturb.
    σ : float
        Standard deviation of the multiplicative noise.

    Returns
    -------
    np.ndarray
        The perturbed data.
    """
    return x * rng.normal(1.0, σ, size=len(x))


def fit_noisy_regression(
    rng: np.random.Generator,
    data: SimpleRegressionData,
    σ: float,
) -> RidgeRegression:
    """Fit a least squares regression to one noisy copy of the data.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    data : SimpleRegressionData
        The original data.
    σ : float
        Standard deviation of the multiplicative noise.

    Returns
    -------
    RidgeRegression
        A least squares fit (λ = 0) to the noisy data.
    """
    x_noisy = add_multiplicative_noise(rng, data.x, σ)
    return RidgeRegression(lam=0.0).fit(x_noisy, data.y)


@dataclass
class NoisyCopiesSlopes:
    """The two ways of combining fits to the same K noisy copies of a data set.

    Attributes
    ----------
    noise_augmented : float
        Slope of one least squares fit to all K copies stacked together.
    averaged : float
        Average slope of K least squares fits, one to each copy.
    """
    noise_augmented: float
    averaged: float


def fit_noisy_copies(
    rng: np.random.Generator,
    data: SimpleRegressionData,
    σ: float,
    K: int,
) -> NoisyCopiesSlopes:
    """Draw K noisy copies of the data, and fit both the noise-augmented and
    the averaged regression to them.

    Both estimates use the same copies, so they differ only in how the copies
    are combined.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    data : SimpleRegressionData
        The original data.
    σ : float
        Standard deviation of the multiplicative noise.
    K : int
        Number of noisy copies.

    Returns
    -------
    NoisyCopiesSlopes
        The noise-augmented and averaged slopes.
    """
    x_copies = add_multiplicative_noise(rng, np.tile(data.x, K), σ)
    y_copies = np.tile(data.y, K)
    noise_augmented = RidgeRegression(lam=0.0).fit(x_copies, y_copies)
    single_fits = [
        RidgeRegression(lam=0.0).fit(x_copy, data.y)
        for x_copy in x_copies.reshape(K, len(data))
    ]
    return NoisyCopiesSlopes(
        noise_augmented=noise_augmented.slope,
        averaged=np.mean([fit.slope for fit in single_fits]),
    )


def equivalent_ridge_penalty(N: int, σ: float) -> float:
    """The ridge penalty equivalent to multiplicative noise with standard
    deviation σ, for standardized data with N observations.

    Parameters
    ----------
    N : int
        Number of observations.
    σ : float
        Standard deviation of the multiplicative noise.

    Returns
    -------
    float
        The penalty λ = Nσ^2.
    """
    return N * σ**2
