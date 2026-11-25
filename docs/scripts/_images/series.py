"""Figures of the discrete-time processes, time series, models and noises."""

from pathlib import Path

import numpy as np

from samplepaths import (
    AGARCHProcess,
    ARCHProcess,
    ARIMAProcess,
    AutoregressiveProcess,
    BatesModel,
    BernoulliProcess,
    ChineseRestaurantProcess,
    ColoredNoise,
    CorrelatedBrownianMotions,
    DirichletProcess,
    EGARCHProcess,
    FractionalGaussianNoise,
    GARCHProcess,
    GaussianNoise,
    HestonModel,
    HullWhiteTwoFactorModel,
    MarkovChain,
    MoranProcess,
    MovingAverageProcess,
    RandomWalk,
    SABRModel,
    SARIMAProcess,
    TGARCHProcess,
    TwoFactorCoxIngersollRossModel,
)

from .helpers import (
    PATHS,
    STEPS,
    plot_components,
    plot_noise,
    plot_sequence,
    plot_series,
    save_plot,
)


def make_discrete(output: Path) -> None:
    """Generate the figures of the discrete-time processes."""
    plot_sequence(output, "bernoulli_process", "Bernoulli process", BernoulliProcess(p=0.5), n=16)
    plot_sequence(
        output,
        "chinese_restaurant_process",
        "Chinese restaurant process",
        ChineseRestaurantProcess(),
        n=64,
    )
    plot_sequence(output, "dirichlet_process", "Dirichlet process", DirichletProcess(), n=64)
    plot_sequence(
        output,
        "markov_chain",
        "Markov chain (5 states)",
        MarkovChain(transition=np.full((5, 5), 0.2)),
        n=32,
    )
    plot_sequence(output, "random_walk", "Random walk", RandomWalk(), n=64, paths=PATHS)

    moran = MoranProcess(maximum=20)
    ys = [moran.sample_with_start(50, start=10) for _ in range(PATHS)]
    xs = [np.arange(len(y), dtype=np.float64) for y in ys]
    save_plot(output, "moran_process", "Moran process", xs, ys, scatter=True, xlabel="Step")


def make_timeseries(output: Path) -> None:
    """Generate the figures of the time series models."""
    plot_series(
        output,
        "autoregressive_process",
        "Autoregressive process (coefficients 0.6 and 0.3)",
        AutoregressiveProcess(coefficients=[0.6, 0.3]),
    )
    plot_series(
        output,
        "moving_average_process",
        "Moving average process (coefficients 0.6 and 0.3)",
        MovingAverageProcess(coefficients=[0.6, 0.3]),
    )
    plot_series(
        output,
        "arima_process",
        "ARIMA(1, 1, 1) process",
        ARIMAProcess(ar=0.5, ma=0.3, d=1),
        paths=PATHS,
    )
    plot_series(
        output,
        "sarima_process",
        "Seasonal ARIMA process with period 12",
        SARIMAProcess(ar=0.5, seasonal_ma=0.4, seasonal_d=1, period=12),
        n=144,
    )
    plot_series(output, "arch_process", "ARCH(1) process", ARCHProcess(), n=1000)
    plot_series(output, "garch_process", "GARCH(1, 1) process", GARCHProcess(), n=1000)
    plot_series(output, "tgarch_process", "Threshold GARCH process", TGARCHProcess(), n=1000)
    plot_series(output, "agarch_process", "Asymmetric GARCH process", AGARCHProcess(), n=1000)
    plot_series(output, "egarch_process", "Exponential GARCH process", EGARCHProcess(), n=1000)


def make_multivariate(output: Path) -> None:
    """Generate the figures of the processes with several components."""
    plot_components(
        output,
        "correlated_brownian_motions",
        "Correlated Brownian motions (correlation=0.8)",
        CorrelatedBrownianMotions(correlation=0.8),
        together=True,
    )
    plot_components(
        output,
        "heston_model",
        "Heston model",
        HestonModel(drift=0.05, speed=3.0, mean=0.04, vol=0.5, correlation=-0.7),
    )
    plot_components(
        output,
        "bates_model",
        "Bates model",
        BatesModel(
            drift=0.05, speed=3.0, mean=0.04, vol=0.5, correlation=-0.7, rate=4.0, jump_mean=-0.08
        ),
    )
    plot_components(
        output,
        "hull_white_two_factor_model",
        "Two-factor Hull-White model",
        HullWhiteTwoFactorModel(
            speed=2.0, drift=0.1, vol=0.01, factor_speed=0.3, factor_vol=0.05, correlation=-0.5, t=5
        ),
    )
    plot_components(
        output,
        "two_factor_cox_ingersoll_ross_model",
        "Two-factor Cox-Ingersoll-Ross model",
        TwoFactorCoxIngersollRossModel(t=5),
        together=True,
    )
    plot_components(
        output,
        "sabr_model",
        "SABR model (beta=0.5)",
        SABRModel(volvol=0.8, beta=0.5, correlation=-0.5, initial_volatility=0.2),
    )


def make_noise(output: Path) -> None:
    """Generate the figures of the noise processes."""
    plot_noise(output, "gaussian_noise", "Gaussian noise", GaussianNoise(), n=2**8)
    plot_noise(
        output,
        "fractional_gaussian_noise",
        "Fractional Gaussian noise (hurst=0.2)",
        FractionalGaussianNoise(hurst=0.2),
        n=2**8,
    )

    colors = {"violet": -2, "blue": -1, "white": 0, "pink": 1, "red": 2}
    for color, beta in colors.items():
        plot_noise(
            output,
            f"{color}_noise",
            f"{color.capitalize()} noise",
            ColoredNoise(beta=beta),
        )

    times = np.linspace(0, 1, STEPS + 1)[1:]
    ys = [ColoredNoise(beta=colors[color]).sample(STEPS) for color in ("white", "pink", "red")]
    save_plot(output, "colored_noise", "Colored noise", [times] * len(ys), ys)
