"""Generate the figures shown in the documentation.

Run from the repository root::

    make docs-images

or directly, optionally with another output directory::

    uv run python docs/scripts/make_images.py --output /tmp/images
"""

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from plot_style import NEUTRAL, PALETTE, use_theme  # noqa: E402

from samplepaths import (  # noqa: E402
    AGARCHProcess,
    ARCHProcess,
    ARIMAProcess,
    AutoregressiveProcess,
    BernoulliProcess,
    BesselProcess,
    BrownianBridge,
    BrownianExcursion,
    BrownianMeander,
    BrownianMotion,
    CauchyProcess,
    ChineseRestaurantProcess,
    ColoredNoise,
    ConstantElasticityVarianceProcess,
    CoxIngersollRossProcess,
    DirichletProcess,
    EGARCHProcess,
    ExtendedVasicekProcess,
    FractionalBrownianMotion,
    FractionalCoxIngersollRossProcess,
    FractionalGaussianNoise,
    FractionalGeometricBrownianMotion,
    FractionalJacobiProcess,
    FractionalOrnsteinUhlenbeckProcess,
    FractionalVasicekProcess,
    GammaProcess,
    GARCHProcess,
    GaussianNoise,
    GeometricBrownianMotion,
    InverseGaussianProcess,
    MarkovChain,
    MixedPoissonProcess,
    MoranProcess,
    MovingAverageProcess,
    MultifractionalBrownianMotion,
    OrnsteinUhlenbeckProcess,
    PoissonProcess,
    RandomWalk,
    SARIMAProcess,
    SquaredBesselProcess,
    TGARCHProcess,
    VarianceGammaProcess,
    VasicekProcess,
    WienerProcess,
    random,  # noqa: E402
)
from samplepaths.base import BaseProcess, BaseTimeProcess  # noqa: E402

DEFAULT_OUTPUT = Path(__file__).parent.parent / "_static"

#: Number of realizations drawn in each figure.
PATHS = 3

#: Number of increments of the continuous-time realizations.
STEPS = 2**12

Array = npt.NDArray[np.float64] | npt.NDArray[np.int64]


def save_plot(
    output: Path,
    name: str,
    title: str,
    xs: Sequence[Array],
    ys: Sequence[Array],
    *,
    scatter: bool = False,
    xlabel: str = "Time",
    ylabel: str = "Value",
) -> None:
    """Plot every (x, y) pair on one figure and save it as ``name``.png."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for x, y in zip(xs, ys, strict=True):
        if scatter:
            ax.scatter(x, y, s=18)
        else:
            ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output / f"{name}.png")
    plt.close(fig)
    print(f"{name}.png saved")


def plot_paths(
    output: Path,
    name: str,
    title: str,
    process: BaseTimeProcess,
    *,
    n: int = STEPS,
    paths: int = PATHS,
) -> None:
    """Plot realizations of a continuous-time process against time."""
    times = np.linspace(0, process.t, n + 1)
    ys = [process.sample(n) for _ in range(paths)]
    save_plot(output, name, title, [times] * paths, ys)


def plot_noise(
    output: Path,
    name: str,
    title: str,
    process: BaseTimeProcess,
    *,
    n: int = STEPS,
) -> None:
    """Plot one realization of a noise process against time."""
    times = np.linspace(0, process.t, n + 1)[1:]
    save_plot(output, name, title, [times], [process.sample(n)])


def plot_sequence(
    output: Path,
    name: str,
    title: str,
    process: BaseProcess,
    *,
    n: int,
    paths: int = 1,
) -> None:
    """Plot realizations of a discrete-time process against the step index."""
    ys = [process.sample(n) for _ in range(paths)]
    xs = [np.arange(len(y), dtype=np.float64) for y in ys]
    save_plot(output, name, title, xs, ys, scatter=True, xlabel="Step")


def plot_arrivals(
    output: Path,
    name: str,
    title: str,
    process: BaseProcess,
    *,
    n: int = 500,
    paths: int = PATHS,
) -> None:
    """Plot the counting process of arrival time realizations."""
    xs = [process.sample(n) for _ in range(paths)]
    ys = [np.arange(len(x), dtype=np.float64) for x in xs]
    save_plot(output, name, title, xs, ys, ylabel="Arrivals")


def make_continuous(output: Path) -> None:
    """Generate the figures of the continuous-time processes."""
    plot_paths(output, "bessel_process", "Bessel process (dim=3)", BesselProcess(dim=3))
    plot_paths(output, "brownian_bridge", "Brownian bridge (b=0)", BrownianBridge(b=0))
    plot_paths(output, "brownian_excursion", "Brownian excursion", BrownianExcursion())
    plot_paths(output, "brownian_meander", "Brownian meander", BrownianMeander())
    plot_paths(output, "brownian_motion", "Brownian motion", BrownianMotion())
    plot_paths(output, "cauchy_process", "Cauchy process", CauchyProcess())
    plot_paths(
        output,
        "fractional_brownian_motion",
        "Fractional Brownian motion (hurst=0.7)",
        FractionalBrownianMotion(hurst=0.7),
    )
    plot_paths(
        output,
        "gamma_process",
        "Gamma process (mean=1, variance=1)",
        GammaProcess(mean=1, variance=1),
    )
    plot_paths(
        output,
        "geometric_brownian_motion",
        "Geometric Brownian motion (drift=0, volatility=1)",
        GeometricBrownianMotion(drift=0, volatility=1),
    )
    plot_paths(
        output,
        "inverse_gaussian",
        "Inverse Gaussian process",
        InverseGaussianProcess(),
    )
    plot_paths(
        output,
        "multifractional_brownian_motion",
        "Multifractional Brownian motion",
        MultifractionalBrownianMotion(
            hurst=lambda t: math.sin(t * 4 * math.pi) * 0.4 + 0.5,
        ),
        n=2**9,
    )
    plot_paths(
        output,
        "squared_bessel_process",
        "Squared Bessel process (dim=3)",
        SquaredBesselProcess(dim=3),
    )
    plot_paths(
        output,
        "variance_gamma_process",
        "Variance Gamma process (drift=0, variance=1, scale=1)",
        VarianceGammaProcess(drift=0, variance=1, scale=1),
    )
    plot_paths(output, "wiener_process", "Wiener process", WienerProcess())

    plot_arrivals(
        output,
        "poisson_process",
        "Poisson process (rate=1)",
        PoissonProcess(rate=1),
    )
    plot_arrivals(
        output,
        "mixed_poisson_process",
        "Mixed Poisson process",
        MixedPoissonProcess(
            rate_func=random.generator.uniform,
            rate_args=(1, 5),
        ),
    )


def make_diffusion(output: Path) -> None:
    """Generate the figures of the diffusion processes."""
    plot_paths(
        output,
        "constant_elasticity_variance_process",
        "CEV process",
        ConstantElasticityVarianceProcess(),
    )
    plot_paths(
        output,
        "cox_ingersoll_ross_process",
        "CIR process",
        CoxIngersollRossProcess(),
    )
    plot_paths(
        output,
        "extended_vasicek_process",
        "Extended Vasicek process",
        ExtendedVasicekProcess(),
    )
    plot_paths(
        output,
        "ornstein_uhlenbeck_process",
        "Ornstein-Uhlenbeck process",
        OrnsteinUhlenbeckProcess(),
    )
    plot_paths(output, "vasicek_process", "Vasicek process", VasicekProcess())
    plot_paths(
        output,
        "fractional_vasicek_process",
        "Fractional Vasicek process (hurst=0.8)",
        FractionalVasicekProcess(hurst=0.8, speed=3.0, mean=1.0, vol=0.5),
    )
    plot_paths(
        output,
        "fractional_ornstein_uhlenbeck_process",
        "Fractional Ornstein-Uhlenbeck process (hurst=0.2)",
        FractionalOrnsteinUhlenbeckProcess(hurst=0.2, speed=3.0, vol=0.5),
    )
    plot_paths(
        output,
        "fractional_cox_ingersoll_ross_process",
        "Fractional CIR process (hurst=0.8)",
        FractionalCoxIngersollRossProcess(hurst=0.8, speed=3.0, mean=1.0, vol=0.5),
    )
    plot_paths(
        output,
        "fractional_jacobi_process",
        "Fractional Jacobi process (hurst=0.8)",
        FractionalJacobiProcess(hurst=0.8, speed=3.0, mean=0.5, vol=0.6),
    )
    plot_paths(
        output,
        "fractional_geometric_brownian_motion",
        "Fractional geometric Brownian motion (hurst=0.8)",
        FractionalGeometricBrownianMotion(hurst=0.8, drift=0.5, volatility=0.5),
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


def plot_series(
    output: Path,
    name: str,
    title: str,
    process: BaseProcess,
    *,
    n: int = 300,
    paths: int = 1,
) -> None:
    """Plot realizations of a time series model against the step index."""
    ys = [process.sample(n) for _ in range(paths)]
    xs = [np.arange(n, dtype=np.float64)] * paths
    save_plot(output, name, title, xs, ys, xlabel="Step")


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


def make_branding(output: Path) -> None:
    """Generate the logo and the banner of the documentation."""
    rng = np.random.default_rng(7)

    # Logo: three sample paths in a square, without axes.
    fig, ax = plt.subplots(1, 1, figsize=(1.6, 1.6))
    bm = BrownianMotion(rng=rng)
    for width in (2.6, 2.6, 2.6):
        ax.plot(np.linspace(0, 1, 49), bm.sample(48), linewidth=width, solid_capstyle="round")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.96)
    fig.savefig(output / "logo.svg")
    plt.close(fig)
    print("logo.svg saved")

    # Banner: many realizations of one process with their quantile bands.
    n, paths = 400, 400
    process = BrownianMotion(drift=0.6, scale=1.0, rng=rng)
    times = np.linspace(0, 1, n + 1)
    samples = np.array([process.sample(n) for _ in range(paths)])

    fig, ax = plt.subplots(1, 1, figsize=(9, 2.8))
    for low, high, alpha in ((0.05, 0.95, 0.12), (0.25, 0.75, 0.2)):
        ax.fill_between(
            times,
            np.quantile(samples, low, axis=0),
            np.quantile(samples, high, axis=0),
            color=PALETTE[0],
            alpha=alpha,
            linewidth=0,
        )
    for index, sample in enumerate(samples[:5]):
        ax.plot(times, sample, color=PALETTE[index % len(PALETTE)], linewidth=1.1)
    ax.plot(times, samples.mean(axis=0), color=NEUTRAL, linewidth=1.4, linestyle="--")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0.02, top=0.98)
    fig.savefig(output / "banner.png", dpi=160)
    plt.close(fig)
    print("banner.png saved")


def make_guides(output: Path) -> None:
    """Generate the explanatory figures of the user guide."""
    rng = np.random.default_rng(11)

    # The five families of processes: three panels above two.
    fig, panels = plt.subplot_mosaic("AABBCC;.DDEE.", figsize=(11, 5))
    ax = panels["A"]
    process: BaseTimeProcess = BrownianMotion(rng=rng)
    for _ in range(3):
        ax.plot(np.linspace(0, 1, 513), process.sample(512))
    ax.set_title("Continuous-time: Brownian motion")
    ax = panels["B"]
    process = VasicekProcess(speed=6.0, mean=0.0, vol=0.6, rng=rng)
    for initial in (2.0, 1.0, -1.5):
        ax.plot(np.linspace(0, 1, 513), process.sample_with_initial(512, initial=initial))
    ax.set_title("Diffusion: Vasicek process")
    ax = panels["C"]
    walk = RandomWalk(rng=rng)
    for _ in range(3):
        ax.step(np.arange(41), walk.sample(40), where="post")
    ax.set_title("Discrete-time: random walk")
    ax = panels["D"]
    ax.plot(GARCHProcess(omega=0.05, alpha=0.15, beta=0.8, rng=rng).sample(500), linewidth=0.8)
    ax.set_title("Time series: GARCH process")
    ax = panels["E"]
    for offset, beta in ((3.0, 0), (0.0, 1), (-3.0, 2)):
        noise = ColoredNoise(beta=beta, rng=rng).sample(512)
        ax.plot(np.linspace(0, 1, 512), offset + noise / noise.std(), linewidth=0.8)
    ax.set_yticks([3.0, 0.0, -3.0], ["white", "pink", "red"])
    ax.set_title("Noise: colored noise")
    fig.tight_layout()
    fig.savefig(output / "families.png")
    plt.close(fig)
    print("families.png saved")

    # The sampling grid: n increments give n + 1 values on [0, t].
    n = 8
    process = BrownianMotion(t=2.0, rng=rng)
    coarse = process.sample(n)
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(process.times, coarse, color=PALETTE[0], marker="o", markersize=6)
    for index, (time, value) in enumerate(zip(process.times, coarse, strict=True)):
        ax.annotate(
            f"$s_{index}$",
            (time, value),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
        )
    ax.set_xticks(process.times, [f"{time:g}" for time in process.times])
    ax.set_xlabel("times  (n = 8 increments on [0, t] with t = 2, giving n + 1 = 9 values)")
    ax.set_ylabel("sample(n)")
    fig.tight_layout()
    fig.savefig(output / "sampling_grid.png")
    plt.close(fig)
    print("sampling_grid.png saved")

    # The effect of the Hurst parameter.
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    for ax, hurst, color in zip(axes, (0.2, 0.5, 0.8), PALETTE, strict=False):
        fbm = FractionalBrownianMotion(hurst=hurst, rng=rng)
        ax.plot(np.linspace(0, 1, 1025), fbm.sample(1024), color=color, linewidth=0.9)
        ax.set_title(f"hurst = {hurst}")
        ax.set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(output / "hurst.png")
    plt.close(fig)
    print("hurst.png saved")

    # Mean reversion of a diffusion process.
    n, paths = 400, 300
    process = VasicekProcess(speed=5.0, mean=1.0, vol=0.5, rng=rng)
    times = np.linspace(0, 1, n + 1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.2))
    for initial, color in ((3.0, PALETTE[0]), (-1.0, PALETTE[1])):
        samples = np.array([process.sample_with_initial(n, initial=initial) for _ in range(paths)])
        ax.fill_between(
            times,
            np.quantile(samples, 0.05, axis=0),
            np.quantile(samples, 0.95, axis=0),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(times, samples[0], color=color, linewidth=1.0, label=f"initial = {initial:g}")
    ax.axhline(1.0, color=NEUTRAL, linestyle="--", linewidth=1.2, label="mean = 1")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(ncols=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "mean_reversion.png")
    plt.close(fig)
    print("mean_reversion.png saved")

    # The power spectral density of the colors of noise.
    n, paths = 2**12, 60
    frequencies = np.fft.rfftfreq(n, d=1.0 / n)[1:]
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.4))
    colors = (("violet", -2), ("blue", -1), ("white", 0), ("pink", 1), ("red", 2))
    for (name, beta), color in zip(colors, PALETTE, strict=True):
        noise_process = ColoredNoise(beta=beta, rng=rng)
        power = np.mean(
            [np.abs(np.fft.rfft(noise_process.sample(n)))[1:] ** 2 for _ in range(paths)],
            axis=0,
        )
        ax.loglog(frequencies, power / power[len(power) // 8], color=color, label=name)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Relative power")
    ax.legend(ncols=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(output / "noise_spectra.png")
    plt.close(fig)
    print("noise_spectra.png saved")


def main() -> None:
    """Generate every figure of the documentation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory to write the images to (default: docs/_static)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed of the random number generator, for reproducible figures",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        random.seed(value=args.seed)

    use_theme()

    make_branding(args.output)
    make_guides(args.output)
    make_continuous(args.output)
    make_diffusion(args.output)
    make_discrete(args.output)
    make_timeseries(args.output)
    make_noise(args.output)


if __name__ == "__main__":
    main()
