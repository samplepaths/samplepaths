"""Figures of the front page and of the user guide."""

from pathlib import Path

import numpy as np
from plot_style import NEUTRAL, PALETTE

from samplepaths import (
    BrownianMotion,
    ColoredNoise,
    FractionalBrownianMotion,
    GARCHProcess,
    HestonModel,
    RandomWalk,
    VasicekProcess,
)
from samplepaths.base import BaseTimeProcess

from .helpers import plt


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

    # The six families of processes.
    fig, panels = plt.subplot_mosaic("ABC;DFE", figsize=(11, 5))
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
    ax = panels["F"]
    price, variance = HestonModel(speed=3.0, mean=0.04, vol=0.5, correlation=-0.7, rng=rng).sample(
        512
    )
    ax.plot(np.linspace(0, 1, 513), price, label="price")
    ax.plot(np.linspace(0, 1, 513), 1 + np.sqrt(variance), label="1 + volatility")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("Several components: Heston model")
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
