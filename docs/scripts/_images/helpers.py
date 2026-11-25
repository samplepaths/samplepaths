"""Functions which draw the figures of the processes."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
from plot_style import PALETTE  # noqa: E402

from samplepaths.base import BaseProcess, BaseTimeProcess  # noqa: E402
from samplepaths.multivariate import VectorProcess  # noqa: E402

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


def plot_components(
    output: Path,
    name: str,
    title: str,
    process: VectorProcess,
    *,
    n: int = 1024,
    together: bool = False,
) -> None:
    """Plot one realization of a process with several components.

    :param together: draw the components on the same axes, for components of
        the same nature; otherwise each gets its own axes
    """
    realization = process.sample(n)
    count = 1 if together else len(realization)
    fig, axes = plt.subplots(count, 1, figsize=(8, 4), sharex=True, squeeze=False)
    for index, (component, values) in enumerate(zip(process.components, realization, strict=True)):
        ax = axes[0 if together else index, 0]
        ax.plot(process.times, values, color=PALETTE[index % len(PALETTE)], label=component)
        if not together:
            ax.set_ylabel(component.capitalize())
    if together:
        axes[0, 0].set_ylabel("Value")
        axes[0, 0].legend()
    axes[0, 0].set_title(title)
    axes[-1, 0].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(output / f"{name}.png")
    plt.close(fig)
    print(f"{name}.png saved")
