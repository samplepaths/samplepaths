"""Plot style shared by the figures and the examples of the documentation."""

import matplotlib.pyplot as plt
from cycler import cycler

#: Colors of the realizations. They are legible on light and dark backgrounds.
PALETTE = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#0ea5e9"]

#: Color of the axes, labels and grid, legible on light and dark backgrounds.
NEUTRAL = "#8b93a1"


def use_theme() -> None:
    """Apply the plot style of the documentation.

    The figures have a transparent background and mid-tone colors, so the
    same image works with the light and the dark theme of the documentation.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.transparent": True,
            "savefig.dpi": 100,
            "axes.prop_cycle": cycler(color=PALETTE),
            "axes.edgecolor": NEUTRAL,
            "axes.labelcolor": NEUTRAL,
            "axes.titlecolor": NEUTRAL,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": NEUTRAL,
            "grid.alpha": 0.25,
            "xtick.color": NEUTRAL,
            "ytick.color": NEUTRAL,
            "text.color": NEUTRAL,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "font.size": 10,
        }
    )


def reset_style(gallery_conf: object, fname: str) -> None:
    """Apply the plot style before each example of the gallery runs.

    :param gallery_conf: the configuration of sphinx-gallery, unused
    :param fname: the file name of the example, unused
    """
    use_theme()
    # Keep the labels inside the figure without a call in every example.
    plt.rcParams["figure.constrained_layout.use"] = True
