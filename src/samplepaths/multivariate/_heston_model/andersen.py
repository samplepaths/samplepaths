"""Discretization of the price of a model with a square-root variance."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..heston_model import HestonModel


def log_price_changes(
    model: "HestonModel",
    variances: npt.NDArray[np.float64],
    delta_t: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Generate the changes of the logarithm of the price over the steps.

    Andersen (2008), equation 33, with the integral of the variance over a
    step approximated by the mean of its two ends.

    :param model: the model, which gives the parameters and the generator
    :param variances: the variances, in an array of shape
        ``(steps + 1, paths)``
    :param delta_t: the length of the steps, a number or a column with one
        length per step
    :returns: an array of shape ``(steps, paths)``
    """
    before, after = variances[:-1], variances[1:]

    ratio = model.correlation / model.vol
    slope = 0.5 * delta_t * (model.speed * ratio - 0.5)
    independent = 0.5 * delta_t * (1.0 - model.correlation**2)

    changes = model.rng.standard_normal(before.shape)
    changes *= np.sqrt(independent * (before + after))
    changes += (model.drift - model.speed * model.mean * ratio) * delta_t
    changes += (slope - ratio) * before
    changes += (slope + ratio) * after
    return changes
