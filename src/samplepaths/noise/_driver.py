"""Helper for the processes driven by a noise process of their own."""

import copy
from typing import TypeVar

from ..base import BaseTimeProcess

Noise = TypeVar("Noise", bound=BaseTimeProcess)


def follow(noise: Noise, process: BaseTimeProcess) -> Noise:
    """Return a noise which uses the generator and the interval of a process.

    A process driven by a noise of its own keeps that noise in an attribute.
    A copy of the process made for a thread shares the attribute, so the
    noise is copied when the generators differ instead of being changed.

    :param noise: the noise kept by the process
    :param process: the process the noise drives
    :returns: the noise to use, which the process keeps in place of the
        given one
    """
    if noise.rng is not process.rng:
        noise = copy.copy(noise)
        noise.rng = process.rng
    if noise.t != process.t:
        noise.t = process.t
    return noise
