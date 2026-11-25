"""Base class of the discrete-time processes."""

from .base_process import BaseProcess


class BaseSequenceProcess(BaseProcess):
    """Base class of the discrete-time processes.

    :param rng: a custom random number generator
    """
