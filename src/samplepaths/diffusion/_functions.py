"""Mathematical functions for single values and for arrays.

The diffusion processes which advance step by step compute one realization
with Python floats, which is fastest for a single value, and many
realizations with arrays. Their coefficients are written once, in terms of
the functions of one of these two namespaces.
"""

import math
from types import SimpleNamespace

import numpy as np


def _clip_scalar(value: float, low: float, high: float) -> float:
    """Confine a value to an interval."""
    return min(max(value, low), high)


#: Functions for Python floats.
SCALAR = SimpleNamespace(sqrt=math.sqrt, log=math.log, clip=_clip_scalar)

#: Functions for arrays.
ARRAY = SimpleNamespace(sqrt=np.sqrt, log=np.log, clip=np.clip)
