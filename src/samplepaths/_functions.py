"""Mathematical functions for single values and for arrays.

The processes which advance step by step compute one realization with
Python floats, which is fastest for a single value, and many realizations
with arrays. Their formulas are written once, in terms of the functions of
one of these two namespaces.
"""

import math
from types import SimpleNamespace

import numpy as np


def _clip_scalar(value: float, low: float, high: float) -> float:
    """Confine a value to an interval."""
    return min(max(value, low), high)


#: Functions for Python floats.
SCALAR = SimpleNamespace(
    sqrt=math.sqrt,
    log=math.log,
    exp=math.exp,
    clip=_clip_scalar,
    maximum=max,
)

#: Functions for arrays.
ARRAY = SimpleNamespace(
    sqrt=np.sqrt,
    log=np.log,
    exp=np.exp,
    clip=np.clip,
    maximum=np.maximum,
)
