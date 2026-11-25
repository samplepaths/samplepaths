"""The Euler-Maruyama method for the generalized diffusion process."""

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter

#: Largest logarithm of a power of the decay used in the linear recurrence.
_LOG_POWER_RANGE = 60.0

#: The speed, the mean, the volatility and its exponent: the values of those
#: which are numbers, ``None`` for the others, and all of them as functions
#: of time.
Values = tuple[float | None, float | None, float | None, float | None]
Functions = tuple[
    Callable[[float], float],
    Callable[[float], float],
    Callable[[float], float],
    Callable[[float], float],
]


def _power(base: float, exponent: float) -> float:
    """Raise a float to a power with the semantics of NumPy scalars.

    A negative base with a fractional exponent gives ``nan`` with a warning
    and an overflow gives ``inf``, where Python floats would return a complex
    number or raise an exception.
    """
    if exponent == 0:
        return 1.0
    if exponent == 1:
        return base
    if base > 0.0:
        try:
            return float(base**exponent)
        except OverflowError:
            pass
    return float(np.float64(base) ** exponent)


def advance_one(
    gns: list[float],
    x: float,
    delta_t: float,
    values: Values,
    functions: Functions,
) -> list[float]:
    """Apply the Euler-Maruyama method to one realization of the noise.

    :param gns: the increments of the Wiener process
    :param x: the initial value
    :param delta_t: the length of a time step
    :param values: the parameters which are numbers
    :param functions: the parameters as functions of time
    :returns: the realization, which starts with the initial value
    """
    speed, mean, vol, volexp = values

    s = [x]

    if speed is not None and mean is not None and vol is not None and volexp is not None:
        # Constant parameters: no function calls inside the loop.
        if volexp == 0:
            for gn in gns:
                x += speed * (mean - x) * delta_t + vol * gn
                s.append(x)
        elif volexp == 1:
            for gn in gns:
                x += speed * (mean - x) * delta_t + vol * x * gn
                s.append(x)
        else:
            for gn in gns:
                x += speed * (mean - x) * delta_t + vol * _power(x, volexp) * gn
                s.append(x)

        return s

    # At least one parameter is a function. The functions are called at
    # every step, in the order speed, mean, vol and volexp.
    speed_function, mean_function, vol_function, volexp_function = functions

    t = 0.0
    for gn in gns:
        t += delta_t
        a = speed if speed is not None else float(speed_function(t))
        m = mean if mean is not None else float(mean_function(t))
        v = vol if vol is not None else float(vol_function(t))
        g = volexp if volexp is not None else float(volexp_function(t))
        x += a * (m - x) * delta_t + v * _power(x, g) * gn
        s.append(x)

    return s


def advance_together(
    noise: npt.NDArray[np.float64],
    initial: float,
    delta_t: float,
    values: Values,
    functions: Functions,
) -> npt.NDArray[np.float64]:
    """Apply the Euler-Maruyama method to many realizations of the noise.

    :param noise: the increments of the Wiener process, one realization per
        row
    :param initial: the initial value
    :param delta_t: the length of a time step
    :param values: the parameters which are numbers
    :param functions: the parameters as functions of time
    :returns: the realizations, one per row, which start with the initial
        value
    """
    speed, mean, vol, volexp = values
    speed_function, mean_function, vol_function, volexp_function = functions
    paths, n = noise.shape

    if speed is not None and volexp == 0:
        # A constant speed without a volatility exponent gives a linear
        # recurrence, X[k + 1] = decay X[k] + c[k], whose solution is a
        # weighted cumulative sum. It is evaluated for all realizations
        # and all steps with array operations.
        times = delta_t * np.arange(1, n + 1)
        means = mean if mean is not None else np.array([mean_function(t) for t in times])
        vols = vol if vol is not None else np.array([vol_function(t) for t in times])
        decay = 1.0 - speed * delta_t
        inputs = speed * means * delta_t + vols * noise

        realizations = np.empty((paths, n + 1))
        realizations[:, 0] = float(initial)

        if decay > 0:
            # The powers of the decay are kept within the range of floats
            # by solving the recurrence over blocks of steps.
            block = n if decay == 1 else max(1, int(_LOG_POWER_RANGE / abs(math.log(decay))))
            for start in range(0, n, block):
                stop = min(n, start + block)
                powers = decay ** np.arange(1, stop - start + 1)
                previous = realizations[:, start : start + 1]
                realizations[:, start + 1 : stop + 1] = powers * (
                    previous + np.cumsum(inputs[:, start:stop] / powers, axis=1)
                )
        else:
            # A time step this large makes the method unstable. A compiled
            # filter evaluates the recurrence as it is written.
            state = np.full((paths, 1), decay * float(initial))
            realizations[:, 1:], _ = lfilter([1.0], [1.0, -decay], inputs, axis=1, zi=state)

        return realizations

    # Advance all realizations together, one step at a time.
    x = np.full(paths, float(initial))
    realizations = np.empty((paths, n + 1))
    realizations[:, 0] = x

    t = 0.0
    for k, gn in enumerate(np.ascontiguousarray(noise.T)):
        t += delta_t
        a = speed if speed is not None else float(speed_function(t))
        m = mean if mean is not None else float(mean_function(t))
        v = vol if vol is not None else float(vol_function(t))
        g = volexp if volexp is not None else float(volexp_function(t))
        if g == 0:
            x = x + a * (m - x) * delta_t + v * gn
        elif g == 1:
            x = x + a * (m - x) * delta_t + v * x * gn
        else:
            x = x + a * (m - x) * delta_t + v * np.power(x, g) * gn
        realizations[:, k + 1] = x

    return realizations
