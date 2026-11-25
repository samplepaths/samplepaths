"""Comparisons with stochastic-rs: time series models."""

import numpy as np
import stochastic_rs as rs

import samplepaths as sp

from .shared import SERIES, Add, final_values


def add_series_cases(add: Add, rng: np.random.Generator, count: int) -> None:
    """Compare the time series models.

    :param add: adds a comparison to the results
    :param rng: the generator of the processes of samplepaths
    :param count: the number of realizations to compare
    """
    # Time series: the last of SERIES values. The parameters have the same
    # meaning in both packages.
    last = f"value {SERIES}"
    add(
        "AR(2)",
        last,
        final_values(rs.PyARp([0.5, -0.3], 2.0, SERIES), count),
        sp.AutoregressiveProcess(coefficients=[0.5, -0.3], scale=2.0, rng=rng).sample_paths(
            SERIES, count
        )[:, -1],
    )
    add(
        "MA(2)",
        last,
        final_values(rs.PyMAq([0.6, 0.2], 1.5, SERIES), count),
        sp.MovingAverageProcess(coefficients=[0.6, 0.2], scale=1.5, rng=rng).sample_paths(
            SERIES, count
        )[:, -1],
    )
    add(
        "ARIMA(1, 1, 1)",
        last,
        final_values(rs.PyArima([0.5], [0.3], 1, 1.0, SERIES), count),
        sp.ARIMAProcess(ar=0.5, ma=0.3, d=1, rng=rng).sample_paths(SERIES, count)[:, -1],
    )
    add(
        "SARIMA(1, 0, 1)(1, 1, 1)4",
        last,
        final_values(rs.PySarima([0.5], [0.3], [0.2], [0.4], 0, 1, 4, 1.0, SERIES), count),
        sp.SARIMAProcess(
            ar=0.5, ma=0.3, seasonal_ar=0.2, seasonal_ma=0.4, seasonal_d=1, period=4, rng=rng
        ).sample_paths(SERIES, count)[:, -1],
    )
    add(
        "ARCH(1)",
        last,
        final_values(rs.PyArch(0.5, [0.5], SERIES), count),
        sp.ARCHProcess(omega=0.5, alpha=0.5, rng=rng).sample_paths(SERIES, count)[:, -1],
    )
    add(
        "GARCH(1, 1)",
        last,
        final_values(rs.PyGarch(0.1, [0.1], [0.8], SERIES), count),
        sp.GARCHProcess(omega=0.1, alpha=0.1, beta=0.8, rng=rng).sample_paths(SERIES, count)[:, -1],
    )
    add(
        "GARCH(2, 2)",
        last,
        final_values(rs.PyGarch(0.1, [0.05, 0.05], [0.5, 0.3], SERIES), count),
        sp.GARCHProcess(omega=0.1, alpha=[0.05, 0.05], beta=[0.5, 0.3], rng=rng).sample_paths(
            SERIES, count
        )[:, -1],
    )
    add(
        "EGARCH(1, 1)",
        last,
        final_values(rs.PyEgarch(-0.1, [0.2], [-0.1], [0.9], SERIES), count),
        sp.EGARCHProcess(omega=-0.1, alpha=0.2, gamma=-0.1, beta=0.9, rng=rng).sample_paths(
            SERIES, count
        )[:, -1],
    )
    add(
        "TGARCH(1, 1)",
        last,
        final_values(rs.PyTgarch(0.1, [0.05], [0.1], [0.8], SERIES), count),
        sp.TGARCHProcess(omega=0.1, alpha=0.05, gamma=0.1, beta=0.8, rng=rng).sample_paths(
            SERIES, count
        )[:, -1],
    )

    # What stochastic-rs calls AGARCH adds a term for the negative values,
    # which is the threshold model here. The AGARCHProcess of samplepaths
    # shifts the value instead, and has no counterpart in stochastic-rs.
    add(
        "AGARCH of stochastic-rs",
        last,
        final_values(rs.PyAgarch(0.1, [0.05], [0.1], [0.8], SERIES), count),
        sp.TGARCHProcess(omega=0.1, alpha=0.05, gamma=0.1, beta=0.8, rng=rng).sample_paths(
            SERIES, count
        )[:, -1],
    )
