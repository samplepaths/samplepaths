"""The loops written out for one realization compute what the general ones do."""

from typing import Any

import numpy as np
import pytest

import samplepaths
from samplepaths.diffusion import EulerMaruyamaProcess
from samplepaths.timeseries import ConditionalVarianceProcess

DIFFUSIONS: list[tuple[str, dict[str, Any], float]] = [
    ("JacobiProcess", {"speed": 2.0, "mean": 0.6, "vol": 0.9}, 0.3),
    ("KimuraProcess", {"selection": 1.5, "vol": 1.2}, 0.3),
    ("VerhulstProcess", {"rate": 1.5, "capacity": 8.0, "vol": 0.4}, 0.5),
    ("GompertzProcess", {"growth": 1.5, "decay": 0.7, "vol": 0.4}, 0.5),
    ("FellerLogisticProcess", {"speed": 1.0, "level": 2.0, "vol": 0.9}, 0.2),
    (
        "FractionalCoxIngersollRossProcess",
        {"hurst": 0.7, "speed": 1.0, "mean": 0.5, "vol": 0.9},
        0.2,
    ),
    ("FractionalJacobiProcess", {"hurst": 0.7, "speed": 1.0, "mean": 0.5, "vol": 0.9}, 0.2),
]

VARIANCE_MODELS: list[tuple[str, dict[str, Any]]] = [
    ("ARCHProcess", {"omega": 0.5, "alpha": 0.5}),
    ("GARCHProcess", {"omega": 0.1, "alpha": 0.1, "beta": 0.8}),
    ("TGARCHProcess", {"omega": 0.1, "alpha": 0.05, "gamma": 0.1, "beta": 0.8}),
    ("AGARCHProcess", {"omega": 0.1, "alpha": 0.1, "shift": 0.5, "beta": 0.8}),
    ("EGARCHProcess", {"omega": -0.1, "alpha": 0.2, "gamma": -0.1, "beta": 0.9}),
    ("EGARCHProcess", {"omega": 50.0, "alpha": 40.0, "gamma": 0.0, "beta": 0.99}),
]


@pytest.mark.parametrize("name, kwargs, initial", DIFFUSIONS)
def test_diffusion_loop_equals_the_method_of_the_base_class(
    name: str, kwargs: dict[str, Any], initial: float
) -> None:
    process = getattr(samplepaths, name)(**kwargs, t=3.0)
    # Steps large enough to reach the boundaries of the domain.
    noise = np.random.default_rng(3).normal(scale=0.3, size=2000).tolist()

    written_out = process._advance(noise, initial, 3.0 / 2000)
    general = EulerMaruyamaProcess._advance(process, noise, initial, 3.0 / 2000)

    assert written_out == general


@pytest.mark.parametrize("name, kwargs", VARIANCE_MODELS)
def test_variance_loop_equals_the_method_of_the_base_class(
    name: str, kwargs: dict[str, Any]
) -> None:
    process = getattr(samplepaths, name)(**kwargs)
    innovations = np.random.default_rng(3).standard_normal(3001).tolist()

    written_out = process._advance_floats(innovations[:1], innovations[1:])
    general = ConditionalVarianceProcess._advance_floats(process, innovations[:1], innovations[1:])

    assert written_out == general


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("ARCHProcess", {"omega": 0.5, "alpha": [0.3, 0.2]}),
        ("GARCHProcess", {"omega": 0.1, "alpha": [0.05, 0.05], "beta": [0.5, 0.3]}),
        ("GARCHProcess", {"omega": 0.5, "alpha": 0.5, "beta": ()}),
    ],
)
def test_models_with_other_orders_use_the_general_loop(name: str, kwargs: dict[str, Any]) -> None:
    process = getattr(samplepaths, name)(**kwargs, rng=np.random.default_rng(3))
    lags = max(len(process.alpha), len(getattr(process, "beta", ())))
    innovations = np.random.default_rng(3).standard_normal(lags + 200).tolist()

    written_out = process._advance_floats(innovations[:lags], innovations[lags:])
    general = ConditionalVarianceProcess._advance_floats(
        process, innovations[:lags], innovations[lags:]
    )

    assert written_out == general
    assert process.sample(50).shape == (50,)
