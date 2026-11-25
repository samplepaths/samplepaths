"""Chinese restaurant tests."""

from typing import Any

import numpy as np
import pytest

from samplepaths.discrete import ChineseRestaurantProcess as ProcessType


@pytest.mark.parametrize(
    "discount, error_type",
    [
        (
            "invalid",
            TypeError,
        ),
        (
            1.0,
            ValueError,
        ),
        (
            2,
            ValueError,
        ),
    ],
)
def test_chinese_restaurant_process_invalid_discount(
    discount: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        ProcessType(discount=discount)


@pytest.mark.parametrize(
    "discount, strength, error_type",
    [
        (
            0.0,
            "invalid",
            TypeError,
        ),
        # discount negative, strength not a multiple
        (
            -1,
            1.1,
            ValueError,
        ),
        # discount negative, strength not positive
        (
            -1,
            -2,
            ValueError,
        ),
        # discount nonnegative, strength not greater than its negative
        (
            0.7,
            -2,
            ValueError,
        ),
        (
            0.0,
            0.0,
            ValueError,
        ),
    ],
)
def test_chinese_restaurant_process_invalid_strength(
    discount: float,
    strength: Any,
    error_type: type,
) -> None:
    with pytest.raises(error_type):
        ProcessType(discount=discount, strength=strength)


@pytest.mark.parametrize(
    "discount, strength",
    [
        (0.0, 1.0),
        (0.5, 2.0),
        (0.5, -0.2),
        (-1, 3),
    ],
)
def test_chinese_restaurant_process_constructor(
    discount: float,
    strength: float,
) -> None:
    instance = ProcessType(discount=discount, strength=strength)

    assert instance.discount == discount
    assert instance.strength == strength


def test_chinese_restaurant_process_str_repr() -> None:
    instance = ProcessType(discount=0.5, strength=1.5)

    assert str(instance) == "Chinese restaurant process with discount=0.5 and strength=1.5"
    assert repr(instance) == "ChineseRestaurantProcess(discount=0.5, strength=1.5)"


@pytest.mark.parametrize(
    "n, error_type",
    [
        (
            0,
            ValueError,
        ),
        (
            -1,
            ValueError,
        ),
        (
            1.5,
            TypeError,
        ),
    ],
)
def test_chinese_restaurant_process_sample_invalid_n(
    n: Any,
    error_type: type,
) -> None:
    instance = ProcessType()

    with pytest.raises(error_type):
        instance.sample(n=n)


@pytest.mark.parametrize(
    "discount, strength",
    [
        (0.0, 1.0),
        (0.5, 2.0),
        (-1, 3),
    ],
)
def test_chinese_restaurant_process_sample_shape(
    discount: float,
    strength: float,
) -> None:
    instance = ProcessType(discount=discount, strength=strength)
    n = 100
    samples = instance.sample(n=n)

    assert samples.shape == (n,)
    # The first customer sits at the first table and every new table gets
    # the next unused index.
    assert samples[0] == 0
    seen = -1
    for table in samples:
        assert 0 <= table <= seen + 1
        seen = max(seen, table)


@pytest.mark.parametrize(
    "discount, strength, n, expected",
    [
        (
            0.0,
            1.0,
            10,
            [0, 1, 0, 1, 1, 1, 0, 2, 1, 1],
        ),
        (
            0.5,
            2.0,
            10,
            [0, 1, 1, 2, 3, 4, 0, 5, 6, 7],
        ),
    ],
)
def test_chinese_restaurant_process_sample_values(
    discount: float,
    strength: float,
    n: int,
    expected: list[int],
) -> None:
    rng = np.random.default_rng(1234567890)

    instance = ProcessType(
        discount=discount,
        strength=strength,
        rng=rng,
    )

    samples = instance.sample(n=n)

    np.testing.assert_array_equal(samples, expected)


def test_chinese_restaurant_process_sample_partition_values() -> None:
    rng = np.random.default_rng(1234567890)

    instance = ProcessType(rng=rng)

    partition = instance.sample_partition(n=10)

    assert [list(table) for table in partition] == [
        [0, 2, 6],
        [1, 3, 4, 5, 8, 9],
        [7],
    ]


@pytest.mark.parametrize("seed", range(5))
def test_chinese_restaurant_process_sample_partition(seed: int) -> None:
    instance = ProcessType(rng=np.random.default_rng(seed))
    n = 50

    partition = instance.sample_partition(n=n)

    # Every customer is seated at exactly one table.
    customers = sorted(customer for table in partition for customer in table)
    assert customers == list(range(n))


@pytest.mark.parametrize("seed", range(5))
def test_chinese_restaurant_process_sequence_partition_round_trip(
    seed: int,
) -> None:
    instance = ProcessType(rng=np.random.default_rng(seed))
    n = 50

    sequence = instance.sample(n=n)
    partition = instance.sequence_to_partition(sequence)

    assert sorted(c for table in partition for c in table) == list(range(n))
    for index, table in enumerate(partition):
        for customer in table:
            assert sequence[customer] == index

    np.testing.assert_array_equal(
        instance.partition_to_sequence(partition),
        sequence,
    )
