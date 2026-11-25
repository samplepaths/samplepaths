"""Chinese restaurant process."""

from bisect import bisect_right
from itertools import accumulate

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

from .._compat import override
from ..base import BaseSequenceProcess
from ..utils.validation import check_numeric, check_positive_integer

#: Number of tables from which the table of a customer is looked up with NumPy.
_MANY_TABLES = 48


def _to_partition_array(partition: list[list[int]]) -> npt.NDArray[np.object_]:
    """Convert the customers of each table to an array of integer arrays."""
    tables = np.empty(len(partition), dtype=object)
    for index, table in enumerate(partition):
        tables[index] = np.array(table, dtype=np.int64)
    return tables


class ChineseRestaurantProcess(BaseSequenceProcess):
    """Chinese restaurant process.

    .. image:: /_static/chinese_restaurant_process.png
        :scale: 50%

    A Chinese restaurant process consists of a sequence of arrivals of
    customers to a Chinese restaurant. Customers may be seated either at an
    occupied table or a new table, there being infinitely many customers
    and tables.

    The first customer sits at the first table. The :math:`n`-th customer
    sits at a new table with probability :math:`1/n`, and at each already
    occupied table with probability :math:`t_k/n`, where :math:`t_k` is the
    number of customers already seated at table :math:`k`. This is the
    canonical process with :math:`discount=0` and :math:`strength=1`.

    The generalized process gives the :math:`n`-th customer a probability of
    :math:`(strength + T * discount) / (n - 1 + strength)` to sit at a new
    table and a probability of :math:`(t_k - discount) / (n - 1 + strength)`
    of sitting at table :math:`k`. :math:`T` is the number of occupied tables.

    Samples provide a sequence of tables selected by a sequence of customers.

    :param discount: the discount value of existing tables.
        Must be strictly less than 1.
    :param strength: the strength of a new table. If discount
        is negative, strength must be a multiple of discount. If discount is
        nonnegative, strength must be strictly greater than the
        negative discount.
    :param rng: a custom random number generator
    """

    def __init__(
        self,
        discount: float = 0.0,
        strength: float = 1.0,
        rng: Generator | None = None,
    ) -> None:
        super().__init__(rng=rng)

        self.discount = discount
        self.strength = strength

    def __str__(self) -> str:
        return (
            f"Chinese restaurant process with discount={self.discount} and strength={self.strength}"
        )

    def __repr__(self) -> str:
        return f"ChineseRestaurantProcess(discount={self.discount}, strength={self.strength})"

    @property
    def discount(self) -> float:
        """Discount parameter."""
        return self.__discount

    @discount.setter
    def discount(self, value: float | int) -> None:
        check_numeric(value=value, name="Discount")

        if value >= 1.0:
            raise ValueError(f"Discount value must be less than 1; got {value}.")

        self.__discount = value

    @property
    def strength(self) -> float:
        """Strength parameter."""
        return self.__strength

    @strength.setter
    def strength(self, value: float | int) -> None:
        check_numeric(value=value, name="Strength")

        if self.discount < 0:
            strength_positive = 1.0 * value / -self.discount <= 0
            strength_not_multiple = (1.0 * value / -self.discount) % 1 != 0
            if strength_positive or strength_not_multiple:
                raise ValueError(
                    "When discount is negative, strength value must be equal to a multiple of"
                    f" the discount value; got {value}."
                )
        elif self.discount < 1:
            if value <= -self.discount:
                raise ValueError(
                    "When discount is between 0 and 1, strength value must be greater than the "
                    f"negative of the discount; got {value}."
                )
        self.__strength = value

    def _sample_chinese_restaurant(
        self,
        n: int,
    ) -> tuple[list[int], list[list[int]]]:
        """Generate a Chinese restaurant process with n customers.

        :returns: the table of each customer and the customers of each table
        """
        check_positive_integer(n=n)

        c: list[list[int]] = [[0]]
        s = [0]
        sizes = [1]
        num_tables = 1

        discount = self.discount
        strength = self.strength
        random = self.rng.random

        # Each customer inverts the cumulative distribution over the tables at
        # a uniform variate, which is what Generator.choice does, without its
        # overhead per call.
        k = 2
        while k <= n and num_tables < _MANY_TABLES:
            denominator = k - 1 + strength
            cumulative = list(accumulate(1.0 * (size - discount) / denominator for size in sizes))
            cumulative.append(
                cumulative[-1] + 1.0 * (strength + num_tables * discount) / denominator
            )
            total = cumulative[-1]

            table = bisect_right([value / total for value in cumulative], random())
            if table == num_tables:
                num_tables += 1
                sizes.append(0)
                c.append([])
            sizes[table] += 1
            c[table].append(k - 1)
            s.append(table)
            k += 1

        if k > n:
            return s, c

        # With many tables the same computation is faster in NumPy. The
        # arithmetic and its order are unchanged, so the result is the same.
        counts = np.zeros(n + 1)
        counts[:num_tables] = sizes
        buffer = np.empty(n + 1)
        for customer in range(k, n + 1):
            denominator = customer - 1 + strength
            lookup = buffer[: num_tables + 1]
            np.cumsum(
                1.0 * (counts[:num_tables] - discount) / denominator,
                out=lookup[:num_tables],
            )
            lookup[num_tables] = (
                lookup[num_tables - 1] + 1.0 * (strength + num_tables * discount) / denominator
            )
            lookup /= lookup[num_tables]

            table = int(np.searchsorted(lookup, random(), side="right"))
            if table == num_tables:
                num_tables += 1
                c.append([])
            counts[table] += 1
            c[table].append(customer - 1)
            s.append(table)

        return s, c

    @override
    def sample(self, n: int) -> npt.NDArray[np.int64]:
        sequence, _ = self._sample_chinese_restaurant(n)
        return np.array(sequence, dtype=np.int64)

    def sample_partition(self, n: int) -> npt.NDArray[np.object_]:
        """Generate a realization as a partition of the customers.

        :param n: the number of customers to simulate
        :returns: one array per table holding the indices of its customers
        """
        _, partition = self._sample_chinese_restaurant(n)
        return _to_partition_array(partition)

    def sequence_to_partition(
        self,
        sequence: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.object_]:
        """Convert a realization to a partition of the customers.

        :param sequence: a realization, as generated by :py:meth:`sample`
        :returns: one array per table holding the indices of its customers
        """
        partition: list[list[int]] = []
        tables = -1
        for idx, table in enumerate(sequence):
            if table > tables:
                tables = int(table)
                partition.append([])
            partition[int(table)].append(idx)

        return _to_partition_array(partition)

    def partition_to_sequence(
        self,
        partition: npt.NDArray[np.object_],
    ) -> npt.NDArray[np.int64]:
        """Convert a partition of the customers to a realization.

        :param partition: a partition, as generated by
            :py:meth:`sample_partition`
        :returns: the table index of each customer
        """
        length = 0
        for table in partition:
            length += len(table)
        sequence = [0] * length
        for idx, table in enumerate(partition):
            for c in table:
                sequence[c] = idx

        return np.array(sequence, dtype=np.int64)
