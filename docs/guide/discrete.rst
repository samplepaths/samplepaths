Discrete-time Processes
=======================

The :py:mod:`samplepaths.discrete` module provides processes which evolve in
steps. They have no time interval; ``sample(n)`` generates ``n`` steps. The
models of time series analysis, which also evolve in steps, have their own
page: :doc:`timeseries`.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Process
     - Realization
   * - :py:class:`~samplepaths.discrete.BernoulliProcess`
     - a sequence of zeros and ones
   * - :py:class:`~samplepaths.discrete.RandomWalk`
     - the position after each step, starting at zero
   * - :py:class:`~samplepaths.discrete.MarkovChain`
     - the index of the state at each step
   * - :py:class:`~samplepaths.discrete.MoranProcess`
     - the size of a population at each step
   * - :py:class:`~samplepaths.discrete.ChineseRestaurantProcess`
     - the index of the table chosen by each customer
   * - :py:class:`~samplepaths.discrete.DirichletProcess`
     - a value at each step, repeating earlier values at random

Bernoulli process
-----------------

.. image:: /_static/bernoulli_process.png
    :scale: 50%

A :py:class:`~samplepaths.discrete.BernoulliProcess` is a sequence of
independent trials which succeed with probability ``p``.

.. code-block:: python

    from samplepaths import BernoulliProcess

    bp = BernoulliProcess(p=0.3)
    trials = bp.sample(100)  # 100 values, each 0 or 1

Random walk
-----------

.. image:: /_static/random_walk.png
    :scale: 50%

A :py:class:`~samplepaths.discrete.RandomWalk` adds a random step at each
time. By default the steps are -1 and 1 with equal probability. Pass
``steps`` and ``weights`` to choose the possible steps and their relative
probabilities; the normalized probabilities are available as ``p``.

.. code-block:: python

    from samplepaths import RandomWalk

    rw = RandomWalk(steps=[-1, 0, 2], weights=[1, 1, 2])
    print(rw.p)
    # [0.25 0.25 0.5 ]

    positions = rw.sample(100)  # 101 values, starting at zero
    increments = rw.sample_increments(100)  # the 100 steps only

Markov chain
------------

.. image:: /_static/markov_chain.png
    :scale: 50%

A :py:class:`~samplepaths.discrete.MarkovChain` moves between a finite
number of states. ``transition`` is a square matrix whose row ``i`` holds the
probabilities of moving from state ``i`` to each state, so each row sums to
one. ``initial`` holds the probabilities of the first state; when it is not
given, every state is equally likely.

.. code-block:: python

    from samplepaths import MarkovChain

    # two states: 0 = "dry", 1 = "wet"
    weather = MarkovChain(
        transition=[
            [0.9, 0.1],
            [0.5, 0.5],
        ],
        initial=[1, 0],
    )
    states = weather.sample(365)

Moran process
-------------

.. image:: /_static/moran_process.png
    :scale: 50%

A :py:class:`~samplepaths.discrete.MoranProcess` models the number of
individuals of one type in a population of fixed size ``maximum``. At each
step the number increases by one, decreases by one or stays the same.

The values zero and ``maximum`` are absorbing: once the process reaches one
of them, the realization ends. A realization can therefore be shorter than
``n``.

.. code-block:: python

    from samplepaths import MoranProcess

    mp = MoranProcess(maximum=20)

    sizes = mp.sample(100)  # starts from 1
    sizes = mp.sample_with_start(100, start=10)

``sample_paths()`` returns rows of equal length: the rest of a row is filled
with the state in which the process was absorbed, where it stays.

Chinese restaurant process
--------------------------

.. image:: /_static/chinese_restaurant_process.png
    :scale: 50%

In a :py:class:`~samplepaths.discrete.ChineseRestaurantProcess`, customers
arrive one at a time and sit either at an occupied table, with a probability
which grows with the number of customers at it, or at a new table. The
result is a random partition of the customers.

``sample(n)`` returns the table of each customer. ``sample_partition(n)``
returns the customers of each table instead, and two methods convert between
the two representations.

.. code-block:: python

    import numpy as np
    from samplepaths import ChineseRestaurantProcess

    crp = ChineseRestaurantProcess(rng=np.random.default_rng(1234567890))

    tables = crp.sample(10)
    print(tables)
    # [0 1 0 1 1 1 0 2 1 1]

    partition = crp.sequence_to_partition(tables)
    print([list(table) for table in partition])
    # [[0, 2, 6], [1, 3, 4, 5, 8, 9], [7]]

The ``discount`` and ``strength`` parameters change how readily new tables
are opened. They must satisfy one of these conditions:

* ``0 <= discount < 1`` and ``strength > -discount``
* ``discount < 0`` and ``strength`` is a positive multiple of ``-discount``

Dirichlet process
-----------------

.. image:: /_static/dirichlet_process.png
    :scale: 50%

A :py:class:`~samplepaths.discrete.DirichletProcess` generates a sequence in
which each value is either a new draw from a base distribution or a repeat of
an earlier value. At step :math:`k` a new value is drawn with probability
:math:`\alpha / (\alpha + k - 1)`, so a larger ``alpha`` gives more distinct
values.

``base`` is a function without arguments which returns one draw of the base
distribution. It defaults to the uniform distribution on ``[0, 1]``.

.. code-block:: python

    import numpy as np
    from samplepaths import DirichletProcess

    rng = np.random.default_rng(5)
    dp = DirichletProcess(base=rng.standard_normal, alpha=5, rng=rng)

    values = dp.sample(200)
    print(len(np.unique(values)))  # far fewer than 200 distinct values
