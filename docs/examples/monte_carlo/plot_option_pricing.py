"""
Pricing an option
=================

The price of a European call option is the discounted mean of its payoff.
With ``sample_paths()`` the estimate converges to the exact Black-Scholes
price as the number of realizations grows.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from samplepaths import GeometricBrownianMotion

rate, volatility, maturity = 0.05, 0.2, 1.0
spot, strike = 100.0, 105.0

# Exact price, for comparison.
d1 = (np.log(spot / strike) + (rate + volatility**2 / 2) * maturity) / (
    volatility * np.sqrt(maturity)
)
d2 = d1 - volatility * np.sqrt(maturity)
exact = spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)

gbm = GeometricBrownianMotion(
    drift=rate,
    volatility=volatility,
    t=maturity,
    rng=np.random.default_rng(25),
)
prices = gbm.sample_paths_with_initial(n=16, paths=200_000, initial=spot)
payoffs = np.exp(-rate * maturity) * np.maximum(prices[:, -1] - strike, 0.0)

counts = np.unique(np.logspace(2, np.log10(len(payoffs)), 60).astype(int))
estimates = np.array([payoffs[:count].mean() for count in counts])
errors = np.array([payoffs[:count].std() / np.sqrt(count) for count in counts])
print(f"exact {exact:.3f}, estimate {estimates[-1]:.3f} +/- {2 * errors[-1]:.3f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(counts, estimates - 2 * errors, estimates + 2 * errors, alpha=0.25)
ax.plot(counts, estimates, label="estimate and 95% band")
ax.axhline(exact, linestyle="--", color="gray", label="exact price")
ax.set_xscale("log")
ax.set_xlabel("Number of realizations")
ax.set_ylabel("Price")
ax.legend()
plt.show()
