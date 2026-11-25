"""Figures of the continuous-time and the diffusion processes."""

import math
from pathlib import Path

from samplepaths import (
    BesselProcess,
    BrownianBridge,
    BrownianExcursion,
    BrownianMeander,
    BrownianMotion,
    CauchyProcess,
    ConstantElasticityVarianceProcess,
    CoxIngersollRossProcess,
    ExtendedVasicekProcess,
    FractionalBrownianMotion,
    FractionalCoxIngersollRossProcess,
    FractionalGeometricBrownianMotion,
    FractionalJacobiProcess,
    FractionalOrnsteinUhlenbeckProcess,
    FractionalVasicekProcess,
    GammaProcess,
    GeometricBrownianMotion,
    HoLeeProcess,
    InverseGaussianProcess,
    MixedPoissonProcess,
    MultifractionalBrownianMotion,
    OrnsteinUhlenbeckProcess,
    PoissonProcess,
    SquaredBesselProcess,
    VarianceGammaProcess,
    VasicekProcess,
    WienerProcess,
    random,
)

from .helpers import plot_arrivals, plot_paths


def make_continuous(output: Path) -> None:
    """Generate the figures of the continuous-time processes."""
    plot_paths(output, "bessel_process", "Bessel process (dim=3)", BesselProcess(dim=3))
    plot_paths(output, "brownian_bridge", "Brownian bridge (b=0)", BrownianBridge(b=0))
    plot_paths(output, "brownian_excursion", "Brownian excursion", BrownianExcursion())
    plot_paths(output, "brownian_meander", "Brownian meander", BrownianMeander())
    plot_paths(output, "brownian_motion", "Brownian motion", BrownianMotion())
    plot_paths(output, "cauchy_process", "Cauchy process", CauchyProcess())
    plot_paths(
        output,
        "fractional_brownian_motion",
        "Fractional Brownian motion (hurst=0.7)",
        FractionalBrownianMotion(hurst=0.7),
    )
    plot_paths(
        output,
        "gamma_process",
        "Gamma process (mean=1, variance=1)",
        GammaProcess(mean=1, variance=1),
    )
    plot_paths(
        output,
        "geometric_brownian_motion",
        "Geometric Brownian motion (drift=0, volatility=1)",
        GeometricBrownianMotion(drift=0, volatility=1),
    )
    plot_paths(
        output,
        "inverse_gaussian",
        "Inverse Gaussian process",
        InverseGaussianProcess(),
    )
    plot_paths(
        output,
        "multifractional_brownian_motion",
        "Multifractional Brownian motion",
        MultifractionalBrownianMotion(
            hurst=lambda t: math.sin(t * 4 * math.pi) * 0.4 + 0.5,
        ),
        n=2**9,
    )
    plot_paths(
        output,
        "squared_bessel_process",
        "Squared Bessel process (dim=3)",
        SquaredBesselProcess(dim=3),
    )
    plot_paths(
        output,
        "variance_gamma_process",
        "Variance Gamma process (drift=0, variance=1, scale=1)",
        VarianceGammaProcess(drift=0, variance=1, scale=1),
    )
    plot_paths(output, "wiener_process", "Wiener process", WienerProcess())

    plot_arrivals(
        output,
        "poisson_process",
        "Poisson process (rate=1)",
        PoissonProcess(rate=1),
    )
    plot_arrivals(
        output,
        "mixed_poisson_process",
        "Mixed Poisson process",
        MixedPoissonProcess(
            rate_func=random.generator.uniform,
            rate_args=(1, 5),
        ),
    )


def make_diffusion(output: Path) -> None:
    """Generate the figures of the diffusion processes."""
    plot_paths(
        output,
        "constant_elasticity_variance_process",
        "CEV process",
        ConstantElasticityVarianceProcess(),
    )
    plot_paths(
        output,
        "cox_ingersoll_ross_process",
        "CIR process",
        CoxIngersollRossProcess(),
    )
    plot_paths(
        output,
        "extended_vasicek_process",
        "Extended Vasicek process",
        ExtendedVasicekProcess(),
    )
    plot_paths(
        output,
        "ornstein_uhlenbeck_process",
        "Ornstein-Uhlenbeck process",
        OrnsteinUhlenbeckProcess(),
    )
    plot_paths(output, "vasicek_process", "Vasicek process", VasicekProcess())
    plot_paths(
        output,
        "ho_lee_process",
        "Ho-Lee process (drift rising with time)",
        HoLeeProcess(drift=lambda t: 2.0 * t, vol=0.5),
    )
    plot_paths(
        output,
        "fractional_vasicek_process",
        "Fractional Vasicek process (hurst=0.8)",
        FractionalVasicekProcess(hurst=0.8, speed=3.0, mean=1.0, vol=0.5),
    )
    plot_paths(
        output,
        "fractional_ornstein_uhlenbeck_process",
        "Fractional Ornstein-Uhlenbeck process (hurst=0.2)",
        FractionalOrnsteinUhlenbeckProcess(hurst=0.2, speed=3.0, vol=0.5),
    )
    plot_paths(
        output,
        "fractional_cox_ingersoll_ross_process",
        "Fractional CIR process (hurst=0.8)",
        FractionalCoxIngersollRossProcess(hurst=0.8, speed=3.0, mean=1.0, vol=0.5),
    )
    plot_paths(
        output,
        "fractional_jacobi_process",
        "Fractional Jacobi process (hurst=0.8)",
        FractionalJacobiProcess(hurst=0.8, speed=3.0, mean=0.5, vol=0.6),
    )
    plot_paths(
        output,
        "fractional_geometric_brownian_motion",
        "Fractional geometric Brownian motion (hurst=0.8)",
        FractionalGeometricBrownianMotion(hurst=0.8, drift=0.5, volatility=0.5),
    )
