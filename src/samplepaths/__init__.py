"""Generate realizations of stochastic processes."""

from importlib.metadata import version

from .continuous import (
    BesselProcess,
    BrownianBridge,
    BrownianExcursion,
    BrownianMeander,
    BrownianMotion,
    CauchyProcess,
    FractionalBrownianMotion,
    GammaProcess,
    GeometricBrownianMotion,
    InverseGaussianProcess,
    MixedPoissonProcess,
    MultifractionalBrownianMotion,
    PoissonProcess,
    SquaredBesselProcess,
    VarianceGammaProcess,
    WienerProcess,
)
from .diffusion import (
    ConstantElasticityVarianceProcess,
    CoxIngersollRossProcess,
    DiffusionProcess,
    ExtendedVasicekProcess,
    OrnsteinUhlenbeckProcess,
    VasicekProcess,
)
from .discrete import (
    BernoulliProcess,
    ChineseRestaurantProcess,
    DirichletProcess,
    MarkovChain,
    MoranProcess,
    RandomWalk,
)
from .noise import (
    BlueNoise,
    BrownianNoise,
    ColoredNoise,
    FractionalGaussianNoise,
    GaussianNoise,
    PinkNoise,
    RedNoise,
    VioletNoise,
    WhiteNoise,
)

__version__ = version("samplepaths")

__all__ = [
    "__version__",
    # continuous
    "BesselProcess",
    "BrownianBridge",
    "BrownianExcursion",
    "BrownianMeander",
    "BrownianMotion",
    "CauchyProcess",
    "FractionalBrownianMotion",
    "GammaProcess",
    "GeometricBrownianMotion",
    "InverseGaussianProcess",
    "MixedPoissonProcess",
    "MultifractionalBrownianMotion",
    "PoissonProcess",
    "SquaredBesselProcess",
    "VarianceGammaProcess",
    "WienerProcess",
    # diffusion
    "ConstantElasticityVarianceProcess",
    "CoxIngersollRossProcess",
    "DiffusionProcess",
    "ExtendedVasicekProcess",
    "OrnsteinUhlenbeckProcess",
    "VasicekProcess",
    # discrete
    "BernoulliProcess",
    "ChineseRestaurantProcess",
    "DirichletProcess",
    "MarkovChain",
    "MoranProcess",
    "RandomWalk",
    # noise
    "BlueNoise",
    "BrownianNoise",
    "ColoredNoise",
    "FractionalGaussianNoise",
    "GaussianNoise",
    "PinkNoise",
    "RedNoise",
    "VioletNoise",
    "WhiteNoise",
]
