"""EIKG regressors for compact Kolmogorov-Gabor modeling."""

from .networks import (
    DeepPolyNetwork,
    DeepPolyNetworkCV,
    PolynomialNetwork,
    PolynomialNetworkCV,
)
from .regressors import EIKGPolynomialRegressor, EIKGPolynomialRegressorCV

__all__ = [
    "EIKGPolynomialRegressor",
    "EIKGPolynomialRegressorCV",
    "DeepPolyNetwork",
    "DeepPolyNetworkCV",
    "PolynomialNetwork",
    "PolynomialNetworkCV",
]
