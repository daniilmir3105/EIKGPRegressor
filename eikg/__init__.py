"""EIKG regressors for compact Kolmogorov-Gabor modeling."""

from .networks import (
    CombinatorialPolynomialNetwork,
    DeepPolyNetwork,
    DeepPolyNetworkCV,
    PolynomialNetwork,
    PolynomialNetworkCV,
)
from .regressors import EIKGPolynomialRegressor, EIKGPolynomialRegressorCV

__all__ = [
    "EIKGPolynomialRegressor",
    "EIKGPolynomialRegressorCV",
    "CombinatorialPolynomialNetwork",
    "DeepPolyNetwork",
    "DeepPolyNetworkCV",
    "PolynomialNetwork",
    "PolynomialNetworkCV",
]
