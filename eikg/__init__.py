"""EIKG regressors for compact Kolmogorov-Gabor modeling."""

from .networks import PolynomialNetwork, PolynomialNetworkCV
from .regressors import EIKGPolynomialRegressor, EIKGPolynomialRegressorCV

__all__ = [
    "EIKGPolynomialRegressor",
    "EIKGPolynomialRegressorCV",
    "PolynomialNetwork",
    "PolynomialNetworkCV",
]
