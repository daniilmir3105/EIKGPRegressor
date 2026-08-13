"""Preprocessing helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class StandardScalerLite:
    """Lightweight scaler with sklearn-like behavior."""

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None
        self.reference_scale_: NDArray[np.float64] | None = None
        self.mean_normalized_: NDArray[np.float64] | None = None
        self.scale_normalized_: NDArray[np.float64] | None = None
        self.constant_mask_: NDArray[np.bool_] | None = None

    def fit(self, x: NDArray[np.float64]) -> StandardScalerLite:
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
                reference = np.max(np.abs(x), axis=0)
                reference[reference == 0.0] = 1.0
                normalized = x / reference
                statistic_mean = np.mean(normalized, axis=0)
                transform_mean = (
                    statistic_mean if self.with_mean else np.zeros(x.shape[1], dtype=x.dtype)
                )
                mean = transform_mean * reference
                if self.with_std:
                    centered = normalized - statistic_mean
                    scale_normalized = np.sqrt(np.mean(centered * centered, axis=0))
                    constant_mask = scale_normalized == 0.0
                    scale = scale_normalized * reference
                    scale[constant_mask] = 1.0
                else:
                    scale_normalized = np.ones(x.shape[1], dtype=x.dtype)
                    constant_mask = np.zeros(x.shape[1], dtype=bool)
                    scale = np.ones(x.shape[1], dtype=x.dtype)
        except FloatingPointError as exc:
            self._clear_state()
            raise FloatingPointError(
                "Overflow or invalid values while fitting the scaler."
            ) from exc
        statistics = (reference, transform_mean, scale_normalized, mean, scale)
        if not all(np.isfinite(statistic).all() for statistic in statistics):
            self._clear_state()
            raise FloatingPointError("Scaler statistics are non-finite.")
        self.reference_scale_ = reference
        self.mean_normalized_ = transform_mean
        self.scale_normalized_ = scale_normalized
        self.constant_mask_ = constant_mask
        self.mean_ = mean
        self.scale_ = scale
        return self

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._is_fitted():
            raise RuntimeError("Scaler must be fitted before transform.")
        assert self.reference_scale_ is not None
        assert self.mean_normalized_ is not None
        assert self.scale_normalized_ is not None
        assert self.constant_mask_ is not None
        assert self.mean_ is not None
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
                normalized = x / self.reference_scale_
                centered = normalized - self.mean_normalized_
                if self.with_std:
                    transformed = np.empty_like(centered)
                    nonconstant = ~self.constant_mask_
                    transformed[:, nonconstant] = (
                        centered[:, nonconstant] / self.scale_normalized_[nonconstant]
                    )
                    transformed[:, self.constant_mask_] = (
                        x[:, self.constant_mask_] - self.mean_[self.constant_mask_]
                    )
                else:
                    transformed = centered * self.reference_scale_
        except FloatingPointError as exc:
            raise FloatingPointError("Overflow or invalid values during scaler transform.") from exc
        if not np.isfinite(transformed).all():
            raise FloatingPointError("Scaler transform produced non-finite values.")
        return transformed

    def fit_transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self._is_fitted():
            raise RuntimeError("Scaler must be fitted before inverse_transform.")
        assert self.reference_scale_ is not None
        assert self.mean_normalized_ is not None
        assert self.scale_normalized_ is not None
        assert self.constant_mask_ is not None
        assert self.mean_ is not None
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
                if self.with_std:
                    transformed = np.empty_like(x)
                    nonconstant = ~self.constant_mask_
                    normalized = (
                        x[:, nonconstant] * self.scale_normalized_[nonconstant]
                        + self.mean_normalized_[nonconstant]
                    )
                    transformed[:, nonconstant] = normalized * self.reference_scale_[nonconstant]
                    transformed[:, self.constant_mask_] = (
                        x[:, self.constant_mask_] + self.mean_[self.constant_mask_]
                    )
                else:
                    transformed = x + self.mean_
        except FloatingPointError as exc:
            raise FloatingPointError(
                "Overflow or invalid values during inverse scaler transform."
            ) from exc
        if not np.isfinite(transformed).all():
            raise FloatingPointError("Inverse scaler transform produced non-finite values.")
        return transformed

    def _is_fitted(self) -> bool:
        return all(
            value is not None
            for value in (
                self.mean_,
                self.scale_,
                self.reference_scale_,
                self.mean_normalized_,
                self.scale_normalized_,
                self.constant_mask_,
            )
        )

    def _clear_state(self) -> None:
        self.mean_ = None
        self.scale_ = None
        self.reference_scale_ = None
        self.mean_normalized_ = None
        self.scale_normalized_ = None
        self.constant_mask_ = None
