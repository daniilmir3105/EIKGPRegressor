"""Deep, fixed-width polynomial networks built from EIKG regressors."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .metrics import mean_squared_error, r2_score
from .regressors import (
    BaseEstimator,
    EIKGPolynomialRegressor,
    RegressorMixin,
)
from .validation import (
    validate_floating_dtype,
    validate_positive_integer,
    validate_x,
    validate_xy_lengths,
    validate_y,
)


def _ensure_finite(values: NDArray[np.float64], *, context: str) -> None:
    if not np.isfinite(values).all():
        raise FloatingPointError(
            f"Non-finite values produced while {context}. "
            "Use fewer layers, enable scaling, or reduce the input magnitude."
        )


def _validate_boolean_parameters(**parameters: Any) -> None:
    for name, value in parameters.items():
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean, got {value!r}.")


def _resolve_layer_degrees(degree: Any, *, n_layers: int) -> tuple[int, ...]:
    """Expand a scalar degree or validate one explicit degree per layer."""

    if isinstance(degree, Integral) and not isinstance(degree, (bool, np.bool_)):
        validated = validate_positive_integer(degree, name="degree")
        return (validated,) * n_layers
    if isinstance(degree, (str, bytes)) or not isinstance(degree, (Sequence, np.ndarray)):
        raise ValueError(
            "degree must be a positive integer or a sequence containing one "
            f"positive integer per layer, got {degree!r}."
        )
    if isinstance(degree, np.ndarray) and degree.ndim != 1:
        raise ValueError("degree array must be one-dimensional.")
    values = list(degree)
    if len(values) != n_layers:
        raise ValueError(
            f"degree sequence must contain exactly n_layers={n_layers} values, "
            f"got {len(values)}."
        )
    return tuple(
        validate_positive_integer(value, name=f"degree[{index}]")
        for index, value in enumerate(values)
    )


def _warn_underflow_loss(
    source: NDArray[np.float64],
    transformed: NDArray[np.float64],
    *,
    context: str,
) -> None:
    lost = (source != 0.0) & (transformed == 0.0)
    if np.any(lost):
        columns = np.flatnonzero(np.any(lost, axis=0)).tolist()
        warnings.warn(
            f"Underflow reduced nonzero values to zero in columns {columns} while {context}.",
            RuntimeWarning,
            stacklevel=3,
        )


def _scale_columns_by_max_abs(
    values: NDArray[np.float64],
    *,
    dtype: type[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return per-column max-absolute scales and safely scaled values."""

    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            scales = np.max(np.abs(values), axis=0)
            scales = np.asarray(scales, dtype=dtype)
            scales[scales == 0.0] = 1.0
            scaled = np.asarray(values / scales, dtype=dtype)
    except FloatingPointError as exc:
        raise FloatingPointError(
            "Could not max-abs scale X without numerical overflow or invalid values."
        ) from exc
    _ensure_finite(scales, context="computing feature scales")
    _ensure_finite(scaled, context="scaling input features")
    _warn_underflow_loss(values, scaled, context="scaling input features")
    return scales, scaled


def _apply_column_scales(
    values: NDArray[np.float64],
    scales: NDArray[np.float64],
    *,
    dtype: type[np.floating],
) -> NDArray[np.float64]:
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            scaled = np.asarray(values / scales, dtype=dtype)
    except FloatingPointError as exc:
        raise FloatingPointError(
            "Prediction data cannot be scaled with the feature scales learned during fit."
        ) from exc
    _ensure_finite(scaled, context="scaling prediction features")
    _warn_underflow_loss(values, scaled, context="scaling prediction features")
    return scaled


def _prediction_scale(prediction: NDArray[np.float64]) -> float:
    try:
        with np.errstate(over="raise", invalid="raise"):
            scale = float(np.max(np.abs(prediction)))
    except FloatingPointError as exc:
        raise FloatingPointError(
            "Could not scale a layer prediction because its magnitude is invalid."
        ) from exc
    if not np.isfinite(scale):
        raise FloatingPointError("A polynomial layer produced a non-finite prediction scale.")
    return 1.0 if scale == 0.0 else scale


def _normalize_prediction(
    prediction: NDArray[np.float64],
    scale: float,
    *,
    dtype: type[np.floating],
    layer_number: int,
) -> NDArray[np.float64]:
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            normalized = np.asarray(prediction / scale, dtype=dtype)
    except FloatingPointError as exc:
        raise FloatingPointError(
            f"Layer {layer_number} prediction cannot be normalized with its training scale."
        ) from exc
    _ensure_finite(normalized, context=f"normalizing layer {layer_number} predictions")
    _warn_underflow_loss(
        prediction.reshape(-1, 1),
        normalized.reshape(-1, 1),
        context=f"normalizing layer {layer_number} predictions",
    )
    return normalized


def _next_power(
    current_power: NDArray[np.float64],
    base_scaled: NDArray[np.float64],
    *,
    dtype: type[np.floating],
    exponent: int,
) -> NDArray[np.float64]:
    """Advance element-wise base powers once, detecting overflow explicitly."""

    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            powered = np.asarray(current_power * base_scaled, dtype=dtype)
    except FloatingPointError as exc:
        raise FloatingPointError(
            f"Overflow or invalid values while building X power {exponent}. "
            "Prediction data may be too far outside the training range."
        ) from exc
    _ensure_finite(powered, context=f"building X power {exponent}")

    source_nonzero = np.any(current_power != 0.0, axis=0)
    collapsed = source_nonzero & np.all(powered == 0.0, axis=0)
    if np.any(collapsed):
        columns = np.flatnonzero(collapsed).tolist()
        warnings.warn(
            f"Underflow reduced all values of powered feature columns {columns} "
            f"to zero at exponent {exponent}.",
            RuntimeWarning,
            stacklevel=3,
        )
    return powered


@dataclass
class _FoldState:
    """Fold-local inputs used by leakage-safe layer-wise model selection."""

    y_train: NDArray[np.float64]
    y_validation: NDArray[np.float64]
    base_train: NDArray[np.float64]
    base_validation: NDArray[np.float64]
    current_train: NDArray[np.float64]
    current_validation: NDArray[np.float64]
    current_power_train: NDArray[np.float64]
    current_power_validation: NDArray[np.float64]


class DeepPolyNetwork(RegressorMixin, BaseEstimator):
    """Greedy deep polynomial network composed of EIKG regression layers.

    The first layer receives max-abs-scaled input features. After layer ``l``,
    the next layer receives the normalized previous prediction together with
    the element-wise power ``X ** (l + 1)`` of the scaled original features.
    Consequently, the explicit feature width is ``n_features + 1`` after the
    first layer instead of growing with network depth.

    Parameters
    ----------
    n_layers : int, default=3
        Number of sequential polynomial layers. Must be a positive integer.
    degree : int or sequence of int, default=2
        A scalar applies one latent polynomial degree to every layer. A sequence
        supplies one positive degree per layer and must have length ``n_layers``.
        Tuples are recommended when the estimator will be cloned or tuned with
        scikit-learn.
    regularization : {"none", "ridge", None}, default="ridge"
        Least-squares regularization used by every layer.
    alpha_ridge : float, default=1e-8
        Ridge strength used when ``regularization="ridge"``.
    fit_intercept, scale, scale_y, normalize_latent, dtype, copy, check_input,
    lstsq_rcond
        Passed unchanged to each :class:`EIKGPolynomialRegressor` layer.
    """

    def __init__(
        self,
        n_layers: int = 3,
        degree: int | Sequence[int] = 2,
        regularization: str | None = "ridge",
        alpha_ridge: float = 1e-8,
        fit_intercept: bool = True,
        scale: bool = True,
        scale_y: bool = False,
        normalize_latent: bool = True,
        dtype: type[np.floating] = np.float64,
        copy: bool = True,
        check_input: bool = True,
        lstsq_rcond: float | None = None,
    ) -> None:
        self.n_layers = n_layers
        self.degree = degree
        self.regularization = regularization
        self.alpha_ridge = alpha_ridge
        self.fit_intercept = fit_intercept
        self.scale = scale
        self.scale_y = scale_y
        self.normalize_latent = normalize_latent
        self.dtype = dtype
        self.copy = copy
        self.check_input = check_input
        self.lstsq_rcond = lstsq_rcond

    def fit(self, x: Any, y: Any) -> DeepPolyNetwork:
        """Fit all layers sequentially and replace any previous fitted state."""

        self._clear_fitted_state()
        n_layers = validate_positive_integer(self.n_layers, name="n_layers")
        degrees = _resolve_layer_degrees(self.degree, n_layers=n_layers)
        validate_floating_dtype(self.dtype)
        _validate_boolean_parameters(
            fit_intercept=self.fit_intercept,
            scale=self.scale,
            scale_y=self.scale_y,
            normalize_latent=self.normalize_latent,
            copy=self.copy,
            check_input=self.check_input,
        )

        x_arr, feature_names = validate_x(
            x, dtype=self.dtype, copy=self.copy, check_input=self.check_input
        )
        y_arr = validate_y(y, dtype=self.dtype, copy=self.copy, check_input=self.check_input)
        validate_xy_lengths(x_arr, y_arr)
        _ensure_finite(x_arr, context="validating network input")
        _ensure_finite(y_arr, context="validating network target")

        base_scale, base_scaled = _scale_columns_by_max_abs(x_arr, dtype=self.dtype)
        current_input = base_scaled
        current_power = base_scaled
        layers: list[EIKGPolynomialRegressor] = []
        prediction_scales: list[float] = []
        input_sizes: list[int] = []
        final_prediction: NDArray[np.float64] | None = None

        for layer_index in range(n_layers):
            layer_number = layer_index + 1
            layer = self._make_layer(degrees[layer_index])
            layer.fit(current_input, y_arr)
            prediction = np.asarray(layer.predict(current_input), dtype=self.dtype)
            _ensure_finite(prediction, context=f"predicting training layer {layer_number}")

            layers.append(layer)
            input_sizes.append(int(current_input.shape[1]))
            final_prediction = prediction
            if layer_number == n_layers:
                continue

            scale = _prediction_scale(prediction)
            prediction_scales.append(scale)
            normalized_prediction = _normalize_prediction(
                prediction,
                scale,
                dtype=self.dtype,
                layer_number=layer_number,
            )
            exponent = layer_number + 1
            current_power = _next_power(
                current_power,
                base_scaled,
                dtype=self.dtype,
                exponent=exponent,
            )
            current_input = np.asarray(
                np.column_stack((normalized_prediction, current_power)), dtype=self.dtype
            )
            _ensure_finite(current_input, context=f"building input for layer {layer_number + 1}")

        if final_prediction is None:  # pragma: no cover - guarded by n_layers validation
            raise RuntimeError("No polynomial layers were fitted.")

        self.layers_ = layers
        self.base_scale_ = base_scale
        self.layer_prediction_scales_ = prediction_scales
        self.layer_input_sizes_ = input_sizes
        self.n_features_in_ = int(x_arr.shape[1])
        self.n_layers_ = n_layers
        self.degrees_ = degrees
        self.degree_ = degrees[0] if len(set(degrees)) == 1 else None
        if feature_names is not None:
            self.feature_names_in_ = feature_names.copy()
        self.is_fitted_ = True
        return self

    def predict(self, x: Any) -> NDArray[np.float64]:
        """Predict with the fitted sequence of polynomial layers."""

        self._check_is_fitted()
        x_arr, feature_names = validate_x(
            x, dtype=self.dtype, copy=self.copy, check_input=self.check_input
        )
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {x_arr.shape[1]} features, expected {self.n_features_in_}.")
        if hasattr(self, "feature_names_in_") and feature_names is not None:
            if not np.array_equal(feature_names, self.feature_names_in_):
                raise ValueError("X columns at predict must match training feature names/order.")
        _ensure_finite(x_arr, context="validating network prediction input")

        base_scaled = _apply_column_scales(x_arr, self.base_scale_, dtype=self.dtype)
        current_input = base_scaled
        current_power = base_scaled
        prediction: NDArray[np.float64] | None = None

        for layer_index, layer in enumerate(self.layers_):
            layer_number = layer_index + 1
            prediction = np.asarray(layer.predict(current_input), dtype=self.dtype)
            _ensure_finite(prediction, context=f"predicting layer {layer_number}")
            if layer_number == self.n_layers_:
                continue

            normalized_prediction = _normalize_prediction(
                prediction,
                self.layer_prediction_scales_[layer_index],
                dtype=self.dtype,
                layer_number=layer_number,
            )
            exponent = layer_number + 1
            current_power = _next_power(
                current_power,
                base_scaled,
                dtype=self.dtype,
                exponent=exponent,
            )
            current_input = np.asarray(
                np.column_stack((normalized_prediction, current_power)), dtype=self.dtype
            )
            _ensure_finite(current_input, context=f"building input for layer {layer_number + 1}")

        if prediction is None:  # pragma: no cover - guarded by fitted state
            raise RuntimeError("The fitted network contains no polynomial layers.")
        return prediction

    def score(self, x: Any, y: Any) -> float:
        """Return the coefficient of determination, R-squared."""

        y_true = validate_y(y, dtype=self.dtype, copy=False, check_input=self.check_input)
        y_pred = self.predict(x)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"X and y row mismatch: X has {y_pred.shape[0]} rows but y has {y_true.shape[0]}."
            )
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            result = r2_score(y_true, y_pred)
        if not np.isfinite(result):
            raise FloatingPointError("R-squared is non-finite for the supplied data.")
        return result

    def _make_layer(self, degree: int) -> EIKGPolynomialRegressor:
        return EIKGPolynomialRegressor(
            degree=degree,
            regularization=self.regularization,
            alpha_ridge=self.alpha_ridge,
            fit_intercept=self.fit_intercept,
            scale=self.scale,
            scale_y=self.scale_y,
            normalize_latent=self.normalize_latent,
            dtype=self.dtype,
            copy=self.copy,
            check_input=self.check_input,
            lstsq_rcond=self.lstsq_rcond,
        )

    def _check_is_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Estimator is not fitted. Call fit(X, y) first.")

    def _clear_fitted_state(self) -> None:
        fitted_attributes = (
            "layers_",
            "base_scale_",
            "layer_prediction_scales_",
            "layer_input_sizes_",
            "n_features_in_",
            "n_layers_",
            "degree_",
            "degrees_",
            "feature_names_in_",
            "is_fitted_",
        )
        for attribute in fitted_attributes:
            if hasattr(self, attribute):
                delattr(self, attribute)


class DeepPolyNetworkCV(RegressorMixin, BaseEstimator):
    """Select a separate polynomial degree for each network layer.

    Selection proceeds greedily from the first layer to the last. Every fold
    maintains its own fitted prefix, scales, and train/validation representation,
    so no prediction derived from a validation target enters a later layer.
    Candidate degrees are compared with ``scoring``; exact ties select the lower
    degree. This greedy search does not guarantee the globally best degree vector.
    """

    def __init__(
        self,
        n_layers: int = 3,
        max_degree: int = 6,
        scoring: str = "neg_mean_squared_error",
        cv: int = 5,
        shuffle: bool = False,
        random_state: int | None = None,
        regularization: str | None = "ridge",
        alpha_ridge: float = 1e-8,
        fit_intercept: bool = True,
        scale: bool = True,
        scale_y: bool = False,
        normalize_latent: bool = True,
        dtype: type[np.floating] = np.float64,
        copy: bool = True,
        check_input: bool = True,
        lstsq_rcond: float | None = None,
    ) -> None:
        self.n_layers = n_layers
        self.max_degree = max_degree
        self.scoring = scoring
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.regularization = regularization
        self.alpha_ridge = alpha_ridge
        self.fit_intercept = fit_intercept
        self.scale = scale
        self.scale_y = scale_y
        self.normalize_latent = normalize_latent
        self.dtype = dtype
        self.copy = copy
        self.check_input = check_input
        self.lstsq_rcond = lstsq_rcond

    def fit(self, x: Any, y: Any) -> DeepPolyNetworkCV:
        """Select each layer degree and refit the resulting network on all data."""

        self._clear_fitted_state()
        n_layers = validate_positive_integer(self.n_layers, name="n_layers")
        max_degree = validate_positive_integer(self.max_degree, name="max_degree")
        cv = validate_positive_integer(self.cv, name="cv", minimum=2)
        validate_floating_dtype(self.dtype)
        _validate_boolean_parameters(
            shuffle=self.shuffle,
            fit_intercept=self.fit_intercept,
            scale=self.scale,
            scale_y=self.scale_y,
            normalize_latent=self.normalize_latent,
            copy=self.copy,
            check_input=self.check_input,
        )
        if self.scoring not in {"neg_mean_squared_error", "r2"}:
            raise ValueError("scoring must be one of {'neg_mean_squared_error', 'r2'}.")
        random_state = self._validated_random_state()

        x_arr, _ = validate_x(x, dtype=self.dtype, copy=self.copy, check_input=self.check_input)
        y_arr = validate_y(y, dtype=self.dtype, copy=self.copy, check_input=self.check_input)
        validate_xy_lengths(x_arr, y_arr)
        _ensure_finite(x_arr, context="validating cross-validation input")
        _ensure_finite(y_arr, context="validating cross-validation target")
        if cv > x_arr.shape[0]:
            raise ValueError(f"cv={cv} cannot exceed the number of samples ({x_arr.shape[0]}).")
        if self.scoring == "r2" and x_arr.shape[0] // cv < 2:
            raise ValueError("scoring='r2' requires at least 2 validation samples in every fold.")

        splits = self._make_splits(x_arr, y_arr, cv=cv, random_state=random_state)
        fold_states = self._initialize_fold_states(x_arr, y_arr, splits)
        selected_degrees: list[int] = []
        layer_mean_scores: list[list[float]] = []
        layer_fold_scores: list[list[list[float]]] = []
        layer_best_scores: list[float] = []

        for layer_index in range(n_layers):
            layer_number = layer_index + 1
            mean_scores: list[float] = []
            all_fold_scores: list[list[float]] = []
            best_index = 0
            best_mean_score = -np.inf
            best_train_predictions: list[NDArray[np.float64]] | None = None
            best_validation_predictions: list[NDArray[np.float64]] | None = None

            for degree in range(1, max_degree + 1):
                fold_scores: list[float] = []
                candidate_layers: list[EIKGPolynomialRegressor] = []
                validation_predictions: list[NDArray[np.float64]] = []
                for state in fold_states:
                    candidate = self._make_layer(degree)
                    candidate.fit(state.current_train, state.y_train)
                    prediction = np.asarray(
                        candidate.predict(state.current_validation), dtype=self.dtype
                    )
                    _ensure_finite(
                        prediction,
                        context=(
                            f"predicting validation data for layer {layer_number}, "
                            f"degree {degree}"
                        ),
                    )
                    candidate_layers.append(candidate)
                    validation_predictions.append(prediction)
                    fold_scores.append(self._score_fold(state.y_validation, prediction))

                all_fold_scores.append(fold_scores)
                mean_score = float(np.mean(np.asarray(fold_scores, dtype=np.float64)))
                if not np.isfinite(mean_score):
                    raise FloatingPointError(
                        f"Layer {layer_number}, degree {degree} produced a non-finite "
                        "mean cross-validation score."
                    )
                mean_scores.append(mean_score)
                if mean_score > best_mean_score:
                    best_index = degree - 1
                    best_mean_score = mean_score
                    best_validation_predictions = validation_predictions
                    if layer_number < n_layers:
                        best_train_predictions = []
                        for state, candidate in zip(
                            fold_states, candidate_layers, strict=True
                        ):
                            train_prediction = np.asarray(
                                candidate.predict(state.current_train), dtype=self.dtype
                            )
                            _ensure_finite(
                                train_prediction,
                                context=(
                                    f"predicting training data for layer {layer_number}, "
                                    f"degree {degree}"
                                ),
                            )
                            best_train_predictions.append(train_prediction)

            selected_degree = best_index + 1
            selected_degrees.append(selected_degree)
            layer_mean_scores.append(mean_scores)
            layer_fold_scores.append(all_fold_scores)
            layer_best_scores.append(best_mean_score)

            if layer_number < n_layers:
                if best_train_predictions is None or best_validation_predictions is None:
                    raise RuntimeError("Cross-validation did not retain the selected predictions.")
                self._advance_fold_states(
                    fold_states,
                    best_train_predictions,
                    best_validation_predictions,
                    next_exponent=layer_number + 1,
                    completed_layer=layer_number,
                )

        estimator = self._make_estimator(tuple(selected_degrees), n_layers=n_layers).fit(x, y)

        self.selected_degrees_ = tuple(selected_degrees)
        self.layer_cv_scores_ = layer_mean_scores
        self.layer_cv_fold_scores_ = layer_fold_scores
        self.layer_best_scores_ = layer_best_scores
        # Backward-compatible scalar diagnostics refer to the final output layer.
        self.selected_degree_ = selected_degrees[-1]
        self.cv_scores_ = layer_mean_scores[-1].copy()
        self.cv_fold_scores_ = [scores.copy() for scores in layer_fold_scores[-1]]
        self.best_score_ = layer_best_scores[-1]
        self.estimator_ = estimator
        self.n_features_in_ = estimator.n_features_in_
        self.n_layers_ = n_layers
        if hasattr(estimator, "feature_names_in_"):
            self.feature_names_in_ = estimator.feature_names_in_.copy()
        self.is_fitted_ = True
        return self

    def predict(self, x: Any) -> NDArray[np.float64]:
        """Predict with the full-data refit of the selected network."""

        self._check_is_fitted()
        return self.estimator_.predict(x)

    def score(self, x: Any, y: Any) -> float:
        """Return R-squared from the selected full-data network."""

        self._check_is_fitted()
        return self.estimator_.score(x, y)

    def _make_estimator(
        self, degree: int | Sequence[int], *, n_layers: int
    ) -> DeepPolyNetwork:
        return DeepPolyNetwork(
            n_layers=n_layers,
            degree=degree,
            regularization=self.regularization,
            alpha_ridge=self.alpha_ridge,
            fit_intercept=self.fit_intercept,
            scale=self.scale,
            scale_y=self.scale_y,
            normalize_latent=self.normalize_latent,
            dtype=self.dtype,
            copy=self.copy,
            check_input=self.check_input,
            lstsq_rcond=self.lstsq_rcond,
        )

    def _make_layer(self, degree: int) -> EIKGPolynomialRegressor:
        return EIKGPolynomialRegressor(
            degree=degree,
            regularization=self.regularization,
            alpha_ridge=self.alpha_ridge,
            fit_intercept=self.fit_intercept,
            scale=self.scale,
            scale_y=self.scale_y,
            normalize_latent=self.normalize_latent,
            dtype=self.dtype,
            copy=self.copy,
            check_input=self.check_input,
            lstsq_rcond=self.lstsq_rcond,
        )

    def _initialize_fold_states(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        splits: list[tuple[NDArray[np.int_], NDArray[np.int_]]],
    ) -> list[_FoldState]:
        states: list[_FoldState] = []
        for train_idx, validation_idx in splits:
            base_scale, base_train = _scale_columns_by_max_abs(
                x[train_idx], dtype=self.dtype
            )
            base_validation = _apply_column_scales(
                x[validation_idx], base_scale, dtype=self.dtype
            )
            states.append(
                _FoldState(
                    y_train=y[train_idx],
                    y_validation=y[validation_idx],
                    base_train=base_train,
                    base_validation=base_validation,
                    current_train=base_train,
                    current_validation=base_validation,
                    current_power_train=base_train,
                    current_power_validation=base_validation,
                )
            )
        return states

    def _advance_fold_states(
        self,
        states: list[_FoldState],
        train_predictions: list[NDArray[np.float64]],
        validation_predictions: list[NDArray[np.float64]],
        *,
        next_exponent: int,
        completed_layer: int,
    ) -> None:
        for state, train_prediction, validation_prediction in zip(
            states, train_predictions, validation_predictions, strict=True
        ):
            prediction_scale = _prediction_scale(train_prediction)
            normalized_train = _normalize_prediction(
                train_prediction,
                prediction_scale,
                dtype=self.dtype,
                layer_number=completed_layer,
            )
            normalized_validation = _normalize_prediction(
                validation_prediction,
                prediction_scale,
                dtype=self.dtype,
                layer_number=completed_layer,
            )
            state.current_power_train = _next_power(
                state.current_power_train,
                state.base_train,
                dtype=self.dtype,
                exponent=next_exponent,
            )
            state.current_power_validation = _next_power(
                state.current_power_validation,
                state.base_validation,
                dtype=self.dtype,
                exponent=next_exponent,
            )
            state.current_train = np.asarray(
                np.column_stack((normalized_train, state.current_power_train)),
                dtype=self.dtype,
            )
            state.current_validation = np.asarray(
                np.column_stack(
                    (normalized_validation, state.current_power_validation)
                ),
                dtype=self.dtype,
            )
            _ensure_finite(
                state.current_train,
                context=f"building fold training input for layer {completed_layer + 1}",
            )
            _ensure_finite(
                state.current_validation,
                context=f"building fold validation input for layer {completed_layer + 1}",
            )

    def _make_splits(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        cv: int,
        random_state: int | None,
    ) -> list[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        indices = np.arange(x.shape[0])
        if self.shuffle:
            indices = np.random.default_rng(random_state).permutation(indices)
        fold_sizes = np.full(cv, x.shape[0] // cv, dtype=int)
        fold_sizes[: x.shape[0] % cv] += 1
        starts = np.cumsum(np.concatenate(([0], fold_sizes[:-1])))
        splits: list[tuple[NDArray[np.int_], NDArray[np.int_]]] = []
        for start, fold_size in zip(starts, fold_sizes):
            validation_idx = indices[start : start + fold_size]
            train_mask = np.ones(x.shape[0], dtype=bool)
            train_mask[validation_idx] = False
            train_idx = np.arange(x.shape[0])[train_mask]
            splits.append((train_idx, validation_idx))
        return splits

    def _score_fold(
        self,
        y_true: NDArray[np.float64],
        prediction: NDArray[np.float64],
    ) -> float:
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
                if self.scoring == "r2":
                    score = r2_score(y_true, prediction)
                else:
                    score = -mean_squared_error(y_true, prediction)
        except FloatingPointError as exc:
            raise FloatingPointError(
                "Cross-validation scoring overflowed. Reduce data magnitude or model complexity."
            ) from exc
        if not np.isfinite(score):
            raise FloatingPointError(
                "Cross-validation produced a non-finite score. "
                "Reduce data magnitude or model complexity."
            )
        return score

    def _validated_random_state(self) -> int | None:
        if self.random_state is None:
            return None
        if isinstance(self.random_state, (bool, np.bool_)) or not isinstance(
            self.random_state, Integral
        ):
            raise ValueError(f"random_state must be an integer or None, got {self.random_state!r}.")
        result = int(self.random_state)
        if result < 0:
            raise ValueError("random_state must be non-negative.")
        return result

    def _check_is_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Estimator is not fitted. Call fit(X, y) first.")

    def _clear_fitted_state(self) -> None:
        fitted_attributes = (
            "selected_degrees_",
            "layer_cv_scores_",
            "layer_cv_fold_scores_",
            "layer_best_scores_",
            "selected_degree_",
            "cv_scores_",
            "cv_fold_scores_",
            "best_score_",
            "estimator_",
            "n_features_in_",
            "n_layers_",
            "feature_names_in_",
            "is_fitted_",
        )
        for attribute in fitted_attributes:
            if hasattr(self, attribute):
                delattr(self, attribute)


# Backward-compatible public names. Keep these as direct aliases so existing
# imports, parameter grids, and serialized references continue to resolve to
# the same estimator classes without introducing duplicate subclasses.
PolynomialNetwork = DeepPolyNetwork
PolynomialNetworkCV = DeepPolyNetworkCV
