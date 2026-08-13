from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from eikg import PolynomialNetwork, PolynomialNetworkCV
from eikg.metrics import mean_squared_error, r2_score
from eikg.regressors import EIKGPolynomialRegressor


def make_data(
    n_samples: int = 72, n_features: int = 3, seed: int = 23
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    weights = np.linspace(1.3, -0.4, n_features)
    latent = 0.6 + x @ weights
    y = latent + 0.14 * latent**2 - 0.01 * latent**3
    y += rng.normal(scale=0.025, size=n_samples)
    return x, y


def contiguous_kfold_indices(
    n_samples: int, n_splits: int
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    all_indices = np.arange(n_samples)
    for fold_size in fold_sizes:
        start, stop = current, current + int(fold_size)
        test_indices = all_indices[start:stop]
        train_indices = np.concatenate((all_indices[:start], all_indices[stop:]))
        folds.append((train_indices, test_indices))
        current = stop
    return folds


def ordered_degree_values(values: Any, max_degree: int) -> NDArray[np.float64]:
    if isinstance(values, Mapping):
        return np.asarray([values[degree] for degree in range(1, max_degree + 1)], dtype=float)
    return np.asarray(values, dtype=float)


def test_network_constructor_exposes_documented_defaults() -> None:
    model = PolynomialNetwork()

    assert model.n_layers == 3
    assert model.degree == 2
    assert model.regularization == "ridge"
    assert model.alpha_ridge == pytest.approx(1e-8)
    assert model.fit_intercept is True
    assert model.scale is True
    assert model.scale_y is False
    assert model.normalize_latent is True
    assert model.dtype is np.float64
    assert model.copy is True
    assert model.check_input is True
    assert model.lstsq_rcond is None
    assert not hasattr(model, "layers_")

    cv_model = PolynomialNetworkCV()
    assert cv_model.n_layers == 3
    assert cv_model.max_degree == 6
    assert cv_model.scoring == "neg_mean_squared_error"
    assert cv_model.cv == 5
    assert cv_model.shuffle is False
    assert cv_model.random_state is None
    assert cv_model.regularization == "ridge"
    assert not hasattr(cv_model, "estimator_")


@pytest.mark.parametrize("n_layers", [0, -1, 1.5, True, "2", None])
def test_network_rejects_invalid_n_layers(n_layers: Any) -> None:
    x, y = make_data(n_samples=20)

    with pytest.raises((TypeError, ValueError), match="n_layers"):
        PolynomialNetwork(n_layers=n_layers).fit(x, y)

    with pytest.raises((TypeError, ValueError), match="n_layers"):
        PolynomialNetworkCV(n_layers=n_layers, max_degree=1, cv=2).fit(x, y)


@pytest.mark.parametrize("n_layers", [1, 3])
def test_network_fit_predict_and_fitted_state(n_layers: int) -> None:
    x, y = make_data()
    model = PolynomialNetwork(n_layers=n_layers, degree=3, alpha_ridge=1e-6)

    returned = model.fit(x, y)
    prediction = model.predict(x)

    assert returned is model
    assert model.is_fitted_ is True
    assert model.n_features_in_ == x.shape[1]
    assert len(model.layers_) == n_layers
    assert all(isinstance(layer, EIKGPolynomialRegressor) for layer in model.layers_)
    assert all(layer.is_fitted_ for layer in model.layers_)
    assert model.layer_input_sizes_ == [x.shape[1]] + [x.shape[1] + 1] * (n_layers - 1)
    assert len(model.layer_prediction_scales_) == max(0, n_layers - 1)
    assert prediction.shape == (x.shape[0],)
    assert np.isfinite(prediction).all()
    assert model.score(x, y) > 0.9
    assert model.score(x, y) == pytest.approx(r2_score(y, prediction))

    base_scale = np.asarray(model.base_scale_, dtype=float)
    assert np.isfinite(base_scale).all()
    assert np.all(base_scale > 0.0)
    prediction_scales = np.asarray(model.layer_prediction_scales_, dtype=float)
    assert np.isfinite(prediction_scales).all()
    assert np.all(prediction_scales > 0.0)


def test_network_prediction_follows_fixed_width_layer_architecture() -> None:
    x, y = make_data(n_samples=60)
    model = PolynomialNetwork(n_layers=4, degree=2, alpha_ridge=1e-6).fit(x, y)

    scaled_x = x / np.asarray(model.base_scale_)
    expected = model.layers_[0].predict(scaled_x)
    for layer_number, (layer, prediction_scale) in enumerate(
        zip(model.layers_[1:], model.layer_prediction_scales_, strict=True), start=2
    ):
        layer_input = np.column_stack(
            (expected / prediction_scale, np.power(scaled_x, layer_number))
        )
        assert layer_input.shape[1] == x.shape[1] + 1
        expected = layer.predict(layer_input)

    np.testing.assert_allclose(model.predict(x), expected, rtol=1e-11, atol=1e-11)


def test_network_repeated_fit_replaces_all_learned_layers() -> None:
    x_first, y_first = make_data(n_samples=50, n_features=3, seed=31)
    x_second, y_second = make_data(n_samples=65, n_features=2, seed=32)
    parameters = {"n_layers": 3, "degree": 2, "alpha_ridge": 1e-6}
    model = PolynomialNetwork(**parameters).fit(x_first, y_first)
    old_layers = tuple(model.layers_)

    model.fit(x_second, y_second)
    fresh_model = PolynomialNetwork(**parameters).fit(x_second, y_second)

    assert len(model.layers_) == parameters["n_layers"]
    assert model.n_features_in_ == x_second.shape[1]
    assert model.layer_input_sizes_ == [2, 3, 3]
    assert all(
        new_layer is not old_layer for new_layer in model.layers_ for old_layer in old_layers
    )
    np.testing.assert_allclose(
        model.predict(x_second), fresh_model.predict(x_second), rtol=1e-11, atol=1e-11
    )


def test_network_unfitted_predict_and_score_raise() -> None:
    x, y = make_data(n_samples=12)
    model = PolynomialNetwork()

    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(x)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.score(x, y)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_network_rejects_nonfinite_training_data(bad_value: float) -> None:
    x, y = make_data(n_samples=24)
    bad_x = x.copy()
    bad_x[0, 0] = bad_value
    bad_y = y.copy()
    bad_y[0] = bad_value

    with pytest.raises(ValueError, match="X contains"):
        PolynomialNetwork(n_layers=2).fit(bad_x, y)
    with pytest.raises(ValueError, match="y contains"):
        PolynomialNetwork(n_layers=2).fit(x, bad_y)


@pytest.mark.parametrize(
    ("bad_x", "bad_y"),
    [
        (np.ones(5), np.ones(5)),
        (np.ones((5, 2, 1)), np.ones(5)),
        (np.empty((0, 2)), np.empty(0)),
        (np.empty((5, 0)), np.ones(5)),
        (np.ones((5, 2)), np.ones(4)),
        (np.ones((5, 2)), np.ones((5, 2))),
    ],
)
def test_network_rejects_invalid_training_shapes(bad_x: Any, bad_y: Any) -> None:
    with pytest.raises(ValueError):
        PolynomialNetwork(n_layers=1).fit(bad_x, bad_y)


def test_network_validates_predict_features_and_values() -> None:
    x, y = make_data(n_samples=30)
    model = PolynomialNetwork(n_layers=2).fit(x, y)

    with pytest.raises(ValueError, match="features"):
        model.predict(x[:, :2])

    bad_x = x.copy()
    bad_x[0, 0] = np.inf
    with pytest.raises(ValueError, match="X contains"):
        model.predict(bad_x)

    with pytest.raises(ValueError, match="row mismatch"):
        model.score(x, np.ones(1))


@pytest.mark.parametrize("dtype", [np.int64, np.bool_, np.float16, np.longdouble])
def test_network_rejects_unsupported_dtype(dtype: Any) -> None:
    x, y = make_data(n_samples=20)

    with pytest.raises(ValueError, match="dtype"):
        PolynomialNetwork(dtype=dtype).fit(x, y)


def test_network_rejects_complex_data_without_discarding_imaginary_part() -> None:
    x, y = make_data(n_samples=20)

    with pytest.raises(ValueError, match="Complex"):
        PolynomialNetwork().fit(x.astype(np.complex128) + 1j, y)
    with pytest.raises(ValueError, match="Complex"):
        PolynomialNetwork().fit(x, y.astype(np.complex128) + 1j)


def test_network_dataframe_feature_names_and_input_immutability() -> None:
    pd = pytest.importorskip("pandas")
    x, y = make_data(n_samples=48)
    index = np.arange(x.shape[0]) * 7 + 101
    x_df = pd.DataFrame(x, columns=["first", "second", "third"], index=index)
    y_series = pd.Series(y, index=index)
    original_x = x_df.copy(deep=True)
    original_y = y_series.copy(deep=True)
    model = PolynomialNetwork(n_layers=2, degree=2).fit(x_df, y_series)

    np.testing.assert_array_equal(model.feature_names_in_, np.asarray(x_df.columns, dtype=str))
    assert np.isfinite(model.predict(x_df)).all()
    pd.testing.assert_frame_equal(x_df, original_x)
    pd.testing.assert_series_equal(y_series, original_y)

    with pytest.raises(ValueError, match="columns|feature names"):
        model.predict(x_df[["second", "first", "third"]])

    model.fit(x, y)
    assert not hasattr(model, "feature_names_in_")


def test_network_is_stable_for_large_finite_features() -> None:
    rng = np.random.default_rng(44)
    magnitude = 1e100
    x = rng.normal(size=(70, 3)) * magnitude
    scaled = x / magnitude
    y = 0.4 + 1.2 * scaled[:, 0] - 0.7 * scaled[:, 1] + 0.08 * scaled[:, 0] ** 2
    model = PolynomialNetwork(n_layers=4, degree=3, alpha_ridge=1e-5)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        model.fit(x, y)
        prediction = model.predict(x * 0.9)

    assert prediction.shape == (x.shape[0],)
    assert np.isfinite(prediction).all()
    assert np.isfinite(np.asarray(model.base_scale_)).all()
    assert np.isfinite(np.asarray(model.layer_prediction_scales_)).all()


@pytest.mark.parametrize("scale_y", [False, True])
def test_network_is_stable_for_large_finite_targets(scale_y: bool) -> None:
    rng = np.random.default_rng(45)
    x = rng.normal(size=(70, 3))
    y = 1e200 * (1.0 + 0.2 * x[:, 0] - 0.05 * x[:, 1])
    model = PolynomialNetwork(
        n_layers=2,
        degree=2,
        scale_y=scale_y,
        alpha_ridge=1e-8,
    ).fit(x, y)

    prediction = model.predict(x)

    assert np.isfinite(prediction).all()
    assert model.score(x, y) > 0.99


def test_network_raises_on_numerically_unsafe_extrapolation() -> None:
    x = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    y = x[:, 0] + 0.2 * x[:, 0] ** 2
    model = PolynomialNetwork(n_layers=3, degree=3).fit(x, y)
    extreme_x = np.full((2, 1), np.finfo(np.float64).max)

    with pytest.raises(FloatingPointError, match="Overflow|non-finite|Non-finite"):
        model.predict(extreme_x)


def test_network_cv_matches_manual_end_to_end_cross_validation() -> None:
    x, y = make_data(n_samples=42, seed=51)
    max_degree = 2
    n_splits = 3
    common_parameters = {
        "n_layers": 2,
        "regularization": "ridge",
        "alpha_ridge": 1e-5,
        "scale": True,
        "normalize_latent": True,
    }
    model = PolynomialNetworkCV(
        max_degree=max_degree,
        cv=n_splits,
        shuffle=False,
        scoring="neg_mean_squared_error",
        **common_parameters,
    ).fit(x, y)

    expected_fold_scores: list[list[float]] = []
    for degree in range(1, max_degree + 1):
        degree_scores: list[float] = []
        for train_indices, test_indices in contiguous_kfold_indices(x.shape[0], n_splits):
            fold_model = PolynomialNetwork(degree=degree, **common_parameters).fit(
                x[train_indices], y[train_indices]
            )
            fold_prediction = fold_model.predict(x[test_indices])
            degree_scores.append(-mean_squared_error(y[test_indices], fold_prediction))
        expected_fold_scores.append(degree_scores)

    fold_scores = ordered_degree_values(model.cv_fold_scores_, max_degree)
    scores = ordered_degree_values(model.cv_scores_, max_degree)
    expected_fold_scores_array = np.asarray(expected_fold_scores)
    expected_scores = np.mean(expected_fold_scores_array, axis=1)
    expected_degree = int(np.argmax(expected_scores)) + 1

    np.testing.assert_allclose(fold_scores, expected_fold_scores_array, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-11, atol=1e-11)
    assert model.selected_degree_ == expected_degree
    assert model.best_score_ == pytest.approx(expected_scores[expected_degree - 1])
    assert model.is_fitted_ is True
    assert isinstance(model.estimator_, PolynomialNetwork)
    assert model.estimator_.degree == model.selected_degree_
    assert len(model.estimator_.layers_) == model.n_layers
    assert np.isfinite(model.predict(x)).all()


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"max_degree": 0}, "max_degree"),
        ({"cv": 1}, "cv"),
        ({"scoring": "mean_absolute_error"}, "scoring"),
    ],
)
def test_network_cv_rejects_invalid_search_parameters(
    parameters: dict[str, Any], message: str
) -> None:
    x, y = make_data(n_samples=20)

    with pytest.raises((TypeError, ValueError), match=message):
        PolynomialNetworkCV(n_layers=1, max_degree=2, cv=2, **parameters).fit(x, y)


def test_network_cv_rejects_more_folds_than_samples() -> None:
    x, y = make_data(n_samples=5)

    with pytest.raises(ValueError, match="cv|splits|samples"):
        PolynomialNetworkCV(n_layers=1, max_degree=1, cv=6).fit(x, y)


def test_network_cv_rejects_single_sample_r2_folds() -> None:
    x, y = make_data(n_samples=5)

    with pytest.raises(ValueError, match="at least 2 validation samples"):
        PolynomialNetworkCV(n_layers=1, max_degree=1, cv=5, scoring="r2").fit(x, y)


def test_network_cv_shuffle_is_reproducible_with_random_state() -> None:
    x, y = make_data(n_samples=36, seed=61)
    parameters = {
        "n_layers": 2,
        "max_degree": 2,
        "cv": 3,
        "shuffle": True,
        "random_state": 123,
        "alpha_ridge": 1e-5,
    }

    first = PolynomialNetworkCV(**parameters).fit(x, y)
    second = PolynomialNetworkCV(**parameters).fit(x, y)

    assert first.selected_degree_ == second.selected_degree_
    np.testing.assert_allclose(
        ordered_degree_values(first.cv_fold_scores_, 2),
        ordered_degree_values(second.cv_fold_scores_, 2),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(first.predict(x), second.predict(x), rtol=1e-12, atol=1e-12)


def test_network_estimators_are_sklearn_cloneable() -> None:
    sklearn = pytest.importorskip("sklearn")
    from sklearn.base import clone, is_regressor

    del sklearn
    model = PolynomialNetwork(
        n_layers=2,
        degree=3,
        regularization="none",
        alpha_ridge=0.25,
        fit_intercept=False,
        scale=False,
        scale_y=True,
        normalize_latent=False,
        dtype=np.float32,
        copy=False,
        check_input=False,
        lstsq_rcond=1e-7,
    )
    cloned = clone(model)

    assert cloned.get_params(deep=False) == model.get_params(deep=False)
    assert not hasattr(cloned, "layers_")
    assert is_regressor(cloned)

    cv_model = PolynomialNetworkCV(
        n_layers=2,
        max_degree=3,
        cv=2,
        shuffle=True,
        random_state=17,
        regularization="none",
        alpha_ridge=0.25,
        fit_intercept=False,
        scale=False,
        scale_y=True,
        normalize_latent=False,
        dtype=np.float32,
        copy=False,
        check_input=False,
        lstsq_rcond=1e-7,
    )
    cloned_cv = clone(cv_model)

    assert cloned_cv.get_params(deep=False) == cv_model.get_params(deep=False)
    assert not hasattr(cloned_cv, "estimator_")
    assert is_regressor(cloned_cv)


def test_network_works_in_sklearn_pipeline() -> None:
    sklearn = pytest.importorskip("sklearn")
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    del sklearn
    x, y = make_data(n_samples=54)
    pipeline = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("network", PolynomialNetwork(n_layers=2, degree=2, scale=False)),
        ]
    )

    pipeline.fit(x, y)
    prediction = pipeline.predict(x)

    assert prediction.shape == (x.shape[0],)
    assert np.isfinite(prediction).all()


def test_network_works_in_sklearn_grid_search() -> None:
    sklearn = pytest.importorskip("sklearn")
    from sklearn.model_selection import GridSearchCV

    del sklearn
    x, y = make_data(n_samples=54, seed=71)
    search = GridSearchCV(
        PolynomialNetwork(alpha_ridge=1e-5),
        param_grid={"n_layers": [1, 2], "degree": [1, 2]},
        cv=2,
        scoring="neg_mean_squared_error",
    )

    search.fit(x, y)
    prediction = search.predict(x)

    assert search.best_estimator_ is not None
    assert search.best_params_["n_layers"] in {1, 2}
    assert search.best_params_["degree"] in {1, 2}
    assert prediction.shape == (x.shape[0],)
    assert np.isfinite(prediction).all()
