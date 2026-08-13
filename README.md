# EIKGPolynomial

`EIKGPolynomial` is a lightweight Python package for regression models based on the **elementary image of the Kolmogorov-Gabor polynomial**.

The package provides four sklearn-style estimators:

* `EIKGPolynomialRegressor`
* `EIKGPolynomialRegressorCV`
* `PolynomialNetwork`
* `PolynomialNetworkCV`

The model uses a compact two-stage representation:

1. Linear latent stage:

```text
z = b0 + b1*x1 + ... + bm*xm
```

2. Polynomial output stage:

```text
y_hat = a0 + a1*z + a2*z^2 + ... + ad*z^d
```

This implementation is designed for numerically stable regression workflows with a compact polynomial-in-latent structure.

## Scientific background

The implementation is inspired by the concept of the elementary image of the Kolmogorov-Gabor polynomial proposed in:

> Svetunkov S. Elementary image of the Kolmogorov-Gabor polynomial in economic modeling. *Technoeconomics*, 2024, Vol. 3, No. 2 (9), pp. 4–21.
> DOI: https://doi.org/10.57809/2024.3.2.9.1

Original Russian title:

> Светуньков С. Элементарный образ полинома Колмогорова-Габора в моделировании экономики // Техноэкономика. 2024. Т. 3, № 2 (9). С. 4–21.

## Important note

`EIKGPolynomialRegressor` implements a **compressed elementary image** of the Kolmogorov-Gabor polynomial.

It is not equivalent to direct estimation of the full multivariate polynomial basis. Instead of constructing all multivariate polynomial terms explicitly, the model first builds one latent linear projection and then applies a univariate polynomial transformation to this latent variable.

## Installation

Install the current released version from PyPI:

```bash
pip install eikgp-regressor
```

Install from source in editable mode:

```bash
git clone https://github.com/daniilmir3105/EIKGPolynomial.git
cd EIKGPolynomial
pip install -e .
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Optional dependencies can be installed separately when needed:

```bash
pip install -e ".[scipy]"
pip install -e ".[sklearn]"
pip install -e ".[pandas]"
```

## Basic usage

```python
import numpy as np
from eikg import EIKGPolynomialRegressor

rng = np.random.default_rng(42)

X = rng.normal(size=(200, 3))
z = 1.0 + 1.8 * X[:, 0] - 0.9 * X[:, 1]
y = z + 0.15 * z**2 + rng.normal(0, 0.1, size=200)

model = EIKGPolynomialRegressor(
    degree=3,
    regularization="ridge",
    alpha_ridge=1e-5,
    scale=True,
    normalize_latent=True,
)

model.fit(X, y)

y_pred = model.predict(X)
r2 = model.score(X, y)

print(r2)
```

## `EIKGPolynomialRegressor`

`EIKGPolynomialRegressor` is the main regression model.

```python
from eikg import EIKGPolynomialRegressor
```

### Main parameters

| Parameter          |                        Type |      Default | Description                                                                                   |
| ------------------ | --------------------------: | -----------: | --------------------------------------------------------------------------------------------- |
| `degree`           |                       `int` |          `2` | Degree of the polynomial transformation applied to the latent variable.                       |
| `regularization`   | `"none"`, `"ridge"`, `None` |     `"none"` | Least-squares regularization mode.                                                            |
| `alpha_ridge`      |                     `float` |       `1e-8` | Ridge regularization strength. Used only when `regularization="ridge"`.                       |
| `fit_intercept`    |                      `bool` |       `True` | Whether to fit intercept terms in both stages.                                                |
| `scale`            |                      `bool` |       `True` | Whether to standardize input features `X` before fitting.                                     |
| `scale_y`          |                      `bool` |      `False` | Whether to standardize target values `y` during fitting and invert scaling during prediction. |
| `normalize_latent` |                      `bool` |       `True` | Whether to normalize the latent variable before building polynomial powers.                   |
| `dtype`            | `np.float32` or `np.float64` | `np.float64` | Internal numeric dtype.                                                                       |
| `copy`             |                      `bool` |       `True` | Whether to copy input arrays during validation.                                               |
| `check_input`      |                      `bool` |       `True` | Whether to check input shapes and finite values.                                              |
| `lstsq_rcond`      |           `float` or `None` |       `None` | Cutoff parameter passed to the least-squares solver.                                          |

### Practical recommendations

For most tasks, start with:

```python
model = EIKGPolynomialRegressor(
    degree=2,
    regularization="ridge",
    alpha_ridge=1e-5,
    scale=True,
    normalize_latent=True,
)
```

Use `regularization="ridge"` when the data are noisy, the features are correlated, or the polynomial degree is greater than 2.

Keep `scale=True` unless the input data are already standardized.

Keep `normalize_latent=True` for better numerical stability, especially for higher polynomial degrees.

Use `scale_y=True` when target values have a large magnitude or a very wide numerical range.

## `EIKGPolynomialRegressorCV`

`EIKGPolynomialRegressorCV` automatically selects the polynomial degree using cross-validation.

```python
from eikg import EIKGPolynomialRegressorCV
```

Example:

```python
cv_model = EIKGPolynomialRegressorCV(
    max_degree=6,
    cv=5,
    scoring="neg_mean_squared_error",
    regularization="ridge",
    alpha_ridge=1e-5,
    scale=True,
    normalize_latent=True,
)

cv_model.fit(X, y)

print(cv_model.selected_degree_)
print(cv_model.best_score_)
```

### Main parameters

| Parameter            |                               Type |                    Default | Description                                                |
| -------------------- | ---------------------------------: | -------------------------: | ---------------------------------------------------------- |
| `max_degree`         |                              `int` |                        `6` | Maximum degree tested during cross-validation.             |
| `cv`                 |                              `int` |                        `5` | Number of cross-validation folds.                          |
| `scoring`            | `"neg_mean_squared_error"`, `"r2"` | `"neg_mean_squared_error"` | Metric used for degree selection.                          |
| `**regressor_kwargs` |                  keyword arguments |                          — | Additional parameters passed to `EIKGPolynomialRegressor`. |

For example, these parameters are passed directly to the inner regressor:

```python
cv_model = EIKGPolynomialRegressorCV(
    max_degree=5,
    regularization="ridge",
    alpha_ridge=1e-4,
    scale=True,
    normalize_latent=True,
)
```

## `PolynomialNetwork`

`PolynomialNetwork` is a fixed-width cascade of independently fitted
`EIKGPolynomialRegressor` layers. Every layer predicts the same target, and every layer after
the first receives both the preceding layer prediction and one element-wise power of the
original features.

For `m` original features and `L = n_layers`, training first computes a train-only max-absolute
scale for every input column:

```text
s_j = max_i(abs(X[i, j]))            (zero scales are replaced by 1)
X_scaled[:, j] = X[:, j] / s_j
```

The layer inputs are then:

```text
H_1 = X_scaled
p_l = EIKGPolynomialRegressor(degree=d_l).fit(H_l, y).predict(H_l)
H_l = [p_(l-1) / r_(l-1), X_scaled ** l]   for l = 2, ..., L
r_l = max_i(abs(p_l[i]))             (zero scales are replaced by 1)
y_pred = p_L
```

Here `d_l` is the latent polynomial degree of layer `l`. It is independent of the explicit
element-wise feature exponent `l`, which continues to be determined by the layer number. A
scalar `degree=d` sets `d_l=d` for every layer; a sequence supplies the complete
`(d_1, ..., d_L)` configuration.

The max-absolute scales are learned only from the fitting data and reused unchanged by
`predict`. Scaling a power column by a nonzero constant does not change the polynomial model
class, but it prevents raw feature magnitudes from being raised directly to high powers.

This construction is fixed-width: layer 1 receives `m` columns and every later layer receives
`m + 1` columns. It therefore avoids the exponential feature-space expansion of a full
multivariate polynomial basis.

### Usage

```python
import numpy as np

from eikg import PolynomialNetwork

rng = np.random.default_rng(42)
X = rng.uniform(-2.0, 2.0, size=(300, 3))
y = 1.5 * X[:, 0] + 0.8 * X[:, 1] ** 2 - 0.3 * X[:, 2] ** 3
y += rng.normal(scale=0.05, size=X.shape[0])

network = PolynomialNetwork(
    n_layers=3,
    degree=(1, 2, 3),
    regularization="ridge",
    alpha_ridge=1e-8,
    scale=True,
    normalize_latent=True,
)

network.fit(X, y)
y_pred = network.predict(X)
r2 = network.score(X, y)
print(network.degrees_)  # (1, 2, 3)
```

### Main parameters

| Parameter          |                        Type |      Default | Description                                                                                   |
| ------------------ | --------------------------: | -----------: | --------------------------------------------------------------------------------------------- |
| `n_layers`         |                       `int` |          `3` | Number of sequential polynomial layers. Must be a positive integer.                           |
| `degree`           | `int` or sequence of `int` |          `2` | A scalar degree used by every layer, or exactly one positive degree per layer.                 |
| `regularization`   | `"none"`, `"ridge"`, `None` |    `"ridge"` | Least-squares regularization mode used by every layer.                                         |
| `alpha_ridge`      |                     `float` |       `1e-8` | Ridge strength passed to every layer.                                                          |
| `fit_intercept`    |                      `bool` |       `True` | Whether each layer fits intercepts.                                                            |
| `scale`            |                      `bool` |       `True` | Whether each layer standardizes its constructed input matrix.                                  |
| `scale_y`          |                      `bool` |      `False` | Whether each layer standardizes the target during fitting.                                     |
| `normalize_latent` |                      `bool` |       `True` | Whether each layer normalizes its latent projection before constructing powers.                |
| `dtype`            | `np.float32` or `np.float64` | `np.float64` | Internal numeric dtype.                                                                        |
| `copy`             |                      `bool` |       `True` | Whether input arrays are copied during validation.                                             |
| `check_input`      |                      `bool` |       `True` | Whether each inner layer performs its standard input checks. The network always rejects non-finite raw or constructed values. |
| `lstsq_rcond`      |           `float` or `None` |       `None` | Cutoff passed to the least-squares backend in every layer.                                     |

Repeated calls to `fit` replace the complete learned cascade; old layers are not accumulated.
`n_layers=1` creates exactly one polynomial layer and does not construct any unused powered
features.

### Practical recommendations

Start with two or three layers, degree 1 or 2, Ridge regularization, and latent normalization.
A scalar is the simplest baseline. Use a tuple such as `degree=(1, 2, 3)` only when layer-wise
validation supports the extra flexibility; its length must equal `n_layers`.
Increase depth only when validation data show that the additional feature powers improve
generalization. Use `scale_y=True` when target values have a very large magnitude or dynamic
range.

## `PolynomialNetworkCV`

`PolynomialNetworkCV` greedily selects a separate degree for each layer. Starting with layer 1,
it evaluates degrees `1` through `max_degree` using `scoring`, retains that layer's best degree,
builds fold-local inputs for the next layer, and repeats until `n_layers` degrees have been
selected. The resulting tuple is then used to fit one final `PolynomialNetwork` on all data
supplied to `fit`.

```python
from eikg import PolynomialNetworkCV

cv_network = PolynomialNetworkCV(
    n_layers=3,
    max_degree=5,
    cv=5,
    scoring="neg_mean_squared_error",
    shuffle=True,
    random_state=42,
    regularization="ridge",
    alpha_ridge=1e-8,
    scale=True,
    normalize_latent=True,
)

cv_network.fit(X, y)
print(cv_network.selected_degrees_)
print(cv_network.layer_best_scores_)
y_pred = cv_network.predict(X)
```

### Why cross-validation covers the complete network

Running `EIKGPolynomialRegressorCV` independently on a later layer would not be leakage-safe if
the preceding prediction column had first been produced by a model fitted on all rows. Such a
column already depends on the validation targets before the later layer creates its folds.

`PolynomialNetworkCV` avoids this by maintaining an independent fitted prefix for each fold.
For a candidate at layer `l`, that fold's previous layers, input scales, and intermediate
prediction scales were learned only from its training rows. Validation rows are passed only to
`predict`; no validation target contributes to an input of a later layer.

### Main parameters

| Parameter      |                               Type |                    Default | Description                                                        |
| -------------- | ---------------------------------: | -------------------------: | ------------------------------------------------------------------ |
| `n_layers`     |                              `int` |                        `3` | Fixed number of layers in every candidate network.                 |
| `max_degree`   |                              `int` |                        `6` | Highest candidate degree evaluated independently at every layer.   |
| `cv`           |                              `int` |                        `5` | Number of folds. Must be between 2 and the number of samples.       |
| `scoring`      | `"neg_mean_squared_error"`, `"r2"` | `"neg_mean_squared_error"` | Metric used to select the degree.                                  |
| `shuffle`      |                             `bool` |                    `False` | Whether rows are shuffled before folds are formed.                 |
| `random_state` |                    `int` or `None` |                     `None` | Non-negative reproducible shuffle seed; ignored when `shuffle=False`. |

The remaining layer parameters are the same explicit parameters as on `PolynomialNetwork`,
except that the per-layer degree tuple is selected by CV and `degree` is therefore replaced by
`max_degree`.

With `K = cv`, `D = max_degree`, and `L = n_layers`, greedy selection performs `K * D * L`
candidate layer fits, followed by `L` fits for the final network. Selected fold models may also
be used to produce training predictions needed for the next fold-local representation; this
does not change the number of fitted layer models. The overall model-fit count is therefore
`K * D * L + L`. Begin with modest values of `L` and `D`. Setting `shuffle=False` is
deterministic but may be unsuitable for data ordered by time or another systematic variable.

If `n` is the number of rows and `q <= m + 1` is a layer's explicit input width, each dense
least-squares stage costs roughly `O(n * q^2)` when `n >= q`; the latent polynomial solve adds a
corresponding term based on the candidate degree. Thin-SVD workspaces are the main memory
bottleneck. Depth does not widen `q`, but CV repeats these dense solves for every candidate and
fold. Candidate polynomial systems grow with degree, so elapsed time can increase faster than
the simple model-fit count suggests.

### Greedy selection is not a global search

At layer `l`, the selected degree maximizes that layer's mean fold score given the already
selected prefix. Earlier degrees are not reconsidered after later layers are added. A candidate
that scores worse immediately could still produce a representation that helps a future layer,
so the selected tuple is not guaranteed to maximize the final `L`-layer score over all
`max_degree ** n_layers` possible tuples.

The layer coefficients are also fitted sequentially against the target: later layers never
update earlier coefficients. Even an exhaustive search over degree tuples would optimize only
the hyperparameters, not all network coefficients jointly. `layer_best_scores_` and the legacy
`best_score_` are model-selection diagnostics and are selection-biased; use an independent test
set or nested cross-validation for an unbiased generalization estimate.

To select depth and degree together, use the ordinary estimator with scikit-learn:

```python
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    PolynomialNetwork(),
    {"n_layers": [1, 2, 3], "degree": [1, 2, 3]},
    cv=5,
    scoring="neg_mean_squared_error",
)
search.fit(X, y)
```

When external preprocessing must itself be fitted inside each fold, put that preprocessing and
`PolynomialNetwork` in one sklearn `Pipeline` and search the complete pipeline. A supervised
transformer placed before `PolynomialNetworkCV` would otherwise be fitted before the estimator's
internal folds and could leak target information.

## Numerical stability of polynomial networks

The network applies max-absolute normalization before constructing element-wise feature powers
and scales every intermediate prediction using a value learned from the corresponding training
data. It computes powers incrementally and checks constructed matrices and layer predictions for
non-finite values. NaN and infinite inputs are rejected; they are never silently replaced with
zeros. If finite nonzero values underflow to zero during scaling or power construction, the
network emits a `RuntimeWarning` identifying the affected columns.

Prediction uses the scales learned by `fit` and does not clip new observations. Consequently,
values far outside the fitting range can still overflow after repeated powers or polynomial
composition. In that case the estimator raises a diagnostic error instead of returning silently
corrupted predictions. Reducing `n_layers` or `degree`, enabling target scaling, and checking for
out-of-distribution feature magnitudes are the preferred remedies.

Max-absolute scaling bounds the explicit powered features on the training set, but it does not
remove all conditioning risks inside a polynomial layer. Keep `scale=True`,
`normalize_latent=True`, and Ridge regularization for the usual workflow. A high reported
training score is not a substitute for validation on data that were excluded from all fitting
and model-selection steps.

Ridge layers apply the regularization filter to a thin SVD rather than forming the normal
equations `X.T @ X`. This avoids squaring the design matrix condition number, does not allocate a
dense penalty identity, and remains well-defined for rank-deficient inputs, including
`alpha_ridge=0`. The Ridge strength must always be finite and non-negative.

## Fitted attributes

After calling `fit`, the model stores learned parameters and diagnostics:

| Attribute           | Description                                        |
| ------------------- | -------------------------------------------------- |
| `beta_`             | Coefficients of the first linear latent stage.     |
| `intercept_1_`      | Intercept of the first stage.                      |
| `alpha_`            | Coefficients of the polynomial output stage.       |
| `intercept_2_`      | Intercept of the second stage.                     |
| `degree_`           | Fitted polynomial degree.                          |
| `n_features_in_`    | Number of input features seen during fitting.      |
| `condition_number_` | Condition number of the first-stage design matrix. |
| `rank_`             | Rank of the first-stage design matrix.             |
| `rank_latent_`      | Rank of the latent polynomial design matrix.       |

For `EIKGPolynomialRegressorCV`, the main fitted attributes are:

| Attribute          | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `selected_degree_` | Degree selected by cross-validation.                 |
| `best_score_`      | Best cross-validation score.                         |
| `cv_scores_`       | Mean cross-validation scores for all tested degrees. |
| `estimator_`       | Final fitted `EIKGPolynomialRegressor`.              |

For `PolynomialNetwork`, the main fitted attributes are:

| Attribute           | Description                                                            |
| ------------------- | ---------------------------------------------------------------------- |
| `layers_`           | Ordered fitted `EIKGPolynomialRegressor` instances.                    |
| `base_scale_`       | Train-only max-absolute scale for every original input feature.        |
| `layer_prediction_scales_` | Train-only scales for predictions passed between adjacent layers. |
| `layer_input_sizes_` | Number of columns received by each fitted layer.                       |
| `n_features_in_`    | Number of original input features seen during fitting.                 |
| `n_layers_`         | Validated number of fitted layers.                                     |
| `degrees_`          | Canonical tuple containing the fitted degree of every layer.            |
| `degree_`           | Common degree when all values in `degrees_` match; otherwise `None`.    |
| `feature_names_in_` | Original DataFrame column names, when fitting used named columns.      |
| `is_fitted_`        | Whether the complete cascade was fitted successfully.                  |

`layer_prediction_scales_` contains `n_layers - 1` values because the final prediction is
returned directly and is not transformed for another layer.

For `PolynomialNetworkCV`, the main fitted attributes are:

| Attribute           | Description                                                        |
| ------------------- | ------------------------------------------------------------------ |
| `selected_degrees_` | Canonical tuple of greedily selected degrees, one per layer.        |
| `layer_cv_scores_`  | Mean candidate scores for every layer and degree.                   |
| `layer_cv_fold_scores_` | Per-fold candidate scores for every layer and degree.          |
| `layer_best_scores_` | Best mean candidate score reached at each greedy layer step.       |
| `selected_degree_`  | Legacy diagnostic: selected degree at the final greedy step.       |
| `best_score_`       | Legacy diagnostic: best mean score at the final greedy step.       |
| `cv_scores_`        | Legacy diagnostic: candidate mean scores at the final greedy step. |
| `cv_fold_scores_`   | Legacy diagnostic: candidate fold scores at the final greedy step. |
| `estimator_`        | Final `PolynomialNetwork` refitted on all supplied training rows.  |
| `n_features_in_`    | Number of original input features seen during fitting.             |
| `n_layers_`         | Validated number of layers in every evaluated network.             |
| `feature_names_in_` | Original DataFrame column names, when fitting used named columns.  |
| `is_fitted_`        | Whether selection and the final refit completed successfully.      |

The nested diagnostics use layer order first: `layer_cv_scores_[l][d - 1]` is the mean score
for degree `d` at zero-based layer index `l`, while
`layer_cv_fold_scores_[l][d - 1]` contains its fold scores. For compatibility, the legacy
attributes are aliases of the final greedy step:

```text
selected_degree_ = selected_degrees_[-1]
cv_scores_ = layer_cv_scores_[-1]
cv_fold_scores_ = layer_cv_fold_scores_[-1]
best_score_ = layer_best_scores_[-1]
```

They do not summarize the earlier decisions and must not be interpreted as an unbiased score
for the final full-data estimator.

## Minimal API

All four estimators follow the standard sklearn-style workflow:

```python
model.fit(X, y)
y_pred = model.predict(X)
score = model.score(X, y)
```

`score(X, y)` returns the coefficient of determination `R^2`.

## Limitations

* The model uses one latent linear projection, so its expressiveness is more limited than a full multivariate polynomial model.
* High polynomial degrees may be numerically unstable without scaling, latent normalization, or regularization.
* The model is most suitable when the target can be reasonably approximated by a polynomial function of a compact latent representation.
* `PolynomialNetwork` is a greedily fitted cascade, not a jointly optimized neural network; later layers do not update earlier-layer coefficients.
* For per-layer latent degrees `d_l`, an upper bound on the effective algebraic degree follows `e_1 = d_1` and `e_l = d_l * max(e_(l-1), l)`. Although explicit width stays at `m + 1`, effective degree and sensitivity can therefore grow rapidly.
* Max-absolute scaling bounds training powers but cannot guarantee safe extrapolation beyond the observed feature range.
* `PolynomialNetworkCV` uses greedy layer-wise selection and does not guarantee the globally best degree tuple. Exhaustive selection would evaluate up to `max_degree ** n_layers` configurations.
* Cross-validation diagnostics are selection-biased; reserve independent data or use nested cross-validation for performance estimation.
* Built-in K-fold selection is not a replacement for a time-series, grouped, or otherwise domain-specific validation design.

## Development checks

```bash
ruff check .
mypy eikg
pytest
```
