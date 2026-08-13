# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and the project adheres to Semantic Versioning.

## [Unreleased]

### Added

- `PolynomialNetwork`, a fixed-width cascade of independently fitted
  `EIKGPolynomialRegressor` layers with an explicit `n_layers` parameter.
- Train-only max-absolute scaling for original features and intermediate predictions before
  constructing successive layer inputs.
- `PolynomialNetworkCV` for leakage-safe, end-to-end selection of a common layer degree and a
  final refit on all supplied training data.
- Validation and diagnostics for non-finite inputs, unstable powered features, fitted state,
  repeated fitting, and feature-name consistency.
- Documentation for the network mathematics, API, numerical-stability behavior,
  cross-validation design, computational cost, and known limitations.

### Changed

- Replaced the Ridge normal-equation solve with a thin-SVD filter to avoid squaring the design
  matrix condition number, avoid a dense penalty identity, and support rank-deficient inputs,
  including `alpha_ridge=0`.
- Added validation requiring `alpha_ridge` to be finite and non-negative.
- Restored modern scikit-learn regressor tags for the existing estimators and made
  `EIKGPolynomialRegressorCV` expose its layer options as explicit, clone-safe parameters.
- Fitted state is now cleared on refit, preventing stale DataFrame metadata or mixed model/scaler
  state after an unsuccessful repeated fit.
- Lightweight scaling now raises diagnostic errors for non-finite statistics and transformed
  values instead of allowing overflow warnings to become corrupted model inputs.
- Scaling and latent normalization use max-abs-preconditioned statistics so very large finite
  targets can be processed without overflowing mean or variance calculations.
- Complex-valued inputs and unsupported numeric dtypes are rejected before lossy conversion;
  scoring validates sample counts and follows the standard constant-target R-squared convention.

## [0.1.1] - 2026-05-09

### Changed
- Renamed distribution package from `eikg-regressor` to `eikgp-regressor`.
- Added release and publishing guidance for repeatable pip/TestPyPI/PyPI workflow.
- Finalized CI + lint/type/test/build validation for publish-ready state.

## [0.1.0] - 2026-05-09

### Added
- Initial `eikg` package structure and public API.
- `EIKGPolynomialRegressor` with numerically stable two-stage fitting.
- `EIKGPolynomialRegressorCV` for automatic degree selection via cross-validation.
- Optional support for `pandas`, `scipy`, and `scikit-learn`.
- Strict input validation utilities and lightweight preprocessing/metrics modules.
- Pytest suite for regressor behavior, validation, and sklearn compatibility.
- README with usage, pipeline, and GridSearch examples.
- Packaging metadata for pip installation and publishing.

