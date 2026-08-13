import numpy as np
import pytest

from eikg.metrics import r2_score
from eikg.validation import validate_degree, validate_x, validate_xy_lengths, validate_y


@pytest.mark.parametrize("degree", [0, -1, 1.5, True, "2"])
def test_validate_degree_raises_for_invalid(degree: object) -> None:
    with pytest.raises(ValueError, match="degree must be >="):
        validate_degree(degree)  # type: ignore[arg-type]


def test_validate_xy_length_mismatch() -> None:
    x = np.ones((5, 2))
    y = np.ones(4)
    with pytest.raises(ValueError, match="row mismatch"):
        validate_xy_lengths(x, y)


def test_validate_x_nan_raises() -> None:
    x = np.array([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(ValueError, match="contains NaN"):
        validate_x(x)


def test_validate_y_multicolumn_df_raises() -> None:
    pd = pytest.importorskip("pandas")
    y = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="exactly one column"):
        validate_y(y)


def test_r2_constant_target_matches_sklearn_convention() -> None:
    y = np.full(5, 3.0)

    assert r2_score(y, y.copy()) == 1.0
    assert r2_score(y, np.zeros_like(y)) == 0.0
