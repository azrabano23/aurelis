import math

from aurelis.metrics import mae, mean, pearson, quadratic_weighted_kappa, wilson_interval


def test_mean_and_mae():
    assert mean([]) == 0.0
    assert mean([2, 4]) == 3.0
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0
    assert math.isclose(mae([1, 2], [2, 4]), 1.5)


def test_pearson_perfect_and_inverse():
    assert math.isclose(pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0, abs_tol=1e-9)
    assert math.isclose(pearson([1, 2, 3, 4], [4, 3, 2, 1]), -1.0, abs_tol=1e-9)
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0  # no variance -> 0


def test_qwk_agreement_levels():
    a = [0, 1, 2, 3, 4]
    assert math.isclose(quadratic_weighted_kappa(a, a), 1.0, abs_tol=1e-9)
    # systematic off-by-one is good-but-imperfect agreement
    near = [1, 2, 3, 4, 4]
    assert 0.5 < quadratic_weighted_kappa(a, near) < 1.0
    # reversed ordering is strongly negative
    assert quadratic_weighted_kappa(a, [4, 3, 2, 1, 0]) < 0


def test_wilson_brackets_and_bounds():
    lo, hi = wilson_interval(7, 10)
    assert lo < 0.7 < hi and 0.0 <= lo <= hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
