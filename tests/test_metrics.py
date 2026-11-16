import numpy as np
import pytest
from sklearn.metrics import f1_score, matthews_corrcoef

from narr.audit import fleiss_kappa_counts, fleiss_kappa_labels
from narr.evaluate import bootstrap_ci, binary_metrics, multiclass_metrics, paired_bootstrap_diff


def test_binary_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.4 + rng.random(300) * 0.6, 0, 1)
    m = binary_metrics(y, p)
    pred = (p >= 0.5).astype(int)
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, pred))
    assert m["f1_pos"] == pytest.approx(f1_score(y, pred))


def test_multiclass_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 400)
    pred = np.where(rng.random(400) < 0.6, y, rng.integers(0, 4, 400))
    m = multiclass_metrics(y, pred, 4)
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, pred))


def test_perfect_and_constant_predictions():
    y = np.array([0, 1] * 50)
    assert binary_metrics(y, y.astype(float))["macro_f1"] == 1.0
    const = binary_metrics(y, np.zeros(100))
    assert const["mcc"] == 0.0 and const["f1_pos"] == 0.0


def test_bootstrap_ci_contains_point_and_is_reproducible():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    P = np.stack([np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1) for _ in range(3)])
    a = bootstrap_ci(y, P, binary_metrics, n_boot=200, seed=5)
    b = bootstrap_ci(y, P, binary_metrics, n_boot=200, seed=5)
    assert a.equals(b)
    mf = a[a.metric == "macro_f1"].iloc[0]
    assert mf.ci_lo <= mf.point <= mf.ci_hi


def test_paired_diff_of_identical_models_is_zero():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 150)
    P = rng.random((2, 150))
    d = paired_bootstrap_diff(y, P, P, binary_metrics, n_boot=100, seed=1)
    assert (d["diff"].abs() < 1e-12).all() and (d["ci_lo"].abs() < 1e-12).all()


def test_fleiss_kappa_wikipedia_example():
    counts = np.array([[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
                       [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])
    assert fleiss_kappa_counts(counts) == pytest.approx(0.210, abs=5e-4)


def test_fleiss_labels_perfect_agreement_is_one():
    import pandas as pd
    r = pd.DataFrame({"a": list("xyxyxy"), "b": list("xyxyxy"), "c": list("xyxyxy")})
    assert fleiss_kappa_labels(r) == pytest.approx(1.0)
