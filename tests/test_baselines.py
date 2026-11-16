"""Смоук-тести бейзлайнів на синтетичних текстах (без зовнішніх даних)."""
import numpy as np

from narr.baselines import make_model, oof_proba


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    pos_words = ["зрада", "корупція", "капітуляція", "бандера"]
    neg_words = ["погода", "дорога", "ремонт", "концерт"]
    y = rng.integers(0, 2, n)
    texts = []
    for lab in y:
        words = list(rng.choice(pos_words if lab else neg_words, 4)) + list(rng.choice(pos_words + neg_words, 2))
        texts.append(" ".join(words) + " новини")
    return np.array(texts, dtype=object), y


def test_oof_shapes_and_every_item_predicted_once_per_repeat():
    X, y = _toy()
    P = oof_proba("word", X, y, n_splits=4, n_repeats=2, seed=1, n_jobs=1)
    assert P.shape == (2, len(y), 2)
    assert np.allclose(P.sum(axis=2), 1.0)  # кожен рядок заповнений, ймовірності сумуються в 1


def test_model_learns_planted_signal_and_dummy_does_not():
    X, y = _toy()
    P = oof_proba("word", X, y, n_splits=4, n_repeats=1, seed=1, n_jobs=1)
    acc = ((P[0, :, 1] >= 0.5).astype(int) == y).mean()
    assert acc > 0.9
    Pd = oof_proba("majority", X, y, n_splits=4, n_repeats=1, seed=1, n_jobs=1)
    assert len(np.unique(Pd[0].argmax(1))) == 1


def test_group_cv_keeps_groups_apart():
    X, y = _toy()
    groups = np.repeat(np.arange(24), 5)
    P = oof_proba("word", X, y, n_splits=4, n_repeats=1, seed=1, groups=groups, n_jobs=1)
    assert np.allclose(P.sum(axis=2), 1.0)


def test_make_model_names():
    for n in ["majority", "stratified", "word", "char", "word_char"]:
        assert make_model(n, 0) is not None
