"""Текстові бейзлайни: TF-IDF (слова, символьні n-грами) + логістична регресія, дамі-моделі, повторна CV."""
from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

MODEL_NAMES = ["majority", "stratified", "word", "char", "word_char"]
CS = [0.1, 1, 10, 100]


def _word() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000)


def _char() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=100000)


def make_model(name: str, seed: int, task: str = "binary"):
    """Підбір C відбувається лише всередині навчальної частини (вкладена CV), тест не бачить підбору."""
    scoring = "average_precision" if task == "binary" else "f1_macro"

    def lr():
        return LogisticRegressionCV(
            Cs=CS, cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring=scoring,
            class_weight="balanced", max_iter=3000, random_state=seed)

    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "stratified":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if name == "word":
        return Pipeline([("tfidf", _word()), ("clf", lr())])
    if name == "char":
        return Pipeline([("tfidf", _char()), ("clf", lr())])
    if name == "word_char":
        return Pipeline([("tfidf", FeatureUnion([("w", _word()), ("c", _char())])), ("clf", lr())])
    raise ValueError(name)


def _fit_predict(name: str, task: str, X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, seed: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = make_model(name, seed, task)
        m.fit(X[tr], y[tr])
        return te, m.predict_proba(X[te])


def oof_proba(name: str, X: np.ndarray, y: np.ndarray, *, task: str = "binary", n_splits: int = 5,
              n_repeats: int = 5, seed: int = 42, groups: np.ndarray | None = None, n_jobs: int = 2) -> np.ndarray:
    """Повертає (n_repeats, N, K) ймовірності out-of-fold: кожне повідомлення передбачене моделлю, що його не бачила."""
    jobs = []
    for r in range(n_repeats):
        if groups is None:
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
            it = cv.split(X, y)
        else:
            cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + r)
            it = cv.split(X, y, groups)
        jobs += [(r, tr, te) for tr, te in it]
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(name, task, X, y, tr, te, seed + r) for r, tr, te in jobs)
    k = len(np.unique(y))
    P = np.zeros((n_repeats, len(y), k))
    for (r, _, _), (te, pr) in zip(jobs, res):
        P[r, te, :] = pr
    return P


def top_features(texts: np.ndarray, y: np.ndarray, n: int = 20, C: float = 1.0) -> tuple[list[str], list[str]]:
    """Описова перевірка змісту: найсильніші слова за коефіцієнтами на всіх даних (не оцінка якості)."""
    vec = _word()
    X = vec.fit_transform(texts)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X, y)
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    return list(names[order[::-1][:n]]), list(names[order[:n]])
