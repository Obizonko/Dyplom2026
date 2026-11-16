"""Метрики, бутстреп-довірчі інтервали, парні порівняння."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def confusion(y: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(np.asarray(y) * k + np.asarray(pred), minlength=k * k).reshape(k, k)


def mcc_from_cm(cm: np.ndarray) -> float:
    """Багатокласовий MCC (Городкін); для k=2 збігається з класичним."""
    t, p = cm.sum(1).astype(float), cm.sum(0).astype(float)
    c, s = float(np.trace(cm)), float(cm.sum())
    den = np.sqrt(s ** 2 - (p * p).sum()) * np.sqrt(s ** 2 - (t * t).sum())
    return float((c * s - (t * p).sum()) / den) if den > 0 else 0.0


def f1_per_class(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)


def binary_metrics(y: np.ndarray, p1: np.ndarray, thr: float = 0.5) -> dict:
    pred = (p1 >= thr).astype(int)
    cm = confusion(y, pred, 2)
    f1 = f1_per_class(cm)
    tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]
    return {
        "macro_f1": float(f1.mean()), "mcc": mcc_from_cm(cm),
        "pr_auc": float(average_precision_score(y, p1)) if y.sum() > 0 else float("nan"),
        "f1_pos": float(f1[1]), "f1_neg": float(f1[0]),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
    }


def multiclass_metrics(y: np.ndarray, pred: np.ndarray, k: int) -> dict:
    cm = confusion(y, pred, k)
    f1 = f1_per_class(cm)
    out = {"macro_f1": float(f1.mean()), "mcc": mcc_from_cm(cm), "accuracy": float(np.trace(cm) / cm.sum())}
    out.update({f"f1_class{i}": float(v) for i, v in enumerate(f1)})
    return out


def _mean_over_repeats(fn: Callable, y: np.ndarray, P: np.ndarray, idx: np.ndarray | None = None) -> dict:
    ms = [fn(y if idx is None else y[idx], p if idx is None else p[idx]) for p in P]
    return {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}


def bootstrap_ci(y: np.ndarray, P: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], dict],
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """P: (R, N) або (R, N, ...) прогнози для R повторів CV. Точкова оцінка: середнє метрики по повторах на всіх повідомленнях.
    CI: процентильний бутстреп по повідомленнях (метрика на кожній вибірці усереднюється по повторах).
    sd_repeats: розкид точкової метрики між повторами CV (окремо від CI)."""
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)
    per_rep = pd.DataFrame([fn(y, p) for p in P])
    point = per_rep.mean()
    boots = pd.DataFrame([_mean_over_repeats(fn, y, P, rng.integers(0, n, n)) for _ in range(n_boot)])
    return pd.DataFrame({
        "point": point, "ci_lo": boots.quantile(alpha / 2), "ci_hi": boots.quantile(1 - alpha / 2),
        "sd_repeats": per_rep.std(ddof=1) if len(per_rep) > 1 else np.nan,
    }).rename_axis("metric").reset_index()


def paired_bootstrap_diff(y: np.ndarray, PA: np.ndarray, PB: np.ndarray, fn: Callable,
                          n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> pd.DataFrame:
    """Різниця метрик A мінус B з парним бутстрепом (ті самі індекси для обох моделей)."""
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)
    pa, pb = _mean_over_repeats(fn, y, PA), _mean_over_repeats(fn, y, PB)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, b = _mean_over_repeats(fn, y, PA, idx), _mean_over_repeats(fn, y, PB, idx)
        diffs.append({k: a[k] - b[k] for k in a})
    d = pd.DataFrame(diffs)
    return pd.DataFrame({
        "diff": {k: pa[k] - pb[k] for k in pa}, "ci_lo": d.quantile(alpha / 2), "ci_hi": d.quantile(1 - alpha / 2),
    }).rename_axis("metric").reset_index()
