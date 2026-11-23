"""S7: LightGBM на графових ознаках, на ембедингах і на обох. Підбір гіперпараметрів Optuna за валідацією слабких міток.

Правила: поріг 0,5 заздалегідь; вибір гіперпараметрів лише за macro-F1 на валідації всередині train (група за text_hash);
золоті повідомлення тут не використовуються. Ранню зупинку за logloss валідації виконує сам LightGBM.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from narr.evaluate import binary_metrics

BASE_PARAMS = dict(objective="binary", is_unbalance=True, verbosity=-1, force_col_wise=True, max_bin=63)


def _fit(params: dict, Xtr, ytr, Xva, yva, num_threads: int, seed: int, max_rounds: int = 400, categorical=None):
    import lightgbm as lgb

    p = {**BASE_PARAMS, **params, "num_threads": num_threads, "seed": seed}
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=categorical or "auto", free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, categorical_feature=categorical or "auto", free_raw_data=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lgb.train(p, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(30, verbose=False)])


def _suggest(trial, wide: bool) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 8, 96, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.05 if wide else 0.5, 0.5 if wide else 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
    }


def tune_and_fit(Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray, Xte: pd.DataFrame, *, n_trials: int, seed: int,
                 num_threads: int = 6, search_rows: int | None = None, categorical=None, log=print) -> dict:
    """Optuna (TPE, seed) на валідації слабких міток; потім найкращі параметри на всьому train. Повертає {"p", "valid_macro_f1", "params", "booster", ...}.
    `search_rows`: у швидкому режимі пошук параметрів іде на підвибірці train (seed), фінальне навчання на всьому train."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    wide = Xtr.shape[1] > 100
    if search_rows and len(Xtr) > search_rows:
        sub = np.random.default_rng(seed).choice(len(Xtr), size=search_rows, replace=False)
        Xs, ys = Xtr.iloc[sub], ytr[sub]
    else:
        Xs, ys = Xtr, ytr
    t0 = time.time()

    def objective(trial):
        params = _suggest(trial, wide)
        b = _fit(params, Xs, ys, Xva, yva, num_threads, seed, categorical=categorical)
        return binary_metrics(yva, b.predict(Xva, num_iteration=b.best_iteration))["macro_f1"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = _fit({**study.best_params, "bagging_freq": 1}, Xtr, ytr, Xva, yva, num_threads, seed, categorical=categorical)
    pva = best.predict(Xva, num_iteration=best.best_iteration)
    f1 = binary_metrics(yva, pva)["macro_f1"]
    log(f"    Optuna {n_trials} спроб за {time.time() - t0:.0f} с, найкращий valid macro-F1 пошуку {study.best_value:.4f}, "
        f"на повному train {f1:.4f}, ітерацій {best.best_iteration}")
    return {"p": best.predict(Xte, num_iteration=best.best_iteration), "valid_macro_f1": float(f1), "params": study.best_params,
            "n_trials": int(n_trials), "best_iteration": int(best.best_iteration), "booster": best, "search_rows": int(len(Xs))}
