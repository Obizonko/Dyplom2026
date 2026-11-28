"""S6: тренування текстових моделей на пулі зі слабкими мітками і прогнози для золотих повідомлень.

Моделі: `tfidf` (слова 1-2 + символьні 2-5 n-грами, LogReg), `bge_lr` (ембединги bge-m3 з S4, LogReg), `xlmr` (дотренування XLM-R base).
Контролі з перемішаними мітками: `tfidf_shuf`, `bge_lr_shuf`, `xlmr_shuf` (лише P1).

Правила, що не залежать від золотих:
- валідація всередині train: 10% за групами text_hash, seed 42 (`splits.group_valid_split`);
- C підбирається за macro-F1 на цій валідації (поріг 0,5, слабкі мітки), навчання лише на 90% train, без повторного навчання на всьому train;
- C підбирається на P1; для P2 і P3 береться те саме C (економія часу, значення пишеться в unit-файли);
- XLM-R: епоха обирається за валідаційним macro-F1; золоті мітки в цьому модулі не використовуються.
Результат одиниці (модель, протокол, seed): `predictions/<unit>.parquet` (ймовірності для золотих тесту) і `units/<unit>.json` (валідація, C, тривалість).
Якщо обидва файли вже є, одиницю пропускаємо (можна відновлювати перервані запуски).
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from narr.baselines import _char, _word
from narr.evaluate import binary_metrics, weighted_macro_f1
from narr.splits import group_valid_split, make_key

SEED = 42
VALID_FRAC = 0.1
CS_TFIDF = (2.0, 8.0, 32.0)
CS_DENSE = (3.0, 30.0, 300.0)
TEXT_MODELS = ("tfidf", "bge_lr", "xlmr")
SHUF_SUFFIX = "_shuf"


def hash_texts(texts: pd.Series) -> pd.Series:
    """Той самий хеш, що в `semantic.build_embedding_set` (за ним S4 зберіг `text_hash`)."""
    return pd.util.hash_pandas_object(texts.astype(str).reset_index(drop=True), index=False).astype("uint64").map(lambda v: f"{int(v):016x}").set_axis(texts.index)


# ---------------------------------------------------------------- спліти і тексти

def load_splits(split_dir: Path | str, protocols=("P1", "P2", "P3")) -> list[dict]:
    """Список сплітів: {"protocol", "fold", "train" (channel_id, message_id, weak_narrative, key), "test" (channel_id, message_id, y, date_utc, key)}.
    P1 і P2 мають один сплiт (fold 0), P3 один на кожну групу каналів (fold 1..K)."""
    d = Path(split_dir)
    out = []
    for pr in protocols:
        if pr in ("P1", "P2"):
            pairs = [(0, d / pr / "train.csv.gz", d / pr / "test.csv")]
        elif pr == "P3":
            ks = sorted(int(p.name.split("_")[0][4:]) for p in (d / "P3").glob("fold*_train.csv.gz"))
            pairs = [(k, d / "P3" / f"fold{k}_train.csv.gz", d / "P3" / f"fold{k}_test.csv") for k in ks]
        else:
            raise ValueError(pr)
        for k, ptr, pte in pairs:
            tr, te = pd.read_csv(ptr), pd.read_csv(pte)
            tr["weak_narrative"] = tr["weak_narrative"].astype(bool)
            tr["key"] = make_key(tr["channel_id"], tr["message_id"])
            te["key"] = make_key(te["channel_id"], te["message_id"])
            out.append({"protocol": pr, "fold": k, "train": tr, "test": te})
    return out


def needed_keys(splits: list[dict]) -> np.ndarray:
    return np.unique(np.concatenate([np.concatenate([s["train"]["key"].to_numpy(), s["test"]["key"].to_numpy()]) for s in splits]))


def load_texts(raw_dir: Path | str, keys: np.ndarray) -> pd.Series:
    """Нормалізовані тексти з telegram_messages.parquet для потрібних ключів (як у S4). Повертає Series key -> text."""
    import pyarrow.parquet as pq

    # Читаємо порціями і лишаємо лише потрібні ключі: повний файл (1 млн текстів) у пам'ять не вантажимо
    # (на ноутбуці з ~14 ГБ RAM це ще й заважало ініціалізації CUDA)
    want = set(keys.tolist())
    parts = []
    pf = pq.ParquetFile(Path(raw_dir) / "telegram_messages.parquet")
    for batch in pf.iter_batches(batch_size=50_000, columns=["channel_id", "message_id", "text"]):
        df = batch.to_pandas()
        df["key"] = make_key(df["channel_id"], df["message_id"])
        df = df[df["key"].isin(want)]
        if len(df):
            parts.append(df[["key", "text"]])
    tg = pd.concat(parts).drop_duplicates("key")
    s = pd.Series(tg["text"].to_numpy(), index=tg["key"].to_numpy())
    missing = np.setdiff1d(keys, s.index.to_numpy())
    if len(missing):
        raise KeyError(f"для {len(missing)} ключів немає тексту в telegram_messages.parquet, наприклад {missing[:3].tolist()}")
    return s.loc[keys]


def attach_text(splits: list[dict], texts: pd.Series) -> None:
    """Додає колонки text і text_hash до train і test кожного спліту (на місці)."""
    th = hash_texts(texts)
    for s in splits:
        for part in ("train", "test"):
            df = s[part]
            df["text"] = df["key"].map(texts)
            df["text_hash"] = df["key"].map(th)
            if df["text"].isna().any():
                raise KeyError(f"{s['protocol']} fold {s['fold']} {part}: є повідомлення без тексту")


def check_hash_against_index(texts: pd.Series, index: pd.DataFrame) -> dict:
    """Звіряє хеш завантаженого тексту з text_hash із індексу ембедингів S4: так перевіряємо, що ембединг і текст належать одному повідомленню."""
    th = hash_texts(texts)
    ref = index.drop_duplicates("key").set_index("key")["text_hash"]
    common = th.index.intersection(ref.index)
    mism = (th.loc[common] != ref.loc[common]).sum()
    return {"n_keys": int(len(th)), "n_in_index": int(len(common)), "n_mismatch": int(mism), "ok": bool(len(common) == len(th) and mism == 0)}


# ---------------------------------------------------------------- лінійні моделі

def fit_logreg(X, y: np.ndarray, C: float, seed: int, max_iter: int = 500, class_weight="balanced", sample_weight=None) -> LogisticRegression:
    """Версія 1: class_weight="balanced", без ваг рядків. Версія 1.1 (преєстрація 11.1): class_weight=None, ваги пулу."""
    m = LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter, solver="lbfgs", random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y, sample_weight=sample_weight)
    return m


def select_C(Xtr, ytr: np.ndarray, Xva, yva: np.ndarray, Cs, seed: int, max_iter: int = 500, log=print, w_tr=None, w_va=None,
             class_weight="balanced") -> dict:
    """Перебір C за macro-F1 на валідації (слабкі мітки; у версії 1.1 зважений вагами пулу). Нічия: менше C.
    Повертає {"model", "C", "valid_macro_f1", "grid"}."""
    best, grid = None, []
    for C in Cs:
        t0 = time.time()
        m = fit_logreg(Xtr, ytr, C, seed, max_iter, class_weight=class_weight, sample_weight=w_tr)
        f1 = weighted_macro_f1(yva, m.predict_proba(Xva)[:, 1], w_va)
        grid.append({"C": float(C), "valid_macro_f1": float(f1), "n_iter": int(np.max(m.n_iter_)), "seconds": round(time.time() - t0, 1)})
        log(f"    C={C:g}: valid macro-F1 {f1:.4f} (ітерацій {grid[-1]['n_iter']}, {grid[-1]['seconds']} с)")
        if best is None or f1 > best[0] + 1e-12:
            best = (f1, C, m)
    return {"model": best[2], "C": float(best[1]), "valid_macro_f1": float(best[0]), "grid": grid}


def _tfidf_features(tr_texts, other: dict[str, list[str]]):
    w, c = _word(), _char()
    Xtr = sparse.hstack([w.fit_transform(tr_texts), c.fit_transform(tr_texts)]).tocsr()
    return Xtr, {k: sparse.hstack([w.transform(v), c.transform(v)]).tocsr() for k, v in other.items()}


def _pred_frame(split: dict, model: str, seed: int, p: np.ndarray) -> pd.DataFrame:
    te = split["test"]
    return pd.DataFrame({"protocol": split["protocol"], "fold": split["fold"], "model": model, "seed": seed,
                         "channel_id": te["channel_id"].to_numpy(), "message_id": te["message_id"].to_numpy(), "p": np.asarray(p, dtype=float)})


def _train_view(split: dict, shuffle_seed: int | None):
    """Навчальна і валідаційна частини спліту. Для контролю мітки перемішуються в усьому train ще до поділу (валідація теж перемішана)."""
    tr = split["train"]
    y = tr["weak_narrative"].astype(int).to_numpy()
    if shuffle_seed is not None:
        y = np.random.default_rng(shuffle_seed).permutation(y)
    if "is_valid" in tr.columns:  # версія 1.1: валідація за компонентами копій (weights.attach_weights)
        va = tr["is_valid"].to_numpy()
        m_tr, m_va = pd.Series(~va, index=tr.index), pd.Series(va, index=tr.index)
    else:
        m_tr, m_va = group_valid_split(tr, VALID_FRAC, SEED)
    return tr[m_tr.to_numpy()], tr[m_va.to_numpy()], y[m_tr.to_numpy()], y[m_va.to_numpy()]


def is_v11(split: dict) -> bool:
    """Спліт підготовлено за преєстрацією 1.1 (є ваги пулу)."""
    return "weight" in split["train"].columns


def run_lr(kind: str, splits: list[dict], emb_lookup, fixed_C: float | None, shuffle_seed: int | None, log=print, unit_seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Одна одиниця для `tfidf` або `bge_lr` на всіх сплітах одного протоколу."""
    name = kind + (SHUF_SUFFIX if shuffle_seed is not None else "")
    preds, per_split = [], []
    C_used = fixed_C
    for s in splits:
        t0 = time.time()
        a, v, ya, yv = _train_view(s, shuffle_seed)
        log(f"  {name} {s['protocol']} fold {s['fold']}: train {len(a)}, valid {len(v)}, test {len(s['test'])}")
        if kind == "tfidf":
            Xa, other = _tfidf_features(a["text"].tolist(), {"v": v["text"].tolist(), "t": s["test"]["text"].tolist()})
            Xv, Xt = other["v"], other["t"]
            grid = CS_TFIDF
        else:
            Xa, Xv, Xt = (emb_lookup(d["key"].to_numpy()) for d in (a, v, s["test"]))
            grid = CS_DENSE
        v11 = is_v11(s)
        res = select_C(Xa, ya, Xv, yv, [C_used] if C_used is not None else grid, SEED, log=log,
                       w_tr=a["weight"].to_numpy() if v11 else None, w_va=v["weight"].to_numpy() if v11 else None,
                       class_weight=None if v11 else "balanced")
        C_used = res["C"]  # якщо C не було задано, його підбирає перший спліт, решта (фолди P3) беруть те саме
        p = res["model"].predict_proba(Xt)[:, 1]
        preds.append(_pred_frame(s, name, unit_seed, p))
        per_split.append({"fold": s["fold"], "n_train": int(len(a)), "n_valid": int(len(v)), "C": res["C"], "valid_macro_f1": res["valid_macro_f1"],
                          "grid": res["grid"], "seconds": round(time.time() - t0, 1)})
    info = {"model": name, "protocol": splits[0]["protocol"], "seed": unit_seed, "shuffle_seed": shuffle_seed, "C": per_split[0]["C"],
            "valid_macro_f1": float(np.mean([r["valid_macro_f1"] for r in per_split])), "splits": per_split}
    return pd.concat(preds, ignore_index=True), info


def run_xlmr(splits: list[dict], seed: int, shuffle_seed: int | None, kwargs: dict, log=print) -> tuple[pd.DataFrame, dict]:
    from narr.xlmr import fine_tune

    name = "xlmr" + (SHUF_SUFFIX if shuffle_seed is not None else "")
    preds, per_split = [], []
    for s in splits:
        t0 = time.time()
        a, v, ya, yv = _train_view(s, shuffle_seed)
        log(f"  {name} {s['protocol']} fold {s['fold']} seed {seed}: train {len(a)}, valid {len(v)}, test {len(s['test'])}")
        wkw = {"w_train": a["weight"].to_numpy(), "w_valid": v["weight"].to_numpy()} if is_v11(s) else {}
        r = fine_tune(a["text"].tolist(), ya, v["text"].tolist(), yv, {"test": s["test"]["text"].tolist()}, seed=seed, log=log, **kwargs, **wkw)
        preds.append(_pred_frame(s, name, seed, r["probs"]["test"]))
        per_split.append({"fold": s["fold"], "n_train": r["n_train"], "n_valid": r["n_valid"], "best_epoch": r["best_epoch"],
                          "valid_macro_f1": r["valid_macro_f1"], "history": r["history"], "tokens_median": r["tokens_median"],
                          "share_truncated": r["share_truncated"], "seconds": round(time.time() - t0, 1)})
    info = {"model": name, "protocol": splits[0]["protocol"], "seed": seed, "valid_macro_f1": float(np.mean([r["valid_macro_f1"] for r in per_split])),
            "splits": per_split}
    return pd.concat(preds, ignore_index=True), info


# ---------------------------------------------------------------- оркестрація з відновленням

def unit_name(model: str, protocol: str, seed: int) -> str:
    return f"{model}__{protocol}__s{seed}"


def run_all(splits: list[dict], out: Path | str, *, models=TEXT_MODELS, protocols=("P1", "P2", "P3"), xlmr_protocols=("P1", "P2"),
            seeds=(0, 1, 2), controls=True, emb_lookup=None, xlmr_kwargs: dict | None = None, log=print, n_perm: int = 1, xlmr_perm: int = 20) -> list[dict]:
    """Проходить усі одиниці у порядку: P1 перед P2 і P3 (щоб C для P2, P3 взяти з P1). Повертає список info усіх одиниць (нових і з диска)."""
    out = Path(out)
    (out / "units").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    by_pr = {pr: [s for s in splits if s["protocol"] == pr] for pr in protocols}
    infos, chosen_C = [], {}

    def unit(model: str, protocol: str, seed: int, fn):
        un = unit_name(model, protocol, seed)
        pj, pp = out / "units" / f"{un}.json", out / "predictions" / f"{un}.parquet"
        if pj.exists() and pp.exists():
            info = json.loads(pj.read_text(encoding="utf-8"))
            log(f"[пропуск] {un}: уже є")
        else:
            log(f"[{time.strftime('%H:%M:%S')}] {un}")
            t0 = time.time()
            pred, info = fn()
            info["seconds"] = round(time.time() - t0, 1)
            pred.to_parquet(pp, index=False)
            pj.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
            log(f"  готово за {info['seconds']} с, valid macro-F1 {info['valid_macro_f1']:.4f}")
        infos.append(info)
        return info

    for kind in [m for m in models if m in ("tfidf", "bge_lr")]:
        for pr in protocols:  # порядок P1, P2, P3 береться з аргументу, P1 має йти першим
            info = unit(kind, pr, 0, lambda: run_lr(kind, by_pr[pr], emb_lookup, chosen_C.get(kind), None, log))
            chosen_C.setdefault(kind, info["C"])
    if "xlmr" in models:
        for pr in [p for p in protocols if p in xlmr_protocols]:
            for sd in seeds:
                unit("xlmr", pr, sd, lambda: run_xlmr(by_pr[pr], sd, None, xlmr_kwargs or {}, log))
    if controls and "P1" in by_pr:
        for kind in [m for m in models if m in ("tfidf", "bge_lr")]:
            if n_perm <= 1:  # версія 1: один контроль, seed перемішування 1234
                unit(kind + SHUF_SUFFIX, "P1", 0, lambda: run_lr(kind, by_pr["P1"], emb_lookup, chosen_C.get(kind), 1234, log))
            else:  # версія 1.1 (преєстрація 11.1, п. 6): n_perm перестановок, seed перемішування 1000 + k, C основної моделі
                for k in range(n_perm):
                    unit(kind + SHUF_SUFFIX, "P1", k, lambda: run_lr(kind, by_pr["P1"], emb_lookup, chosen_C.get(kind), 1000 + k, log, unit_seed=k))
        if "xlmr" in models and "P1" in xlmr_protocols:
            if n_perm <= 1:
                unit("xlmr" + SHUF_SUFFIX, "P1", 0, lambda: run_xlmr(by_pr["P1"], 0, 1234, {**(xlmr_kwargs or {}), "epochs": 1}, log))
            else:  # версія 1.1: xlmr_perm перестановок по 1 епосі (дорого; кількість задається окремо)
                for k in range(xlmr_perm):
                    unit("xlmr" + SHUF_SUFFIX, "P1", k, lambda: run_xlmr(by_pr["P1"], k, 1000 + k, {**(xlmr_kwargs or {}), "epochs": 1}, log))
    return infos
