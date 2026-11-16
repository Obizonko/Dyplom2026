"""S5: протоколи оцінювання P1-P3, фільтр витоку, канальний пріор лише з train.

Навчальний пул: повідомлення розміченої партії зі слабкими мітками, з яких прибрано
  1) самі золоті повідомлення,
  2) повідомлення з тим самим нормалізованим текстом, що й у якогось золотого,
  3) семантичні копії золотих (cos >= 0,985 за S4, список `exclude_labeled_keys_cos985.csv`).
Золоті повідомлення в навчання не потрапляють ніколи (мітки золотих лише для оцінки).

P1: пул, тест усі 412 золотих.
P2: пул з датою до CUTOFF, тест золоті з датою від CUTOFF.
P3: золоті канали ділимо на K груп (за каналами), для кожної групи тест її золоті, а з пулу прибираємо всі повідомлення її каналів.
    (План казав «канали, яких не було в навчанні», але всі 56 золотих каналів мають слабко розмічені повідомлення, див. reports/05_protocol.md.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY_MULT = 10_000_000  # ключ повідомлення = channel_id * KEY_MULT + message_id (як в eda.py і semantic.py)
CUTOFF = "2025-09-01"
SEM_THRESHOLD = 0.985
REASONS = ("is_gold", "same_text_as_gold", "semantic_copy_of_gold")


def make_key(channel_id, message_id) -> np.ndarray:
    return np.asarray(channel_id, dtype="int64") * KEY_MULT + np.asarray(message_id, dtype="int64")


def gold_frame(gold: pd.DataFrame, text_hash: pd.Series | None = None, binary: str = "Binary label",
               date: str = "date_utc") -> pd.DataFrame:
    """Золоті: key, channel_id, message_id, date_utc, y (+ text_hash, якщо передано відображення key -> хеш)."""
    out = pd.DataFrame({"channel_id": gold["channel_id"].astype("int64"), "message_id": gold["message_id"].astype("int64"),
                        "date_utc": pd.to_datetime(gold[date], utc=True), "y": gold[binary].astype(int)})
    out["key"] = make_key(out["channel_id"], out["message_id"])
    if text_hash is not None:
        out["text_hash"] = out["key"].map(text_hash)
    return out.sort_values("key").reset_index(drop=True)


def build_pool(index: pd.DataFrame, gold_keys: set, gold_hashes: set, exclude_keys: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """index: labeled_index (key, channel_id, message_id, date_utc, text_hash, in_gold, in_labeled, weak_narrative).
    Повертає (pool, excluded). excluded має булеві колонки за причинами, pool це розмічені без жодної причини."""
    lab = index[index["in_labeled"].astype(bool)].copy()
    lab["date_utc"] = pd.to_datetime(lab["date_utc"], utc=True)
    lab["is_gold"] = lab["key"].isin(gold_keys)
    lab["same_text_as_gold"] = lab["text_hash"].isin(gold_hashes)
    lab["semantic_copy_of_gold"] = lab["key"].isin(exclude_keys)
    bad = lab[list(REASONS)].any(axis=1)
    pool = lab.loc[~bad, ["key", "channel_id", "message_id", "date_utc", "text_hash", "weak_narrative"]].reset_index(drop=True)
    pool["weak_narrative"] = pool["weak_narrative"].astype(bool)
    excluded = lab.loc[bad, ["key", "channel_id", "message_id", *REASONS]].reset_index(drop=True)
    return pool, excluded


def exclusion_summary(n_labeled: int, excluded: pd.DataFrame) -> dict:
    """Скільки прибрано і чому (за кожною причиною окремо і за першою причиною в порядку REASONS)."""
    out = {"n_labeled": int(n_labeled), "n_excluded": int(len(excluded)), "n_pool": int(n_labeled - len(excluded))}
    for r in REASONS:
        out[f"reason_{r}"] = int(excluded[r].sum()) if len(excluded) else 0
    first = excluded[list(REASONS)].idxmax(axis=1) if len(excluded) else pd.Series([], dtype=object)
    for r in REASONS:
        out[f"first_reason_{r}"] = int((first == r).sum())
    return out


def split_p1(pool: pd.DataFrame, gold: pd.DataFrame) -> dict:
    return {"train": pool, "test": gold}


def split_p2(pool: pd.DataFrame, gold: pd.DataFrame, cutoff: str = CUTOFF) -> dict:
    t = pd.Timestamp(cutoff, tz="UTC")
    return {"train": pool[pool["date_utc"] < t].reset_index(drop=True), "test": gold[gold["date_utc"] >= t].reset_index(drop=True)}


def channel_folds(gold: pd.DataFrame, k: int = 5, seed: int = 42) -> pd.DataFrame:
    """Жадібний розподіл золотих каналів по k групах. Канали йдуть від тих, що мають найбільше російськомовних, потім наративних, потім усіх повідомлень; кожен потрапляє в групу, де після додавання
    найменша сума квадратів нормованих навантажень за трьома показниками: кількість повідомлень, кількість наративних, кількість російськомовних
    (останнє лише якщо в gold є колонка lang). Рівні за розміром канали розставляє seed. Повертає channel_id, fold (1..k), n, n_pos, n_ru."""
    g = gold.assign(is_ru=(gold["lang"] == "ru").astype(int) if "lang" in gold else 0)
    ch = g.groupby("channel_id").agg(n=("key", "size"), n_pos=("y", "sum"), n_ru=("is_ru", "sum")).reset_index()
    rng = np.random.default_rng(seed)
    ch["tie"] = rng.permutation(len(ch))
    ch = ch.sort_values(["n_ru", "n_pos", "n", "tie"], ascending=[False, False, False, True]).reset_index(drop=True)
    cols = [c for c in ("n", "n_pos", "n_ru") if ch[c].sum() > 0]
    target = {c: ch[c].sum() / k for c in cols}
    load = {c: np.zeros(k) for c in cols}
    fold = np.zeros(len(ch), dtype=int)
    for i in range(len(ch)):
        cost = sum(((load[c] + ch.loc[i, c]) / target[c]) ** 2 for c in cols)
        j = int(np.argmin(cost))
        fold[i] = j + 1
        for c in cols:
            load[c][j] += ch.loc[i, c]
    ch["fold"] = fold
    return ch.drop(columns="tie").sort_values(["fold", "channel_id"]).reset_index(drop=True)


def split_p3(pool: pd.DataFrame, gold: pd.DataFrame, folds: pd.DataFrame) -> dict:
    """Словник {fold: {train, test}}. Для групи f з пулу прибрано всі повідомлення каналів цієї групи."""
    out = {}
    for f in sorted(folds["fold"].unique()):
        chs = set(folds.loc[folds["fold"] == f, "channel_id"])
        out[int(f)] = {"train": pool[~pool["channel_id"].isin(chs)].reset_index(drop=True),
                       "test": gold[gold["channel_id"].isin(chs)].reset_index(drop=True)}
    return out


def group_valid_split(train: pd.DataFrame, frac: float = 0.1, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Валідація всередині train для Optuna і ранньої зупинки. Групи за text_hash: однакові тексти не розходяться між частинами.
    Повертає (train_mask, valid_mask) у порядку рядків train."""
    groups = train["text_hash"].to_numpy()
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    valid_groups = set(rng.choice(uniq, size=max(1, int(round(frac * len(uniq)))), replace=False).tolist())
    v = pd.Series([g in valid_groups for g in groups], index=train.index)
    return ~v, v


def channel_prior(train: pd.DataFrame, alpha: float = 1.0) -> pd.Series:
    """Згладжена частка наративних слабких міток каналу, рахується ЛИШЕ з переданого train. Для каналів, яких там немає,
    значення відсутнє (викликач підставляє глобальну частку train). Згладжування до глобальної частки: (n_pos + alpha * p) / (n + alpha)."""
    p = float(train["weak_narrative"].mean())
    g = train.groupby("channel_id")["weak_narrative"].agg(["sum", "size"])
    return ((g["sum"] + alpha * p) / (g["size"] + alpha)).rename("channel_prior")


def leakage_checks(train: pd.DataFrame, test: pd.DataFrame, gold_copy_pairs: pd.DataFrame | None = None,
                   threshold: float = SEM_THRESHOLD, by_channel: bool = False) -> list[dict]:
    """Перевірки витоку для однієї пари train/test. gold_copy_pairs: gold_key, neighbor_key, cos (S4, повний пошук копій золотих)."""
    checks = []
    inter = set(train["key"]) & set(test["key"])
    checks.append({"check": "Перетин ключів train і test порожній", "passed": not inter, "detail": f"{len(inter)} спільних ключів"})
    if "text_hash" in train and "text_hash" in test:
        ht = set(test["text_hash"].dropna())
        n_same = int(train["text_hash"].isin(ht).sum())
        checks.append({"check": "Немає train-повідомлень з тим самим нормалізованим текстом, що й у test", "passed": n_same == 0, "detail": f"{n_same} збігів"})
    if gold_copy_pairs is not None and len(gold_copy_pairs):
        gp = gold_copy_pairs[(gold_copy_pairs["cos"] >= threshold) & gold_copy_pairs["gold_key"].isin(set(test["key"]))]
        n_bad = int(gp["neighbor_key"].isin(set(train["key"])).sum())
        checks.append({"check": f"Немає train-повідомлень зі схожістю до test не нижче {threshold}".replace(".", ","), "passed": n_bad == 0,
                       "detail": f"{n_bad} пар (з {len(gp)} пар копій test)"})
    if by_channel:
        both = set(train["channel_id"]) & set(test["channel_id"])
        checks.append({"check": "Канали train і test не перетинаються", "passed": not both, "detail": f"{len(both)} спільних каналів"})
    return checks


def size_row(protocol: str, fold: int | None, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    return {"protocol": protocol, "fold": fold, "n_train": int(len(train)), "n_train_narrative": int(train["weak_narrative"].sum()),
            "n_train_channels": int(train["channel_id"].nunique()), "n_test": int(len(test)), "n_test_narrative": int(test["y"].sum()),
            "n_test_channels": int(test["channel_id"].nunique())}
