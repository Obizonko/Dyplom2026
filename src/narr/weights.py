"""Преєстрація v1.1 (розділ 11.1): ваги навчального пулу до розподілу корпусу і компоненти копій для валідації.

- Цільовий розподіл: очищений корпус `telegram_messages.parquet` за правилом TeleNarratives (текст не порожній, щонайменше 20 символів,
  містить літеру), бо золоті отримано стратифікованою за тижнями простою випадковою вибіркою саме з нього.
- Компоненти копій: зв'язні компоненти на повідомленнях train, ребро = однаковий `text_hash` або cos >= 0,985 (будь-які канали).
- Вага рядка: (1 / розмір компоненти) * w_c, w_c = q_c / p_c, обрізання [0,05; 20], нормування до середнього 1.
Мітки тут не використовуються, золоті теж.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

LETTER = r"[A-Za-zА-Яа-яІіЇїЄєҐґ]"
CLIP = (0.05, 20.0)


def corpus_channel_counts(raw_parquet: Path | str, batch_size: int = 100_000) -> pd.DataFrame:
    """Кількість повідомлень очищеного корпусу за каналами (правило cleaning_and_preprocessing TeleNarratives без clean_text:
    непорожній текст довжини >= 20, є літера, дата парситься; точні дублікати рядків (channel_id, message_id) прибрано)."""
    import pyarrow.parquet as pq

    parts = []
    pf = pq.ParquetFile(raw_parquet)
    for b in pf.iter_batches(batch_size=batch_size, columns=["channel_id", "message_id", "text", "date_utc"]):
        df = b.to_pandas()
        t = df["text"]
        ok = t.notna() & (t.astype(str).str.len() >= 20) & t.astype(str).str.contains(LETTER, regex=True)
        ok &= pd.to_datetime(df["date_utc"], errors="coerce", utc=True).notna()
        parts.append(df.loc[ok, ["channel_id", "message_id"]])
    d = pd.concat(parts, ignore_index=True).drop_duplicates()
    out = d.groupby("channel_id").size().rename("n_corpus").reset_index()
    return out.sort_values("n_corpus", ascending=False).reset_index(drop=True)


def components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Зв'язні компоненти неспрямованого графа на n вершинах з ребрами (i, j). Повертає мітку компоненти (найменший індекс)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if len(i) == 0:
        return np.arange(n)
    a = coo_matrix((np.ones(len(i), dtype=np.int8), (np.asarray(i), np.asarray(j))), shape=(n, n))
    _, lab = connected_components(a, directed=False)
    return lab


def copy_components(train: pd.DataFrame, pairs_keys: pd.DataFrame) -> np.ndarray:
    """Компоненти копій для рядків train (порядок рядків train).
    train: колонки key, text_hash. pairs_keys: колонки key_i, key_j (cos >= 0,985, будь-які канали; надмножина допустима)."""
    n = len(train)
    pos = pd.Series(np.arange(n), index=train["key"].to_numpy())
    ii, jj = [], []
    if pairs_keys is not None and len(pairs_keys):
        a = pos.reindex(pairs_keys["key_i"].to_numpy()).to_numpy()
        b = pos.reindex(pairs_keys["key_j"].to_numpy()).to_numpy()
        ok = ~(np.isnan(a) | np.isnan(b))
        ii.append(a[ok].astype(np.int64)), jj.append(b[ok].astype(np.int64))
    h = pd.factorize(train["text_hash"].fillna("").to_numpy())[0]  # однаковий текст: ланцюжок до першого рядка з тим самим хешем
    first = pd.Series(np.arange(n)).groupby(h).transform("first").to_numpy()
    m = first != np.arange(n)
    ii.append(np.arange(n)[m]), jj.append(first[m])
    return components(n, np.concatenate(ii), np.concatenate(jj))


def pool_weights(train: pd.DataFrame, comp: np.ndarray, corpus_counts: pd.DataFrame, clip=CLIP) -> tuple[np.ndarray, dict]:
    """Ваги рядків train за розділом 11.1 преєстрації. Повертає (ваги з середнім 1, зведення)."""
    comp = np.asarray(comp)
    size = pd.Series(comp).map(pd.Series(comp).value_counts()).to_numpy()
    w0 = 1.0 / size
    ch = train["channel_id"].to_numpy()
    p = pd.Series(w0).groupby(ch).sum()
    p = p / p.sum()
    q = corpus_counts.set_index("channel_id")["n_corpus"].reindex(p.index).fillna(0.0)
    if q.sum() == 0:
        raise ValueError("жоден канал train не знайдено в корпусі")
    q = q / q.sum()
    wc = (q / p).clip(*clip)
    w = w0 * pd.Series(ch).map(wc).to_numpy()
    w = w / w.mean()
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    info = {"n_rows": int(len(w)), "n_components": int(len(np.unique(comp))), "kish_ess": ess, "n_channels": int(len(wc)),
            "n_channels_clipped_low": int((q / p < clip[0]).sum()), "n_channels_clipped_high": int((q / p > clip[1]).sum()),
            "n_channels_missing_in_corpus": int(q.eq(0).sum()), "top_channel_share_before": float(p.max()),
            "top_channel_share_after": float(pd.Series(w).groupby(ch).sum().max() / w.sum())}
    return w, info


def group_split(groups: np.ndarray, frac: float = 0.1, seed: int = 42) -> np.ndarray:
    """Маска валідації: частка frac груп (компонент), seed. Той самий механізм, що `splits.group_valid_split`, але за довільними групами."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    vg = rng.choice(uniq, size=max(1, int(round(frac * len(uniq)))), replace=False)
    return np.isin(groups, vg)


def attach_weights(splits: list[dict], pairs_keys: pd.DataFrame, corpus_counts: pd.DataFrame, text_hash: pd.Series,
                   frac: float = 0.1, seed: int = 42) -> list[dict]:
    """Додає до train кожного спліту колонки text_hash, comp, is_valid і weight (ваги рахуються окремо для навчальної і валідаційної частин,
    кожна нормована до середнього 1). Повертає зведення по сплітах."""
    out = []
    for s in splits:
        tr = s["train"]
        tr["text_hash"] = tr["key"].map(text_hash)
        if tr["text_hash"].isna().any():
            raise KeyError(f"{s['protocol']} fold {s['fold']}: немає text_hash для {int(tr['text_hash'].isna().sum())} рядків")
        tr["comp"] = copy_components(tr, pairs_keys)
        tr["is_valid"] = group_split(tr["comp"].to_numpy(), frac, seed)
        w = np.zeros(len(tr))
        infos = {}
        for part, m in (("train", ~tr["is_valid"].to_numpy()), ("valid", tr["is_valid"].to_numpy())):
            wi, info = pool_weights(tr[m].reset_index(drop=True), tr.loc[m, "comp"].to_numpy(), corpus_counts)
            w[m] = wi
            infos[part] = info
        tr["weight"] = w
        out.append({"protocol": s["protocol"], "fold": s["fold"], **{f"{k}_{a}": v for k, d in infos.items() for a, v in d.items()}})
    return out


def load_v11_inputs(ROOT: Path, log=print):
    """Вхід версії 1.1 (кеш, рахується за відсутності; див. scripts/v11_prepare.py). Кількості корпусу, пари копій (ключі) і відображення key -> text_hash; рахує й кешує, якщо файлів немає."""
    cc_path = ROOT / "data_small" / "corpus_channel_counts.csv"
    pairs_path = ROOT / "data" / "interim" / "features" / "sem_pairs_any_0985.parquet"
    emb_dir = ROOT / "data" / "interim" / "embeddings"
    index = pd.read_parquet(emb_dir / "bge_m3_index.parquet")
    if cc_path.exists():
        cc = pd.read_csv(cc_path)
    else:
        t0 = time.time()
        cc = corpus_channel_counts(ROOT / "data" / "raw" / "telegram_messages.parquet")
        cc.to_csv(cc_path, index=False)
        log(f"Корпус: {int(cc['n_corpus'].sum())} повідомлень після очищення, {len(cc)} каналів ({time.time() - t0:.0f} с)")
    if pairs_path.exists():  # пари з усіх каналів; ft.semantic_pairs з cross_channel=False
        pk = pd.read_parquet(pairs_path)
    else:
        t0 = time.time()
        E = np.asarray(np.load(emb_dir / "bge_m3_fp16.npy", mmap_mode="r"), dtype=np.float32)
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        from narr import features as ft

        p = ft.semantic_pairs(E, index["channel_id"].to_numpy(), threshold=0.985, block=1024, cross_channel=False)
        keys = index["key"].to_numpy()
        pk = pd.DataFrame({"key_i": keys[p["i"].to_numpy().astype(np.int64)], "key_j": keys[p["j"].to_numpy().astype(np.int64)], "cos": p["cos"].to_numpy()})
        pairs_path.parent.mkdir(parents=True, exist_ok=True)
        pk.to_parquet(pairs_path, index=False)
        log(f"Пари cos >= 0,985 (будь-які канали): {len(pk)} ({time.time() - t0:.0f} с)")
    th = index.drop_duplicates("key").set_index("key")["text_hash"]
    return cc, pk, th
