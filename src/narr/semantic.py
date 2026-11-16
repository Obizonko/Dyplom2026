"""S4: семантичний шар. Ембединги bge-m3, семантичні ребра, порівняння з ребрами набору, копії золотих, досяжність.

Модуль не читає диск (крім `load_inputs`) і не імпортує torch: кодувальник передається як функція `encode(list[str]) -> ndarray`,
тому всі обчислення тестуються на малих синтетичних даних. Ключ повідомлення = channel_id * 10^7 + message_id (як в eda.py).
Усі косинуси рахуються на float32-векторах одиничної довжини (скалярний добуток).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from narr.eda import KEY_MULT, keys, normalize_text

MODEL_NAME = "BAAI/bge-m3"
THRESHOLDS = (0.95, 0.975, 0.985, 0.99)


# ---------------------------------------------------------------- набір для ембедингів

def load_inputs(raw_dir: Path | str, golden_path: Path | str) -> dict:
    """Читає лише те, що потрібно S4. Тексти беремо з telegram_messages.parquet (звідти золоті, S2)."""
    import pyarrow.parquet as pq

    raw_dir = Path(raw_dir)
    mpath = next(iter(sorted(raw_dir.rglob("messages.parquet"))), None)
    if mpath is None:
        raise FileNotFoundError(f"не знайдено messages.parquet у {raw_dir}")
    gd = mpath.parent
    msgs = pq.read_table(mpath, columns=["channel_id", "message_id", "text", "is_labeled_batch", "narrative_id", "date_utc"]).to_pandas()
    tg = pq.read_table(raw_dir / "telegram_messages.parquet", columns=["channel_id", "message_id", "text"]).to_pandas()
    fwd = pq.read_table(gd / "forward_edges.parquet", columns=["logical_src_channel_id", "logical_src_message_id",
                                                              "logical_dst_channel_id", "logical_dst_message_id"]).to_pandas()
    sem = pq.read_table(gd / "semantic_edges.parquet", columns=["src_channel_id", "src_message_id", "dst_channel_id",
                                                               "dst_message_id", "similarity_score"]).to_pandas()
    return {"msgs": msgs, "tg": tg, "gold": pd.read_csv(golden_path), "fwd": fwd, "sem": sem,
            "coding": pd.read_csv(gd / "narrative_subnarrative_coding_en.csv")}


def build_embedding_set(msgs: pd.DataFrame, tg: pd.DataFrame, gold: pd.DataFrame, sem: pd.DataFrame,
                        fwd: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Що ембедимо: усі золоті, уся розмічена партія, усі кінці семантичних ребер набору.

    Текст: нормалізований із telegram_messages (як у золотих); якщо ключа там немає, нормалізуємо сирий текст із messages.
    Порожні після нормалізації тексти відкидаємо (їхні ембединги однакові і дали б фальшиві ребра), кількість повертаємо.
    Колонки: key, channel_id, message_id, text, in_gold, in_labeled, in_sem, in_fwd, narrative_id, date_utc, text_hash."""
    m = msgs[[c for c in ("channel_id", "message_id", "is_labeled_batch", "narrative_id", "date_utc") if c in msgs.columns]].assign(key=keys(msgs))
    t = pd.Series(tg["text"].to_numpy(), index=keys(tg).to_numpy()).loc[lambda x: ~x.index.duplicated()]
    g = gold.assign(key=keys(gold))
    sk = pd.concat([keys(sem, "src_channel_id", "src_message_id"), keys(sem, "dst_channel_id", "dst_message_id")])
    fk = (pd.concat([keys(fwd, "logical_src_channel_id", "logical_src_message_id"),
                     keys(fwd, "logical_dst_channel_id", "logical_dst_message_id")]) if fwd is not None else pd.Series([], dtype="int64"))
    lab = m.loc[m["is_labeled_batch"].eq(True), "key"]
    allkeys = pd.Index(pd.concat([g["key"], lab, sk]).unique())
    base = pd.DataFrame({"key": allkeys.to_numpy()})
    base["channel_id"] = base["key"] // KEY_MULT
    base["message_id"] = base["key"] % KEY_MULT
    mi = m.drop_duplicates("key").set_index("key")
    text = base["key"].map(t)
    missing_tg = text.isna()
    if missing_tg.any():
        raw_all = pd.Series(msgs["text"].to_numpy(), index=keys(msgs).to_numpy())
        raw = base.loc[missing_tg, "key"].map(raw_all.loc[lambda x: ~x.index.duplicated()])
        text.loc[missing_tg] = normalize_text(raw.fillna("")).to_numpy()
    gt = g.drop_duplicates("key").set_index("key")["text"]
    still = text.isna() | (text.astype(str).str.strip() == "")
    from_gold = base["key"].isin(gt.index) & still
    text.loc[from_gold] = base.loc[from_gold, "key"].map(gt)
    base["text"] = text.fillna("").astype(str)
    base["in_gold"] = base["key"].isin(set(g["key"]))
    base["in_labeled"] = base["key"].isin(set(lab))
    base["in_sem"] = base["key"].isin(set(sk))
    base["in_fwd"] = base["key"].isin(set(fk))
    nar = base["key"].map(mi["narrative_id"]) if "narrative_id" in mi else pd.Series(np.nan, index=base.index)
    base["narrative_id"] = nar
    dt = pd.to_datetime(base["key"].map(mi["date_utc"]), utc=True, errors="coerce") if "date_utc" in mi else pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns, UTC]")
    gd = pd.to_datetime(g.drop_duplicates("key").set_index("key")["date_utc"], utc=True, errors="coerce") if "date_utc" in g else None
    if gd is not None:  # золоті мають власну дату (збігається з корпусом, S2); беремо її, якщо в корпусі ключа немає
        dt = dt.fillna(base["key"].map(gd))
    base["date_utc"] = dt
    empty = base["text"].str.strip() == ""
    info = {"n_candidates": int(len(base)), "n_empty_dropped": int(empty.sum()), "n_text_from_raw_fallback": int(missing_tg.sum()),
            "n_gold": int(base["in_gold"].sum()), "n_labeled": int(base["in_labeled"].sum()), "n_sem_endpoints": int(base["in_sem"].sum()),
            "n_gold_empty": int((empty & base["in_gold"]).sum())}
    base = base[~empty].reset_index(drop=True)
    base["text_hash"] = pd.util.hash_pandas_object(base["text"], index=False).astype("uint64").map(lambda v: f"{int(v):016x}")
    info["n_embedded"] = int(len(base))
    return base, info


def embed_sorted(texts: list[str], encode, batch_size: int = 128, progress=None) -> np.ndarray:
    """Кодує в порядку зростання довжини (менше доповнень), повертає в початковому порядку. Рядки нормуються до одиничної довжини.
    progress(done, total) викликається після кожної 50-ї партії."""
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = None
    for bi, s in enumerate(range(0, len(texts), batch_size)):
        if progress is not None and bi % 50 == 0:
            progress(s, len(texts))
        idx = order[s:s + batch_size]
        v = np.asarray(encode([texts[i] for i in idx]), dtype=np.float32)
        if out is None:
            out = np.zeros((len(texts), v.shape[1]), dtype=np.float32)
        out[idx] = v
    if out is None:
        return np.zeros((0, 0), dtype=np.float32)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-12)


def hash_encoder(dim: int = 256):
    """Детермінований кодувальник на хешованих символьних тригрaмах: лише для тестів і сухого запуску без GPU."""
    import zlib

    def enc(batch: list[str]) -> np.ndarray:
        out = np.zeros((len(batch), dim), dtype=np.float32)
        for r, t in enumerate(batch):
            t = f"  {t}  "
            for i in range(len(t) - 2):
                out[r, zlib.crc32(t[i:i + 3].encode("utf-8")) % dim] += 1.0
        return out

    return enc


def quantize_fp16(emb: np.ndarray) -> np.ndarray:
    """Те, що буде збережено (float16), повертаємо назад у float32 і знову нормуємо: аналіз завжди йде на тих самих числах,
    що лежать на диску, тож повторний запуск з кешу дає той самий результат."""
    q = emb.astype(np.float16).astype(np.float32)
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------- сусіди

def knn_pairs(emb: np.ndarray, k: int = 80, threshold: float = 0.975, ef: int = 300, M: int = 32, seed: int = 42,
              report_thresholds: tuple = ()) -> tuple[pd.DataFrame, dict]:
    """Наближені k найближчих сусідів (hnswlib, внутрішній добуток = косинус для одиничних векторів).
    Повертає унікальні неспрямовані пари i < j з cos >= threshold і лічильники насичення: вузол, у якого всі k сусідів вище порога,
    міг мати більше сусідів, ніж ми знайшли. Лічильники дано для threshold і для кожного з report_thresholds."""
    import hnswlib

    n, d = emb.shape
    idx = hnswlib.Index(space="ip", dim=d)
    idx.init_index(max_elements=n, ef_construction=200, M=M, random_seed=seed)
    idx.add_items(emb, np.arange(n), num_threads=-1)
    idx.set_ef(max(ef, k + 1))
    kk = min(k + 1, n)
    lab, dist = idx.knn_query(emb, k=kk, num_threads=-1)
    cos = 1.0 - dist
    src = np.repeat(np.arange(n), kk)
    dst, c = lab.ravel(), cos.ravel()
    keep = (src != dst) & (c >= threshold)
    a, b = np.minimum(src[keep], dst[keep]), np.maximum(src[keep], dst[keep])
    df = pd.DataFrame({"i": a, "j": b, "cos": c[keep]}).groupby(["i", "j"], as_index=False)["cos"].max()
    df = df.astype({"i": "int64", "j": "int64", "cos": "float32"})
    by = {}
    for t in sorted({threshold, *report_thresholds}):
        above = (cos[:, 1:] >= t).sum(axis=1) if kk > 1 else np.zeros(n, dtype=int)
        by[float(t)] = {"n_pairs": int((df["cos"] >= t).sum()), "n_nodes_saturated": int((above >= kk - 1).sum()) if kk > 1 else 0,
                        "n_nodes_with_neighbor": int((above > 0).sum())}
    info = {"k": k, "threshold": threshold, "n_pairs": int(len(df)), "n_nodes_saturated": by[float(threshold)]["n_nodes_saturated"],
            "n_nodes_with_neighbor": by[float(threshold)]["n_nodes_with_neighbor"], "by_threshold": by}
    return df, info


def exact_neighbors(emb: np.ndarray, query_idx: np.ndarray, threshold: float, exclude_self: bool = True, block: int = 256) -> pd.DataFrame:
    """Точний пошук: усі вектори з cos >= threshold до кожного запиту (без обмеження на k). Повертає колонки q, i, cos."""
    rows = []
    for s in range(0, len(query_idx), block):
        q = query_idx[s:s + block]
        sims = emb[q] @ emb.T
        if exclude_self:
            sims[np.arange(len(q)), q] = -2.0
        qi, ii = np.nonzero(sims >= threshold)
        rows.append(pd.DataFrame({"q": q[qi], "i": ii, "cos": sims[qi, ii]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame({"q": [], "i": [], "cos": []})


def hnsw_recall(emb: np.ndarray, found: pd.DataFrame, threshold: float, n_check: int = 500, seed: int = 42, k: int = 80) -> dict:
    """Повнота hnsw за порогом: для випадкових вузлів порівнюємо число знайдених сусідів з cos >= threshold з точним (обмеженим k)."""
    rng = np.random.default_rng(seed)
    q = rng.choice(len(emb), size=min(n_check, len(emb)), replace=False)
    ex = exact_neighbors(emb, q, threshold)
    exact_n = ex.groupby("q").size().reindex(q).fillna(0).clip(upper=k).to_numpy()
    deg = pd.concat([found.groupby("i").size(), found.groupby("j").size()]).groupby(level=0).sum()
    found_n = deg.reindex(q).fillna(0).clip(upper=k).to_numpy()
    tot = exact_n.sum()
    return {"n_queries": int(len(q)), "n_with_neighbors": int((exact_n > 0).sum()), "exact_neighbors_capped": int(tot),
            "recall": float(np.minimum(found_n, exact_n).sum() / tot) if tot > 0 else float("nan")}


# ---------------------------------------------------------------- порівняння з ребрами набору

def pair_cosines(emb: np.ndarray, key2idx: dict, a_keys: np.ndarray, b_keys: np.ndarray) -> np.ndarray:
    ia = np.array([key2idx.get(int(x), -1) for x in a_keys])
    ib = np.array([key2idx.get(int(x), -1) for x in b_keys])
    ok = (ia >= 0) & (ib >= 0)
    out = np.full(len(ia), np.nan, dtype=np.float32)
    out[ok] = np.einsum("ij,ij->i", emb[ia[ok]], emb[ib[ok]])
    return out


def compare_scores(score: np.ndarray, cos: np.ndarray) -> dict:
    """Наскільки збережена оцінка схожості набору збігається з нашим косинусом. Моделі могли відрізнятись, тому крім різниці
    даємо кореляції (інваріантні до монотонного перетворення, зокрема (cos + 1) / 2) і лінійну підгонку score ~ a * cos + b."""
    m = np.isfinite(score) & np.isfinite(cos)
    s, c = score[m].astype(float), cos[m].astype(float)
    a, b = np.polyfit(c, s, 1) if len(s) > 1 else (np.nan, np.nan)
    d = s - c
    dm = s - (c + 1) / 2
    nt = s < 0.995  # майже однакові тексти дають оцінки ~1 з шумом, ранги на них не інформативні: окремі показники для решти пар
    extra = {"share_score_ge_0995": float((~nt).mean()), "n_nontrivial": int(nt.sum())}
    if nt.sum() > 2 and np.ptp(s[nt]) > 0 and np.ptp(c[nt]) > 0:
        extra.update({"pearson_nontrivial": float(pearsonr(s[nt], c[nt])[0]), "spearman_nontrivial": float(spearmanr(s[nt], c[nt])[0]),
                      "mean_abs_diff_direct_nontrivial": float(np.abs(d[nt]).mean()),
                      "mean_abs_diff_mapped_half_nontrivial": float(np.abs(dm[nt]).mean())})
    else:
        extra.update({k: float("nan") for k in ("pearson_nontrivial", "spearman_nontrivial", "mean_abs_diff_direct_nontrivial", "mean_abs_diff_mapped_half_nontrivial")})
    return {**extra, "n": int(m.sum()), "pearson": float(pearsonr(s, c)[0]), "spearman": float(spearmanr(s, c)[0]),
            "mean_abs_diff_direct": float(np.abs(d).mean()), "mean_abs_diff_mapped_half": float(np.abs(dm).mean()),
            "median_diff_direct": float(np.median(d)), "diff_p05": float(np.quantile(d, 0.05)), "diff_p95": float(np.quantile(d, 0.95)),
            "fit_slope": float(a), "fit_intercept": float(b), "share_cos_ge_score_min": float((c >= s.min()).mean()),
            "score_min": float(s.min()), "score_max": float(s.max()), "cos_min": float(c.min()), "cos_max": float(c.max())}


def overlap(own: set, orig: set) -> dict:
    inter = len(own & orig)
    return {"n_own": len(own), "n_orig": len(orig), "n_intersection": inter,
            "recall_of_orig": inter / len(orig) if orig else float("nan"), "share_of_own_in_orig": inter / len(own) if own else float("nan")}


def pair_set(a: np.ndarray, b: np.ndarray) -> set:
    return {(min(int(x), int(y)), max(int(x), int(y))) for x, y in zip(a, b)}


# ---------------------------------------------------------------- копії золотих

def gold_copies(emb: np.ndarray, emb_set: pd.DataFrame, thresholds=THRESHOLDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Для кожного золотого повідомлення всі інші ембедовані повідомлення з cos >= min(thresholds) (точний пошук).
    Повертає (пари, зведення за порогами). Сусіди, що самі золоті, теж потрапляють (позначено in_gold)."""
    q = np.flatnonzero(emb_set["in_gold"].to_numpy())
    pairs = exact_neighbors(emb, q, min(thresholds))
    key = emb_set["key"].to_numpy()
    txt = emb_set["text"].to_numpy()
    pairs = pairs.assign(gold_key=key[pairs["q"]], neighbor_key=key[pairs["i"]],
                         neighbor_in_labeled=emb_set["in_labeled"].to_numpy()[pairs["i"]],
                         neighbor_in_gold=emb_set["in_gold"].to_numpy()[pairs["i"]],
                         same_text=txt[pairs["q"]] == txt[pairs["i"]]).drop(columns=["q", "i"])
    rows = []
    for t in thresholds:
        p = pairs[pairs["cos"] >= t]
        rows.append({"threshold": t, "n_gold": int(len(q)), "gold_with_neighbor": int(p["gold_key"].nunique()),
                     "gold_with_labeled_neighbor": int(p.loc[p["neighbor_in_labeled"], "gold_key"].nunique()),
                     "neighbor_messages": int(p["neighbor_key"].nunique()),
                     "neighbor_messages_in_labeled": int(p.loc[p["neighbor_in_labeled"], "neighbor_key"].nunique()),
                     "pairs": int(len(p)), "pairs_exact_text": int(p["same_text"].sum())})
    return pairs, pd.DataFrame(rows)


# ---------------------------------------------------------------- досяжність

def _adj(n: int, a: np.ndarray, b: np.ndarray) -> sparse.csr_matrix:
    m = sparse.coo_matrix((np.ones(len(a), dtype=np.float32), (a, b)), shape=(n, n)).tocsr()
    m = ((m + m.T) > 0).astype(np.float32)
    m.setdiag(0)
    m.eliminate_zeros()
    return m.tocsr()


def reach_within(adj: sparse.csr_matrix, source_mask: np.ndarray, kmax: int) -> list[np.ndarray]:
    """x_k[v] = True, якщо від v можна дійти до вузла з source_mask не більш ніж за k кроків (неспрямований граф)."""
    x = source_mask.astype(np.float32)
    out = []
    for _ in range(kmax):
        x = ((x + adj @ x) > 0).astype(np.float32)
        out.append(x > 0)
    return out


def reachability(n_nodes: int, a: np.ndarray, b: np.ndarray, group: np.ndarray, kmax: int = 4) -> pd.DataFrame:
    """Для вузлів групи h частка тих, від кого за <= k кроків можна дійти до вузла іншої групи.
    group: bool на вузол (True = канал групи n), вузли поза шаром (ступінь 0) не враховуються."""
    adj = _adj(n_nodes, a, b)
    present = np.asarray(adj.sum(axis=1)).ravel() > 0
    rows = []
    for name, h in (("n_to_nn", True), ("nn_to_n", False)):
        start = present & (group == h)
        reach = reach_within(adj, present & (group != h), kmax)
        for k, r in enumerate(reach, 1):
            rows.append({"direction": name, "k": k, "n_start": int(start.sum()), "n_reach": int((r & start).sum()),
                         "share": float((r & start).sum() / start.sum()) if start.sum() else float("nan")})
    return pd.DataFrame(rows)


def reachability_null(n_nodes: int, a: np.ndarray, b: np.ndarray, channel_of: np.ndarray, channel_group: dict, kmax: int = 4,
                      n_perm: int = 20, seed: int = 42) -> pd.DataFrame:
    """Нульова модель: групи розкидані по каналах випадково (розміри груп збережено), повідомлення успадковують групу каналу."""
    rng = np.random.default_rng(seed)
    chs = np.array(sorted(channel_group))
    vals = np.array([channel_group[c] for c in chs])
    pos = {c: i for i, c in enumerate(chs)}
    ci = np.array([pos.get(int(c), -1) for c in channel_of])
    ok = ci >= 0
    acc = []
    for _ in range(n_perm):
        g = np.zeros(n_nodes, dtype=bool)
        g[ok] = rng.permutation(vals)[ci[ok]]
        acc.append(reachability(n_nodes, a, b, g, kmax).set_index(["direction", "k"])["share"])
    df = pd.concat(acc, axis=1)
    return pd.DataFrame({"null_mean": df.mean(axis=1), "null_p05": df.quantile(0.05, axis=1), "null_p95": df.quantile(0.95, axis=1)}).reset_index()


# ---------------------------------------------------------------- перевірка форвардів проти channel_flow (залишок S3)

def forward_vs_channel_flow(fwd: pd.DataFrame, flow: pd.DataFrame) -> dict:
    """Агрегує message-рівневі forward-ребра до пар каналів і звіряє з таблицею channel_flow_edges (S3)."""
    f = fwd.rename(columns={"logical_src_channel_id": "s", "logical_dst_channel_id": "d"})[["s", "d"]]
    n_all = len(f)
    loops = f["s"] == f["d"]
    agg = f[~loops].groupby(["s", "d"]).size().rename("w").reset_index()
    fl = flow.rename(columns={"src_channel_id": "s", "dst_channel_id": "d", "weight": "wf"})[["s", "d", "wf"]]
    mm = agg.merge(fl, on=["s", "d"], how="outer").fillna(0)
    diff = mm[mm["w"] != mm["wf"]]
    return {"n_forward_edges": int(n_all), "n_self_forward": int(loops.sum()), "n_channel_pairs_from_forward": int(len(agg)),
            "weight_sum_without_self": int(agg["w"].sum()), "channel_flow_rows": int(len(fl)), "channel_flow_weight_sum": int(fl["wf"].sum()),
            "n_pairs_with_different_weight": int(len(diff)), "equal_to_channel_flow": bool(len(diff) == 0)}


def pair_group_stats(ch_a: np.ndarray, ch_b: np.ndarray, group: dict) -> dict:
    """Частки пар за групами каналів (n, nn) і частка пар усередині одного каналу. Пари з каналом без групи рахуються окремо."""
    ga = pd.Series(ch_a).map(group)
    gb = pd.Series(ch_b).map(group)
    known = ga.notna().to_numpy() & gb.notna().to_numpy()
    a, b = ga[known].astype(bool).to_numpy(), gb[known].astype(bool).to_numpy()
    n = int(known.sum())
    out = {"n_pairs": int(len(ch_a)), "n_unknown_group": int((~known).sum()), "same_channel_share": float((np.asarray(ch_a) == np.asarray(ch_b)).mean()) if len(ch_a) else float("nan")}
    out.update({"n_n": int((a & b).sum()), "n_nn": int((a != b).sum()), "nn_nn": int((~a & ~b).sum()),
                "cross_share": float((a != b).sum() / n) if n else float("nan")})
    return out
