"""S7: ознаки повідомлення з графа і ембедингів для табличних моделей (LightGBM).

Ознаки без міток (мережа каналів, семантичні копії, форвард-прапорець) рахуються з усієї мережі/набору (преєстрація, розділ 2).
Ознака з мітками (канальний пріор) рахується лише з train відповідного спліту; для рядків самого train використовується
позафолдовий (OOF) варіант за групами `text_hash`, щоб модель не бачила мітку рядка в його ж ознаці.
Членство каналу в мережі як ознака не береться (S3, п. 12: воно пов'язане з класом).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from narr import network as nw
from narr.splits import channel_prior

SEM_THRESHOLD = 0.985

NET_COLS = ["net_in_strength_log", "net_out_strength_log", "net_pagerank_forwarder", "net_pagerank_source", "net_betweenness",
            "net_coreness", "net_in_degree", "net_out_degree", "net_community"]
SEM_COLS = ["sem_n_copies_xch", "sem_n_channels_xch", "sem_max_cos_xch", "sem_first_copy_delta_h", "sem_n_copies_any"]
FLAG_COLS = ["is_forward_endpoint"]
PRIOR_COL = "channel_prior"
GRAPH_COLS = NET_COLS + SEM_COLS + FLAG_COLS + [PRIOR_COL]
GRAPH_COLS_NO_PRIOR = NET_COLS + SEM_COLS + FLAG_COLS


def channel_network_features(edges: pd.DataFrame, communities: pd.DataFrame, weight: str = "log") -> pd.DataFrame:
    """Ознаки каналу з мережі S3 з вагами log(1 + w). Індекс channel_id. Канали поза мережею тут відсутні (NaN після злиття).
    Спільнота береться з `community_log` (найкращий за модулярністю з 10 запусків Leiden, S3)."""
    e, _ = nw.clean_edges(edges)
    if weight == "log":
        e = e.assign(weight=np.log1p(e["weight"]))
    g = nw.build_graph(e)
    c = nw.centrality(g)
    out = pd.DataFrame({
        "channel_id": c["channel_id"],
        "net_in_strength_log": c["in_strength"], "net_out_strength_log": c["out_strength"],
        "net_pagerank_forwarder": c["pagerank_forwarder"], "net_pagerank_source": c["pagerank_source"],
        "net_betweenness": c["betweenness"], "net_coreness": c["coreness"],
        "net_in_degree": c["in_degree"], "net_out_degree": c["out_degree"]})
    out = out.merge(communities[["channel_id", "community_log"]].rename(columns={"community_log": "net_community"}), on="channel_id", how="left")
    return out.set_index("channel_id")


def semantic_copy_features(emb: np.ndarray, channel_id: np.ndarray, date: pd.Series, threshold: float = SEM_THRESHOLD,
                           block: int = 2048, device: str | None = None, log=print) -> pd.DataFrame:
    """Для кожного повідомлення: скільки копій (cos >= threshold) у інших каналах, у скількох каналах, найбільший cos,
    різниця часу (години) до найранішої копії (від'ємна, якщо копія раніше), і число копій у будь-яких каналах.
    Точний пошук блоками (hnswlib на Windows недоступний). Копії шукаються в усьому наборі S4 (не лише в train), мітки не використовуються."""
    import torch

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    E = torch.as_tensor(np.ascontiguousarray(emb, dtype=np.float32), device=dev)
    n = len(emb)
    ch = np.asarray(channel_id)
    ch_t = torch.as_tensor(ch, device=dev)
    ts = pd.to_datetime(date, utc=True, errors="coerce")
    t_h = ((ts - pd.Timestamp("2024-01-01", tz="UTC")).dt.total_seconds() / 3600.0).to_numpy(dtype="float64")
    n_x = np.zeros(n, dtype=np.int32)
    n_ch = np.zeros(n, dtype=np.int32)
    n_any = np.zeros(n, dtype=np.int32)
    max_c = np.full(n, np.nan, dtype=np.float32)
    first_d = np.full(n, np.nan, dtype=np.float64)
    pairs_rows = []
    for s in range(0, n, block):
        sims = E[s:s + block] @ E.T
        idx = torch.arange(s, min(s + block, n), device=dev)
        sims[torch.arange(len(idx), device=dev), idx] = -2.0
        qi, ji = torch.nonzero(sims >= threshold, as_tuple=True)
        if len(qi) == 0:
            continue
        c = sims[qi, ji].float().cpu().numpy()
        qi, ji = qi.cpu().numpy() + s, ji.cpu().numpy()
        pairs_rows.append(pd.DataFrame({"i": qi, "j": ji, "cos": c}))
        if (s // block) % 10 == 0:
            log(f"    семантичні копії: {min(s + block, n)} з {n}")
    if not pairs_rows:
        return pd.DataFrame({k: [] for k in SEM_COLS})
    p = pd.concat(pairs_rows, ignore_index=True)
    p["any"] = 1
    n_any[:] = np.bincount(p["i"], minlength=n)
    x = p[ch[p["i"].to_numpy()] != ch[p["j"].to_numpy()]].copy()
    x["ch_j"] = ch[x["j"].to_numpy()]
    x["delta"] = t_h[x["j"].to_numpy()] - t_h[x["i"].to_numpy()]
    n_x[:] = np.bincount(x["i"], minlength=n)
    g = x.groupby("i").agg(n_ch=("ch_j", "nunique"), max_c=("cos", "max"))
    n_ch[g.index.to_numpy()] = g["n_ch"].to_numpy()
    max_c[g.index.to_numpy()] = g["max_c"].to_numpy()
    xe = x.dropna(subset=["delta"])
    first = xe.loc[xe.groupby("i")["delta"].idxmin()] if len(xe) else xe
    first_d[first["i"].to_numpy()] = first["delta"].to_numpy()
    return pd.DataFrame({"sem_n_copies_xch": n_x, "sem_n_channels_xch": n_ch, "sem_max_cos_xch": max_c,
                         "sem_first_copy_delta_h": first_d, "sem_n_copies_any": n_any})


def oof_channel_prior(train: pd.DataFrame, n_folds: int = 5, seed: int = 42, alpha: float = 1.0) -> np.ndarray:
    """Канальний пріор для рядків train з позафолдовим обчисленням (групи за text_hash)."""
    groups = (train["comp"] if "comp" in train.columns else train["text_hash"]).to_numpy()  # версія 1.1: компоненти копій
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq.tolist(), rng.integers(0, n_folds, size=len(uniq)).tolist()))
    fold = np.array([fold_of[g] for g in groups])
    out = np.full(len(train), np.nan)
    for f in range(n_folds):
        te = fold == f
        rest = train[~te]
        pr = channel_prior(rest, alpha)
        glob = float(rest["weak_narrative"].mean())
        out[te] = train.loc[te, "channel_id"].map(pr).fillna(glob).to_numpy()
    return out


def prior_for(test: pd.DataFrame, train: pd.DataFrame, alpha: float = 1.0) -> np.ndarray:
    pr = channel_prior(train, alpha)
    return test["channel_id"].map(pr).fillna(float(train["weak_narrative"].mean())).to_numpy()


def assemble(frame: pd.DataFrame, base: pd.DataFrame, channel_net: pd.DataFrame) -> pd.DataFrame:
    """Додає до рядків (колонки key, channel_id) ознаки мережі каналу і ознаки повідомлення з `base` (індекс key)."""
    net = channel_net.reindex(frame["channel_id"].to_numpy()).reset_index(drop=True)
    b = base.reindex(frame["key"].to_numpy()).reset_index(drop=True)
    out = pd.concat([net[NET_COLS], b[SEM_COLS + FLAG_COLS]], axis=1)
    out.index = frame.index
    return out


def rewire_edges(edges: pd.DataFrame, seed: int = 42, n_mult: int = 20) -> pd.DataFrame:
    """Негативний контроль 2 (преєстрація, розділ 8): перезв'язування ребер мережі каналів зі збереженням вхідних і вихідних ступенів.
    Ваги лишаються за номером ребра (розподіл ваг той самий, зв'язок вага-пара втрачено)."""
    import random

    e, _ = nw.clean_edges(edges)
    g = nw.build_graph(e)
    w0 = np.asarray(g.es["weight"], dtype=float)  # rewire не зберігає атрибути ребер: ваги призначаємо за номером ребра
    random.seed(seed)
    g.rewire(n=n_mult * g.ecount(), mode="simple")
    ids = g.vs["channel_id"]
    el = g.get_edgelist()
    return pd.DataFrame({"src_channel_id": [ids[a] for a, _ in el], "dst_channel_id": [ids[b] for _, b in el], "weight": w0})


def semantic_pairs(emb: np.ndarray, channel_id: np.ndarray, threshold: float = SEM_THRESHOLD, block: int = 2048,
                   device: str | None = None, cross_channel: bool = True) -> pd.DataFrame:
    """Унікальні неспрямовані пари i < j з cos >= threshold (за замовчуванням лише між різними каналами; основний шар H2, преєстрація п. 6).
    Точний пошук блоками на GPU (hnswlib на Windows недоступний)."""
    import torch

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    E = torch.as_tensor(np.ascontiguousarray(emb, dtype=np.float32), device=dev)
    n = len(emb)
    ch = np.asarray(channel_id)
    rows = []
    for s in range(0, n, block):
        sims = E[s:s + block] @ E.T
        idx = torch.arange(s, min(s + block, n), device=dev)
        sims[torch.arange(len(idx), device=dev), idx] = -2.0
        qi, ji = torch.nonzero(sims >= threshold, as_tuple=True)
        if len(qi) == 0:
            continue
        c = sims[qi, ji].float().cpu().numpy()
        qi, ji = qi.cpu().numpy() + s, ji.cpu().numpy()
        keep = qi < ji
        qi, ji, c = qi[keep], ji[keep], c[keep]
        if cross_channel:
            k2 = ch[qi] != ch[ji]
            qi, ji, c = qi[k2], ji[k2], c[k2]
        rows.append(pd.DataFrame({"i": qi, "j": ji, "cos": c}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame({"i": [], "j": [], "cos": []})


def copublication_pairs(date: pd.Series, channel_id: np.ndarray, window_h: float = 1.0, max_offset: int = 30, k: int = 10) -> pd.DataFrame:
    """Ребра co_publication: повідомлення різних каналів, опубліковані в межах window_h годин одне від одного.
    Щоб не отримати мільйони пар у години піку, кожне повідомлення з'єднується не більше ніж з k наступними (за часом) повідомленнями
    інших каналів серед max_offset сусідів у впорядкуванні за часом. Повертає унікальні неспрямовані пари i < j (індекси початкового порядку), колонку dt_h."""
    t = (pd.to_datetime(date, utc=True, errors="coerce") - pd.Timestamp("2024-01-01", tz="UTC")).dt.total_seconds().to_numpy() / 3600.0
    ok = np.flatnonzero(~np.isnan(t))
    o = ok[np.argsort(t[ok], kind="stable")]
    ts, cs = t[o], np.asarray(channel_id)[o]
    rows = []
    for d in range(1, max_offset + 1):
        a, b = np.arange(0, len(o) - d), np.arange(d, len(o))
        m = (ts[b] - ts[a] <= window_h) & (cs[a] != cs[b])
        rows.append(pd.DataFrame({"a": a[m], "b": b[m], "d": d}))
    p = pd.concat(rows, ignore_index=True).sort_values(["a", "d"])
    p = p.groupby("a", sort=False).head(k)
    i, j = o[p["a"].to_numpy()], o[p["b"].to_numpy()]
    out = pd.DataFrame({"i": np.minimum(i, j), "j": np.maximum(i, j), "dt_h": (ts[p["b"].to_numpy()] - ts[p["a"].to_numpy()])})
    return out.drop_duplicates(["i", "j"]).reset_index(drop=True)
