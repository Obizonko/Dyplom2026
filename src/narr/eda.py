"""S2: описова аналітика повних даних TeleNarratives.

Модуль самодостатній (pandas, numpy, pyarrow, matplotlib), тому той самий код працює в Colab і в тестах.
Усі функції приймають таблиці як DataFrame і не читають диск, крім `load_inputs`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

KEY_MULT = 10_000_000          # ключ повідомлення = channel_id * KEY_MULT + message_id
DATE_MIN, DATE_MAX = "2024-12-16", "2026-02-27"
# Кількості з README і MANIFEST.json графового набору (знімок 11.03.2026)
README_COUNTS = {"channels": 111, "messages": 1_052_563, "forward_edges": 35_697, "semantic_edges": 52_414,
                 "channel_flow_edges": 214, "sem_channel_flow_edges": 1_273}
GOLD_COLS = {"text": "text", "channel": "channel_id", "message": "message_id", "date": "date_utc",
             "binary": "Binary label", "level1": "General label (level-1)"}
MSG_COLS = ["channel_id", "message_id", "date_utc", "channel_username", "text", "views", "forwards", "has_media",
            "binary_label", "is_labeled_batch", "narrative_id", "sub_narrative_id", "label_confidence"]
TG_COLS = ["channel_id", "message_id", "is_reply", "reply_to_msg_id", "is_forwarded", "fwd_from_channel_id",
           "fwd_from_message_id", "date_utc", "views", "forwards", "replies_count", "has_media"]

_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)


# ---------------------------------------------------------------- допоміжні

def keys(df: pd.DataFrame, ch: str = "channel_id", msg: str = "message_id") -> pd.Series:
    return df[ch].astype("int64") * KEY_MULT + df[msg].astype("int64")


def _flag(s: pd.Series) -> pd.Series:
    """Булева маска з порожніми значеннями як False (працює для bool, object з None, nullable boolean)."""
    return s.eq(True).fillna(False).astype(bool)


def normalize_text(s: pd.Series) -> pd.Series:
    out = s.fillna("").astype(str).str.lower().str.replace(_NON_WORD, "", regex=True)
    return out.str.replace(r"\s+", " ", regex=True).str.strip()


def _share(a, b) -> float:
    return float(a) / float(b) if b else float("nan")


def _binary_scores(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())

    def f1(t, f_p, f_n):
        d = 2 * t + f_p + f_n
        return 2 * t / d if d else float("nan")

    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"n": int(len(y)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": _share(tp + tn, len(y)), "precision": _share(tp, tp + fp), "recall": _share(tp, tp + fn),
            "f1_pos": f1(tp, fp, fn), "macro_f1": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
            "mcc": (tp * tn - fp * fn) / den if den else float("nan")}


# ---------------------------------------------------------------- читання

def load_inputs(raw_dir: Path | str, golden_path: Path | str) -> dict:
    """Читає потрібні таблиці з data/raw (повні parquet не тримаємо в пам'яті, якщо вони не потрібні)."""
    import pyarrow.parquet as pq

    raw_dir = Path(raw_dir)
    mpath = next(iter(sorted(raw_dir.rglob("messages.parquet"))), None)
    if mpath is None:
        raise FileNotFoundError(f"не знайдено messages.parquet у {raw_dir}")
    gd = mpath.parent
    tg_path = raw_dir / "telegram_messages.parquet"

    msgs = pq.read_table(mpath, columns=MSG_COLS).to_pandas()
    tg = pq.read_table(tg_path, columns=TG_COLS).to_pandas()
    gold = pd.read_csv(golden_path)

    gold_ids = [int(x) for x in gold[GOLD_COLS["message"]].unique()]
    tg_text = pq.read_table(tg_path, columns=["channel_id", "message_id", "text"],
                            filters=[("message_id", "in", gold_ids)]).to_pandas()

    fwd = pq.read_table(gd / "forward_edges.parquet",
                        columns=["logical_src_channel_id", "logical_src_message_id",
                                 "logical_dst_channel_id", "logical_dst_message_id"]).to_pandas()
    sem = pq.read_table(gd / "semantic_edges.parquet",
                        columns=["src_channel_id", "src_message_id", "dst_channel_id", "dst_message_id",
                                 "similarity_score"]).to_pandas()
    return {"msgs": msgs, "tg": tg, "tg_gold_text": tg_text, "gold": gold, "fwd": fwd, "sem": sem,
            "channels": pq.read_table(gd / "channels.parquet").to_pandas(),
            "coding": pd.read_csv(gd / "narrative_subnarrative_coding_en.csv"),
            "paths": {"messages": str(mpath), "telegram_messages": str(tg_path)}}


# ---------------------------------------------------------------- таблиці

def check_integrity(msgs: pd.DataFrame, tg: pd.DataFrame, date_min: str = DATE_MIN, date_max: str = DATE_MAX) -> dict:
    km, kt = keys(msgs).to_numpy(), keys(tg).to_numpy()
    both = np.intersect1d(km, kt)
    lo = pd.Timestamp(date_min, tz="UTC")
    hi = pd.Timestamp(date_max, tz="UTC") + pd.Timedelta(days=1)
    out = {"n_msgs": int(len(msgs)), "n_tg": int(len(tg)),
           "n_channels_msgs": int(msgs["channel_id"].nunique()), "n_channels_tg": int(tg["channel_id"].nunique()),
           "dup_keys_msgs": int(pd.Series(km).duplicated().sum()), "dup_keys_tg": int(pd.Series(kt).duplicated().sum()),
           "keys_both": int(len(both)), "keys_only_msgs": int(len(np.unique(km)) - len(both)),
           "keys_only_tg": int(len(np.unique(kt)) - len(both))}
    for name, df in (("msgs", msgs), ("tg", tg)):
        d = pd.to_datetime(df["date_utc"], utc=True, errors="coerce")
        out[f"unparsed_dates_{name}"] = int(d.isna().sum())
        out[f"date_min_{name}"] = str(d.min())
        out[f"date_max_{name}"] = str(d.max())
        out[f"out_of_range_{name}"] = int(((d < lo) | (d >= hi)).sum())
    return out


def monthly_table(msgs: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(msgs["date_utc"], utc=True)
    lab = _flag(msgs["is_labeled_batch"])
    t = pd.DataFrame({"month": d.dt.strftime("%Y-%m"), "n": 1, "n_labeled": lab.astype(int),
                      "n_narrative": (lab & msgs["narrative_id"].notna()).astype(int)})
    t = t.groupby("month", as_index=False).sum()
    t["narrative_share_of_labeled"] = t["n_narrative"] / t["n_labeled"].where(t["n_labeled"] > 0)
    t["partial_month"] = t["month"].isin([t["month"].min(), t["month"].max()])
    return t


def per_channel(msgs: pd.DataFrame, tg: pd.DataFrame, channels: pd.DataFrame) -> pd.DataFrame:
    lab = _flag(msgs["is_labeled_batch"])
    g = pd.DataFrame({"channel_id": msgs["channel_id"], "n_messages": 1, "n_labeled": lab.astype(int),
                      "n_narrative": (lab & msgs["narrative_id"].notna()).astype(int),
                      "length": msgs["text"].fillna("").str.len()})
    agg = g.groupby("channel_id").agg(n_messages=("n_messages", "sum"), n_labeled=("n_labeled", "sum"),
                                      n_narrative=("n_narrative", "sum"), median_length=("length", "median")).reset_index()
    t2 = tg.groupby("channel_id").agg(forwarded_share=("is_forwarded", "mean"), reply_share=("is_reply", "mean")).reset_index()
    ch = channels[["channel_id", "username", "narrative_presence", "narrative_presence_status", "channel_label",
                   "labeled_messages", "narrative_messages", "narrative_share"]].rename(columns={
        "narrative_presence": "table_narrative_presence", "narrative_presence_status": "table_status",
        "channel_label": "table_label", "labeled_messages": "table_labeled", "narrative_messages": "table_narrative",
        "narrative_share": "table_share"})
    out = ch.merge(agg, on="channel_id", how="outer").merge(t2, on="channel_id", how="left")
    for c in ("n_messages", "n_labeled", "n_narrative"):
        out[c] = out[c].fillna(0).astype(int)
    out["share_recomputed"] = out["n_narrative"] / out["n_labeled"].where(out["n_labeled"] > 0)
    out["counts_match"] = (out["n_labeled"] == out["table_labeled"]) & (out["n_narrative"] == out["table_narrative"])
    return out.sort_values("n_messages", ascending=False).reset_index(drop=True)


def label_tables(msgs: pd.DataFrame, coding: pd.DataFrame) -> tuple[dict, dict[str, pd.DataFrame]]:
    lab = _flag(msgs["is_labeled_batch"])
    narr = msgs["narrative_id"].notna()
    n_lab, n_narr = int(lab.sum()), int((lab & narr).sum())
    summary = {"n_messages": int(len(msgs)), "n_labeled": n_lab, "share_labeled": _share(n_lab, len(msgs)),
               "n_narrative_any": int(narr.sum()), "n_narrative_labeled": n_narr,
               "narrative_not_labeled": int((narr & ~lab).sum()),
               "narrative_share_of_labeled": _share(n_narr, n_lab),
               "sub_without_narrative": int((msgs["sub_narrative_id"].notna() & ~narr).sum()),
               "confidence_missing_among_labeled": int(msgs.loc[lab, "label_confidence"].isna().sum())}

    # consistency of sub_narrative_id -> narrative_id with the coding file
    sub2narr = coding.drop_duplicates("sub_narrative_id").set_index("sub_narrative_id")["narrative_id"]
    both = msgs.loc[msgs["sub_narrative_id"].notna() & narr, ["sub_narrative_id", "narrative_id"]]
    mapped = both["sub_narrative_id"].astype("int64").map(sub2narr)
    summary["sub_narrative_unknown_in_coding"] = int(mapped.isna().sum())
    summary["sub_to_narrative_match_share"] = _share(int((mapped == both["narrative_id"]).sum()), len(both))
    summary["max_sub_narrative_id"] = int(msgs["sub_narrative_id"].max()) if msgs["sub_narrative_id"].notna().any() else None
    summary["sub_ids_not_in_coding"] = int(msgs["sub_narrative_id"].dropna().astype("int64").map(sub2narr).isna().sum())

    # confidence of labels
    conf = pd.DataFrame({"label_confidence": msgs.loc[lab, "label_confidence"], "narrative": narr[lab]})
    conf_t = (conf.groupby("label_confidence")["narrative"].agg(n="size", n_narrative="sum").reset_index()
              .assign(n_not_narrative=lambda d: d["n"] - d["n_narrative"]))
    conf_t["n_narrative"] = conf_t["n_narrative"].astype(int)

    # binary_label vs labeled batch
    bl = msgs["binary_label"].astype(object).where(msgs["binary_label"].notna(), "(порожньо)").astype(str)
    cross = (pd.DataFrame({"binary_label": bl, "is_labeled_batch": lab, "has_narrative_id": narr})
             .groupby(["binary_label", "is_labeled_batch", "has_narrative_id"]).size().rename("n").reset_index())

    # meta-narratives of weak labels (through narrative_id, not sub_narrative_id)
    n2m = coding.drop_duplicates("narrative_id")[["narrative_id", "level_1_id", "level_1"]]
    summary["narrative_to_meta_unique"] = bool((coding.groupby("narrative_id")["level_1_id"].nunique() == 1).all())
    m = msgs.loc[lab & narr, ["narrative_id"]].merge(n2m, on="narrative_id", how="left")
    meta = (m.groupby(["level_1_id", "level_1"], dropna=False).size().rename("n_weak").reset_index()
            .sort_values("n_weak", ascending=False))
    meta["share_weak"] = meta["n_weak"] / meta["n_weak"].sum()

    # text length
    L = msgs["text"].fillna("").str.len()
    groups = {"усі": pd.Series(True, index=msgs.index), "розмічені": lab, "розмічені, наративні": lab & narr,
              "розмічені, ненаративні": lab & ~narr, "нерозмічені": ~lab}
    rows = []
    for name, mask in groups.items():
        x = L[mask]
        rows.append({"group": name, "n": int(len(x)), "mean": float(x.mean()), "p10": float(x.quantile(.1)),
                     "median": float(x.median()), "p90": float(x.quantile(.9)), "max": int(x.max())})
    return summary, {"label_confidence": conf_t, "binary_label_crosstab": cross, "meta_weak": meta,
                     "length_stats": pd.DataFrame(rows)}


def language_tables(msgs: pd.DataFrame, gold: pd.DataFrame, lang_fn, n_unlabeled: int, seed: int,
                    gcols: dict = GOLD_COLS) -> dict[str, pd.DataFrame]:
    """Мова всіх розмічених + випадкової вибірки нерозмічених + золотих. lang_fn(Series) -> масив кодів."""
    lab = _flag(msgs["is_labeled_batch"]).to_numpy()
    rng = np.random.default_rng(seed)
    idx_lab = np.flatnonzero(lab)
    idx_unl = np.flatnonzero(~lab)
    take = np.sort(rng.choice(idx_unl, size=min(n_unlabeled, len(idx_unl)), replace=False)) if len(idx_unl) else idx_unl
    sub = msgs.iloc[np.concatenate([idx_lab, take])][["channel_id", "message_id", "text", "narrative_id"]].copy()
    sub["is_labeled"] = np.r_[np.ones(len(idx_lab), bool), np.zeros(len(take), bool)]
    sub["narrative"] = sub["narrative_id"].notna() & sub["is_labeled"]
    sub["lang"] = lang_fn(sub["text"].fillna("").astype(str).str.slice(0, 300))

    by_group = (sub.assign(group=np.where(sub["is_labeled"], "розмічені (усі)", "нерозмічені (вибірка)"))
                .groupby(["group", "lang"]).size().rename("n").reset_index())
    by_group["share"] = by_group["n"] / by_group.groupby("group")["n"].transform("sum")

    wl = sub[sub["is_labeled"]]
    x_label = (wl.groupby("lang")["narrative"].agg(n="size", n_narrative="sum").reset_index()
               .assign(source="слабкі мітки (розмічені)"))

    g = gold.copy()
    g["lang"] = lang_fn(g[gcols["text"]].fillna("").astype(str).str.slice(0, 300))
    gx = (g.groupby("lang")[gcols["binary"]].agg(n="size", n_narrative="sum").reset_index()
          .assign(source="золота розмітка"))
    x_label = pd.concat([x_label, gx], ignore_index=True)
    x_label["n_narrative"] = x_label["n_narrative"].astype(int)
    x_label["narrative_share"] = x_label["n_narrative"] / x_label["n"]

    ch = sub.groupby(["channel_id", "lang"]).size().unstack(fill_value=0)
    ch["n_language_sample"] = ch.sum(axis=1)
    for c in [c for c in ch.columns if c != "n_language_sample"]:
        ch[f"share_{c}"] = ch[c] / ch["n_language_sample"]
    return {"language_by_group": by_group, "language_x_label": x_label, "language_by_channel": ch.reset_index()}


def graph_membership(msgs: pd.DataFrame, fwd: pd.DataFrame, sem: pd.DataFrame) -> pd.DataFrame:
    km = keys(msgs).to_numpy()
    f = np.union1d(keys(fwd, "logical_src_channel_id", "logical_src_message_id").to_numpy(),
                   keys(fwd, "logical_dst_channel_id", "logical_dst_message_id").to_numpy())
    s = np.union1d(keys(sem, "src_channel_id", "src_message_id").to_numpy(),
                   keys(sem, "dst_channel_id", "dst_message_id").to_numpy())
    in_f, in_s = np.isin(km, f), np.isin(km, s)
    lab = _flag(msgs["is_labeled_batch"]).to_numpy()
    narr = (lab & msgs["narrative_id"].notna().to_numpy())
    grp = np.where(in_f & in_s, "forward і semantic", np.where(in_f, "лише forward",
                                                              np.where(in_s, "лише semantic", "поза графом")))
    t = (pd.DataFrame({"group": grp, "n": 1, "n_labeled": lab.astype(int), "n_narrative": narr.astype(int)})
         .groupby("group", as_index=False).sum())
    t["labeled_share"] = t["n_labeled"] / t["n"]
    t["narrative_share_of_labeled"] = t["n_narrative"] / t["n_labeled"].where(t["n_labeled"] > 0)
    return t


def forwards_summary(tg: pd.DataFrame, msgs: pd.DataFrame, channels: pd.DataFrame, n_forward_edges: int) -> dict:
    fw = tg[_flag(tg["is_forwarded"])]
    corpus = set(channels["channel_id"].astype("int64")) | set(msgs["channel_id"].astype("int64"))
    src = fw["fwd_from_channel_id"].dropna().astype("int64")
    in_corpus_channel = src.isin(corpus)
    ok = fw["fwd_from_channel_id"].notna() & fw["fwd_from_message_id"].notna()
    ofk = (fw.loc[ok, "fwd_from_channel_id"].astype("int64") * KEY_MULT + fw.loc[ok, "fwd_from_message_id"].astype("int64"))
    orig_present = ofk.isin(set(keys(tg)))
    return {"n_forwarded": int(len(fw)), "forwarded_share": _share(len(fw), len(tg)),
            "n_unique_source_channels": int(src.nunique()),
            "n_forwarded_from_corpus_channel": int(in_corpus_channel.sum()),
            "n_forwarded_with_original_in_corpus": int(orig_present.sum()),
            "n_forward_edges_file": int(n_forward_edges),
            "n_replies": int(_flag(tg["is_reply"]).sum()), "reply_share": _share(int(_flag(tg["is_reply"]).sum()), len(tg))}


def _gold_merge(gold, msgs, gcols):
    g = gold.copy()
    g["_key"] = keys(g, gcols["channel"], gcols["message"])
    m = msgs[["channel_id", "message_id", "text", "is_labeled_batch", "narrative_id", "label_confidence"]].copy()
    m["_key"] = keys(m)
    return g.merge(m[["_key", "text", "is_labeled_batch", "narrative_id", "label_confidence"]].rename(
        columns={"text": "text_graph"}), on="_key", how="left")


def gold_overlap(gold: pd.DataFrame, msgs: pd.DataFrame, coding: pd.DataFrame, tg_gold_text: pd.DataFrame,
                 fwd: pd.DataFrame, sem: pd.DataFrame, gcols: dict = GOLD_COLS) -> tuple[dict, dict[str, pd.DataFrame]]:
    m = _gold_merge(gold, msgs, gcols)
    present = m["text_graph"].notna()
    lab = _flag(m["is_labeled_batch"]) & present
    y = m[gcols["binary"]].astype(bool)
    summary = {"n_gold": int(len(m)), "n_gold_in_graph_messages": int(present.sum()),
               "n_gold_labeled_weak": int(lab.sum()),
               "n_gold_present_not_labeled": int((present & ~lab).sum()),
               "gold_narrative_share_present_not_labeled": _share(int((y & present & ~lab).sum()), int((present & ~lab).sum()))}
    km_gold = keys(gold, gcols["channel"], gcols["message"])
    f = np.union1d(keys(fwd, "logical_src_channel_id", "logical_src_message_id").to_numpy(),
                   keys(fwd, "logical_dst_channel_id", "logical_dst_message_id").to_numpy())
    s = np.union1d(keys(sem, "src_channel_id", "src_message_id").to_numpy(),
                   keys(sem, "dst_channel_id", "dst_message_id").to_numpy())
    summary["n_gold_in_forward_edges"] = int(np.isin(km_gold.to_numpy(), f).sum())
    summary["n_gold_in_semantic_edges"] = int(np.isin(km_gold.to_numpy(), s).sum())

    tables: dict[str, pd.DataFrame] = {}
    if lab.any():
        weak = m.loc[lab, "narrative_id"].notna().to_numpy()
        sc = _binary_scores(y[lab].to_numpy(), weak)
        summary["weak_vs_gold"] = sc
        tables["weak_vs_gold_confusion"] = pd.DataFrame(
            [[sc["tn"], sc["fp"]], [sc["fn"], sc["tp"]]], index=["золота: ні", "золота: так"],
            columns=["слабка: ні", "слабка: так"]).reset_index().rename(columns={"index": ""})
        # мета-наративи: назви рівня 1 у золотому наборі проти слабких міток
        n2m = coding.drop_duplicates("narrative_id").set_index("narrative_id")["level_1"]
        both = lab & y & m["narrative_id"].notna()
        a = m.loc[both, gcols["level1"]].astype(str)
        b = m.loc[both, "narrative_id"].astype("int64").map(n2m).astype(str)
        summary["meta_pairs_both_narrative"] = int(both.sum())
        summary["meta_match_share"] = _share(int((a.to_numpy() == b.to_numpy()).sum()), int(both.sum()))
        summary["gold_meta_names_not_in_coding"] = int((~a.isin(set(coding["level_1"]))).sum())
        tables["weak_vs_gold_meta"] = pd.crosstab(a.rename("золотий мета-наратив"), b.rename("слабкий мета-наратив")).reset_index()
    # варіанти тексту
    tt = tg_gold_text.copy()
    tt["_key"] = keys(tt)
    mm = m.merge(tt[["_key", "text"]].rename(columns={"text": "text_tg"}), on="_key", how="left")
    gt = mm[gcols["text"]].fillna("").astype(str)
    gn = normalize_text(gt)
    rows = []
    for name, col in (("messages.parquet (сирий)", "text_graph"), ("telegram_messages.parquet (нормалізований)", "text_tg")):
        has = mm[col].notna()
        other = mm.loc[has, col].astype(str)
        on = normalize_text(other)
        gnn = gn[has]
        rows.append({"варіант тексту": name, "n_знайдено": int(has.sum()),
                     "точно збігається": int((gt[has].to_numpy() == other.to_numpy()).sum()),
                     "збігається після нормалізації": int((gnn.to_numpy() == on.to_numpy()).sum()),
                     "збігаються перші 60 символів після нормалізації": int((gnn.str[:60].to_numpy() == on.str[:60].to_numpy()).sum()),
                     "медіана довжини": float(other.str.len().median())})
    summary["gold_text_median_length"] = float(gt.str.len().median())
    tables["text_variants"] = pd.DataFrame(rows)
    # копії золотих текстів у корпусі (для виключення в P1)
    norm_all = normalize_text(msgs["text"])
    gold_norm = set(gn[gn != ""])
    is_copy = norm_all.isin(gold_norm).to_numpy()
    not_self = ~keys(msgs).isin(set(km_gold)).to_numpy()
    summary["n_corpus_copies_of_gold_text"] = int((is_copy & not_self).sum())
    gcopy = gn.isin(set(norm_all[is_copy & not_self]))
    summary["n_gold_with_corpus_copy"] = int(gcopy.sum())
    dup = norm_all.duplicated(keep=False)
    labd = _flag(msgs["is_labeled_batch"])
    summary["duplicates"] = {"n_rows_in_duplicate_groups": int(dup.sum()), "share_all": float(dup.mean()),
                             "share_labeled": float(dup[labd.to_numpy()].mean()), "n_unique_texts": int(norm_all.nunique())}
    return summary, tables


# ---------------------------------------------------------------- запуск

def run_all(inp: dict, lang_fn, lang_sample_unlabeled: int = 150_000, seed: int = 42,
            gcols: dict = GOLD_COLS) -> tuple[dict, dict, list]:
    msgs, tg, channels, coding = inp["msgs"], inp["tg"], inp["channels"], inp["coding"]
    gold = inp["gold"]
    tables: dict[str, pd.DataFrame] = {}
    summary: dict = {"readme_counts": README_COUNTS}
    summary["integrity"] = check_integrity(msgs, tg)
    tables["monthly"] = monthly_table(msgs)
    tables["per_channel"] = per_channel(msgs, tg, channels)
    s, t = label_tables(msgs, coding)
    summary["labels"] = s
    tables.update(t)
    tables.update(language_tables(msgs, gold, lang_fn, lang_sample_unlabeled, seed, gcols))
    tables["graph_membership"] = graph_membership(msgs, inp["fwd"], inp["sem"])
    summary["forwards"] = forwards_summary(tg, msgs, channels, len(inp["fwd"]))
    gs, gt = gold_overlap(gold, msgs, coding, inp["tg_gold_text"], inp["fwd"], inp["sem"], gcols)
    summary["gold"] = gs
    tables.update(gt)
    gb = gold[gcols["binary"]].astype(bool)
    summary["gold_narrative_share"] = float(gb.mean())
    gm = gold.loc[gb, gcols["level1"]].astype(str).value_counts()
    tables["gold_meta_counts"] = pd.DataFrame({"level_1": gm.index, "n_gold": gm.to_numpy(),
                                               "share_gold": gm.to_numpy() / gm.sum()})
    summary["edges"] = {"forward_edges": int(len(inp["fwd"])), "semantic_edges": int(len(inp["sem"])),
                        "semantic_min_similarity": float(inp["sem"]["similarity_score"].min())}
    return tables, summary, run_checks(tables, summary)


def run_checks(tables: dict, summary: dict) -> list[dict]:
    """hard=True: інваріанти (дублікати ключів, непарсовані дати), їх порушення ламає наступні етапи.
    hard=False: очікування з README і плану, розбіжність не блокує етап, а потрапляє у звіт як знахідка."""
    it, lb, rc = summary["integrity"], summary["labels"], summary["readme_counts"]
    pc = tables["per_channel"]
    pc_msgs = pc[pc["n_messages"] > 0]
    checks = [
        (True, "Немає дублікатів (channel_id, message_id) у messages", it["dup_keys_msgs"] == 0, f"дублікатів {it['dup_keys_msgs']}"),
        (True, "Немає дублікатів (channel_id, message_id) у telegram_messages", it["dup_keys_tg"] == 0, f"дублікатів {it['dup_keys_tg']}"),
        (True, "Усі дати розпарсені", it["unparsed_dates_msgs"] == 0 and it["unparsed_dates_tg"] == 0,
         f"непарсованих {it['unparsed_dates_msgs']} і {it['unparsed_dates_tg']}"),
        (False, f"Дати в межах {DATE_MIN} .. {DATE_MAX}", it["out_of_range_msgs"] == 0 and it["out_of_range_tg"] == 0,
         f"поза межами {it['out_of_range_msgs']} (messages), {it['out_of_range_tg']} (telegram_messages)"),
        (False, "Наративні повідомлення входять до розмічених", lb["narrative_not_labeled"] == 0, f"поза розміченими {lb['narrative_not_labeled']}"),
        (False, "sub_narrative_id у мітках відомі файлу кодування", lb["sub_ids_not_in_coding"] == 0, f"невідомих {lb['sub_ids_not_in_coding']}"),
        (False, "Кожен sub_narrative_id веде до того самого narrative_id, що й у мітці", lb["sub_to_narrative_match_share"] == 1.0,
         f"збіг {lb['sub_to_narrative_match_share']:.4f}"),
        (False, "Кількість повідомлень збігається з README", it["n_msgs"] == rc["messages"], f"{it['n_msgs']} проти {rc['messages']}"),
        (False, "Кількість каналів у таблиці каналів збігається з README", len(pc) == rc["channels"], f"{len(pc)} проти {rc['channels']}"),
        (False, "Кількості ребер збігаються з README",
         summary["edges"]["forward_edges"] == rc["forward_edges"] and summary["edges"]["semantic_edges"] == rc["semantic_edges"],
         f"forward {summary['edges']['forward_edges']}, semantic {summary['edges']['semantic_edges']}"),
        (False, "Частка наративних серед розмічених близька до золотої (різниця до 0,10)",
         abs(lb["narrative_share_of_labeled"] - summary["gold_narrative_share"]) <= 0.10,
         f"слабкі {lb['narrative_share_of_labeled']:.3f}, золота {summary['gold_narrative_share']:.3f}"),
        (False, "Мітки каналів у таблиці збігаються з перерахунком за повідомленнями", bool(pc_msgs["counts_match"].all()),
         f"збігається {int(pc_msgs['counts_match'].sum())} з {len(pc_msgs)} каналів"),
        (False, "Усі золоті повідомлення знайдені в messages", summary["gold"]["n_gold_in_graph_messages"] == summary["gold"]["n_gold"],
         f"{summary['gold']['n_gold_in_graph_messages']} з {summary['gold']['n_gold']}"),
    ]
    return [{"hard": h, "check": n, "passed": bool(ok), "detail": d} for h, n, ok, d in checks]


# ---------------------------------------------------------------- збереження

def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    raise TypeError(type(o))


def save_all(tables: dict, summary: dict, checks: list, out_dir: Path | str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tables.items():
        p = out_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")
    (out_dir / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths + [out_dir / "summary.json", out_dir / "checks.json"]


# ---------------------------------------------------------------- графіки

def make_figures(tables: dict, summary: dict, out_dir: Path | str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9})
    paths = []

    def save(fig, name):
        p = out_dir / name
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # 1. Повідомлення за місяцями
    mo = tables["monthly"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    colors = np.where(mo["partial_month"], "#b8b8b8", "#3b6fb6")
    ax[0].bar(mo["month"], mo["n"] / 1000, color=colors)
    ax[0].set_ylabel("повідомлень, тис.")
    ax[0].set_title("Усі повідомлення за місяцями (сірі: неповні місяці)")
    ax[1].bar(mo["month"], mo["n_labeled"] / 1000, color="#8aa9d6", label="розмічені")
    ax[1].bar(mo["month"], mo["n_narrative"] / 1000, color="#c0392b", label="з наративом")
    ax[1].set_ylabel("повідомлень, тис.")
    ax[1].set_title("Слабкі мітки за місяцями")
    ax[1].legend(frameon=False)
    for a in ax:
        a.tick_params(axis="x", rotation=70)
    save(fig, "01_monthly.png")

    # 2. Канали: обсяг і частка наративних
    pc = tables["per_channel"]
    top = pc[pc["n_messages"] > 0].head(25).iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    ax[0].barh(top["username"], top["n_messages"] / 1000, color="#3b6fb6")
    ax[0].set_xlabel("повідомлень, тис.")
    ax[0].set_title("25 найбільших каналів")
    ax[1].barh(top["username"], top["share_recomputed"].fillna(0), color="#c0392b")
    ax[1].set_xlabel("частка наративних серед розмічених")
    ax[1].set_xlim(0, 1)
    ax[1].set_title("Слабкі мітки")
    save(fig, "02_channels.png")

    # 3. Мова і частка наративних
    lx = tables["language_x_label"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    langs = sorted(lx["lang"].unique())
    w = 0.38
    for i, (src, col) in enumerate((("слабкі мітки (розмічені)", "#8aa9d6"), ("золота розмітка", "#c0392b"))):
        d = lx[lx["source"] == src].set_index("lang").reindex(langs)
        xs = np.arange(len(langs)) + (i - 0.5) * w
        ax.bar(xs, d["narrative_share"].fillna(0), w, label=src, color=col)
        for x, (_, r) in zip(xs, d.iterrows()):
            if pd.notna(r["n"]):
                ax.text(x, (r["narrative_share"] if pd.notna(r["narrative_share"]) else 0) + 0.01, f"n={int(r['n'])}",
                        ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(langs)))
    ax.set_xticklabels(langs)
    ax.set_ylabel("частка наративних")
    ax.set_ylim(0, 1.1)
    ax.set_title("Мова і частка наративних повідомлень")
    ax.legend(frameon=False)
    save(fig, "03_language.png")

    # 4. Мета-наративи: слабкі мітки проти золотої розмітки
    meta = tables["meta_weak"].copy()
    gold_meta = tables.get("gold_meta_counts")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    meta = meta.sort_values("n_weak")
    ax.barh(meta["level_1"].astype(str).str.slice(0, 48), meta["share_weak"], color="#8aa9d6", label="слабкі мітки")
    if gold_meta is not None:
        gm = gold_meta.set_index("level_1")["share_gold"]
        ax.scatter(meta["level_1"].map(gm).fillna(0), range(len(meta)), color="#c0392b", zorder=3, label="золота розмітка")
    ax.set_xlabel("частка серед наративних")
    ax.set_title("Розподіл мета-наративів")
    ax.legend(frameon=False)
    save(fig, "04_meta.png")

    # 5. Довжина тексту
    ls = tables["length_stats"].set_index("group")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    cols = ["розмічені, ненаративні", "розмічені, наративні", "нерозмічені"]
    ax.bar(range(len(cols)), [ls.loc[c, "median"] for c in cols], color=["#8aa9d6", "#c0392b", "#b8b8b8"])
    ax.errorbar(range(len(cols)), [ls.loc[c, "median"] for c in cols],
                yerr=[[ls.loc[c, "median"] - ls.loc[c, "p10"] for c in cols], [ls.loc[c, "p90"] - ls.loc[c, "median"] for c in cols]],
                fmt="none", color="k", capsize=4)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_ylabel("символів (медіана, діапазон p10-p90)")
    ax.set_title("Довжина тексту")
    save(fig, "05_length.png")

    # 6. Упевненість міток і слабкі мітки проти золотих
    cf = tables["label_confidence"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar(cf["label_confidence"].astype(str), cf["n_not_narrative"], label="без наративу", color="#8aa9d6")
    ax[0].bar(cf["label_confidence"].astype(str), cf["n_narrative"], bottom=cf["n_not_narrative"], label="з наративом", color="#c0392b")
    ax[0].set_xlabel("label_confidence")
    ax[0].set_ylabel("розмічених повідомлень")
    ax[0].set_title("Упевненість слабких міток")
    ax[0].tick_params(axis="x", rotation=60)
    ax[0].legend(frameon=False)
    cm = tables.get("weak_vs_gold_confusion")
    if cm is not None:
        arr = cm.iloc[:, 1:].to_numpy(dtype=float)
        ax[1].imshow(arr, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax[1].text(j, i, f"{int(arr[i, j])}", ha="center", va="center", color="k")
        ax[1].set_xticks([0, 1])
        ax[1].set_xticklabels(["слабка: ні", "слабка: так"])
        ax[1].set_yticks([0, 1])
        ax[1].set_yticklabels(["золота: ні", "золота: так"])
        ax[1].set_title("Слабкі мітки проти золотих (де є обидві)")
    save(fig, "06_confidence_and_agreement.png")
    return paths
