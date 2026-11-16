"""S6: оцінка текстових моделей на золотих повідомленнях за протоколами P1-P3.

Вхід: прогнози з Colab (`protocol, fold, model, seed, channel_id, message_id, p`), таблиця золотих (S5), розмітка явний/неявний (S5),
спліти (для компараторів, що рахуються з train). Золоті мітки використовуються лише тут, у функціях оцінки.

Правила (преєстрація, розділи 3, 4, 7):
- поріг 0,5 заздалегідь; для моделей з кількома seed ймовірності усереднюються по seed, метрика рахується від середнього;
- 95% ДІ: бутстреп по повідомленнях, 1000 вибірок, seed 42; чутливість: бутстреп по каналах;
- «найкраща текстова» обирається за валідаційним macro-F1 на слабких мітках, не за золотими;
- обов'язкові компаратори: мажоритарний клас train, правило «російська = наратив», канальний пріор лише з train.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from narr.evaluate import binary_metrics, paired_bootstrap_diff
from narr.splits import channel_prior, make_key

TEXT_MODELS = ("tfidf", "bge_lr", "xlmr")
CONTROLS = ("tfidf_shuf", "bge_lr_shuf", "xlmr_shuf")
COMPARATORS = ("majority", "lang_rule", "channel_prior")
PROTOCOLS = ("P1", "P2", "P3")
SLICES = ("all", "ru", "uk")
EXPL_DEFS = ("a", "narrow", "wide", "b")


# ---------------------------------------------------------------- золота таблиця і підмножини

def make_gold_table(membership: pd.DataFrame, lab: pd.DataFrame, auto_b: pd.DataFrame) -> pd.DataFrame:
    """Золоті: key, channel_id, message_id, y, lang, fold, in_p2_test, expl (explicit/implicit/unclear, порожньо для ненаративних),
    border (сумнівний рядок розмітки), imp_auto (неявний за визначенням (б))."""
    g = membership[["key", "channel_id", "message_id", "y", "lang", "fold", "in_p2_test"]].copy()
    lab = lab.assign(key=make_key(lab["channel_id"], lab["message_id"]))
    expl = lab.set_index("key")["explicitness"].fillna("").astype(str).str.strip()
    border = lab.set_index("key")["comment"].fillna("").astype(str).str.contains("borderline")
    g["expl"] = g["key"].map(expl).fillna("")
    g["border"] = g["key"].map(border).fillna(False).astype(bool)
    g["imp_auto"] = g["key"].map(auto_b.set_index("key")["implicit_auto"]).fillna(False).astype(bool)
    bad = g[(g["y"] == 1) & ~g["expl"].isin(["explicit", "implicit", "unclear"])]
    if len(bad):
        raise ValueError(f"{len(bad)} наративних золотих без розмітки явний/неявний")
    return g.sort_values("key").reset_index(drop=True)


def explicitness_masks(g: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Для кожного визначення (а, вузьке, широке, б): (неявні наративні, явні наративні). Рядки `unclear` не входять ні туди, ні туди."""
    nar = g["y"].to_numpy() == 1
    imp, exp = g["expl"].eq("implicit").to_numpy(), g["expl"].eq("explicit").to_numpy()
    bd, auto = g["border"].to_numpy(), g["imp_auto"].to_numpy()
    return {"a": (imp, exp), "narrow": (imp & ~bd, exp), "wide": (imp | (exp & bd), exp & ~bd),
            "b": (nar & auto, nar & ~auto)}


def subset_masks(g: pd.DataFrame) -> dict[str, np.ndarray]:
    """Підмножини для H1: «неявні» = усі ненаративні + неявні наративні, «явні» = усі ненаративні + явні наративні."""
    neg = g["y"].to_numpy() == 0
    out = {"all": np.ones(len(g), dtype=bool)}
    for d, (imp, exp) in explicitness_masks(g).items():
        out[f"implicit_{d}"] = neg | imp
        out[f"explicit_{d}"] = neg | exp
    return out


def slice_mask(g: pd.DataFrame, name: str) -> np.ndarray:
    if name == "all":
        return np.ones(len(g), dtype=bool)
    return g["lang"].eq(name).to_numpy()


# ---------------------------------------------------------------- прогнози

def check_predictions(pred: pd.DataFrame, g: pd.DataFrame) -> dict:
    """Структурні перевірки прогнозів: дублікати, покриття золотих за протоколом, діапазон ймовірностей, невідомі ключі."""
    key = make_key(pred["channel_id"], pred["message_id"])
    gk = set(g["key"].tolist())
    p2 = set(g.loc[g["in_p2_test"].astype(bool), "key"].tolist())
    dup = pred.assign(key=key).duplicated(["protocol", "model", "seed", "key"]).sum()
    unknown = int((~pd.Series(key).isin(gk)).sum())
    cover = {}
    for (pr, m, sd), d in pred.assign(key=key).groupby(["protocol", "model", "seed"]):
        need = p2 if pr == "P2" else gk
        cover[f"{m}/{pr}/s{sd}"] = bool(set(d["key"]) == need)
    return {"n_duplicates": int(dup), "n_unknown_keys": unknown, "all_covered": bool(all(cover.values())) if cover else False,
            "not_covered": [k for k, v in cover.items() if not v],
            "p_in_range": bool(pred["p"].between(0, 1).all()), "p_finite": bool(np.isfinite(pred["p"]).all())}


def mean_over_seeds(pred: pd.DataFrame) -> pd.DataFrame:
    """Одна ймовірність на (протокол, модель, ключ): середнє по seed (для P3 фолди не перетинаються, тож середнє по одному значенню)."""
    d = pred.assign(key=make_key(pred["channel_id"], pred["message_id"]))
    out = d.groupby(["protocol", "model", "key"]).agg(p=("p", "mean"), n_seeds=("seed", "nunique")).reset_index()
    return out


def comparator_frame(g: pd.DataFrame, splits: list[dict]) -> pd.DataFrame:
    """Компаратори на тестових ключах кожного протоколу.
    majority: клас, що переважає в train за слабкими мітками; lang_rule: російська = наратив (мова за S5);
    channel_prior: згладжена частка наративних слабких міток каналу з train, для каналу без train глобальна частка train."""
    lang = g.set_index("key")["lang"]
    rows = []
    for s in splits:
        tr, te = s["train"], s["test"]
        k = te["key"].to_numpy()
        prev = float(tr["weak_narrative"].mean())
        pri = channel_prior(tr)
        cp = te["channel_id"].map(pri).fillna(prev).to_numpy()
        rows.append(pd.DataFrame({"protocol": s["protocol"], "model": "majority", "key": k, "p": float(prev >= 0.5)}))
        rows.append(pd.DataFrame({"protocol": s["protocol"], "model": "lang_rule", "key": k, "p": lang.loc[k].eq("ru").astype(float).to_numpy()}))
        rows.append(pd.DataFrame({"protocol": s["protocol"], "model": "channel_prior", "key": k, "p": cp}))
    d = pd.concat(rows, ignore_index=True)
    d["n_seeds"] = 1
    return d


# ---------------------------------------------------------------- метрики з ДІ

def _arrays(pm: pd.DataFrame, g: pd.DataFrame, protocol: str, model: str) -> pd.DataFrame | None:
    d = pm[(pm["protocol"] == protocol) & (pm["model"] == model)]
    if d.empty:
        return None
    return g.merge(d[["key", "p"]], on="key", how="inner")


def metrics_table(pm: pd.DataFrame, g: pd.DataFrame, protocols, models, slices, subsets, n_boot: int = 1000, seed: int = 42,
                  min_class: int = 5) -> pd.DataFrame:
    """Метрики з 95% ДІ (бутстреп по повідомленнях) для кожної комбінації (протокол, модель, зріз, підмножина)."""
    from narr.evaluate import bootstrap_ci

    rows = []
    for pr in protocols:
        for m in models:
            d = _arrays(pm, g, pr, m)
            if d is None:
                continue
            sm = subset_masks(d)
            for sl in slices:
                for sb in subsets:
                    mask = slice_mask(d, sl) & sm[sb]
                    y, p = d["y"].to_numpy()[mask], d["p"].to_numpy()[mask]
                    n_pos = int(y.sum())
                    if n_pos < min_class or len(y) - n_pos < min_class:
                        continue
                    t = bootstrap_ci(y, p[None, :], binary_metrics, n_boot=n_boot, seed=seed).drop(columns=["sd_repeats"])
                    t.insert(0, "subset", sb), t.insert(0, "slice", sl), t.insert(0, "model", m), t.insert(0, "protocol", pr)
                    t["n"], t["n_pos"] = int(len(y)), n_pos
                    rows.append(t)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cluster_bootstrap(y: np.ndarray, p: np.ndarray, channels: np.ndarray, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Чутливість: бутстреп по каналах (вибираємо канали з поверненням, беремо всі їхні повідомлення)."""
    uniq = np.unique(channels)
    groups = [np.flatnonzero(channels == c) for c in uniq]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([groups[i] for i in pick])
        rows.append(binary_metrics(y[idx], p[idx]))
    b = pd.DataFrame(rows)
    point = binary_metrics(y, p)
    return pd.DataFrame({"point": pd.Series(point), "ci_lo": b.quantile(0.025), "ci_hi": b.quantile(0.975)}).rename_axis("metric").reset_index()


def cluster_table(pm: pd.DataFrame, g: pd.DataFrame, protocol: str, models, slices, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    rows = []
    for m in models:
        d = _arrays(pm, g, protocol, m)
        if d is None:
            continue
        for sl in slices:
            mask = slice_mask(d, sl)
            t = cluster_bootstrap(d["y"].to_numpy()[mask], d["p"].to_numpy()[mask], d["channel_id"].to_numpy()[mask], n_boot, seed)
            t.insert(0, "slice", sl), t.insert(0, "model", m), t.insert(0, "protocol", protocol)
            t["n_channels"] = int(d.loc[mask, "channel_id"].nunique())
            rows.append(t)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def paired_table(pm: pd.DataFrame, g: pd.DataFrame, protocol: str, pairs, slices, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Парні різниці A мінус B (парний бутстреп на тих самих повідомленнях)."""
    rows = []
    for a, b in pairs:
        da, db = _arrays(pm, g, protocol, a), _arrays(pm, g, protocol, b)
        if da is None or db is None:
            continue
        d = da.merge(db[["key", "p"]], on="key", suffixes=("", "_b"))
        for sl in slices:
            mask = slice_mask(d, sl)
            y = d["y"].to_numpy()[mask]
            if y.sum() < 5 or len(y) - y.sum() < 5:
                continue
            t = paired_bootstrap_diff(y, d["p"].to_numpy()[mask][None, :], d["p_b"].to_numpy()[mask][None, :], binary_metrics, n_boot=n_boot, seed=seed)
            t.insert(0, "slice", sl), t.insert(0, "B", b), t.insert(0, "A", a), t.insert(0, "protocol", protocol)
            t["n"] = int(len(y))
            rows.append(t)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def recall_by_explicitness(pm: pd.DataFrame, g: pd.DataFrame, protocol: str, models, slices, n_boot: int = 1000, seed: int = 42,
                           thr: float = 0.5) -> pd.DataFrame:
    """Частка знайдених (recall) серед явних і неявних наративних повідомлень, різниця явні мінус неявні, бутстреп по повідомленнях
    усередині кожної групи. Визначення (б) збудовано з похибок TF-IDF, тому для TF-IDF воно упереджене за конструкцією."""
    rows = []
    for m in models:
        d = _arrays(pm, g, protocol, m)
        if d is None:
            continue
        hit = (d["p"].to_numpy() >= thr)
        for df_, (imp, exp) in explicitness_masks(d).items():
            for sl in slices:
                s = slice_mask(d, sl)
                hi, he = hit[imp & s], hit[exp & s]
                if len(hi) == 0 or len(he) == 0:
                    continue
                rng = np.random.default_rng(seed)
                bi = np.array([hi[rng.integers(0, len(hi), len(hi))].mean() for _ in range(n_boot)])
                be = np.array([he[rng.integers(0, len(he), len(he))].mean() for _ in range(n_boot)])
                diff = be - bi
                rows.append({"protocol": protocol, "model": m, "slice": sl, "definition": df_, "n_implicit": int(len(hi)), "n_explicit": int(len(he)),
                             "recall_implicit": float(hi.mean()), "ri_lo": float(np.quantile(bi, 0.025)), "ri_hi": float(np.quantile(bi, 0.975)),
                             "recall_explicit": float(he.mean()), "re_lo": float(np.quantile(be, 0.025)), "re_hi": float(np.quantile(be, 0.975)),
                             "diff_explicit_minus_implicit": float(he.mean() - hi.mean()),
                             "diff_lo": float(np.quantile(diff, 0.025)), "diff_hi": float(np.quantile(diff, 0.975))})
    return pd.DataFrame(rows)


def seed_table(pred: pd.DataFrame, g: pd.DataFrame, protocols=PROTOCOLS, slices=("all", "ru")) -> pd.DataFrame:
    """Метрики кожного seed окремо (описово): розкид між seed."""
    d = pred.assign(key=make_key(pred["channel_id"], pred["message_id"]))
    rows = []
    for (pr, m, sd), part in d.groupby(["protocol", "model", "seed"]):
        if pr not in protocols:
            continue
        x = g.merge(part[["key", "p"]], on="key")
        for sl in slices:
            mk = slice_mask(x, sl)
            r = binary_metrics(x["y"].to_numpy()[mk], x["p"].to_numpy()[mk])
            rows.append({"protocol": pr, "model": m, "seed": int(sd), "slice": sl, "macro_f1": r["macro_f1"], "mcc": r["mcc"], "pr_auc": r["pr_auc"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- вибір найкращої текстової за валідацією

def valid_table(units: list[dict]) -> pd.DataFrame:
    rows = []
    for u in units:
        rows.append({"protocol": u["protocol"], "model": u["model"], "seed": u["seed"], "valid_macro_f1": u["valid_macro_f1"],
                     "C": u.get("C", np.nan), "best_epoch": np.mean([s["best_epoch"] for s in u["splits"]]) if "best_epoch" in u["splits"][0] else np.nan,
                     "seconds": u.get("seconds", np.nan)})
    return pd.DataFrame(rows)


def select_best(vt: pd.DataFrame, protocol: str, candidates=TEXT_MODELS) -> pd.DataFrame:
    """Середній валідаційний macro-F1 (слабкі мітки) по seed для кожної кандидатної моделі протоколу; selected = найвищий. Золоті не використовуються."""
    d = vt[(vt["protocol"] == protocol) & vt["model"].isin(candidates)]
    t = d.groupby("model", as_index=False).agg(valid_macro_f1=("valid_macro_f1", "mean"), n_units=("seed", "size"))
    t["selected"] = t["valid_macro_f1"].eq(t["valid_macro_f1"].max()) if len(t) else False
    t.insert(0, "protocol", protocol)
    return t.sort_values("valid_macro_f1", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- перевірки етапу

def run_checks(main: pd.DataFrame, pcheck: dict, vt: pd.DataFrame, seeds: pd.DataFrame, integrity: dict | None, models_present) -> list[dict]:
    """Список перевірок {check, hard, passed, detail}."""
    from narr.report import fmt

    out = []

    def add(name, hard, ok, detail):
        out.append({"check": name, "hard": bool(hard), "passed": bool(ok), "detail": detail})

    add("Прогнози: без дублікатів, усі ключі золоті, ймовірності в [0; 1], покриття тесту повне", True,
        pcheck["n_duplicates"] == 0 and pcheck["n_unknown_keys"] == 0 and pcheck["p_in_range"] and pcheck["p_finite"] and pcheck["all_covered"],
        f"дублікатів {pcheck['n_duplicates']}, невідомих {pcheck['n_unknown_keys']}, без повного покриття {pcheck['not_covered'][:5]}")
    if integrity is not None:
        add("Тексти збігаються з хешами індексу ембедингів S4 (ембединг і текст належать одному повідомленню)", True, integrity.get("ok", False),
            f"ключів {integrity.get('n_keys')}, розбіжностей {integrity.get('n_mismatch')}")

    def pt(model, protocol, sl, metric, col="point", subset="all"):
        r = main[(main["model"] == model) & (main["protocol"] == protocol) & (main["slice"] == sl) & (main["subset"] == subset) & (main["metric"] == metric)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    maj = pt("majority", "P1", "all", "macro_f1")
    for m in [x for x in TEXT_MODELS if x in models_present]:
        v = pt(m, "P1", "all", "macro_f1")
        add(f"{m}: macro-F1 на всіх 412 (P1) вище за мажоритарний клас", True, v > maj, f"{fmt(v)} проти {fmt(maj)}")
    for c in [x for x in CONTROLS if x in models_present]:
        lo, hi = pt(c, "P1", "all", "mcc", "ci_lo"), pt(c, "P1", "all", "mcc", "ci_hi")
        add(f"{c}: з перемішаними мітками MCC на золотих не відрізняється від нуля (ДІ включає 0)", True, lo <= 0 <= hi, f"[{fmt(lo)}; {fmt(hi)}]")
    ladder = [m for m in ("tfidf", "bge_lr", "xlmr") if m in models_present]
    for a, b in zip(ladder[1:], ladder[:-1]):
        va, vb = pt(a, "P1", "all", "macro_f1"), pt(b, "P1", "all", "macro_f1")
        add(f"Складніша модель не гірша за простішу: {a} не нижче {b} за macro-F1 на всіх 412 (P1)", False, va >= vb, f"{fmt(va)} проти {fmt(vb)}")
    for m in [x for x in TEXT_MODELS if x in models_present]:
        r = vt[(vt["protocol"] == "P1") & (vt["model"] == m)]
        if len(r):
            add(f"{m}: валідаційний macro-F1 на слабких мітках (P1) вище 0,6", False, bool(r["valid_macro_f1"].mean() > 0.6), fmt(float(r["valid_macro_f1"].mean())))
    sd = seeds[(seeds["model"] == "xlmr") & (seeds["protocol"] == "P1") & (seeds["slice"] == "all")]
    if len(sd) > 1:
        add("XLM-R: розкид macro-F1 між seed (P1, всі 412) не більший за 0,03 (стандартне відхилення)", False, bool(sd["macro_f1"].std(ddof=1) <= 0.03),
            f"sd {fmt(float(sd['macro_f1'].std(ddof=1)))}, seed {len(sd)}")
    for m in [x for x in ("tfidf", "bge_lr") if x in models_present]:
        r = vt[(vt["model"] == m) & (vt["protocol"] == "P1")]
        if len(r) and not np.isnan(r["C"].iloc[0]):
            grid = (2.0, 32.0) if m == "tfidf" else (3.0, 300.0)
            add(f"{m}: підібране C (P1) не на краю сітки (інакше сітку треба розширити)", False, float(r["C"].iloc[0]) not in grid, f"C = {r['C'].iloc[0]:g}")
    return out
