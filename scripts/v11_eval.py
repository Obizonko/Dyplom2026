"""Оцінка на золотих за преєстрацією 1.1 (розділ 11.1): S6 (TF-IDF, bge-m3 + LogReg) і S7 (LightGBM) з вагами пулу.

Запуск:  python -I scripts/v11_eval.py [--n-boot 1000]
Читає results/v11/06_text і results/v11/07_lgbm (а для порівняння з версією 1 ще results/06_text_models і results/quick/07_simple_hybrid).
Пише results/v11/eval/*.csv, checks.json, run.json і results/v11/eval/tables.md (таблиці для звіту).

Правила: поріг 0,5; основна метрика macro-F1; ДІ бутстрепом по повідомленнях і кластерним по каналах (1000, seed 42);
«найкраща текстова» серед tfidf, bge_lr, lgbm_emb і «найкраща гібридна» серед гібридів обираються за зваженим валідаційним macro-F1 (усі моделі мають
однакову валідацію v1.1), не за золотими; критерій H1 виконано лише за нижньої межі ДІ > 0 за обома бутстрепами.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narr import text_eval as te, textmodels as tm  # noqa: E402
from narr.evaluate import binary_metrics  # noqa: E402
from narr.io import ROOT, write_run_json  # noqa: E402
from narr.splits import make_key  # noqa: E402

TEXT = ("tfidf", "bge_lr", "lgbm_emb")
HYBRID = ("lgbm_both",)
GRAPH = ("lgbm_graph", "lgbm_graph_noprior")
COMP = ("majority", "lang_rule", "channel_prior")
SHUF = ("tfidf_shuf", "bge_lr_shuf", "lgbm_graph_shuf", "lgbm_both_shuf")
PROTOCOLS = ("P1", "P2", "P3")
METRICS = ("macro_f1", "mcc", "roc_auc", "pr_auc", "specificity", "recall")


def metrics(y, p, thr: float = 0.5) -> dict:
    """Швидкі метрики (numpy) для бутстрепу: macro-F1, MCC, ROC-AUC (Манна-Уітні), PR-AUC (average precision як у sklearn), специфічність, повнота."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = p >= thr
    tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum())
    fn = float((~pred & (y == 1)).sum()); tn = float((~pred & (y == 0)).sum())
    f1p = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1n = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    npos, nneg = int(y.sum()), int(len(y) - y.sum())
    out = {"macro_f1": (f1p + f1n) / 2, "mcc": float((tp * tn - fp * fn) / den) if den > 0 else 0.0,
           "specificity": tn / (tn + fp) if (tn + fp) else float("nan"), "recall": tp / (tp + fn) if (tp + fn) else float("nan")}
    if npos == 0 or nneg == 0:
        out["roc_auc"] = out["pr_auc"] = float("nan")
        return out
    r = rankdata(p)
    out["roc_auc"] = float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    o = np.argsort(-p, kind="mergesort")
    ps, ys = p[o], y[o]
    last = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]  # останній індекс кожного порогу
    ctp = np.cumsum(ys)[last]
    prec = ctp / (last + 1)
    rec = ctp / npos
    out["pr_auc"] = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
    return out


class Boot:
    """Однакові індекси бутстрепу для всіх моделей на тій самій множині повідомлень: по повідомленнях і по каналах."""

    def __init__(self, channels: np.ndarray, n_boot: int, seed: int):
        rng = np.random.default_rng(seed)
        n = len(channels)
        uc = np.unique(channels)
        by = [np.flatnonzero(channels == c) for c in uc]
        self.idx = {"message": [rng.integers(0, n, n) for _ in range(n_boot)],
                    "channel": [np.concatenate([by[i] for i in rng.integers(0, len(uc), len(uc))]) for _ in range(n_boot)]}

    def ci(self, fn, *arrays):
        out = {}
        for kind, ids in self.idx.items():
            vals = [fn(*[a[ix] for a in arrays]) for ix in ids]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            out[kind] = (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))
        return out

    def ci_all(self, y, p):
        """ДІ для всіх METRICS одним проходом: {метрика: {kind: (lo, hi)}}."""
        out = {m: {} for m in METRICS}
        for kind, ids in self.idx.items():
            vals = pd.DataFrame([metrics(y[ix], p[ix]) for ix in ids if 0 < y[ix].sum() < len(ix)])
            for m in METRICS:
                out[m][kind] = (float(vals[m].quantile(0.025)), float(vals[m].quantile(0.975)))
        return out


def safe(metric):
    def f(y, p):
        if y.sum() == 0 or y.sum() == len(y):
            return None
        return metrics(y, p)[metric]
    return f


def load_preds(dirs):
    parts, units = [], []
    for d in dirs:
        if (d / "predictions").exists():
            parts += [pd.read_parquet(p) for p in sorted((d / "predictions").glob("*.parquet"))]
            units += [json.loads(p.read_text(encoding="utf-8")) for p in sorted((d / "units").glob("*.json"))]
    pred = pd.concat(parts, ignore_index=True)
    pred["key"] = make_key(pred["channel_id"], pred["message_id"])
    return pred, units


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "v11" / "eval"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s5 = ROOT / "results" / "05_protocol"
    g = te.make_gold_table(pd.read_csv(s5 / "gold_membership.csv"), pd.read_csv(s5 / "implicit_labeling.csv"), pd.read_csv(s5 / "implicit_auto_b.csv"))
    gi = g.set_index("key")
    pred, units = load_preds([ROOT / "results" / "v11" / "06_text", ROOT / "results" / "v11" / "07_lgbm"])
    ctrl = pred[pred["model"].isin(SHUF)]
    main_pred = pred[~pred["model"].isin(SHUF) & ~pred["model"].str.endswith("_rewired")]
    pm = main_pred.groupby(["protocol", "model", "key"], as_index=False).agg(p=("p", "mean"), n_seeds=("seed", "nunique"))
    splits = tm.load_splits(ROOT / "splits", PROTOCOLS)
    pm = pd.concat([pm, te.comparator_frame(g, splits)], ignore_index=True)
    rew = pred[pred["model"] == "lgbm_both_rewired"][["protocol", "model", "key", "p"]]
    pm = pd.concat([pm, rew], ignore_index=True)
    # версія 1 для порівняння (ті самі моделі без ваг пулу)
    old, _ = load_preds([ROOT / "results" / "06_text_models", ROOT / "results" / "quick" / "07_simple_hybrid"])
    old = old[old["model"].isin(("tfidf", "bge_lr", "lgbm_emb", "lgbm_both", "lgbm_graph"))]
    old = old.groupby(["protocol", "model", "key"], as_index=False)["p"].mean()
    old["model"] = old["model"] + "_v1"
    pm = pd.concat([pm, old], ignore_index=True)

    # ---------------- основна таблиця
    rows = []
    boots = {}
    for pr in PROTOCOLS:
        for sl in ("all", "ru"):
            keys = g.loc[(g["in_p2_test"].astype(bool) if pr == "P2" else np.ones(len(g), bool)) & (te.slice_mask(g, sl)), "key"].to_numpy()
            boots[(pr, sl)] = (keys, Boot(gi.loc[keys, "channel_id"].to_numpy(), args.n_boot, args.seed))
    models = [m for m in (*COMP, *TEXT, *GRAPH, *HYBRID, "lgbm_both_rewired", *[x + "_v1" for x in ("tfidf", "bge_lr", "lgbm_emb", "lgbm_graph", "lgbm_both")])
              if m in set(pm["model"])]
    for (pr, sl), (keys, bt) in boots.items():
        y = gi.loc[keys, "y"].to_numpy()
        for m in models:
            d = pm[(pm["protocol"] == pr) & (pm["model"] == m)].set_index("key")
            if d.empty or not set(keys) <= set(d.index):
                continue
            p = d.loc[keys, "p"].to_numpy()
            pt = metrics(y, p)
            cis = bt.ci_all(y, p)
            for met in METRICS:
                ci = cis[met]
                rows.append({"protocol": pr, "slice": sl, "model": m, "metric": met, "point": pt[met], "msg_lo": ci["message"][0], "msg_hi": ci["message"][1],
                             "ch_lo": ci["channel"][0], "ch_hi": ci["channel"][1], "n": int(len(y)), "n_pos": int(y.sum()),
                             "n_seeds": int(d["n_seeds"].max()) if "n_seeds" in d and d["n_seeds"].notna().any() else 1})
        print(f"метрики {pr} {sl}: {time.time() - t0:.0f} с", flush=True)
    main_t = pd.DataFrame(rows)
    main_t.to_csv(out / "metrics.csv", index=False)

    # ---------------- вибір найкращої текстової і гібридної за валідацією (зваженою, v1.1)
    vt = pd.DataFrame([{"protocol": u["protocol"], "model": u["model"], "seed": u["seed"], "valid_macro_f1": u["valid_macro_f1"]} for u in units
                       if not u["model"].endswith(("_shuf", "_rewired"))])
    sel = vt[vt["protocol"] == "P1"].groupby("model", as_index=False)["valid_macro_f1"].mean()
    best_text = sel[sel["model"].isin(TEXT)].sort_values("valid_macro_f1", ascending=False)["model"].iloc[0]
    best_hyb = sel[sel["model"].isin(HYBRID)].sort_values("valid_macro_f1", ascending=False)["model"].iloc[0]
    sel["role"] = np.where(sel["model"] == best_text, "best_text", np.where(sel["model"] == best_hyb, "best_hybrid", ""))
    sel.to_csv(out / "valid_selection.csv", index=False)
    print("Найкраща текстова:", best_text, "| найкраща гібридна:", best_hyb, flush=True)

    # ---------------- парні різниці
    pairs = [(best_hyb, best_text), ("lgbm_both", "lgbm_emb"), ("lgbm_emb", "bge_lr"), ("bge_lr", "tfidf"), ("lgbm_emb", "tfidf"),
             ("tfidf", "lang_rule"), ("bge_lr", "lang_rule"), ("lgbm_emb", "lang_rule"), ("lgbm_both", "lang_rule"),
             ("tfidf", "tfidf_v1"), ("bge_lr", "bge_lr_v1"), ("lgbm_emb", "lgbm_emb_v1"), ("lgbm_both", "lgbm_both_v1"), ("lgbm_both", "lgbm_both_rewired")]
    prow = []
    for pr in PROTOCOLS:
        for sl in ("all", "ru"):
            keys, bt = boots[(pr, sl)]
            y = gi.loc[keys, "y"].to_numpy()
            for a, b in dict.fromkeys(pairs):
                da, db = (pm[(pm["protocol"] == pr) & (pm["model"] == x)].set_index("key") for x in (a, b))
                if da.empty or db.empty or not (set(keys) <= set(da.index) and set(keys) <= set(db.index)):
                    continue
                pa, pb = da.loc[keys, "p"].to_numpy(), db.loc[keys, "p"].to_numpy()
                for met in ("macro_f1", "mcc", "roc_auc"):
                    f = safe(met)
                    diff = lambda yy, xa, xb, f=f: (None if f(yy, xa) is None else f(yy, xa) - f(yy, xb))
                    ci = bt.ci(diff, y, pa, pb)
                    prow.append({"protocol": pr, "slice": sl, "A": a, "B": b, "metric": met, "diff": diff(y, pa, pb), "msg_lo": ci["message"][0], "msg_hi": ci["message"][1],
                                 "ch_lo": ci["channel"][0], "ch_hi": ci["channel"][1], "n": int(len(y))})
    paired = pd.DataFrame(prow)
    paired.to_csv(out / "paired.csv", index=False)

    # ---------------- H1 за преєстрацією (P1, ru-зріз, підмножини неявних і явних, різниця різниць)
    h1 = []
    keys, bt = boots[("P1", "ru")]
    gg = gi.loc[keys].reset_index()
    sm = te.subset_masks(gg)
    y = gg["y"].to_numpy()
    pa = pm[(pm["protocol"] == "P1") & (pm["model"] == best_hyb)].set_index("key").loc[keys, "p"].to_numpy()
    pb = pm[(pm["protocol"] == "P1") & (pm["model"] == best_text)].set_index("key").loc[keys, "p"].to_numpy()
    for defn in ("a", "narrow", "wide", "b"):
        mi, me = sm[f"implicit_{defn}"], sm[f"explicit_{defn}"]
        n_imp = int((mi & (y == 1)).sum())

        def d_sub(yy, xa, xb, mm):
            if yy[mm].sum() == 0 or yy[mm].sum() == mm.sum():
                return None
            return metrics(yy[mm], xa[mm])["macro_f1"] - metrics(yy[mm], xb[mm])["macro_f1"]

        di = lambda yy, xa, xb, mm: d_sub(yy, xa, xb, mm)
        ci_i = bt.ci(di, y, pa, pb, mi)
        ci_e = bt.ci(di, y, pa, pb, me)
        dod = lambda yy, xa, xb, m1, m2: (None if d_sub(yy, xa, xb, m1) is None or d_sub(yy, xa, xb, m2) is None
                                          else d_sub(yy, xa, xb, m1) - d_sub(yy, xa, xb, m2))
        ci_d = bt.ci(dod, y, pa, pb, mi, me)
        enough = n_imp >= 30
        met = enough and ci_i["message"][0] > 0 and ci_i["channel"][0] > 0
        h1.append({"A": best_hyb, "B": best_text, "definition": defn, "n_implicit_pos": n_imp, "n_explicit_pos": int((me & (y == 1)).sum()),
                   "diff_implicit": d_sub(y, pa, pb, mi), "imp_msg_lo": ci_i["message"][0], "imp_msg_hi": ci_i["message"][1], "imp_ch_lo": ci_i["channel"][0], "imp_ch_hi": ci_i["channel"][1],
                   "diff_explicit": d_sub(y, pa, pb, me), "exp_msg_lo": ci_e["message"][0], "exp_msg_hi": ci_e["message"][1],
                   "diff_of_diffs": dod(y, pa, pb, mi, me), "dod_msg_lo": ci_d["message"][0], "dod_msg_hi": ci_d["message"][1], "dod_ch_lo": ci_d["channel"][0], "dod_ch_hi": ci_d["channel"][1],
                   "verdict": "недостатньо даних" if not enough else ("критерій виконано (до Холма)" if met else "критерій не виконано")})
    h1 = pd.DataFrame(h1)
    h1.to_csv(out / "h1.csv", index=False)

    # ---------------- перестановочні контролі (11.1, п. 6)
    crow = []
    yk = gi.loc[g["key"], "y"].to_numpy()
    for m, d in ctrl[ctrl["protocol"] == "P1"].groupby("model"):
        for sd, part in d.groupby("seed"):
            q = part.set_index("key").loc[g["key"], "p"].to_numpy()
            crow.append({"model": m, "perm": int(sd), "mcc": binary_metrics(yk, q)["mcc"], "macro_f1": binary_metrics(yk, q)["macro_f1"]})
    cdf = pd.DataFrame(crow)
    cdf.to_csv(out / "permutations.csv", index=False)
    csum = cdf.groupby("model")["mcc"].agg(n="size", mean="mean", q025=lambda s: s.quantile(0.025), q975=lambda s: s.quantile(0.975),
                                           mean_abs=lambda s: s.abs().mean()).reset_index() if len(cdf) else pd.DataFrame()
    if len(csum):
        csum["passed"] = (csum["q025"] <= 0) & (0 <= csum["q975"]) & (csum["mean_abs"] < 0.1)
    csum.to_csv(out / "permutations_summary.csv", index=False)

    # ---------------- перевірки
    def pt(m, pr="P1", sl="all", met="macro_f1"):
        r = main_t[(main_t["model"] == m) & (main_t["protocol"] == pr) & (main_t["slice"] == sl) & (main_t["metric"] == met)]
        return float(r["point"].iloc[0]) if len(r) else float("nan")

    checks = []
    for m in (*TEXT, *HYBRID):
        checks.append({"check": f"{m}: macro-F1 P1 (усі 412) вище за мажоритарний", "hard": True, "passed": bool(pt(m) > pt("majority")),
                       "detail": f"{pt(m):.3f} проти {pt('majority'):.3f}"})
    for _, r in csum.iterrows():
        checks.append({"check": f"{r['model']}: перестановочний контроль ({int(r['n'])} перестановок)", "hard": True, "passed": bool(r["passed"]),
                       "detail": f"MCC середнє {r['mean']:.3f}, квантилі [{r['q025']:.3f}; {r['q975']:.3f}], середнє |MCC| {r['mean_abs']:.3f}"})
    for m in GRAPH:
        checks.append({"check": f"{m}: MCC на всіх 412 не вище 0,8 (поріг витоку)", "hard": True, "passed": bool(pt(m, met="mcc") <= 0.8), "detail": f"{pt(m, met='mcc'):.3f}"})
    sd_rows = []
    for (pr, m), d in main_pred[main_pred["model"].isin((*GRAPH, *HYBRID, "lgbm_emb"))].groupby(["protocol", "model"]):
        for sd, part in d.groupby("seed"):
            q = part.set_index("key").loc[g["key"] if pr != "P2" else g.loc[g["in_p2_test"].astype(bool), "key"], "p"].to_numpy()
            yy = gi.loc[g["key"] if pr != "P2" else g.loc[g["in_p2_test"].astype(bool), "key"], "y"].to_numpy()
            sd_rows.append({"protocol": pr, "model": m, "seed": int(sd), "macro_f1": binary_metrics(yy, q)["macro_f1"]})
    sdt = pd.DataFrame(sd_rows)
    sdt.to_csv(out / "seeds.csv", index=False)
    for (pr, m), d in sdt.groupby(["protocol", "model"]):
        if pr == "P1":
            checks.append({"check": f"{m}: розкид macro-F1 між seed (P1) не більше 0,03", "hard": False, "passed": bool(d["macro_f1"].std(ddof=1) <= 0.03),
                           "detail": f"sd {d['macro_f1'].std(ddof=1):.3f}, seed {len(d)}"})
    (out / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- таблиці для звіту
    def fmt(x):
        return "" if pd.isna(x) else f"{x:.3f}".replace(".", ",")

    lines = []
    for pr in PROTOCOLS:
        for sl in ("all", "ru"):
            t = main_t[(main_t["protocol"] == pr) & (main_t["slice"] == sl)]
            if t.empty:
                continue
            n = int(t["n"].iloc[0])
            lines += [f"### {pr}, {'усі' if sl == 'all' else 'російськомовні'} (n = {n})", "",
                      "| Модель | macro-F1 | ДІ повідомл. | ДІ канали | MCC | ROC-AUC | PR-AUC | Специфічність |", "|---|---|---|---|---|---|---|---|"]
            for m in models:
                r = t[t["model"] == m].set_index("metric")
                if r.empty:
                    continue
                lines.append(f"| {m} | {fmt(r.loc['macro_f1', 'point'])} | [{fmt(r.loc['macro_f1', 'msg_lo'])}; {fmt(r.loc['macro_f1', 'msg_hi'])}] | "
                             f"[{fmt(r.loc['macro_f1', 'ch_lo'])}; {fmt(r.loc['macro_f1', 'ch_hi'])}] | {fmt(r.loc['mcc', 'point'])} | {fmt(r.loc['roc_auc', 'point'])} | "
                             f"{fmt(r.loc['pr_auc', 'point'])} | {fmt(r.loc['specificity', 'point'])} |")
            lines.append("")
    lines += ["### Парні різниці (P1)", "", "| Зріз | A | B | метрика | різниця | ДІ повідомл. | ДІ канали |", "|---|---|---|---|---|---|---|"]
    for _, r in paired[(paired["protocol"] == "P1") & (paired["metric"].isin(["macro_f1", "mcc"]))].iterrows():
        lines.append(f"| {r['slice']} | {r['A']} | {r['B']} | {r['metric']} | {fmt(r['diff'])} | [{fmt(r['msg_lo'])}; {fmt(r['msg_hi'])}] | [{fmt(r['ch_lo'])}; {fmt(r['ch_hi'])}] |")
    lines += ["", "### H1", "", h1.round(3).to_string(index=False), "",
              "### Перестановочні контролі", "", csum.round(3).to_string(index=False), ""]
    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    write_run_json(out, "scripts/v11_eval.py", {"n_boot": args.n_boot, "seed": args.seed, "best_text": best_text, "best_hybrid": best_hyb,
                                                "seconds": round(time.time() - t0)})
    print(f"Готово за {time.time() - t0:.0f} с; перевірок {len(checks)}, не пройдено: {[c['check'] for c in checks if not c['passed']]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
