"""Діагностика для критичного аналізу (розвідувальна, не змінює результати репозиторію).
Запуск: .venv/Scripts/python.exe -I scripts/review_diag.py . > results/review/diag.log
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
from narr import text_eval as te, textmodels as tm  # noqa: E402
from narr.evaluate import binary_metrics  # noqa: E402
from narr.splits import make_key  # noqa: E402

s5 = ROOT / "results" / "05_protocol"
g = te.make_gold_table(pd.read_csv(s5 / "gold_membership.csv"), pd.read_csv(s5 / "implicit_labeling.csv"), pd.read_csv(s5 / "implicit_auto_b.csv"))
g = g.set_index("key")
src = [ROOT / "results" / "06_text_models", ROOT / "results" / "quick" / "07_simple_hybrid", ROOT / "results" / "quick" / "08_gnn", ROOT / "results" / "quick" / "08_gnn_h2"]
pred = pd.concat([pd.read_parquet(p) for d in src for p in sorted((d / "predictions").glob("*.parquet"))], ignore_index=True)
pred = pred[pred["protocol"] == "P1"]
pm = te.mean_over_seeds(pred)
splits = tm.load_splits(ROOT / "splits", ("P1",))
pm = pd.concat([pm, te.comparator_frame(g.reset_index(), splits)], ignore_index=True)
pm = pm[pm["protocol"] == "P1"]

MODELS = ["lang_rule", "channel_prior", "tfidf", "bge_lr", "lgbm_emb", "lgbm_graph", "gnn_sage_notext", "gnn_sage", "h2_sem985", "lgbm_both"]
rng = np.random.default_rng(42)


def boot(y, p, ch, fn, n=1000, cluster=False):
    out = []
    uc = np.unique(ch)
    idx_by = {c: np.flatnonzero(ch == c) for c in uc}
    for _ in range(n):
        if cluster:
            cs = rng.choice(uc, len(uc), replace=True)
            ix = np.concatenate([idx_by[c] for c in cs])
        else:
            ix = rng.integers(0, len(y), len(y))
        if y[ix].min() == y[ix].max():
            continue
        out.append(fn(y[ix], p[ix]))
    return np.quantile(out, [0.025, 0.975])


rows = []
for sl in ("all", "ru"):
    for m in MODELS:
        d = pm[pm["model"] == m].set_index("key")
        if d.empty:
            continue
        gg = g.loc[d.index]
        if sl == "ru":
            msk = (gg["lang"] == "ru").to_numpy()
        else:
            msk = np.ones(len(gg), bool)
        y = gg["y"].to_numpy()[msk]
        p = d["p"].to_numpy()[msk]
        ch = gg["channel_id"].to_numpy()[msk]
        bm = binary_metrics(y, p)
        spec = float(((p < 0.5) & (y == 0)).sum() / max((y == 0).sum(), 1))
        roc = roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5
        f1fn = lambda a, b: binary_metrics(a, b)["macro_f1"]
        ci_m = boot(y, p, ch, f1fn)
        ci_c = boot(y, p, ch, f1fn, cluster=True)
        # оракульний поріг (верхня межа втрат від порогу, НЕ для звітування як результату)
        ths = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
        best = max((binary_metrics(y, p, t)["macro_f1"], t) for t in ths) if len(ths) > 1 else (bm["macro_f1"], 0.5)
        rows.append({"slice": sl, "model": m, "n": int(msk.sum()), "macro_f1": bm["macro_f1"], "ci_msg": f"[{ci_m[0]:.3f}; {ci_m[1]:.3f}]",
                     "ci_channel": f"[{ci_c[0]:.3f}; {ci_c[1]:.3f}]", "mcc": bm["mcc"], "roc_auc": roc, "pr_auc": bm["pr_auc"],
                     "specificity": spec, "mean_p": float(p.mean()), "prev": float(y.mean()), "oracle_thr_f1": best[0], "oracle_thr": best[1]})
res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(res.round(3).to_string(index=False))

# склад навчального пулу P1
tr = splits[0]["train"]
li = pd.read_csv(ROOT / "results" / "04_semantic" / "labeled_index.csv.gz")
print("\nПул P1:", len(tr), "частка наративних", round(tr["weak_narrative"].mean(), 3))
top = tr["channel_id"].value_counts()
print("топ-5 каналів пулу (частка):", (top.head(5) / len(tr)).round(3).to_dict())
print("каналів у пулі:", tr["channel_id"].nunique(), "; у золотих:", g["channel_id"].nunique(),
      "; золотих каналів, яких у пулі < 20 повідомлень:", int((g["channel_id"].drop_duplicates().map(top).fillna(0) < 20).sum()))
th = li.set_index("key")["text_hash"]
tr_h = tr["key"].map(th)
print("унікальних text_hash у пулі:", tr_h.nunique(), f"({tr_h.nunique() / len(tr):.3f} від пулу)")
eff_w = 1 / tr.groupby("channel_id")["key"].transform("count")
print("ефективна к-сть каналів (1/sum p^2):", round(1 / ((top / len(tr)) ** 2).sum(), 1))
# пріор пулу за каналами золотих: наскільки канальні частки в пулі відповідають золотим
gc = g.groupby("channel_id")["y"].agg(["mean", "size"])
pc = tr.groupby("channel_id")["weak_narrative"].agg(["mean", "size"])
j = gc.join(pc, lsuffix="_gold", rsuffix="_pool", how="inner")
j = j[j["size_gold"] >= 5]
print("кореляція частки наративних каналу (золоті vs пул, канали з >=5 золотих):", round(j["mean_gold"].corr(j["mean_pool"]), 3), "n каналів", len(j))
print(j.sort_values("size_gold", ascending=False).head(15).round(2).to_string())
res.to_csv(ROOT / "results" / "review" / "diag_metrics.csv", index=False)

# покриття графовими ознаками: train проти золотих
idx = pd.read_parquet(ROOT / "data" / "interim" / "embeddings" / "bge_m3_index.parquet")
sem = pd.read_parquet(ROOT / "data" / "interim" / "features" / "sem_features.parquet").set_index("key")
for name, keys in (("train P1", tr["key"]), ("золоті", pd.Series(g.index))):
    s = sem.reindex(keys.to_numpy())
    print(f"{name}: частка з міжканальною копією {float((s['sem_n_copies_xch'] > 0).mean()):.3f}, з будь-якою копією {float((s['sem_n_copies_any'] > 0).mean()):.3f}")
cp = pd.read_parquet(ROOT / "data" / "interim" / "features" / "copub_pairs.parquet")
row = pd.Series(np.arange(len(idx)), index=idx["key"].to_numpy())
deg = np.bincount(np.concatenate([cp["i"].to_numpy(), cp["j"].to_numpy()]), minlength=len(idx))
for name, keys in (("train P1", tr["key"]), ("золоті", pd.Series(g.index))):
    d = deg[row.reindex(keys.to_numpy()).dropna().astype(int).to_numpy()]
    print(f"{name}: co_publication ступінь>0 у {float((d > 0).mean()):.3f}, медіана {float(np.median(d)):.0f}")
# мова пулу (через labeled_index немає мови; беремо з ембединг-індексу, якщо є)
print("колонки індексу:", list(idx.columns))
