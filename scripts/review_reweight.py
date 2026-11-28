"""Розвідувальний експеримент (не преєстрований): чи змінює дедуплікація і балансування пулу за каналами якість bge-m3 + LogReg на золотих.
Жоден вибір тут не робиться за золотими: C фіксоване (30, як обрано валідацією в S6), варіанти визначені заздалегідь.
Запуск: .venv/Scripts/python.exe -I scripts/review_reweight.py . results/review/reweight.csv > results/review/reweight.log
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "src"))
from narr import text_eval as te, textmodels as tm  # noqa: E402
from narr.evaluate import binary_metrics  # noqa: E402

s5 = ROOT / "results" / "05_protocol"
g = te.make_gold_table(pd.read_csv(s5 / "gold_membership.csv"), pd.read_csv(s5 / "implicit_labeling.csv"), pd.read_csv(s5 / "implicit_auto_b.csv")).set_index("key")
sp = tm.load_splits(ROOT / "splits", ("P1",))[0]
tr, test = sp["train"].copy(), sp["test"].copy()
emb_dir = ROOT / "data" / "interim" / "embeddings"
idx = pd.read_parquet(emb_dir / "bge_m3_index.parquet")
E = np.load(emb_dir / "bge_m3_fp16.npy", mmap_mode="r")
row = pd.Series(np.arange(len(idx)), index=idx["key"].to_numpy())
tr["text_hash"] = tr["key"].map(idx.set_index("key")["text_hash"])


def X(keys):
    a = np.asarray(E[row.loc[keys].to_numpy()], dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


Xte = X(test["key"].to_numpy())
yte = g.loc[test["key"], "y"].to_numpy()
ru = (g.loc[test["key"], "lang"] == "ru").to_numpy()


def evaluate(name, p, extra):
    out = []
    for sl, m in (("all", np.ones(len(yte), bool)), ("ru", ru)):
        y, q = yte[m], p[m]
        bm = binary_metrics(y, q)
        out.append({"variant": name, "slice": sl, "macro_f1@0.5": bm["macro_f1"], "mcc": bm["mcc"], "roc_auc": roc_auc_score(y, q),
                    "pr_auc": bm["pr_auc"], "specificity": float(((q < 0.5) & (y == 0)).sum() / (y == 0).sum()), "mean_p": float(q.mean()), **extra})
    return out


def fit(Xtr, y, w, cw):
    m = LogisticRegression(C=30.0, class_weight=cw, max_iter=500, solver="lbfgs", random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xtr, y, sample_weight=w)
    return m


rows, P = [], {}
# A: як у S6 (усі рядки, class_weight balanced); B: без балансування класів
# C: дедуплікація за text_hash (мітка більшості, канал першого входження); D: C + вага 1/sqrt(n_каналу); E: C + вага 1/n_каналу (кожен канал однаковий внесок)
y_all = tr["weak_narrative"].astype(int).to_numpy()
Xall = X(tr["key"].to_numpy())
dd = tr.assign(y=y_all).groupby("text_hash").agg(key=("key", "first"), channel_id=("channel_id", "first"), y=("y", "mean"), n=("key", "size")).reset_index()
dd["y"] = (dd["y"] >= 0.5).astype(int)
Xdd = X(dd["key"].to_numpy())
cnt = dd["channel_id"].map(dd["channel_id"].value_counts()).to_numpy()
variants = {
    "A_as_S6": (Xall, y_all, None, "balanced"),
    "B_no_classweight": (Xall, y_all, None, None),
    "C_dedup": (Xdd, dd["y"].to_numpy(), None, "balanced"),
    "D_dedup_w_sqrt_channel": (Xdd, dd["y"].to_numpy(), 1 / np.sqrt(cnt), "balanced"),
    "E_dedup_w_channel": (Xdd, dd["y"].to_numpy(), 1 / cnt, "balanced"),
}
for name, (Xa, ya, w, cw) in variants.items():
    t0 = time.time()
    if w is not None:
        w = w / w.mean()
    m = fit(Xa, ya, w, cw)
    P[name] = m.predict_proba(Xte)[:, 1]
    rows += evaluate(name, P[name], {"n_train": len(ya), "seconds": round(time.time() - t0)})
    print(name, "done", round(time.time() - t0), "s", flush=True)

# парний бутстреп (1000, seed 42) різниці кожного варіанта з A, по повідомленнях і по каналах
ch = g.loc[test["key"], "channel_id"].to_numpy()
rng = np.random.default_rng(42)
boot_rows = []
for sl, msk in (("all", np.ones(len(yte), bool)), ("ru", ru)):
    y, c = yte[msk], ch[msk]
    uc = np.unique(c)
    by = {k: np.flatnonzero(c == k) for k in uc}
    samples = {"message": [rng.integers(0, len(y), len(y)) for _ in range(1000)],
               "channel": [np.concatenate([by[k] for k in rng.choice(uc, len(uc))]) for _ in range(1000)]}
    for name in P:
        if name == "A_as_S6":
            continue
        a, b = P[name][msk], P["A_as_S6"][msk]
        for metric in ("macro_f1", "mcc"):
            d0 = binary_metrics(y, a)[metric] - binary_metrics(y, b)[metric]
            for kind, ss in samples.items():
                ds = [binary_metrics(y[ix], a[ix])[metric] - binary_metrics(y[ix], b[ix])[metric] for ix in ss]
                lo, hi = np.quantile(ds, [0.025, 0.975])
                boot_rows.append({"variant": name, "minus": "A_as_S6", "slice": sl, "metric": metric, "bootstrap": kind, "diff": d0, "ci_lo": lo, "ci_hi": hi})
pd.DataFrame(boot_rows).to_csv(OUT.with_name("reweight_paired.csv"), index=False)
pd.DataFrame({"key": test["key"].to_numpy(), **P}).to_parquet(OUT.with_name("reweight_predictions.parquet"), index=False)

# перестановочний контроль: 10 перемішувань міток (варіант A), розподіл MCC на золотих
mccs = []
for k in range(10):
    yp = np.random.default_rng(1000 + k).permutation(y_all)
    m = fit(Xall, yp, None, "balanced")
    mccs.append(binary_metrics(yte, m.predict_proba(Xte)[:, 1])["mcc"])
print("перемішані мітки, MCC на 412 золотих за 10 перестановками:", np.round(mccs, 3).tolist(), "середнє", round(float(np.mean(mccs)), 3))
pd.DataFrame({"permutation_seed": [1000 + k for k in range(10)], "mcc_gold_all": mccs}).to_csv(OUT.with_name("reweight_permutations.csv"), index=False)
res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
pd.set_option("display.width", 220)
print(res.round(3).to_string(index=False))
