"""Аудит даних TeleNarratives: факти про золоту вибірку, повторний підрахунок IAA, метрики оригінальних моделей."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------- золота вибірка ----------

def gold_facts(df: pd.DataFrame, cfg: dict) -> dict:
    col = cfg["columns"]
    month = df[col["date"]].dt.strftime("%Y-%m")
    per_month = month.value_counts().sort_index()
    per_channel = df[col["channel"]].value_counts()
    length = df[col["text"]].str.len()
    return {
        "n_messages": int(len(df)),
        "n_narrative": int(df[col["binary"]].sum()),
        "share_narrative": float(df[col["binary"]].mean()),
        "n_channels": int(df[col["channel"]].nunique()),
        "share_forwarded": float(df[col["forwarded"]].mean()),
        "date_min": str(df[col["date"]].min().date()),
        "date_max": str(df[col["date"]].max().date()),
        "n_months": int(len(per_month)),
        "per_month_min": int(per_month.min()),
        "per_month_median": float(per_month.median()),
        "per_month_max": int(per_month.max()),
        "text_len_median": float(length.median()),
        "text_len_mean": float(length.mean()),
        "text_len_p90": float(length.quantile(0.9)),
        "top1_channel_share": float(per_channel.iloc[0] / len(df)),
        "top5_channel_share": float(per_channel.iloc[:5].sum() / len(df)),
        "channels_with_ge5_messages": int((per_channel >= 5).sum()),
    }


def monthly_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    col = cfg["columns"]
    m = df[col["date"]].dt.strftime("%Y-%m")
    t = pd.crosstab(m, df[col["binary"]]).rename(columns={True: "n_narrative", False: "n_other"})
    for c in ("n_narrative", "n_other"):
        if c not in t:
            t[c] = 0
    return t[["n_other", "n_narrative"]].reset_index().rename(columns={col["date"]: "month", "row_0": "month"})


def meta_distribution(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    col = cfg["columns"]
    pos = df[df[col["binary"]]]
    t = pos[col["level1"]].value_counts().rename_axis("label").reset_index(name="n")
    t["share_of_narrative"] = t["n"] / t["n"].sum()
    return t


def channel_counts(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    col = cfg["columns"]
    g = df.groupby(col["channel"]).agg(n=(col["message"], "size"), n_narrative=(col["binary"], "sum"))
    if "channel_username" in df.columns:
        g["channel_username"] = df.groupby(col["channel"])["channel_username"].first()
    g["share_narrative"] = g["n_narrative"] / g["n"]
    return g.sort_values("n", ascending=False).reset_index()


# ---------- міжанотаторська згода ----------

def fleiss_kappa_counts(counts: np.ndarray) -> float:
    """counts: матриця N x k, кількість рецензентів, що обрали категорію j для об'єкта i (однакова кількість рецензентів)."""
    counts = np.asarray(counts, dtype=float)
    n_items = counts.shape[0]
    n_raters = counts[0].sum()
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_i = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    return float((p_bar - p_e) / (1 - p_e))


def fleiss_kappa_labels(ratings: pd.DataFrame) -> float:
    """ratings: рядок = об'єкт, колонка = рецензент, значення = категорія."""
    cats = sorted(pd.unique(ratings.values.ravel()), key=str)
    counts = np.stack([(ratings == c).sum(axis=1).to_numpy() for c in cats], axis=1)
    return fleiss_kappa_counts(counts)


def iaa_recomputed(shared: pd.DataFrame, iaa_file: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    col = cfg["columns"]
    no_narr = col["no_narrative_label"]
    cases = {
        "binary": shared[col["reviewers"]].ne(no_narr).astype(str),
        "multiclass": shared[col["reviewers"]].astype(str),
        "multiclass_level_1": shared[col["reviewers_level1"]].astype(str),
    }
    rows = []
    for case, ratings in cases.items():
        rec = fleiss_kappa_labels(ratings)
        ref = iaa_file[(iaa_file["case"] == case) & (iaa_file["metric"] == "fleiss_kappa")]["value"]
        ref_v = float(ref.iloc[0]) if len(ref) else np.nan
        rows.append({"case": case, "n_items": len(ratings), "fleiss_kappa_recomputed": rec,
                     "fleiss_kappa_in_file": ref_v, "abs_diff": abs(rec - ref_v)})
    return pd.DataFrame(rows)


# ---------- метрики оригінальних моделей ----------

MODEL_NAMES = {
    "text-baseline-10": "Текстовий бейзлайн (RoBERTa)",
    "gnn-posted-forwards": "GNN: posted + forwards",
    "gnn-posted-similar-to": "GNN: posted + similar-to",
    "gnn-posted-forwards-similar-to": "GNN: posted + forwards + similar-to",
    "gnn-posted-similar-to-subgraph": "GNN: posted + similar-to (підграф)",
}


def _cm_metrics(tn, fp, fn, tp) -> dict:
    n0, n1, n = tn + fp, fn + tp, tn + fp + fn + tp
    f1_0 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    f1_1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"n": int(n), "share_pos": n1 / n, "f1_neg": f1_0, "f1_pos": f1_1,
            "macro_f1": (f1_0 + f1_1) / 2, "weighted_f1": (n0 * f1_0 + n1 * f1_1) / n, "mcc": mcc}


def original_models(gnn_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Повертає (таблицю фінальних метрик на test, статистику val-кривих по епохах)."""
    rows, curves = [], []
    for d in sorted(Path(gnn_dir).iterdir()):
        f = d / "metrics.csv"
        if not f.exists():
            continue
        m = pd.read_csv(f)
        fin = m[m["phase"] == "final"].set_index("split")
        t = fin.loc["test"]
        rec = _cm_metrics(t.tn, t.fp, t.fn, t.tp)
        sizes = {s: int(fin.loc[s, ["tn", "fp", "fn", "tp"]].sum()) for s in ("train", "val", "test")}
        rows.append({
            "model": MODEL_NAMES.get(d.name, d.name), "dir": d.name,
            "n_train": sizes["train"], "n_val": sizes["val"], "n_test": sizes["test"],
            "n_total": sum(sizes.values()),
            "general_f1_reported": float(t.general_f1), "weighted_f1_recomputed": rec["weighted_f1"],
            "macro_f1_recomputed": rec["macro_f1"], "f1_pos": rec["f1_pos"],
            "mcc_reported": float(t.mcc), "mcc_recomputed": rec["mcc"],
            "pr_auc": float(t.pr_auc), "roc_auc": float(t.roc_auc), "share_pos_test": rec["share_pos"],
        })
        ep = m[(m["phase"] != "final") & (m["split"] == "val")]
        if len(ep):
            f1 = ep["general_f1"].to_numpy()
            curves.append({"model": MODEL_NAMES.get(d.name, d.name), "n_epochs_val": len(ep),
                           "val_f1_min": float(f1.min()), "val_f1_max": float(f1.max()),
                           "val_f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else np.nan,
                           "max_abs_step": float(np.abs(np.diff(f1)).max()) if len(f1) > 1 else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(curves)


def weak_label_quality(eval_dir: Path) -> pd.DataFrame:
    """Бінарні метрики оцінки слабких міток на золотій розмітці (файли з TeleNarratives, як є)."""
    rows = []
    for f in sorted(Path(eval_dir).rglob("*.csv")):
        try:
            e = pd.read_csv(f)
        except Exception:
            continue
        if not {"task", "subset"} <= set(e.columns):
            continue
        b = e[(e["task"] == "binary") & (e["subset"] == "all")]
        if b.empty:
            continue
        r = b.iloc[0]
        rows.append({"method": f.parent.name, "file": f.stem.replace("evaluation_results_", ""),
                     "support": r.get("support"), "precision": r.get("precision"), "recall": r.get("recall"),
                     "f1_pos": r.get("f1"), "mcc": r.get("mcc"), "roc_auc": r.get("roc_auc")})
    return pd.DataFrame(rows)
