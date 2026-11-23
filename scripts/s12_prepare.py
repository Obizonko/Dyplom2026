"""S12: підготовка даних прототипу Streamlit з збережених прогнозів (без перенавчання).

Запуск:  python -I scripts/s12_prepare.py --hybrid results/quick/07_simple_hybrid [--mode "швидкий (попередній)"]
Читає: результати s07_eval (predictions_mean.parquet, run.json), shap_gold_groups_lgbm_both.parquet, ембединги S4, золотий набір, тексти сусідів
з data/raw/telegram_messages.parquet. Пише data/processed/prototype/{messages,neighbors}.parquet і meta.json (папка не комітиться).
Для кожного золотого повідомлення: ймовірності найкращої текстової, графової (lgbm_graph) і гібридної (lgbm_both) моделей, знакові внески SHAP,
5 найсхожіших повідомлень серед 108 тис. набору S4 (косинус, не саме повідомлення).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narr import text_eval as te, textmodels as tm  # noqa: E402
from narr.io import ROOT  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid", default=str(ROOT / "results" / "07_simple_hybrid"))
    ap.add_argument("--emb-dir", default=str(ROOT / "data" / "interim" / "embeddings"))
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "prototype"))
    ap.add_argument("--mode", default="остаточний")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args(argv)
    hyb, out, emb_dir = Path(args.hybrid), Path(args.out), Path(args.emb_dir)
    out.mkdir(parents=True, exist_ok=True)

    run = json.loads((hyb / "run.json").read_text(encoding="utf-8"))
    best_text = run["params"]["best_text"]["P1"]
    pm = pd.read_parquet(hyb / "predictions_mean.parquet")
    pm = pm[pm["protocol"] == "P1"]
    s5 = ROOT / "results" / "05_protocol"
    g = te.make_gold_table(pd.read_csv(s5 / "gold_membership.csv"), pd.read_csv(s5 / "implicit_labeling.csv"), pd.read_csv(s5 / "implicit_auto_b.csv"))
    piv = pm.pivot(index="key", columns="model", values="p")
    m = g[["key", "channel_id", "message_id", "y", "lang"]].copy()
    m["p_text"] = m["key"].map(piv[best_text])
    m["p_graph"] = m["key"].map(piv["lgbm_graph"])
    m["p_hybrid"] = m["key"].map(piv["lgbm_both"])
    sh = pd.read_parquet(hyb / "shap_gold_groups_lgbm_both.parquet")
    m = m.merge(sh, on="key", how="left")

    index = pd.read_parquet(emb_dir / "bge_m3_index.parquet")
    keys = index["key"].to_numpy()
    E = np.asarray(np.load(emb_dir / "bge_m3_fp16.npy", mmap_mode="r"), dtype=np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    row = pd.Series(np.arange(len(keys)), index=keys)
    qi = row.loc[m["key"]].to_numpy()
    sims = E[qi] @ E.T
    sims[np.arange(len(qi)), qi] = -2.0
    top = np.argsort(-sims, axis=1)[:, :args.k]
    rows = []
    for a, (k, t) in enumerate(zip(m["key"], top)):
        for rank, j in enumerate(t, 1):
            rows.append({"key": int(k), "rank": rank, "nkey": int(keys[j]), "cos": float(sims[a, j])})
    nb = pd.DataFrame(rows)

    need = np.unique(np.concatenate([m["key"].to_numpy(), nb["nkey"].to_numpy()]))
    texts = tm.load_texts(args.raw_dir, need)
    ch_name = pd.read_csv(ROOT / "results" / "02_eda" / "per_channel.csv").set_index("channel_id")
    name_col = next((c for c in ("username", "channel", "channel_name") if c in ch_name.columns), None)
    cname = (lambda c: str(ch_name.loc[c, name_col]) if name_col and c in ch_name.index else str(c))
    m["text"] = m["key"].map(texts)
    m["channel"] = [cname(c) for c in m["channel_id"]]
    weak = index.set_index("key")["narrative_id"].notna()
    nb["text"] = nb["nkey"].map(texts)
    nb["channel"] = [cname(int(k) // 10_000_000) for k in nb["nkey"]]
    nb["weak_label"] = nb["nkey"].map(weak).map({True: "наратив", False: "не наратив"}).fillna("немає")
    m.to_parquet(out / "messages.parquet", index=False)
    nb.to_parquet(out / "neighbors.parquet", index=False)
    (out / "meta.json").write_text(json.dumps({"mode": args.mode, "text_model": best_text, "n": int(len(m))}, ensure_ascii=False), encoding="utf-8")
    print(f"Прототип: {len(m)} повідомлень, {len(nb)} сусідів, текстова модель {best_text}, папка {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
