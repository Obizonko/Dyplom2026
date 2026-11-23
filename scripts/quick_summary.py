"""Зведення швидкого режиму (S6 + S7 + S8) для презентації: метрики з 95% ДІ на золотих, P1, зріз «усі 412» і російськомовний.

Запуск:  python -I scripts/quick_summary.py [--n-boot 1000]
Читає прогнози з results/06_text_models (TF-IDF, bge-m3+LogReg), results/quick/06_xlmr (XLM-R, зменшений бюджет), results/quick/07_simple_hybrid (LightGBM),
results/quick/08_gnn (GraphSAGE). Пише results/quick/summary/*.csv, figures/*.png і reports/quick_summary.md. Усі числа в звіті беруться з цих CSV.
Це описове зведення: жодного вибору моделі, порога чи зрізу за золотими. Попередній режим, не підсумкові результати.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narr import text_eval as te, textmodels as tm  # noqa: E402
from narr.io import ROOT, write_run_json  # noqa: E402

SOURCES = {"text": ROOT / "results" / "06_text_models", "xlmr": ROOT / "results" / "quick" / "06_xlmr",
           "hybrid": ROOT / "results" / "quick" / "07_simple_hybrid", "gnn": ROOT / "results" / "quick" / "08_gnn"}
ORDER = [("majority", "мажоритарний", "k"), ("lang_rule", "правило «російська = наратив»", "k"), ("channel_prior", "канальний пріор", "k"),
         ("tfidf", "TF-IDF + LogReg", "t"), ("bge_lr", "bge-m3 + LogReg", "t"), ("xlmr", "XLM-R (зменшений бюджет)", "t"),
         ("lgbm_emb", "LightGBM: ембединги", "t"), ("lgbm_graph", "LightGBM: лише граф", "g"), ("lgbm_graph_noprior", "LightGBM: граф без пріору", "g"),
         ("gnn_sage_notext", "GraphSAGE без тексту", "g"), ("gnn_sage", "GraphSAGE з текстом", "h"), ("lgbm_both", "LightGBM: ембединги + граф", "h")]
COLORS = {"k": "#8c8c8c", "t": "#2a6fb0", "g": "#d9822b", "h": "#2e8b57"}


def fmt(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "quick" / "summary"))
    ap.add_argument("--report", default=str(ROOT / "reports" / "quick_summary.md"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    preds, units, used = [], [], {}
    for k, d in SOURCES.items():
        if (d / "predictions").exists() and any((d / "predictions").glob("*.parquet")):
            preds.append(pd.concat([pd.read_parquet(p) for p in sorted((d / "predictions").glob("*.parquet"))], ignore_index=True))
            units += [json.loads(p.read_text(encoding="utf-8")) for p in sorted((d / "units").glob("*.json"))]
            used[k] = str(d.relative_to(ROOT))
    pred = pd.concat(preds, ignore_index=True)
    pred = pred[pred["protocol"] == "P1"]
    s5 = ROOT / "results" / "05_protocol"
    g = te.make_gold_table(pd.read_csv(s5 / "gold_membership.csv"), pd.read_csv(s5 / "implicit_labeling.csv"), pd.read_csv(s5 / "implicit_auto_b.csv"))
    splits = tm.load_splits(ROOT / "splits", ("P1",))
    pm = pd.concat([te.mean_over_seeds(pred), te.comparator_frame(g, splits)], ignore_index=True)
    pm.to_parquet(out / "predictions_mean_p1.parquet", index=False)
    models = [m for m, _, _ in ORDER if m in set(pm["model"])]
    mt = te.metrics_table(pm, g, ["P1"], models, ("all", "ru"), ["all"], args.n_boot, args.seed)
    mt.to_csv(out / "metrics_p1.csv", index=False)
    sub = te.metrics_table(pm, g, ["P1"], models, ("ru",), ["implicit_a", "explicit_a"], args.n_boot, args.seed)
    sub.to_csv(out / "metrics_p1_ru_explicitness.csv", index=False)
    vt = te.valid_table(units)
    vt = vt[vt["protocol"] == "P1"].groupby("model", as_index=False)["valid_macro_f1"].mean()
    vt.to_csv(out / "valid_macro_f1_p1.csv", index=False)
    pairs = [("lgbm_both", "lgbm_emb"), ("lgbm_both", "tfidf"), ("lgbm_emb", "tfidf"), ("gnn_sage", "gnn_sage_notext"), ("gnn_sage", "bge_lr"), ("lgbm_graph", "channel_prior")]
    pt = te.paired_table(pm, g, "P1", [(a, b) for a, b in pairs if a in models and b in models], ["all", "ru"], args.n_boot, args.seed)
    pt.to_csv(out / "paired_p1.csv", index=False)

    def get(model, sl, metric):
        r = mt[(mt["model"] == model) & (mt["slice"] == sl) & (mt["metric"] == metric)]
        return None if r.empty else (float(r["point"].iloc[0]), float(r["ci_lo"].iloc[0]), float(r["ci_hi"].iloc[0]))

    # рисунок: macro-F1 і MCC, два зрізи
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharey=True)
    labels = {m: (lab, c) for m, lab, c in ORDER}
    for r_i, metric in enumerate(("macro_f1", "mcc")):
        for c_i, sl in enumerate(("all", "ru")):
            ax = axes[r_i, c_i]
            ys = np.arange(len(models))[::-1]
            for y, m in zip(ys, models):
                v = get(m, sl, metric)
                if v is None:
                    continue
                ax.barh(y, v[0], color=COLORS[labels[m][1]], alpha=0.9)
                ax.plot([v[1], v[2]], [y, y], color="black", lw=1.2)
                ax.text(max(v[2], v[0]) + 0.01, y, fmt(v[0], 2), va="center", fontsize=8)
            ax.set_yticks(ys)
            ax.set_yticklabels([labels[m][0] for m in models], fontsize=9)
            ax.set_xlim(0, 1.0 if metric == "macro_f1" else 0.95)
            ax.set_title(f"{'macro-F1' if metric == 'macro_f1' else 'MCC'}: {'усі 412 золотих' if sl == 'all' else 'російськомовні золоті (199)'}", fontsize=10)
            ax.grid(axis="x", alpha=0.25)
    fig.suptitle("P1, золота розмітка, 95% ДІ (бутстреп). Швидкий (попередній) режим", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "figures" / "quick_p1_metrics.png", dpi=150)
    plt.close(fig)

    # звіт
    L = []
    L.append("# Швидке зведення S6, S7, S8 (попередній режим для презентації)\n")
    L.append("Усі числа взято з `results/quick/summary/*.csv` (скрипт `scripts/quick_summary.py`) і прогнозів у `results/`. Це описове зведення: "
             "ні модель, ні поріг, ні зріз за золотими не обирались. **Не підсумкові результати**: нижче перелічено, чим цей режим слабший за преєстрований.\n")
    L.append("## Чим швидкий режим відрізняється від плану\n")
    L.append("- XLM-R: підвибірка train 20 000 рядків, 1 seed, 2 епохи, словникові ембединги заморожено (повний режим: весь пул, 3 seed, 3 епохи).\n"
             "- LightGBM: 20 спроб Optuna замість 50, пошук параметрів на підвибірці 20 000 рядків, 1 seed, лише P1 і P2, без P3.\n"
             "- GraphSAGE: ребра лише семантичні (між каналами, cos ≥ 0,985) і «канал публікує повідомлення»; форвардів на рівні повідомлень і co_publication немає; "
             "лише GraphSAGE (без R-GCN і HGT), 3 seed, лише P1.\n"
             "- Семантичні копії шукаються точним пошуком на GPU (hnswlib на Windows недоступний).\n"
             "- Преєстровані негативні контролі виконано для LightGBM; для XLM-R контроль із перемішаними мітками у швидкому режимі не запускався.\n")
    L.append("## Метрики на золотих, P1, 95% ДІ\n")
    L.append("Макро-F1, MCC, PR-AUC. Перший блок: усі 412 золотих; другий: російськомовні (199), основний зріз за преєстрацією. Приклад читання: «0,731 [0,66; 0,79]».\n")
    for sl, title in (("ru", "Російськомовний зріз (основний)"), ("all", "Усі 412 золотих (вторинний)")):
        L.append(f"\n### {title}\n")
        L.append("| Модель | macro-F1 | MCC | PR-AUC |")
        L.append("|---|---|---|---|")
        for m, lab, _ in ORDER:
            if m not in models:
                continue
            cells = []
            for metric in ("macro_f1", "mcc", "pr_auc"):
                v = get(m, sl, metric)
                cells.append("н/д" if v is None else f"{fmt(v[0])} [{fmt(v[1], 2)}; {fmt(v[2], 2)}]")
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("\n![Метрики P1](../results/quick/summary/figures/quick_p1_metrics.png)\n")
    L.append("## Парні різниці (P1, парний бутстреп)\n")
    L.append("| A мінус B | зріз | різниця macro-F1 | 95% ДІ |")
    L.append("|---|---|---|---|")
    for _, r in pt[pt["metric"] == "macro_f1"].iterrows() if "metric" in pt.columns else []:
        L.append(f"| {r['A']} мінус {r['B']} | {r['slice']} | {fmt(r['diff'])} | [{fmt(r['ci_lo'], 3)}; {fmt(r['ci_hi'], 3)}] |")
    L.append("\n## Валідаційний macro-F1 на слабких мітках (за ним обирають моделі, не за золотими)\n")
    L.append("| Модель | valid macro-F1 |")
    L.append("|---|---|")
    for _, r in vt.sort_values("valid_macro_f1", ascending=False).iterrows():
        L.append(f"| {r['model']} | {fmt(r['valid_macro_f1'], 4)} |")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    write_run_json(out, "scripts/quick_summary.py", {"n_boot": args.n_boot, "seed": args.seed, "sources": used, "models": models, "seconds": round(time.time() - t0)})
    print(f"Готово за {time.time() - t0:.0f} с; моделі {models}; звіт {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
