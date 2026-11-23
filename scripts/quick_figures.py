"""Додаткові рисунки швидкого зведення: PR-криві, повнота за явністю, SHAP графа і тексту, криві GraphSAGE.
Запуск: python -I scripts/quick_figures.py   (після scripts/quick_summary.py і s07_run.py). Пише results/quick/summary/figures/*.png.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narr import text_eval as te  # noqa: E402
from narr.io import ROOT  # noqa: E402

OUT = ROOT / "results" / "quick" / "summary"
FIG = OUT / "figures"
LAB = {"tfidf": "TF-IDF", "bge_lr": "bge-m3 + LogReg", "xlmr": "XLM-R (швидкий)", "lgbm_emb": "LGBM: ембединги", "lgbm_graph": "LGBM: граф",
       "lgbm_both": "LGBM: ембединги + граф", "gnn_sage": "GraphSAGE + текст", "gnn_sage_notext": "GraphSAGE без тексту",
       "channel_prior": "канальний пріор", "lang_rule": "мовне правило"}
COL = {"tfidf": "#2a6fb0", "bge_lr": "#5aa0d8", "xlmr": "#12406f", "lgbm_emb": "#7a4fb5", "lgbm_graph": "#d9822b", "lgbm_both": "#2e8b57",
       "gnn_sage": "#b22d5a", "gnn_sage_notext": "#e49ab8", "channel_prior": "#8c8c8c", "lang_rule": "#444444"}


def main():
    s5 = ROOT / "results" / "05_protocol"
    g = te.make_gold_table(pd.read_csv(s5 / "gold_membership.csv"), pd.read_csv(s5 / "implicit_labeling.csv"), pd.read_csv(s5 / "implicit_auto_b.csv"))
    pm = pd.read_parquet(OUT / "predictions_mean_p1.parquet")
    models = [m for m in LAB if m in set(pm["model"])]
    ru = g["lang"].eq("ru")

    # 1. PR-криві (російськомовний зріз)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for m in models:
        if m in ("channel_prior", "lang_rule"):
            continue
        d = g.merge(pm[pm["model"] == m][["key", "p"]], on="key")
        d = d[d["lang"] == "ru"]
        pr, rc, _ = precision_recall_curve(d["y"], d["p"])
        ax.plot(rc, pr, label=LAB[m], color=COL[m], lw=1.8)
    prev = g.loc[ru, "y"].mean()
    ax.axhline(prev, color="k", ls="--", lw=1, label=f"випадковий (частка наративних {prev:.3f})".replace(".", ","))
    ax.set_xlabel("повнота (recall)")
    ax.set_ylabel("точність (precision)")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("PR-криві, російськомовні золоті (199), P1")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "quick_pr_ru.png", dpi=150)
    plt.close(fig)

    # 2. Повнота на явних і неявних наративних (поріг 0,5, російськомовні), розмітка (а)
    imp, exp = te.explicitness_masks(g)["a"]
    rows = []
    for m in models:
        d = g.merge(pm[pm["model"] == m][["key", "p"]], on="key")
        for name, mask in (("неявні", imp), ("явні", exp), ("ненаративні", g["y"].to_numpy() == 0)):
            mm = (mask & ru.to_numpy())
            hit = (d["p"].to_numpy() >= 0.5) if name != "ненаративні" else (d["p"].to_numpy() < 0.5)
            k = int((hit & mm).sum()) if len(d) == len(g) else 0
            n = int(mm.sum())
            rows.append({"model": m, "group": name, "recall": k / n if n else np.nan, "k": k, "n": n})
    rec = pd.DataFrame(rows)
    rec.to_csv(OUT / "recall_explicitness_ru_p1.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.38
    xs = np.arange(len(models))
    for i, (grp, c) in enumerate((("неявні", "#c0504d"), ("явні", "#4f81bd"), ("ненаративні", "#6aa84f"))):
        v = [rec[(rec.model == m) & (rec.group == grp)]["recall"].iloc[0] for m in models]
        nm = {"неявні": "повнота на неявних наративних", "явні": "повнота на явних наративних", "ненаративні": "специфічність на ненаративних"}[grp]
        ax.bar(xs + (i - 1) * 0.27, v, 0.27, label=f"{nm} (n={int(rec[(rec.group == grp)]['n'].iloc[0])})", color=c)
    ax.set_xticks(xs)
    ax.set_xticklabels([LAB[m] for m in models], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("частка правильних при порозі 0,5")
    ax.set_ylim(0, 1.42)
    ax.set_title("Російськомовні золоті, P1: що саме модель визначає правильно (поріг 0,5, розмітка явності (а))")
    ax.legend(fontsize=8, loc="upper center", ncol=1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "quick_recall_explicitness.png", dpi=150)
    plt.close(fig)

    # 3. SHAP: внесок графа і тексту
    hy = ROOT / "results" / "quick" / "07_simple_hybrid"
    if (hy / "shap_features.csv").exists():
        sh = pd.read_csv(hy / "shap_features.csv")
        b = sh[sh["model"] == "lgbm_both"]
        gr = b[b["group"] == "graph"].sort_values("mean_abs_shap_gold", ascending=False).head(10)
        tot = b.groupby("group")["mean_abs_shap_gold"].sum()
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 2]})
        a1.bar(["текст\n(1024 ембединг-ознак)", "граф\n(ознаки каналу, копії)"], [tot.get("text", 0), tot.get("graph", 0)], color=["#2a6fb0", "#d9822b"])
        a1.set_ylabel("сума середніх |SHAP| на золотих")
        a1.set_title("Гібрид LightGBM: внесок груп")
        a2.barh(gr["feature"][::-1], gr["mean_abs_shap_gold"][::-1], color="#d9822b")
        a2.set_title("Топ-10 графових ознак за середнім |SHAP|")
        a2.set_xlabel("середній |SHAP| на золотих")
        fig.tight_layout()
        fig.savefig(FIG / "quick_shap.png", dpi=150)
        plt.close(fig)
    sg = hy / "shap_gold_groups_lgbm_both.parquet"
    if sg.exists():
        s = pd.read_parquet(sg).merge(g[["key", "y", "lang"]], on="key")
        s = s[s["lang"] == "ru"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        data, labs = [], []
        for y, nm in ((1, "наратив"), (0, "не наратив")):
            for col, cn in (("shap_text", "текст"), ("shap_graph", "граф")):
                data.append(s[s["y"] == y][col].to_numpy())
                labs.append(f"{nm}\n{cn}")
        bp = ax.boxplot(data, tick_labels=labs, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#2a6fb0", "#d9822b"] * 2):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("внесок у log-odds наративу (сума SHAP)")
        ax.set_title("Знакові внески тексту і графа, російськомовні золоті")
        fig.tight_layout()
        fig.savefig(FIG / "quick_shap_signed.png", dpi=150)
        plt.close(fig)

    # 4. Криві GraphSAGE
    gd = ROOT / "results" / "quick" / "08_gnn" / "units"
    if gd.exists():
        fig, ax = plt.subplots(figsize=(7, 4.4))
        for v, c in (("gnn_sage", "#b22d5a"), ("gnn_sage_notext", "#e49ab8")):
            for k, f in enumerate(sorted(gd.glob(f"{v}__P1__s*.json"))):
                h = json.loads(f.read_text(encoding="utf-8"))["splits"][0]["history"]
                ax.plot([x["epoch"] for x in h], [x["valid_macro_f1"] for x in h], color=c, alpha=0.8, label=LAB[v] if k == 0 else None)
        ax.set_xlabel("епоха")
        ax.set_ylabel("valid macro-F1 (слабкі мітки)")
        ax.set_title("GraphSAGE: криві валідації (3 seed)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG / "quick_gnn_curves.png", dpi=150)
        plt.close(fig)
    print("Рисунки збережено в", FIG)


if __name__ == "__main__":
    main()
