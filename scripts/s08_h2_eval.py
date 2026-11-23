"""H2 (преєстрація, розділ 5): семантичні ребра додають інформацію понад форварди.

Основний тест: різниця macro-F1 моделі з семантичними ребрами (cos >= 0,985, лише між каналами) і тієї самої моделі без них (лише форварди),
P1, російськомовний зріз, 5 seed, ті самі спліти. Прогнози усереднюються по seed, ДІ парний бутстреп по повідомленнях (1000, seed 42).
Підтверджено, якщо різниця додатна з нижньою межею ДІ > 0 і знак той самий за порогів 0,975, 0,985, 0,99.
Також: покриття золотих ребрами (скільки золотих узагалі мають форвард чи семантичного сусіда), різниця за seed окремо, внесок форвардів (розвідувально).

Запуск: python -I scripts/s08_h2_eval.py [--dir results/quick/08_gnn_h2] [--n-boot 1000]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narr import text_eval as te  # noqa: E402
from narr.evaluate import binary_metrics  # noqa: E402
from narr.io import ROOT, write_run_json  # noqa: E402
from narr.splits import make_key  # noqa: E402

THR = {}  # заповнюється за --arch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", default=None, help="папки з прогнозами (можна кілька)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arch", default=None, help="sage|rgcn|hgt: повна модель; без аргументу: старий швидкий варіант h2_*")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    dirs = [Path(x) for x in (args.dir or [str(ROOT / "results" / "quick" / "08_gnn_h2")])]
    d = Path(args.out) if args.out else dirs[0]
    d.mkdir(parents=True, exist_ok=True)
    pred = pd.concat([pd.read_parquet(p) for dd in dirs for p in sorted((dd / "predictions").glob("*.parquet"))], ignore_index=True)
    if args.arch:
        a = args.arch
        THR.update({f"{a}_sem975": 0.975, f"{a}_full": 0.985, f"{a}_sem990": 0.99})
        MAIN, NOSEM, NOFWD = f"{a}_full", f"{a}_nosem", f"{a}_nofwd"
    else:
        THR.update({"h2_sem975": 0.975, "h2_sem985": 0.985, "h2_sem990": 0.99})
        MAIN, NOSEM, NOFWD = "h2_sem985", "h2_nosem", "h2_nofwd"
    s5 = ROOT / "results" / "05_protocol"
    g = te.make_gold_table(pd.read_csv(s5 / "gold_membership.csv"), pd.read_csv(s5 / "implicit_labeling.csv"), pd.read_csv(s5 / "implicit_auto_b.csv"))
    pm = te.mean_over_seeds(pred)
    n_seeds = int(pred.groupby("model")["seed"].nunique().min())

    rows = []
    pairs = [(m, NOSEM) for m in THR] + [(MAIN, NOFWD)]
    for a, b in pairs:
        t = te.paired_table(pm, g, "P1", [(a, b)], ["all", "ru"], args.n_boot, args.seed)
        t = t[t["metric"].isin(["macro_f1", "mcc"])]
        rows.append(t)
    diffs = pd.concat(rows, ignore_index=True)
    # різниця по seed окремо (середнє)
    ps = []
    for (a, b) in pairs:
        for sl in ("all", "ru"):
            vals = []
            for sd in sorted(pred["seed"].unique()):
                da, db = pred[(pred["model"] == a) & (pred["seed"] == sd)], pred[(pred["model"] == b) & (pred["seed"] == sd)]
                if da.empty or db.empty:
                    continue
                ma = g.merge(da.assign(key=make_key(da["channel_id"], da["message_id"]))[["key", "p"]], on="key")
                mb = g.merge(db.assign(key=make_key(db["channel_id"], db["message_id"]))[["key", "p"]], on="key")
                mk = te.slice_mask(ma, sl)
                vals.append(binary_metrics(ma["y"].to_numpy()[mk], ma["p"].to_numpy()[mk])["macro_f1"] - binary_metrics(mb["y"].to_numpy()[mk], mb["p"].to_numpy()[mk])["macro_f1"])
            ps.append({"A": a, "B": b, "slice": sl, "mean_seed_diff_macro_f1": float(np.mean(vals)), "sd_seed_diff": float(np.std(vals, ddof=1)), "n_seeds": len(vals),
                       "n_positive_seeds": int(sum(v > 0 for v in vals))})
    per_seed = pd.DataFrame(ps)
    diffs.to_csv(d / "h2_diffs.csv", index=False)
    per_seed.to_csv(d / "h2_per_seed.csv", index=False)

    # критерій H2
    mf = diffs[(diffs["metric"] == "macro_f1") & (diffs["B"] == NOSEM) & (diffs["slice"] == "ru")].set_index("A")
    main = mf.loc[MAIN]
    sign_ok = all(float(mf.loc[m, "diff"]) > 0 for m in THR if m in mf.index)
    confirmed = bool(float(main["diff"]) > 0 and float(main["ci_lo"]) > 0 and sign_ok)
    verdict = {"confirmed_before_holm": confirmed, "main_diff_macro_f1_ru": float(main["diff"]), "ci": [float(main["ci_lo"]), float(main["ci_hi"])],
               "same_positive_sign_all_thresholds": bool(sign_ok), "n_seeds": n_seeds,
               "diff_by_threshold_ru": {m: float(mf.loc[m, "diff"]) for m in THR if m in mf.index}}
    (d / "h2_verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")

    # покриття золотих ребрами
    import pyarrow.parquet as pq

    index = pd.read_parquet(ROOT / "data" / "interim" / "embeddings" / "bge_m3_index.parquet")
    row = pd.Series(np.arange(len(index)), index=index["key"].to_numpy())
    gi = row.loc[g["key"]].to_numpy()
    fe = pq.read_table(ROOT / "data" / "raw" / "extracted" / "parquet_set" / "forward_edges.parquet",
                       columns=["logical_src_channel_id", "logical_src_message_id", "logical_dst_channel_id", "logical_dst_message_id"]).to_pandas()
    fk = set(make_key(fe["logical_src_channel_id"], fe["logical_src_message_id"]).tolist()) | set(make_key(fe["logical_dst_channel_id"], fe["logical_dst_message_id"]).tolist())
    pairs_df = pd.read_parquet(ROOT / "data" / "interim" / "features" / "sem_pairs_0975.parquet")
    cov = [{"edge_layer": "форварди (обидва напрямки)", "n_gold_with_edge": int(g["key"].isin(fk).sum()), "n_gold": int(len(g)),
            "n_gold_ru_with_edge": int((g["key"].isin(fk) & g["lang"].eq("ru")).sum())}]
    for m, t in THR.items():
        pp = pairs_df[pairs_df["cos"] >= t - 1e-9]
        nodes = set(pp["i"]) | set(pp["j"])
        has = np.isin(gi, list(nodes))
        cov.append({"edge_layer": f"семантичні, cos >= {t}", "n_gold_with_edge": int(has.sum()), "n_gold": int(len(g)), "n_gold_ru_with_edge": int((has & g["lang"].eq("ru").to_numpy()).sum())})
    pd.DataFrame(cov).to_csv(d / "gold_edge_coverage.csv", index=False)

    # рисунок
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, sl, title in zip(axes, ("ru", "all"), ("російськомовні (199)", "усі 412")):
        sub = diffs[(diffs["metric"] == "macro_f1") & (diffs["B"] == NOSEM) & (diffs["slice"] == sl)]
        xs = np.arange(len(sub))
        ax.errorbar(xs, sub["diff"], yerr=[sub["diff"] - sub["ci_lo"], sub["ci_hi"] - sub["diff"]], fmt="o", color="#2e8b57", capsize=5, lw=2)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"cos ≥ {THR[a]}" for a in sub["A"]])
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Δ macro-F1: з семантикою мінус без неї")
    fig.suptitle(f"H2: GraphSAGE з форвардами, ефект семантичних ребер ({n_seeds} seed, 95% ДІ)", fontsize=11)
    fig.tight_layout()
    (d / "figures").mkdir(exist_ok=True)
    fig.savefig(d / "figures" / "h2_diffs.png", dpi=150)
    plt.close(fig)
    if args.arch:
        names = {f"{args.arch}_full": "Повна модель", f"{args.arch}_nosem": "без семантичних ребер", f"{args.arch}_nofwd": "без форвардів",
                 f"{args.arch}_nocopub": "без ко-публікації", f"{args.arch}_notext": "без тексту вузлів"}
        rows = []
        for m, nm in names.items():
            if m not in set(pm["model"]):
                continue
            dm = g.merge(pm[pm["model"] == m][["key", "p"]], on="key")
            mk = te.slice_mask(dm, "ru")
            r = binary_metrics(dm["y"].to_numpy()[mk], dm["p"].to_numpy()[mk])
            rows.append({"model": m, "name": nm, "f1": r["macro_f1"], "mcc": r["mcc"], "pr_auc": r["pr_auc"]})
        pd.DataFrame(rows).to_csv(d / "ablation_table.csv", index=False)
    write_run_json(d, "scripts/s08_h2_eval.py", {"n_boot": args.n_boot, "seed": args.seed, "verdict": verdict})
    print(json.dumps(verdict, ensure_ascii=False))
    print(pd.DataFrame(cov).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
