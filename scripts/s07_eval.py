"""S7: оцінка простого гібрида на золотих. Метрики з 95% ДІ, порівняння з текстовими моделями S6 і компараторами, перевірки з розділу 8 преєстрації.

Запуск:  python -I scripts/s07_eval.py --hybrid results/07_simple_hybrid --text results/06_text_models [--text results/quick/06_xlmr] --out results/07_simple_hybrid
Найкраща текстова і найкраща гібридна обираються за валідаційним macro-F1 на слабких мітках, не за золотими.
H1 (попередньо): різниця macro-F1 гібрид мінус текст на підмножині неявних у російськомовному зрізі і різниця різниць, парний бутстреп по повідомленнях.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narr import text_eval as te, textmodels as tm  # noqa: E402
from narr.evaluate import binary_metrics  # noqa: E402
from narr.io import ROOT, write_run_json  # noqa: E402

HYBRID = ("lgbm_graph", "lgbm_graph_noprior", "lgbm_emb", "lgbm_both")
CONTROLS = ("lgbm_graph_shuf", "lgbm_both_shuf", "lgbm_both_rewired", "tfidf_shuf", "bge_lr_shuf", "xlmr_shuf")


def load_dir(d: Path):
    pred = pd.concat([pd.read_parquet(p) for p in sorted((d / "predictions").glob("*.parquet"))], ignore_index=True)
    units = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((d / "units").glob("*.json"))]
    return pred, units


def f1_macro(y, p):
    return binary_metrics(y, p)["macro_f1"]


def boot_diff(y, pa, pb, mask_a, mask_b, n_boot, seed):
    """Парний бутстреп по повідомленнях. Повертає дві різниці (на підмножинах a і b), їхню різницю і 95% ДІ кожної."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d_a, d_b = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        ya, yb = y[i], y[i]
        ma, mb = mask_a[i], mask_b[i]
        if ya[ma].sum() < 2 or ya[mb].sum() < 2 or (1 - ya[ma]).sum() < 2:
            continue
        d_a.append(f1_macro(ya[ma], pa[i][ma]) - f1_macro(ya[ma], pb[i][ma]))
        d_b.append(f1_macro(yb[mb], pa[i][mb]) - f1_macro(yb[mb], pb[i][mb]))
    d_a, d_b = np.array(d_a), np.array(d_b)
    pt_a = f1_macro(y[mask_a], pa[mask_a]) - f1_macro(y[mask_a], pb[mask_a])
    pt_b = f1_macro(y[mask_b], pa[mask_b]) - f1_macro(y[mask_b], pb[mask_b])
    q = lambda x: (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))
    return {"diff_implicit": pt_a, "ci_implicit": q(d_a), "diff_explicit": pt_b, "ci_explicit": q(d_b),
            "diff_of_diffs": pt_a - pt_b, "ci_dod": q(d_a - d_b), "n_boot_used": int(len(d_a))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid", default=str(ROOT / "results" / "07_simple_hybrid"))
    ap.add_argument("--text", action="append", default=None, help="папки з прогнозами текстових моделей S6 (можна кілька, прогнози зливаються)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--splits", default=str(ROOT / "splits"))
    ap.add_argument("--s5", default=str(ROOT / "results" / "05_protocol"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    hyb = Path(args.hybrid)
    out = Path(args.out) if args.out else hyb
    text_dirs = [Path(t) for t in (args.text or [str(ROOT / "results" / "06_text_models")])]
    t0 = time.time()

    ph, uh = load_dir(hyb)
    pts, uts = zip(*[load_dir(d) for d in text_dirs])
    pred = pd.concat([ph, *pts], ignore_index=True)
    units = uh + [u for us in uts for u in us]
    protocols = [p for p in ("P1", "P2", "P3") if p in set(ph["protocol"])]
    s5 = Path(args.s5)
    g = te.make_gold_table(pd.read_csv(s5 / "gold_membership.csv"), pd.read_csv(s5 / "implicit_labeling.csv"), pd.read_csv(s5 / "implicit_auto_b.csv"))
    splits = tm.load_splits(args.splits, protocols)
    pm = pd.concat([te.mean_over_seeds(pred), te.comparator_frame(g, splits)], ignore_index=True)
    pm.to_parquet(out / "predictions_mean.parquet", index=False)
    pcheck = te.check_predictions(pred[pred["protocol"].isin(protocols)], g)

    vt = te.valid_table(units)
    vt.to_csv(out / "valid_units.csv", index=False)
    best_text = {pr: te.select_best(vt, pr).query("selected")["model"].tolist() for pr in protocols}
    best_text = {pr: (v[0] if v else None) for pr, v in best_text.items()}
    # найкраща гібридна: серед моделей, що мають і граф, і текст (lgbm_both); лише граф і лише текст це абляції
    best_hyb = {pr: "lgbm_both" for pr in protocols}
    print("Найкраща текстова (валідація слабких міток):", best_text, "| гібрид:", best_hyb, flush=True)

    present = set(pm["model"])
    text_models = [m for m in te.TEXT_MODELS if m in present]
    models = [m for m in (*text_models, *HYBRID, *te.COMPARATORS, *CONTROLS) if m in present]
    nb, sd = args.n_boot, args.seed
    main_t = te.metrics_table(pm, g, protocols, models, ("all", "ru"), ["all"], nb, sd)
    main_t.to_csv(out / "metrics_main.csv", index=False)
    sub_t = te.metrics_table(pm, g, ["P1"], [m for m in models if m not in CONTROLS], ("ru",),
                             [s for s in te.subset_masks(g) if s != "all"], nb, sd)
    sub_t.to_csv(out / "metrics_subsets.csv", index=False)
    print(f"[{time.time() - t0:4.0f} с] метрики з ДІ готові", flush=True)

    # H1 (попередньо): гібрид проти найкращої текстової, ru-зріз, P1
    rows = []
    for pr in protocols:
        a, b = best_hyb[pr], best_text[pr]
        da, db = te._arrays(pm, g, pr, a), te._arrays(pm, g, pr, b)
        if da is None or db is None:
            continue
        d = da.merge(db[["key", "p"]], on="key", suffixes=("", "_b"))
        ru = te.slice_mask(d, "ru")
        for defn in ("a", "narrow", "wide", "b"):
            sm = te.subset_masks(d)
            ma, mb = sm[f"implicit_{defn}"][ru], sm[f"explicit_{defn}"][ru]
            y = d["y"].to_numpy()[ru]
            r = boot_diff(y, d["p"].to_numpy()[ru], d["p_b"].to_numpy()[ru], ma, mb, nb, sd)
            n_imp = int((y[ma] == 1).sum())
            rows.append({"protocol": pr, "A": a, "B": b, "definition": defn, "n_implicit_pos": n_imp, **{k: (v if not isinstance(v, tuple) else None) for k, v in r.items()},
                         "ci_implicit_lo": r["ci_implicit"][0], "ci_implicit_hi": r["ci_implicit"][1], "ci_explicit_lo": r["ci_explicit"][0], "ci_explicit_hi": r["ci_explicit"][1],
                         "ci_dod_lo": r["ci_dod"][0], "ci_dod_hi": r["ci_dod"][1]})
    h1 = pd.DataFrame(rows).drop(columns=["ci_implicit", "ci_explicit", "ci_dod"], errors="ignore")
    h1.to_csv(out / "h1_preliminary.csv", index=False)

    # парні різниці на всьому ru-зрізі і на всіх 412
    pairs = []
    for pr in protocols:
        a, b = best_hyb[pr], best_text[pr]
        cand = [(a, b), (a, "lgbm_emb"), ("lgbm_graph", "channel_prior"), (a, "lang_rule"), ("lgbm_graph_noprior", "channel_prior")]
        pairs.append(te.paired_table(pm, g, pr, [(x, y) for x, y in cand if x and y and x != y], ["all", "ru"], nb, sd))
    pd.concat(pairs, ignore_index=True).to_csv(out / "paired_diffs.csv", index=False)

    # перевірки з розділу 8 преєстрації
    checks = []
    def mget(model, metric, sl="all"):
        r = main_t[(main_t["protocol"] == "P1") & (main_t["slice"] == sl) & (main_t["model"] == model) & (main_t["metric"] == metric)]
        return None if r.empty else (float(r["point"].iloc[0]), float(r["ci_lo"].iloc[0]), float(r["ci_hi"].iloc[0]))

    for m in ("lgbm_graph", "lgbm_graph_noprior"):
        r = mget(m, "mcc")
        if r:
            checks.append({"check": f"{m}: MCC не вище 0,8 (інакше підозра на витік)", "passed": bool(r[0] <= 0.8), "detail": f"MCC {r[0]:.3f} [{r[1]:.3f}; {r[2]:.3f}]"})
    for m in ("lgbm_graph_shuf", "lgbm_both_shuf", "tfidf_shuf", "bge_lr_shuf", "xlmr_shuf"):
        r = mget(m, "mcc")
        if r:
            checks.append({"check": f"{m}: перемішані мітки, ДІ MCC включає 0", "passed": bool(r[1] <= 0 <= r[2]), "detail": f"MCC {r[0]:.3f} [{r[1]:.3f}; {r[2]:.3f}]"})
    r_rw, r_b = mget("lgbm_both_rewired", "macro_f1"), mget("lgbm_both", "macro_f1")
    if r_rw and r_b:
        checks.append({"check": "перезв'язана мережа і розв'язані ознаки копій: lgbm_both не кращий за оригінал", "passed": bool(r_rw[0] <= r_b[0] + 1e-9),
                       "detail": f"macro-F1 {r_rw[0]:.3f} проти {r_b[0]:.3f}"})
    checks.append({"check": "структура прогнозів (дублікати, покриття, діапазон)", "passed": bool(pcheck["n_duplicates"] == 0 and pcheck["all_covered"] and pcheck["p_in_range"]),
                   "detail": json.dumps(pcheck, ensure_ascii=False)[:300]})
    (out / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1), encoding="utf-8")
    write_run_json(out, "scripts/s07_eval.py", {"n_boot": nb, "seed": sd, "best_text": best_text, "best_hybrid": best_hyb, "text_dirs": [str(d) for d in text_dirs], "hybrid_dir": str(hyb)})
    for c in checks:
        print(f"  [{'ok' if c['passed'] else 'УВАГА'}] {c['check']}: {c['detail']}")
    print(f"Готово за {time.time() - t0:.0f} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
