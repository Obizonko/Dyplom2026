"""S6 (хмара): оцінка текстових моделей на золотих за P1-P3 з прогнозів Colab.

Вхід: results/06_text_models/{predictions,units,colab_run.json} (з Colab), splits/, results/05_protocol/ (золоті, розмітка явний/неявний).
Вихід у ту саму папку: metrics_main.csv, metrics_subsets.csv, metrics_cluster.csv, paired_diffs.csv, recall_by_explicitness.csv, seeds.csv,
model_selection.csv, valid_units.csv, predictions_mean.parquet, checks.json, run.json.

Запуск:  python3 -I scripts/s06_eval.py            (повний, 1000 бутстреп-вибірок)
         python3 -I scripts/s06_eval.py --n-boot 50   (швидка перевірка)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from narr import text_eval as te  # noqa: E402
from narr import textmodels as tm  # noqa: E402
from narr.io import ROOT, write_run_json  # noqa: E402


def load_units(res: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((res / "units").glob("*.json"))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "06_text_models"))
    ap.add_argument("--splits", default=str(ROOT / "splits"))
    ap.add_argument("--s5", default=str(ROOT / "results" / "05_protocol"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    res, s5 = Path(args.results), Path(args.s5)
    t0 = time.time()

    pred = pd.concat([pd.read_parquet(p) for p in sorted((res / "predictions").glob("*.parquet"))], ignore_index=True)
    units = load_units(res)
    colab = json.loads((res / "colab_run.json").read_text(encoding="utf-8")) if (res / "colab_run.json").exists() else {}
    g = te.make_gold_table(pd.read_csv(s5 / "gold_membership.csv"), pd.read_csv(s5 / "implicit_labeling.csv"), pd.read_csv(s5 / "implicit_auto_b.csv"))
    protocols = [p for p in te.PROTOCOLS if p in set(pred["protocol"])]
    splits = tm.load_splits(args.splits, protocols)
    models_present = set(pred["model"])
    print(f"Прогнозів {len(pred)}, одиниць {len(units)}, моделі {sorted(models_present)}, протоколи {protocols}", flush=True)

    pcheck = te.check_predictions(pred, g)
    pm = pd.concat([te.mean_over_seeds(pred), te.comparator_frame(g, splits)], ignore_index=True)
    pm.to_parquet(res / "predictions_mean.parquet", index=False)

    vt = te.valid_table(units)
    vt.to_csv(res / "valid_units.csv", index=False)
    sel = pd.concat([te.select_best(vt, p) for p in protocols], ignore_index=True)
    sel.to_csv(res / "model_selection.csv", index=False)
    best = {p: (sel[(sel["protocol"] == p) & sel["selected"]]["model"].iloc[0] if ((sel["protocol"] == p) & sel["selected"]).any() else None) for p in protocols}
    print("Найкраща текстова за валідацією:", best, flush=True)

    all_models = [m for m in (*te.TEXT_MODELS, *te.COMPARATORS, *te.CONTROLS) if m in set(pm["model"])]
    nb, sd = args.n_boot, args.seed
    main_t = te.metrics_table(pm, g, protocols, all_models, te.SLICES, ["all"], nb, sd)
    sub_t = te.metrics_table(pm, g, ["P1"], [m for m in all_models if m not in te.CONTROLS], ["all", "ru"],
                             [s for s in te.subset_masks(g) if s != "all"], nb, sd)
    print(f"[{time.time() - t0:5.0f} с] метрики з ДІ готові", flush=True)
    clu_t = te.cluster_table(pm, g, "P1", [m for m in all_models if m not in te.CONTROLS], ["all", "ru"], nb, sd)
    rec_t = pd.concat([te.recall_by_explicitness(pm, g, "P1", [m for m in all_models if m not in te.CONTROLS], ["all", "ru"], nb, sd)], ignore_index=True)

    pairs = []
    for pr in protocols:
        b = best[pr]
        cand = [(b, "lang_rule"), (b, "channel_prior"), (b, "majority"), ("xlmr", "bge_lr"), ("bge_lr", "tfidf"), ("xlmr", "tfidf")]
        pairs.append(te.paired_table(pm, g, pr, [(x, y) for x, y in cand if x and x != y], ["all", "ru"], nb, sd).assign(best_text=b))
    pair_t = pd.concat(pairs, ignore_index=True)
    seed_t = te.seed_table(pred, g)
    print(f"[{time.time() - t0:5.0f} с] парні різниці, кластерний бутстреп, recall за явністю готові", flush=True)

    for name, t in (("metrics_main", main_t), ("metrics_subsets", sub_t), ("metrics_cluster", clu_t), ("paired_diffs", pair_t),
                    ("recall_by_explicitness", rec_t), ("seeds", seed_t)):
        t.to_csv(res / f"{name}.csv", index=False)

    checks = te.run_checks(main_t, pcheck, vt, seed_t, colab.get("integrity"), models_present)
    (res / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1), encoding="utf-8")
    write_run_json(res, "scripts/s06_eval.py", {"n_boot": nb, "seed": sd, "best_text": best, "protocols": protocols, "models": sorted(models_present),
                                                "colab": {k: colab.get(k) for k in ("seconds_total", "gpu", "torch", "split_sizes")}})
    n_hard = sum(1 for c in checks if c["hard"] and not c["passed"])
    for c in checks:
        print(f"  [{'ok' if c['passed'] else ('ПОМИЛКА' if c['hard'] else 'розбіжність')}] {c['check']}: {c['detail']}")
    print(f"Готово за {time.time() - t0:.0f} с, жорстких не пройдено {n_hard}")
    return 1 if n_hard else 0


if __name__ == "__main__":
    sys.exit(main())
