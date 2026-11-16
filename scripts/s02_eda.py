"""S2: описова аналітика повних даних TeleNarratives (дані з OSF, знімок графу 11.03.2026).

Запуск:  python3 -I scripts/s02_eda.py --raw-dir <data/raw> [--out results/02_eda] [--lang-sample 150000]
Дані читаються з parquet-набору (messages, channels, forward_edges, semantic_edges, кодування наративів)
і з telegram_messages.parquet. Золотий набір береться за шляхом з configs/base.yaml.
Результат: CSV-таблиці, summary.json, checks.json, run.json, figures/*.png.
Код завершується з ненульовим статусом, якщо порушено хоч один жорсткий інваріант (hard=True у checks.json).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narr import eda  # noqa: E402
from narr.io import ROOT, load_config, path_of, validate_golden, write_run_json  # noqa: E402
from narr.language import detect_language_parallel  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="папка data/raw (з telegram_messages.parquet і розпакованим parquet-набором)")
    ap.add_argument("--out", default=None, help="папка результатів (типово results/02_eda у корені репозиторію)")
    ap.add_argument("--golden", default=None, help="шлях до golden set CSV (типово з configs/base.yaml)")
    ap.add_argument("--lang-sample", type=int, default=150_000, help="розмір вибірки нерозмічених повідомлень для визначення мови")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--no-validate-gold", action="store_true", help="лише для тестів на синтетичних даних")
    args = ap.parse_args(argv)

    cfg = load_config()
    seed = cfg["seed"]
    out = Path(args.out) if args.out else ROOT / "results" / "02_eda"
    golden = Path(args.golden) if args.golden else path_of(cfg, "golden")

    t0 = time.time()
    print("Читаю дані ...", flush=True)
    inp = eda.load_inputs(args.raw_dir, golden)
    print(f"  messages {len(inp['msgs']):,}, telegram_messages {len(inp['tg']):,}, золотих {len(inp['gold'])}, "
          f"{time.time() - t0:.0f} с", flush=True)

    print("Рахую таблиці і мову ...", flush=True)
    lang_fn = lambda s: detect_language_parallel(s, n_jobs=args.jobs)  # noqa: E731
    tables, summary, checks = eda.run_all(inp, lang_fn, lang_sample_unlabeled=args.lang_sample, seed=seed)

    if not args.no_validate_gold:
        problems = validate_golden(inp["gold"], cfg)
        checks.append({"hard": True, "check": "Золотий набір проходить validate_golden", "passed": not problems,
                       "detail": "; ".join(problems) if problems else "412 рядків, 148 наративних, 56 каналів, мітки узгоджені"})

    eda.save_all(tables, summary, checks, out)
    figs = eda.make_figures(tables, summary, out / "figures")
    write_run_json(out, "scripts/s02_eda.py", {"raw_dir": str(args.raw_dir), "lang_sample": args.lang_sample, "seed": seed,
                                               "golden": str(golden), "seconds": round(time.time() - t0)})
    print(f"Збережено {len(tables)} таблиць і {len(figs)} графіків у {out}", flush=True)

    bad = [c for c in checks if c["hard"] and not c["passed"]]
    for c in checks:
        mark = "ok  " if c["passed"] else ("ПОМИЛКА" if c["hard"] else "розбіжність")
        print(f"[{mark}] {c['check']}: {c['detail']}")
    if bad:
        print(f"\nЖорстких порушень: {len(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
