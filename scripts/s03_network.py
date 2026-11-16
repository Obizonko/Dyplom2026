"""S3: мережа каналів за пересиланнями (канал = вершина, ребро = пересилання, вага = кількість повідомлень).

Запуск:  python3 -I scripts/s03_network.py [--edges data_small/channel_flow_edges.csv] [--channels results/02_eda/per_channel.csv]
                                           [--out results/03_network] [--n-perm 10000]
Результат: centrality.csv, communities.csv, community_summary.csv, cross_group.json, summary.json, checks.json, run.json,
channel_network.gexf (для Gephi), figures/01_network_map.png. Код завершується з ненульовим статусом при порушенні жорсткого інваріанта.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narr import eda, network as nw  # noqa: E402
from narr.io import ROOT, load_config, path_of, write_run_json  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default=str(ROOT / "data_small" / "channel_flow_edges.csv"))
    ap.add_argument("--channels", default=str(ROOT / "results" / "02_eda" / "per_channel.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "03_network"))
    ap.add_argument("--golden", default=None, help="золотий набір (типово з configs/base.yaml)")
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--lenient", action="store_true", help="не звіряти з перевіреною таблицею (лише для тестів)")
    args = ap.parse_args(argv)

    cfg = load_config()
    seed = cfg["seed"]
    out = Path(args.out)
    t0 = time.time()
    edges = pd.read_csv(args.edges)
    per_channel = pd.read_csv(args.channels)
    print(f"Ребер {len(edges)}, каналів у таблиці {len(per_channel)}", flush=True)

    tables, summary, checks, g = nw.run_all(edges, per_channel, n_perm=args.n_perm, seed=seed, strict=not args.lenient)
    golden = Path(args.golden) if args.golden else path_of(cfg, "golden")
    if golden.exists():
        gold = pd.read_csv(golden)
        summary["gold_coverage"] = nw.gold_coverage(gold, set(tables["communities"]["channel_id"]), cfg["columns"]["channel"], cfg["columns"]["binary"])
    else:
        print(f"Золотий файл не знайдено ({golden}), покриття золотого набору пропущено", flush=True)
    eda.save_all(tables, summary, checks, out)
    cg = {k: summary[k] for k in ("cross_group", "cross_group_permutation", "assortativity", "nmi_partition_vs_group")}
    (out / "cross_group.json").write_text(__import__("json").dumps(cg, ensure_ascii=False, indent=2), encoding="utf-8")

    nodes = tables["centrality"].merge(tables["communities"][["channel_id", "community_log", "community_unweighted"]], on="channel_id")
    nodes["narrative_presence"] = nodes["table_label"].eq("n")
    nw.save_gexf(g, nodes, out / "channel_network.gexf")
    nw.make_map(g, nodes, out / "figures" / "01_network_map.png", seed=seed)
    write_run_json(out, "scripts/s03_network.py", {"edges": str(args.edges), "channels": str(args.channels), "n_perm": args.n_perm,
                                                   "seeds": summary["seeds"], "seed": seed, "golden": str(golden) if golden.exists() else None, "seconds": round(time.time() - t0)})
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
