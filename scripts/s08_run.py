"""S8: GraphSAGE на гетерогенному графі (повідомлення, канали). Швидкий режим: ребра семантичні (між каналами, cos >= 0,985) і публікація каналом.

Запуск:  python -I scripts/s08_run.py --out results/quick/08_gnn [--seeds 3] [--epochs 12]
Перевірка (sanity): python -I scripts/s08_run.py --sanity   (перенавчання на 32 прикладах, втрата має впасти майже до нуля)
Золоті повідомлення в графі присутні зі своїми ребрами, але без міток. Мітки слабкі лише в train-вузлів. Вибір епохи за valid macro-F1 слабких міток.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narr import features as ft, gnn, textmodels as tm  # noqa: E402
from narr.io import ROOT, write_run_json  # noqa: E402
from narr.splits import group_valid_split  # noqa: E402


def load_graph_inputs(emb_dir: Path, edges_path: Path, comm_path: Path, cache: Path, log=print):
    index = pd.read_parquet(emb_dir / "bge_m3_index.parquet")
    keys = index["key"].to_numpy()
    E = np.asarray(np.load(emb_dir / "bge_m3_fp16.npy", mmap_mode="r"), dtype=np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    if cache.exists():
        pairs = pd.read_parquet(cache)
    else:
        t0 = time.time()
        pairs = ft.semantic_pairs(E, index["channel_id"].to_numpy(), threshold=0.975)  # надмножина: пороги 0,985 і 0,99 відбираються за cos
        cache.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_parquet(cache, index=False)
        log(f"Семантичні пари між каналами: {len(pairs)} за {time.time() - t0:.0f} с")
    net = ft.channel_network_features(pd.read_csv(edges_path), pd.read_csv(comm_path))
    chan_ids = np.sort(index["channel_id"].unique())
    cf = net.reindex(chan_ids)[[c for c in ft.NET_COLS if c != "net_community"]]
    cf = np.log1p(cf.clip(lower=0))  # довгі хвости ступенів і PageRank
    cf = (cf - cf.mean()) / cf.std(ddof=0).replace(0, 1)
    cf = cf.fillna(0.0)  # канали поза мережею: нулі (середнє), членства як ознаки немає
    return index, keys, E, pairs, chan_ids, cf.to_numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default=str(ROOT / "data" / "interim" / "embeddings"))
    ap.add_argument("--splits-dir", default=str(ROOT / "splits"))
    ap.add_argument("--out", default=str(ROOT / "results" / "08_gnn"))
    ap.add_argument("--protocols", default="P1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--arch", default="sage", choices=["sage", "rgcn", "hgt"], help="для --sanity")
    ap.add_argument("--variants", default="gnn_sage,gnn_sage_notext",
                    help="gnn_sage, gnn_sage_notext (швидкий режим) або H2: h2_nosem, h2_sem975, h2_sem985, h2_sem990, h2_nofwd")
    ap.add_argument("--fwd", default=str(ROOT / "data" / "raw" / "extracted" / "parquet_set" / "forward_edges.parquet"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    t_all = time.time()
    index, keys, E, pairs, chan_ids, cf = load_graph_inputs(
        Path(args.emb_dir), ROOT / "data_small" / "channel_flow_edges.csv", ROOT / "results" / "03_network" / "communities.csv",
        ROOT / "data" / "interim" / "features" / "sem_pairs_0975.parquet")
    row = pd.Series(np.arange(len(keys)), index=keys)
    th_map = index.set_index("key")["text_hash"]
    msg_channel = index["channel_id"].to_numpy()
    print(f"Вузли msg {len(keys)}, chan {len(chan_ids)}, семантичних пар {len(pairs)}", flush=True)

    if args.sanity:
        splits = tm.load_splits(args.splits_dir, ("P1",))
        tr = splits[0]["train"]
        rng = np.random.default_rng(0)
        pos, neg = tr[tr["weak_narrative"]], tr[~tr["weak_narrative"]]
        sub = pd.concat([pos.sample(16, random_state=0), neg.sample(16, random_state=0)])
        idx = row.loc[sub["key"]].to_numpy()
        y = np.zeros(len(keys), dtype=np.int64)
        y[idx] = sub["weak_narrative"].astype(int).to_numpy()
        data = gnn.build_graph(E, msg_channel, pairs, chan_ids, cf, text=True)
        r = gnn.train_gnn(data, y, idx, idx, {"train": idx}, seed=0, epochs=60, hidden=args.hidden, batch_size=32, patience=60, arch=args.arch, log=lambda *_: None)
        h = r["history"]
        print(f"САНІТІ: втрата {h[0]['train_loss']:.4f} -> {h[-1]['train_loss']:.4f}; valid(=train) macro-F1 {r['valid_macro_f1']:.3f} на епосі {r['epoch']}")
        assert h[-1]["train_loss"] < 0.05 and r["valid_macro_f1"] > 0.99, "перенавчання на 32 прикладах не вдалося"
        print("САНІТІ ПРОЙДЕНО")
        return 0

    (out / "units").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    protocols = tuple(args.protocols.split(","))
    splits = tm.load_splits(args.splits_dir, protocols)
    for s in splits:
        for part in ("train", "test"):
            s[part]["text_hash"] = s[part]["key"].map(th_map)
    fwd = None
    if any((v.startswith("h2_") and v != "h2_nofwd") or (v.split("_")[0] in ("sage", "rgcn", "hgt") and v.split("_")[-1] != "nofwd") for v in args.variants.split(",")):
        import pyarrow.parquet as pq

        fe = pq.read_table(args.fwd, columns=["logical_src_channel_id", "logical_src_message_id", "logical_dst_channel_id", "logical_dst_message_id"]).to_pandas()
        ks, kd = tm.make_key(fe["logical_src_channel_id"], fe["logical_src_message_id"]), tm.make_key(fe["logical_dst_channel_id"], fe["logical_dst_message_id"])
        ok = pd.Series(ks).isin(row.index).to_numpy() & pd.Series(kd).isin(row.index).to_numpy()
        fwd = pd.DataFrame({"src": row.loc[ks[ok]].to_numpy(), "dst": row.loc[kd[ok]].to_numpy()})
        fwd = fwd[fwd["src"] != fwd["dst"]].drop_duplicates()
        print(f"Форвард-ребра: {len(fe)} у наборі, обидва кінці в наборі ембедингів {int(ok.sum())}, унікальних без петель {len(fwd)}", flush=True)
    cp_path = ROOT / "data" / "interim" / "features" / "copub_pairs.parquet"
    if any(v.split("_")[0] in ("sage", "rgcn", "hgt") and v.split("_")[-1] != "nocopub" for v in args.variants.split(",")):
        if cp_path.exists():
            copub = pd.read_parquet(cp_path)
        else:
            copub = ft.copublication_pairs(index["date_utc"], index["channel_id"].to_numpy())
            cp_path.parent.mkdir(parents=True, exist_ok=True)
            copub.to_parquet(cp_path, index=False)
        print(f"co_publication: {len(copub)} пар (вікно 1 год, різні канали, до 10 на вузол)", flush=True)
    else:
        copub = None
    SPECS = {"gnn_sage": dict(text=True, thr=0.985, fwd=False), "gnn_sage_notext": dict(text=False, thr=0.985, fwd=False),
             "h2_nosem": dict(text=True, thr=None, fwd=True), "h2_sem975": dict(text=True, thr=0.975, fwd=True),
             "h2_sem985": dict(text=True, thr=0.985, fwd=True), "h2_sem990": dict(text=True, thr=0.99, fwd=True),
             "h2_nofwd": dict(text=True, thr=0.985, fwd=False)}
    def spec_of(variant: str) -> dict:
        """sage|rgcn|hgt + _full|_nosem|_sem975|_sem990|_nofwd|_nocopub|_notext: повний граф (cos 0,985, форварди, co_publication, текст) з однією зміною."""
        if variant in SPECS:
            return {**SPECS[variant], "arch": "sage", "copub": False}
        arch, _, suf = variant.partition("_")
        sp = dict(text=True, thr=0.985, fwd=True, copub=True, arch=arch)
        sp.update({"full": {}, "nosem": {"thr": None}, "sem975": {"thr": 0.975}, "sem990": {"thr": 0.99}, "nofwd": {"fwd": False},
                   "nocopub": {"copub": False}, "notext": {"text": False}}[suf])
        return sp

    for variant in args.variants.split(","):
        sp = spec_of(variant)
        pr_v = None if sp["thr"] is None else pairs[pairs["cos"] >= sp["thr"] - 1e-9]
        data = gnn.build_graph(E, msg_channel, pr_v, chan_ids, cf, text=sp["text"], fwd=fwd if sp["fwd"] else None, copub=copub if sp["copub"] else None)
        print(f"Граф {variant}: семантичних пар {0 if pr_v is None else len(pr_v)}, форвардів {len(fwd) if sp['fwd'] else 0}", flush=True)
        for pr in protocols:
            for sd in range(args.seeds):
                un = tm.unit_name(variant, pr, sd)
                pj, pp = out / "units" / f"{un}.json", out / "predictions" / f"{un}.parquet"
                if pj.exists() and pp.exists():
                    print(f"[пропуск] {un}", flush=True)
                    continue
                print(f"[{time.strftime('%H:%M:%S')}] {un}", flush=True)
                t0 = time.time()
                preds, per = [], []
                for s in [x for x in splits if x["protocol"] == pr]:
                    tr, te = s["train"].reset_index(drop=True), s["test"].reset_index(drop=True)
                    m_tr, m_va = group_valid_split(tr, 0.1, 42)
                    y = np.zeros(len(keys), dtype=np.int64)
                    ti = row.loc[tr["key"]].to_numpy()
                    y[ti] = tr["weak_narrative"].astype(int).to_numpy()
                    te_idx = row.loc[te["key"]].to_numpy()
                    r = gnn.train_gnn(data, y, ti[m_tr.to_numpy()], ti[m_va.to_numpy()], {"test": te_idx}, seed=sd, epochs=args.epochs, hidden=args.hidden, arch=sp["arch"])
                    ids, p = r["probs"]["test"]
                    pm_ = pd.Series(p, index=ids).loc[te_idx].to_numpy()
                    preds.append(pd.DataFrame({"protocol": pr, "fold": s["fold"], "model": variant, "seed": sd, "channel_id": te["channel_id"].to_numpy(),
                                               "message_id": te["message_id"].to_numpy(), "p": pm_}))
                    per.append({"fold": s["fold"], "valid_macro_f1": r["valid_macro_f1"], "best_epoch": r["epoch"], "history": r["history"]})
                pd.concat(preds, ignore_index=True).to_parquet(pp, index=False)
                info = {"model": variant, "protocol": pr, "seed": sd, "valid_macro_f1": float(np.mean([x["valid_macro_f1"] for x in per])), "splits": per,
                        "seconds": round(time.time() - t0, 1)}
                pj.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"  готово за {info['seconds']} с, valid macro-F1 {info['valid_macro_f1']:.4f}", flush=True)
        del data
    write_run_json(out, "scripts/s08_run.py", {"args": vars(args), "n_semantic_pairs": int(len(pairs)), "seconds": round(time.time() - t_all),
                                               "edges": "sem(cross-channel)+pub, форварди повідомлень лише у варіантах h2_*; co_publication немає"})
    print(f"Готово за {time.time() - t_all:.0f} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
