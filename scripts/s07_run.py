"""S7: простий гібрид без GNN. LightGBM на графових ознаках, на ембедингах bge-m3 і на обох. Optuna за валідацією слабких міток.

Запуск (локально):
  python -I scripts/s07_run.py --out results/07_simple_hybrid [--protocols P1,P2] [--n-trials 50]
Швидкий режим (попередній, для презентації): --n-trials 10 --search-rows 20000 --out results/quick/07_simple_hybrid
Результат: predictions/<модель>__<протокол>__s<seed>.parquet (як в S6), units/*.json, shap_groups.csv, feature_importance.csv, run.json.
Золоті повідомлення використовуються лише як тест; вибір параметрів лише за валідацією слабких міток.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narr import features as ft, hybrid as hy, textmodels as tm  # noqa: E402
from narr.io import ROOT, write_run_json  # noqa: E402
from narr.splits import group_valid_split  # noqa: E402

MODELS = ("lgbm_graph", "lgbm_graph_noprior", "lgbm_emb", "lgbm_both")


def load_base(emb_dir: Path, cache: Path, log=print):
    """Індекс S4, ембединги (float32, нормовані) і семантичні ознаки (кеш)."""
    index = pd.read_parquet(emb_dir / "bge_m3_index.parquet")
    keys = np.load(emb_dir / "bge_m3_keys.npy")
    assert (index["key"].to_numpy() == keys).all(), "індекс і ключі ембедингів не збігаються"
    emb = np.load(emb_dir / "bge_m3_fp16.npy", mmap_mode="r")
    if cache.exists():
        sem = pd.read_parquet(cache)
        log(f"Семантичні ознаки з кешу {cache}")
    else:
        t0 = time.time()
        E = np.asarray(emb, dtype=np.float32)
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        sem = ft.semantic_copy_features(E, index["channel_id"].to_numpy(), index["date_utc"], log=log)
        sem.insert(0, "key", keys)
        cache.parent.mkdir(parents=True, exist_ok=True)
        sem.to_parquet(cache, index=False)
        log(f"Семантичні ознаки пораховано за {time.time() - t0:.0f} с")
    base = sem.set_index("key")
    base["is_forward_endpoint"] = index.set_index("key")["in_fwd"].astype(int)
    return index, keys, emb, base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default=str(ROOT / "data" / "interim" / "embeddings"))
    ap.add_argument("--splits-dir", default=str(ROOT / "splits"))
    ap.add_argument("--out", default=str(ROOT / "results" / "07_simple_hybrid"))
    ap.add_argument("--edges", default=str(ROOT / "data_small" / "channel_flow_edges.csv"))
    ap.add_argument("--communities", default=str(ROOT / "results" / "03_network" / "communities.csv"))
    ap.add_argument("--protocols", default="P1,P2,P3")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--search-rows", type=int, default=0, help="швидкий режим: підбір параметрів на підвибірці train цього розміру")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--control-trials", type=int, default=5)
    ap.add_argument("--v11", action="store_true", help="преєстрація 1.1 (11.1): ваги пулу, валідація за компонентами копій, без is_unbalance, "
                    "Optuna лише на P1 seed 0, решта з параметрами P1, перестановочні контролі")
    ap.add_argument("--n-perm", type=int, default=20, help="версія 1.1: кількість перестановок міток для контролів")
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "units").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    emb_dir = Path(args.emb_dir)
    t_all = time.time()
    models = tuple(args.models.split(","))
    protocols = tuple(args.protocols.split(","))

    index, keys, emb, base = load_base(emb_dir, ROOT / "data" / "interim" / "features" / "sem_features.parquet")
    row = pd.Series(np.arange(len(keys)), index=keys)
    th_map = index.set_index("key")["text_hash"]
    edges = pd.read_csv(args.edges)
    comm = pd.read_csv(args.communities)
    net = ft.channel_network_features(edges, comm)
    net_rw = ft.channel_network_features(ft.rewire_edges(edges, seed=42), comm)
    rng = np.random.default_rng(42)
    base_perm = base.iloc[rng.permutation(len(base))].set_axis(base.index)  # для контролю: ознаки копій розв'язано з повідомленнями
    print(f"Мережа каналів: {len(net)} каналів; повідомлень у наборі {len(keys)}; режим: {args.n_trials} спроб Optuna, "
          f"пошук на {args.search_rows or 'усьому train'} рядках", flush=True)

    splits = tm.load_splits(args.splits_dir, protocols)
    for s in splits:
        for part in ("train", "test"):
            s[part]["text_hash"] = s[part]["key"].map(th_map)
    if args.v11:
        from narr import weights as wt

        cc, pk, _ = wt.load_v11_inputs(ROOT)
        pd.DataFrame(wt.attach_weights(splits, pk, cc, th_map)).to_csv(out / "weights_summary.csv", index=False)
        print("Версія 1.1: ваги пулу і валідація за компонентами копій додано", flush=True)
    by_pr = {pr: [s for s in splits if s["protocol"] == pr] for pr in protocols}
    importance_rows, shap_rows = [], []

    def run_unit(name: str, pr: str, seed: int, n_trials: int, feat: str, shuffle: bool = False, rewired: bool = False, fixed: dict | None = None,
                 shuffle_seed: int = 1234):
        un = tm.unit_name(name, pr, seed)
        pj, pp = out / "units" / f"{un}.json", out / "predictions" / f"{un}.parquet"
        if pj.exists() and pp.exists():
            print(f"[пропуск] {un}", flush=True)
            return json.loads(pj.read_text(encoding="utf-8"))
        print(f"[{time.strftime('%H:%M:%S')}] {un}", flush=True)
        t0 = time.time()
        preds, per = [], []
        for s in by_pr[pr]:
            tr, te = s["train"].reset_index(drop=True), s["test"].reset_index(drop=True)
            if args.v11:
                m_va = tr["is_valid"].astype(bool)
                m_tr = ~m_va
            else:
                m_tr, m_va = group_valid_split(tr, 0.1, 42)
            y = tr["weak_narrative"].astype(int).to_numpy()
            if shuffle:
                y = np.random.default_rng(shuffle_seed).permutation(y)
            nt, bs = (net_rw, base_perm) if rewired else (net, base)
            Xg_tr, Xg_te = ft.assemble(tr, bs, nt), ft.assemble(te, bs, nt)
            Xg_tr[ft.PRIOR_COL] = ft.oof_channel_prior(tr)
            Xg_te[ft.PRIOR_COL] = ft.prior_for(te, tr)
            parts_tr, parts_te = [], []
            if feat in ("graph", "both"):
                cols = ft.GRAPH_COLS
                parts_tr.append(Xg_tr[cols]), parts_te.append(Xg_te[cols])
            if feat == "graph_noprior":
                parts_tr.append(Xg_tr[ft.GRAPH_COLS_NO_PRIOR]), parts_te.append(Xg_te[ft.GRAPH_COLS_NO_PRIOR])
            if feat in ("emb", "both"):
                ecols = [f"e{i}" for i in range(emb.shape[1])]
                parts_tr.append(pd.DataFrame(np.asarray(emb[row.loc[tr["key"]].to_numpy()], dtype=np.float32), columns=ecols, index=tr.index))
                parts_te.append(pd.DataFrame(np.asarray(emb[row.loc[te["key"]].to_numpy()], dtype=np.float32), columns=ecols, index=te.index))
            Xtr, Xte = pd.concat(parts_tr, axis=1), pd.concat(parts_te, axis=1)
            cat = ["net_community"] if "net_community" in Xtr.columns else None
            if cat:  # LightGBM: категорія як ціле, NaN залишається
                for X in (Xtr, Xte):
                    X["net_community"] = X["net_community"].astype("float64")
            a, v = m_tr.to_numpy(), m_va.to_numpy()
            wkw = dict(wtr=tr["weight"].to_numpy()[a], wva=tr["weight"].to_numpy()[v], balance=False) if args.v11 else {}
            r = hy.tune_and_fit(Xtr[a], y[a], Xtr[v], y[v], Xte, n_trials=n_trials, seed=seed, num_threads=args.threads,
                                search_rows=args.search_rows or None, categorical=cat, log=print, fixed_params=fixed, **wkw)
            preds.append(pd.DataFrame({"protocol": pr, "fold": s["fold"], "model": name, "seed": seed, "channel_id": te["channel_id"].to_numpy(),
                                       "message_id": te["message_id"].to_numpy(), "p": r["p"]}))
            per.append({"fold": s["fold"], "n_train": int(a.sum()), "n_valid": int(v.sum()), "valid_macro_f1": r["valid_macro_f1"], "params": r["params"],
                        "best_iteration": r["best_iteration"], "n_trials": r["n_trials"], "search_rows": r["search_rows"], "n_features": int(Xtr.shape[1])})
            if name in ("lgbm_graph", "lgbm_graph_noprior", "lgbm_emb", "lgbm_both") and pr == "P1" and seed == 0 and not shuffle and not rewired:
                imp = pd.Series(r["booster"].feature_importance("gain"), index=Xtr.columns)
                importance_rows.append(imp.rename(name))
                import shap

                samp = Xtr[a].sample(n=min(1500, int(a.sum())), random_state=0)
                sv = shap.TreeExplainer(r["booster"]).shap_values(Xte)
                sv = sv[1] if isinstance(sv, list) else sv
                sv = np.asarray(sv)
                grp = {c: ("graph" if not c.startswith("e") or c in ("is_forward_endpoint",) else "text") for c in Xte.columns}
                absv = pd.Series(np.abs(sv).mean(axis=0), index=Xte.columns)
                shap_rows.append(pd.DataFrame({"model": name, "feature": Xte.columns, "mean_abs_shap_gold": absv.to_numpy(), "group": [grp[c] for c in Xte.columns]}))
                del samp
                if name in ("lgbm_both", "lgbm_graph"):  # знакові внески груп ознак для прототипу (S12): сума SHAP по графових і по текстових колонках
                    isg = np.array([grp[c] == "graph" for c in Xte.columns])
                    ev = shap.TreeExplainer(r["booster"]).expected_value
                    ev = float(np.ravel(ev)[-1])
                    pd.DataFrame({"key": te["key"].to_numpy(), "shap_graph": sv[:, isg].sum(axis=1), "shap_text": sv[:, ~isg].sum(axis=1),
                                  "base_value": ev}).to_parquet(out / f"shap_gold_groups_{name}.parquet", index=False)
        pred = pd.concat(preds, ignore_index=True)
        info = {"model": name, "protocol": pr, "seed": seed, "valid_macro_f1": float(np.mean([p["valid_macro_f1"] for p in per])),
                "splits": per, "seconds": round(time.time() - t0, 1), "shuffled_labels": shuffle, "rewired_graph": rewired,
                "shuffle_seed": shuffle_seed if shuffle else None, "fixed_params": fixed is not None, "rules": "v1.1" if args.v11 else "v1"}
        pred.to_parquet(pp, index=False)
        pj.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  готово за {info['seconds']} с, valid macro-F1 {info['valid_macro_f1']:.4f}", flush=True)
        return info

    feat_of = {"lgbm_graph": "graph", "lgbm_graph_noprior": "graph_noprior", "lgbm_emb": "emb", "lgbm_both": "both"}
    if args.v11:  # Optuna лише на P1 seed 0; P1 seed 1-4, P2, P3 з тими самими параметрами; контролі з параметрами основної моделі
        assert protocols[0] == "P1", "версія 1.1: P1 має йти першим"
        best = {}
        for name in models:
            best[name] = run_unit(name, "P1", 0, args.n_trials, feat_of[name])["splits"][0]["params"]
            for pr in protocols:
                for sd in range(args.seeds):
                    if (pr, sd) != ("P1", 0):
                        run_unit(name, pr, sd, 0, feat_of[name], fixed=best[name])
        if not args.no_controls:
            for name in [m for m in ("lgbm_graph", "lgbm_both") if m in models]:
                for k in range(args.n_perm):
                    run_unit(name + "_shuf", "P1", k, 0, feat_of[name], shuffle=True, fixed=best[name], shuffle_seed=1000 + k)
            if "lgbm_both" in models:
                run_unit("lgbm_both_rewired", "P1", 0, 0, "both", rewired=True, fixed=best["lgbm_both"])
        protocols = ()  # нижче лише версія 1
    for pr in protocols:
        for name in models:
            for sd in range(args.seeds):
                run_unit(name, pr, sd, args.n_trials, feat_of[name])
    if not args.no_controls and "P1" in by_pr and not args.v11:
        if "lgbm_graph" in models:
            run_unit("lgbm_graph_shuf", "P1", 0, args.control_trials, "graph", shuffle=True)
        if "lgbm_both" in models:
            run_unit("lgbm_both_shuf", "P1", 0, args.control_trials, "both", shuffle=True)
            run_unit("lgbm_both_rewired", "P1", 0, args.control_trials, "both", rewired=True)
    if importance_rows:
        pd.concat(importance_rows, axis=1).fillna(0).to_csv(out / "feature_importance_gain.csv")
    if shap_rows:
        sh = pd.concat(shap_rows, ignore_index=True)
        sh.to_csv(out / "shap_features.csv", index=False)
        sh.groupby(["model", "group"], as_index=False)["mean_abs_shap_gold"].sum().to_csv(out / "shap_groups.csv", index=False)
    write_run_json(out, "scripts/s07_run.py", {"args": vars(args), "seconds": round(time.time() - t_all), "n_channels_in_network": int(len(net))})
    print(f"Готово за {time.time() - t_all:.0f} с", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
