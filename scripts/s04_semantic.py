"""S4: семантичний шар. Ембединги bge-m3, власні семантичні ребра, порівняння з ребрами набору, копії золотих, досяжність, UMAP.

Запуск (Colab з GPU):
  python3 -I scripts/s04_semantic.py --raw-dir <data/raw> --out <results/04_semantic> --interim <data/interim/embeddings>
Без GPU і без bge-m3 можна перевірити логіку: `--encoder hash --skip-umap` (хеш-ембединги, лише для тестів).

Що робить: (1) збирає набір повідомлень для ембедингів (золоті, розмічена партія, кінці семантичних ребер набору), (2) кодує їх bge-m3
у float16 і кешує на диску, (3) звіряє наш косинус з `similarity_score` ребер набору, (4) будує власні ребра hnswlib (k=80),
(5) шукає точні копії золотих, (6) рахує досяжність між групами каналів для форвардів і семантики, (7) UMAP,
(8) звіряє forward_edges з channel_flow_edges (залишок S3). Код завершується з ненульовим статусом, якщо порушено жорсткий інваріант.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narr import eda, semantic as sm  # noqa: E402
from narr.eda import KEY_MULT  # noqa: E402
from narr.io import ROOT, load_config, path_of, write_run_json  # noqa: E402

KMAX = 4


def th(n) -> str:
    return f"{int(n):,}".replace(",", " ")

LAYER_ORDER = ["forward", "semantic_orig", "semantic_own_0985", "semantic_own_0975", "forward+semantic_orig"]


def make_bge_encoder(max_len: int, device: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(sm.MODEL_NAME, device=device)
    model.max_seq_length = max_len
    if device == "cuda":
        model.half()

    def enc(batch):
        return model.encode(batch, batch_size=len(batch), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    return enc, model.tokenizer


def token_stats(tokenizer, texts: list[str], max_len: int, chunk: int = 10_000) -> dict:
    lens = []
    for s in range(0, len(texts), chunk):
        lens.extend(len(x) for x in tokenizer(texts[s:s + chunk], add_special_tokens=True, truncation=False)["input_ids"])
    a = np.array(lens)
    return {"tokens_median": float(np.median(a)), "tokens_p95": float(np.quantile(a, 0.95)), "tokens_max": int(a.max()),
            "max_len": max_len, "share_truncated": float((a > max_len).mean())}


def get_embeddings(es: pd.DataFrame, args) -> tuple[np.ndarray, dict]:
    keys = es["key"].to_numpy()
    interim = Path(args.interim) if args.interim else None
    if interim is not None and (interim / "bge_m3_keys.npy").exists() and args.encoder == "bge-m3":
        k0 = np.load(interim / "bge_m3_keys.npy")
        if len(k0) == len(keys) and (k0 == keys).all():
            print(f"Ембединги з кешу: {interim}", flush=True)
            return sm.quantize_fp16(np.load(interim / "bge_m3_fp16.npy").astype(np.float32)), {"from_cache": True, "encoder": args.encoder}
        print("Кеш не збігається з поточним набором ключів, рахую заново", flush=True)
    texts = es["text"].tolist()
    info = {"from_cache": False, "encoder": args.encoder}
    t0 = time.time()
    if args.encoder == "bge-m3":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and not args.allow_cpu:
            raise SystemExit("GPU не знайдено. У Colab: Runtime -> Change runtime type -> T4 GPU (або запустіть з --allow-cpu, але це годинами).")
        enc, tok = make_bge_encoder(args.max_len, device)
        info.update({"model": sm.MODEL_NAME, "device": device, "batch_size": args.batch_size, **token_stats(tok, texts, args.max_len)})
    else:
        enc = sm.hash_encoder()
        info.update({"model": "hash-trigrams (лише тест)", "device": "cpu", "batch_size": args.batch_size})
    emb = sm.embed_sorted(texts, enc, args.batch_size,
                          progress=lambda d, n: print(f"  закодовано {th(d)} з {th(n)}, {time.time() - t0:.0f} с", flush=True))
    emb = sm.quantize_fp16(emb)
    info["seconds"] = round(time.time() - t0)
    if interim is not None and args.encoder == "bge-m3":
        interim.mkdir(parents=True, exist_ok=True)
        np.save(interim / "bge_m3_fp16.npy", emb.astype(np.float16))
        np.save(interim / "bge_m3_keys.npy", keys)
        es.drop(columns=["text"]).to_parquet(interim / "bge_m3_index.parquet")
        print(f"Збережено ембединги: {interim}", flush=True)
    return emb, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--interim", default=None, help="папка для кешу ембедингів (float16 .npy), не потрапляє в репозиторій")
    ap.add_argument("--golden", default=None)
    ap.add_argument("--per-channel", default=str(ROOT / "results" / "02_eda" / "per_channel.csv"))
    ap.add_argument("--channel-flow", default=str(ROOT / "data_small" / "channel_flow_edges.csv"))
    ap.add_argument("--encoder", choices=["bge-m3", "hash"], default="bge-m3")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--k", type=int, default=80)
    ap.add_argument("--umap-n", type=int, default=20_000)
    ap.add_argument("--skip-umap", action="store_true")
    ap.add_argument("--n-perm", type=int, default=20)
    args = ap.parse_args(argv)

    cfg = load_config()
    seed = cfg["seed"]
    out = Path(args.out) if args.out else ROOT / "results" / "04_semantic"
    out.mkdir(parents=True, exist_ok=True)
    golden = Path(args.golden) if args.golden else path_of(cfg, "golden")
    t0 = time.time()

    print("Читаю дані ...", flush=True)
    inp = sm.load_inputs(args.raw_dir, golden)
    es, set_info = sm.build_embedding_set(inp["msgs"], inp["tg"], inp["gold"], inp["sem"], inp["fwd"])
    del inp["msgs"], inp["tg"]  # великі тексти далі не потрібні (економія пам'яті)
    key = es["key"].to_numpy()
    key2idx = {int(k): i for i, k in enumerate(key)}
    print(f"  до ембедингів {th(len(es))} повідомлень (золотих {set_info['n_gold']}, розмічених {th(set_info['n_labeled'])}, "
          f"кінців семантичних ребер {th(set_info['n_sem_endpoints'])}), порожніх відкинуто {set_info['n_empty_dropped']}", flush=True)

    emb, enc_info = get_embeddings(es, args)
    norms = np.linalg.norm(emb, axis=1)
    enc_info.update({"n": int(len(emb)), "dim": int(emb.shape[1]), "max_abs_norm_dev": float(np.abs(norms - 1).max()),
                     "all_finite": bool(np.isfinite(emb).all())})

    pc = pd.read_csv(args.per_channel)
    channel_group = {int(c): bool(v) for c, v in zip(pc["channel_id"], pc["table_narrative_presence"])}
    sem = inp["sem"]
    ka = eda.keys(sem, "src_channel_id", "src_message_id").to_numpy()
    kb = eda.keys(sem, "dst_channel_id", "dst_message_id").to_numpy()
    score = sem["similarity_score"].to_numpy()

    # ---- (3) відтворення оцінок набору
    cos_orig = sm.pair_cosines(emb, key2idx, ka, kb)
    repro = sm.compare_scores(score, cos_orig)
    repro.update({"n_pairs_total": int(len(sem)), "n_pairs_without_cos": int(np.isnan(cos_orig).sum())})
    rng = np.random.default_rng(seed)
    samp = rng.choice(len(sem), size=min(8000, len(sem)), replace=False)
    sample = pd.DataFrame({"score": score[samp], "cos": cos_orig[samp]}).dropna()

    # ---- (4) власні ребра
    print("Будую hnsw-ребра ...", flush=True)
    pairs, kinfo = sm.knn_pairs(emb, k=args.k, threshold=min(sm.THRESHOLDS), report_thresholds=sm.THRESHOLDS, seed=seed)
    pairs = pairs.assign(key_a=key[pairs["i"]], key_b=key[pairs["j"]])
    own = {t: pairs[pairs["cos"] >= t] for t in sm.THRESHOLDS}
    recall = {str(t): sm.hnsw_recall(emb, own[t], t, n_check=500, seed=seed, k=args.k) for t in (0.975, 0.985)}
    orig_set = sm.pair_set(ka, kb)
    sem_node = es["in_sem"].to_numpy()
    ov, ov_sem = {}, {}
    for t in sm.THRESHOLDS:
        ps = sm.pair_set(own[t]["key_a"].to_numpy(), own[t]["key_b"].to_numpy())
        ov[str(t)] = sm.overlap(ps, orig_set)
        both = own[t][sem_node[own[t]["i"].to_numpy()] & sem_node[own[t]["j"].to_numpy()]]
        ov_sem[str(t)] = sm.overlap(sm.pair_set(both["key_a"].to_numpy(), both["key_b"].to_numpy()), orig_set)
    pg = {"orig": sm.pair_group_stats(ka // KEY_MULT, kb // KEY_MULT, channel_group)}
    for t in (0.975, 0.985):
        pg[f"own_{t}"] = sm.pair_group_stats(own[t]["key_a"].to_numpy() // KEY_MULT, own[t]["key_b"].to_numpy() // KEY_MULT, channel_group)

    # ---- (5) копії золотих
    print("Шукаю копії золотих ...", flush=True)
    gpairs, gsumm = sm.gold_copies(emb, es)
    excl = (gpairs[(gpairs["cos"] >= 0.985) & gpairs["neighbor_in_labeled"] & ~gpairs["neighbor_in_gold"]]
            .groupby("neighbor_key").agg(max_cos=("cos", "max"), n_gold_matched=("gold_key", "nunique")).reset_index()
            .rename(columns={"neighbor_key": "key"}))
    excl["channel_id"], excl["message_id"] = excl["key"] // KEY_MULT, excl["key"] % KEY_MULT
    same_text_min = float(gpairs.loc[gpairs["same_text"], "cos"].min()) if gpairs["same_text"].any() else float("nan")

    # ---- (6) досяжність
    print("Рахую досяжність ...", flush=True)
    fa = eda.keys(inp["fwd"], "logical_src_channel_id", "logical_src_message_id").to_numpy()
    fb = eda.keys(inp["fwd"], "logical_dst_channel_id", "logical_dst_message_id").to_numpy()
    layers = {"forward": (fa, fb), "semantic_orig": (ka, kb),
              "semantic_own_0985": (own[0.985]["key_a"].to_numpy(), own[0.985]["key_b"].to_numpy()),
              "semantic_own_0975": (own[0.975]["key_a"].to_numpy(), own[0.975]["key_b"].to_numpy()),
              "forward+semantic_orig": (np.concatenate([fa, ka]), np.concatenate([fb, kb]))}
    universe = np.unique(np.concatenate([np.concatenate(v) for v in layers.values()]))
    ch_of = universe // KEY_MULT
    gser = pd.Series(ch_of).map(channel_group)
    known = gser.notna().to_numpy()
    garr = gser.fillna(False).astype(bool).to_numpy()
    reach_rows, dropped = [], {}
    for name in LAYER_ORDER:
        a, b = (np.searchsorted(universe, x) for x in layers[name])
        ok = known[a] & known[b]
        dropped[name] = int((~ok).sum())
        a, b = a[ok], b[ok]
        r = sm.reachability(len(universe), a, b, garr, KMAX)
        nl = sm.reachability_null(len(universe), a, b, ch_of, channel_group, KMAX, n_perm=args.n_perm, seed=seed)
        r = r.merge(nl, on=["direction", "k"], how="left")
        tot = r.groupby("k").agg(n_start=("n_start", "sum"), n_reach=("n_reach", "sum")).reset_index()
        tot["direction"], tot["share"] = "all", tot["n_reach"] / tot["n_start"]
        r = pd.concat([r, tot], ignore_index=True)
        r["layer"] = name
        r["n_edges"] = int(len(a))
        reach_rows.append(r)
    reach = pd.concat(reach_rows, ignore_index=True)[["layer", "direction", "k", "n_start", "n_reach", "share", "null_mean", "null_p05", "null_p95", "n_edges"]]

    # ---- 50 пар біля порогу
    near = sem[(sem["similarity_score"] >= 0.975) & (sem["similarity_score"] < 0.985)].copy()
    near["cos_ours"] = sm.pair_cosines(emb, key2idx, eda.keys(near, "src_channel_id", "src_message_id").to_numpy(),
                                       eda.keys(near, "dst_channel_id", "dst_message_id").to_numpy())
    near = near.sample(n=min(50, len(near)), random_state=seed).reset_index(drop=True)
    txt = dict(zip(es["key"], es["text"]))
    near["text_src"] = [txt.get(int(k), "")[:300] for k in eda.keys(near, "src_channel_id", "src_message_id")]
    near["text_dst"] = [txt.get(int(k), "")[:300] for k in eda.keys(near, "dst_channel_id", "dst_message_id")]
    near["same_channel"] = near["src_channel_id"] == near["dst_channel_id"]

    # ---- (8) залишок S3
    flow = pd.read_csv(args.channel_flow)
    fwd_check = sm.forward_vs_channel_flow(inp["fwd"], flow)

    # ---- перевірки
    gold_ok = set_info["n_gold"] == len(inp["gold"].drop_duplicates(["channel_id", "message_id"])) and set_info["n_gold_empty"] == 0
    miss_share = repro["n_pairs_without_cos"] / max(repro["n_pairs_total"], 1)
    checks = [
        {"hard": True, "check": "Усі золоті повідомлення ембедовані, порожніх текстів немає", "passed": bool(gold_ok),
         "detail": f"{set_info['n_gold']} золотих, порожніх {set_info['n_gold_empty']}"},
        {"hard": True, "check": "Ключі набору унікальні", "passed": bool(len(key) == len(set(key.tolist()))), "detail": f"{th(len(key))} ключів"},
        {"hard": True, "check": "Ембединги скінченні й одиничної довжини (відхилення норми менше 1e-3)", "passed": bool(enc_info["all_finite"] and enc_info["max_abs_norm_dev"] < 1e-3),
         "detail": f"макс. відхилення {enc_info['max_abs_norm_dev']:.2e}, розмірність {enc_info['dim']}"},
        {"hard": True, "check": "Для 99% і більше семантичних ребер набору є наш косинус (обидва кінці ембедовані)", "passed": bool(miss_share <= 0.01),
         "detail": f"без косинуса {repro['n_pairs_without_cos']} з {th(repro['n_pairs_total'])}"},
        {"hard": True, "check": "Повнота hnsw за порогом 0,985 не нижча за 0,98 (500 випадкових вузлів, точний пошук)", "passed": bool(recall["0.985"]["recall"] >= 0.98 or np.isnan(recall["0.985"]["recall"])),
         "detail": f"{recall['0.985']['recall']:.4f} ({recall['0.985']['exact_neighbors_capped']} сусідів)"},
        {"hard": True, "check": "Повнота hnsw за порогом 0,975 не нижча за 0,98", "passed": bool(recall["0.975"]["recall"] >= 0.98 or np.isnan(recall["0.975"]["recall"])),
         "detail": f"{recall['0.975']['recall']:.4f} ({recall['0.975']['exact_neighbors_capped']} сусідів)"},
        {"hard": False, "check": "Однакові тексти мають косинус не нижчий за 0,995", "passed": bool(np.isnan(same_text_min) or same_text_min >= 0.995),
         "detail": f"мінімум {same_text_min:.4f}" if not np.isnan(same_text_min) else "пар з однаковим текстом серед копій золотих немає"},
        {"hard": False, "check": "Оцінки набору збігаються з нашим косинусом за рангами на нетривіальних парах (score < 0,995; спірмен не нижчий за 0,9)",
         "passed": bool(repro["spearman_nontrivial"] >= 0.9),
         "detail": f"спірмен {repro['spearman_nontrivial']:.3f}, пірсон {repro['pearson_nontrivial']:.3f}, пар {th(repro['n_nontrivial'])}, пар з score від 0,995 {repro['share_score_ge_0995'] * 100:.1f}%"},
        {"hard": False, "check": "Оцінки набору збігаються з нашим косинусом за значенням (середній модуль різниці менший за 0,01)", "passed": bool(repro["mean_abs_diff_direct"] < 0.01),
         "detail": f"усі пари {repro['mean_abs_diff_direct']:.4f}, нетривіальні {repro['mean_abs_diff_direct_nontrivial']:.4f} (з відображенням (cos + 1) / 2: {repro['mean_abs_diff_mapped_half']:.4f})"},
        {"hard": False, "check": "Агреговані forward_edges збігаються з channel_flow_edges", "passed": bool(fwd_check["equal_to_channel_flow"]),
         "detail": f"forward {th(fwd_check['n_forward_edges'])}, самопересилань {fwd_check['n_self_forward']}, сума без них {th(fwd_check['weight_sum_without_self'])}, у таблиці {th(fwd_check['channel_flow_weight_sum'])}"},
    ]

    summary = {"embedding_set": set_info, "encoder": enc_info, "reproduction": repro, "own_edges": {**kinfo, "hnsw_recall": recall},
               "overlap_with_orig_all_embedded": ov, "overlap_with_orig_within_sem_nodes": ov_sem, "pair_groups": pg,
               "gold_copies_same_text_min_cos": same_text_min, "n_excluded_labeled_cos985": int(len(excl)),
               "layers_dropped_edges_unknown_group": dropped, "forward_vs_channel_flow": fwd_check, "n_orig_pairs_near_threshold": int(len(near)),
               "seconds": round(time.time() - t0)}

    # вхід для S5 (спліти): розмічені слабкими мітками і золоті, без текстів
    li = es.loc[es["in_labeled"] | es["in_gold"], ["key", "channel_id", "message_id", "date_utc", "narrative_id", "text_hash", "in_gold", "in_labeled"]].copy()
    li["weak_narrative"] = li["in_labeled"] & li["narrative_id"].notna()
    li["date_utc"] = li["date_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    li = li.drop(columns=["narrative_id"]).sort_values("key")
    summary["labeled_index"] = {"n_rows": int(len(li)), "n_labeled": int(li["in_labeled"].sum()), "n_gold": int(li["in_gold"].sum()),
                                "n_missing_date": int(li["date_utc"].isna().sum())}

    tables = {"labeled_index": li, "reachability": reach, "gold_copies_summary": gsumm, "gold_copies_pairs": gpairs[gpairs["cos"] >= 0.95].sort_values(["gold_key", "cos"], ascending=[True, False]),
              "exclude_labeled_keys_cos985": excl.sort_values("key"), "threshold_pairs_sample": near, "repro_sample": sample}

    # ---- (7) UMAP
    if not args.skip_umap:
        print("UMAP ...", flush=True)
        try:
            import umap

            from narr.language import detect_language_parallel
            pool = np.flatnonzero((es["in_fwd"] | es["in_sem"]).to_numpy() & ~es["in_gold"].to_numpy())
            pick = np.sort(np.random.default_rng(seed).choice(pool, size=min(args.umap_n, len(pool)), replace=False))
            gi = np.flatnonzero(es["in_gold"].to_numpy())
            sel = np.concatenate([gi, pick])
            xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(emb[sel])
            lang = detect_language_parallel(es["text"].to_numpy()[sel], n_jobs=2)
            lvl = inp["coding"].drop_duplicates("narrative_id").set_index("narrative_id")["level_1"]
            gold_meta = inp["gold"].assign(key=eda.keys(inp["gold"])).drop_duplicates("key").set_index("key")["General label (level-1)"]
            nar = es["narrative_id"].to_numpy()[sel]
            meta = np.array([lvl.get(int(v), "без наративу") if pd.notna(v) else "без наративу" for v in nar], dtype=object)
            src = np.where(es["in_gold"].to_numpy()[sel], "gold", "graph")
            gm = es["key"].to_numpy()[sel][src == "gold"]
            meta[src == "gold"] = [gold_meta.get(int(k), "без наративу") for k in gm]
            meta = np.where(meta == cfg["columns"]["no_narrative_label"], "без наративу", meta)
            coords = pd.DataFrame({"key": es["key"].to_numpy()[sel], "x": xy[:, 0], "y": xy[:, 1], "source": src, "language": lang, "meta": meta})
            tables["umap_coords"] = coords
            from narr.semantic_viz import fig_umap
            (out / "figures").mkdir(exist_ok=True)
            fig_umap(coords, out / "figures" / "03_umap.png")
            summary["umap"] = {"n_points": int(len(coords)), "n_gold": int((src == "gold").sum()), "n_graph_sample": int((src == "graph").sum())}
        except ImportError as e:
            print(f"UMAP пропущено: {e}", flush=True)
            summary["umap"] = {"skipped": str(e)}

    eda.save_all(tables, summary, checks, out)
    from narr.semantic_viz import fig_reach, fig_repro
    (out / "figures").mkdir(exist_ok=True)
    fig_repro(sample, repro, out / "figures" / "01_repro_scores.png")
    fig_reach(reach, out / "figures" / "02_reachability.png")
    from importlib import metadata

    libs = {}
    for lib in ("torch", "sentence-transformers", "transformers", "hnswlib", "umap-learn", "scipy", "langid"):
        try:
            libs[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            libs[lib] = None
    write_run_json(out, "scripts/s04_semantic.py", {"libs": libs, "raw_dir": str(args.raw_dir), "encoder": args.encoder, "k": args.k, "max_len": args.max_len,
                                                    "batch_size": args.batch_size, "seed": seed, "umap_n": args.umap_n, "n_perm": args.n_perm,
                                                    "model": sm.MODEL_NAME, "seconds": round(time.time() - t0)})
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
