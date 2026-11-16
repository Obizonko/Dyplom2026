"""S5: спліти P1-P3, перевірки витоку, шаблон розмітки «явний / неявний».

Дві частини:
  золота (працює одразу): групи каналів для P3, тестові множини P1-P3, автоматичне визначення неявних (б) за S6, шаблон для ручної розмітки (а);
  навчальна (потрібен вихід S4: labeled_index.csv, exclude_labeled_keys_cos985.csv, gold_copies_pairs.csv): навчальний пул, три протоколи, перевірки витоку.

Запуск:  python3 -I scripts/s05_splits.py                      (читає results/04_semantic/, якщо файли є)
         python3 -I scripts/s05_splits.py --s04 PATH --out PATH
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from narr import eda, splits as sp  # noqa: E402
from narr.io import ROOT, data_available, load_config, load_golden, write_run_json  # noqa: E402
from narr.language import detect_language  # noqa: E402

PLAN_P2_TEST = 116  # з PLAN_vykonannya.md, етап S5


def th(n) -> str:
    return f"{int(n):,}".replace(",", " ")


def implicit_auto(gold: pd.DataFrame, pred_path: Path, seed: int) -> pd.DataFrame:
    """Визначення (б): наративні золоті, для яких TF-IDF word_char при 5-кратній CV (5 повторів, середнє) дає ймовірність < 0,5."""
    p = pd.read_parquet(pred_path)
    q = (p[(p["setting"] == "random_cv") & (p["model"] == "word_char")].groupby(["channel_id", "message_id"])
         .agg(proba_tfidf=("proba", "mean"), n_repeats=("repeat", "size")).reset_index())
    m = gold.merge(q, on=["channel_id", "message_id"], how="left")
    m["implicit_auto"] = m["y"].eq(1) & (m["proba_tfidf"] < 0.5)
    return m[["key", "channel_id", "message_id", "y", "lang", "proba_tfidf", "n_repeats", "implicit_auto"]]


def labeling_template(g_raw: pd.DataFrame, gold: pd.DataFrame, out_csv: Path, seed: int) -> str:
    """Шаблон розмітки 148 наративних золотих. Якщо файл уже містить відповіді, не перезаписуємо (різні набори ключів дають .new.csv)."""
    pos = g_raw.merge(gold[["channel_id", "message_id", "y", "lang"]], on=["channel_id", "message_id"])
    pos = pos[pos["y"] == 1].sample(frac=1.0, random_state=seed).reset_index(drop=True)  # випадковий порядок
    tpl = pd.DataFrame({"n": np.arange(1, len(pos) + 1), "channel_id": pos["channel_id"], "message_id": pos["message_id"],
                        "channel_username": pos["channel_username"], "date_utc": pos["date_utc"].astype(str),
                        "lang": pos["lang"], "narrative_label": pos["General label"], "meta_label": pos["General label (level-1)"],
                        "text": pos["text"], "explicitness": "", "comment": ""})
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        if old.get("explicitness", pd.Series(dtype=object)).fillna("").astype(str).str.strip().ne("").any():
            same = set(zip(old["channel_id"], old["message_id"])) == set(zip(tpl["channel_id"], tpl["message_id"]))
            if same:
                return "labeled"
            tpl.to_csv(out_csv.with_name(out_csv.stem + ".new.csv"), index=False, encoding="utf-8-sig")
            return "exists_with_answers_keys_differ"
    tpl.to_csv(out_csv, index=False, encoding="utf-8-sig")  # utf-8-sig, щоб Excel правильно показав кирилицю
    return "written"


LABEL_README = """# Розмітка «явний / неявний» наратив

Файл `implicit_labeling.csv`: 148 повідомлень, які золота розмітка визнала наративними, у випадковому порядку (seed 42).

Розмітку виконала Claude (LLM) за рубрикатором `implicit_rubric.md` на прохання користувача, сліпо до прогнозів моделей.
У колонці `explicitness` одне зі значень: `explicit`, `implicit`, `unclear`. У колонці `comment` код причини
(`direct`, `framing`, `irony`, `quote`, `context`, `fragment`, `mixed`, `unrelated`), а для сумнівних випадків ще й `;borderline`.
`unclear` в аналізі H1 не враховуються, їх кількість наведено у звіті.

Перевірка користувачем (близько 15 хвилин): файл `implicit_spotcheck.csv` містить 20 рядків (10 явних і 10 неявних за розміткою Claude)
без самої розмітки. У колонці `explicitness_user` поставте `explicit`, `implicit` або `unclear` і поверніть файл.
Скрипт порахує збіг, і його буде додано до звіту як зовнішню перевірку розмітки.
Текст нормалізований (малі літери, без розділових знаків), бо інших у наборі немає.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s04", default=str(ROOT / "results" / "04_semantic"))
    ap.add_argument("--s06", default=str(ROOT / "results" / "06_text" / "predictions_tfidf_binary.parquet"))
    ap.add_argument("--per-channel", default=str(ROOT / "results" / "02_eda" / "per_channel.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "05_protocol"))
    ap.add_argument("--splits", default=str(ROOT / "splits"))
    ap.add_argument("--cutoff", default=sp.CUTOFF)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)

    cfg = load_config()
    if not data_available(cfg):
        sys.exit("Немає external/TeleNarratives")
    seed = cfg["seed"]
    out, spl, s04 = Path(args.out), Path(args.splits), Path(args.s04)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    g_raw = load_golden(cfg)
    gold = sp.gold_frame(g_raw)
    lang = pd.Series(detect_language(g_raw["text"].astype(str).tolist()), index=sp.make_key(g_raw["channel_id"], g_raw["message_id"]))
    gold["lang"] = gold["key"].map(lang)

    # ---- золота частина
    folds = sp.channel_folds(gold, k=args.folds, seed=seed)  # gold містить lang, тож групи балансуються й за мовою
    gold["fold"] = gold["channel_id"].map(folds.set_index("channel_id")["fold"])
    pc = pd.read_csv(args.per_channel)
    folds = folds.merge(pc[["channel_id", "username", "n_labeled", "n_narrative", "table_label"]].rename(columns={"n_narrative": "n_weak_narrative"}),
                        on="channel_id", how="left")
    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    gold["in_p2_test"] = gold["date_utc"] >= cutoff
    imp = implicit_auto(gold, Path(args.s06), seed) if Path(args.s06).exists() else None
    tpl_status = labeling_template(g_raw, gold, out / "implicit_labeling.csv", seed)
    (out / "implicit_labeling_README.md").write_text(LABEL_README, encoding="utf-8")

    fold_sizes = (gold.groupby("fold").agg(n_test=("key", "size"), n_narrative=("y", "sum"), n_channels=("channel_id", "nunique"),
                                           n_ru=("lang", lambda s: int((s == "ru").sum())), n_uk=("lang", lambda s: int((s == "uk").sum()))).reset_index())
    p2_lang = gold.groupby(["in_p2_test", "lang"]).agg(n=("key", "size"), n_narrative=("y", "sum")).reset_index()
    summary = {"gold": {"n": int(len(gold)), "n_narrative": int(gold["y"].sum()), "n_channels": int(gold["channel_id"].nunique()),
                        "n_p2_test": int(gold["in_p2_test"].sum()), "n_p2_test_narrative": int(gold.loc[gold["in_p2_test"], "y"].sum()),
                        "n_p2_test_channels": int(gold.loc[gold["in_p2_test"], "channel_id"].nunique()), "cutoff": args.cutoff,
                        "p3_folds": int(args.folds)},
               "gold_channels": {"n_without_labeled": int((folds["n_labeled"].fillna(0) == 0).sum()), "min_n_labeled": int(folds["n_labeled"].fillna(0).min()),
                                 "median_n_labeled": float(folds["n_labeled"].fillna(0).median()), "n_in_table": int(folds["n_labeled"].notna().sum())},
               "implicit_template": tpl_status}
    if imp is not None:
        pos = imp[imp["y"] == 1]
        summary["implicit_auto_b"] = {"n_narrative": int(len(pos)), "n_implicit": int(pos["implicit_auto"].sum()),
                                      "n_implicit_ru": int((pos["implicit_auto"] & (pos["lang"] == "ru")).sum()),
                                      "n_implicit_uk": int((pos["implicit_auto"] & (pos["lang"] == "uk")).sum()),
                                      "n_narrative_ru": int((pos["lang"] == "ru").sum()), "n_narrative_uk": int((pos["lang"] == "uk").sum())}
    ab_table = None
    spot_path = out / "implicit_spotcheck.csv"
    if tpl_status == "labeled":
        lab = pd.read_csv(out / "implicit_labeling.csv")
        summary["implicit_manual_a"] = sp.implicit_manual_summary(lab)
        keep = spot_path.exists() and pd.read_csv(spot_path)["explicitness_user"].fillna("").astype(str).str.strip().ne("").any()
        if not keep:
            sp.spotcheck_sample(lab, seed=seed).to_csv(spot_path, index=False, encoding="utf-8-sig")
        agree = sp.spotcheck_agreement(lab, pd.read_csv(spot_path))
        if agree is not None:
            summary["implicit_spotcheck"] = agree
        if imp is not None:
            mm = lab.merge(imp[["channel_id", "message_id", "implicit_auto"]], on=["channel_id", "message_id"])
            ab = pd.crosstab(mm["explicitness"], mm["implicit_auto"].map({True: "b_implicit", False: "b_not_implicit"})).reset_index()
            summary["implicit_a_vs_b"] = {r["explicitness"]: {k: int(r[k]) for k in ab.columns if k != "explicitness"} for _, r in ab.iterrows()}
            ab_table = ab
    tables = {"gold_membership": gold.assign(date_utc=gold["date_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))[
        ["key", "channel_id", "message_id", "date_utc", "y", "lang", "fold", "in_p2_test"]],
        "gold_channel_folds": folds, "p3_fold_sizes": fold_sizes, "p2_test_by_language": p2_lang}
    if imp is not None:
        tables["implicit_auto_b"] = imp
    if ab_table is not None:
        tables["implicit_a_vs_b"] = ab_table
    checks = [
        {"hard": True, "check": "Золотих 412, наративних 148, каналів 56", "passed": bool(len(gold) == 412 and gold["y"].sum() == 148 and gold["channel_id"].nunique() == 56),
         "detail": f"{len(gold)} / {int(gold['y'].sum())} / {gold['channel_id'].nunique()}"},
        {"hard": True, "check": "Кожен золотий канал у рівно одній групі P3, кожне золоте повідомлення тестується в P3 рівно раз",
         "passed": bool(folds["channel_id"].is_unique and fold_sizes["n_test"].sum() == len(gold) and gold["fold"].notna().all()),
         "detail": f"групи {sorted(fold_sizes['fold'].tolist())}, розміри {fold_sizes['n_test'].tolist()}"},
        {"hard": True, "check": "Усі золоті канали є в таблиці каналів S2 (для підрахунку розмічених)", "passed": bool(folds["n_labeled"].notna().all()),
         "detail": f"{summary['gold_channels']['n_in_table']} з {len(folds)}"},
        {"hard": False, "check": f"Тест P2 має {PLAN_P2_TEST} повідомлень, як у плані", "passed": bool(summary["gold"]["n_p2_test"] == PLAN_P2_TEST),
         "detail": f"{summary['gold']['n_p2_test']}"},
        {"hard": False, "check": "Групи P3 збалансовані за розміром, кількістю наративних і російськомовних (найбільша не більш ніж у 1,5 раза більша за найменшу)",
         "passed": bool(all(fold_sizes[c].max() <= 1.5 * fold_sizes[c].min() for c in ("n_test", "n_narrative", "n_ru"))),
         "detail": "; ".join(f"{c}: {fold_sizes[c].min()} до {fold_sizes[c].max()}" for c in ("n_test", "n_narrative", "n_ru"))},
    ]

    if "implicit_manual_a" in summary:
        m = summary["implicit_manual_a"]
        checks.append({"hard": True, "check": "Розмітку (а) заповнено для всіх 148 рядків, значення й коди причин допустимі",
                       "passed": bool(m["n_rows"] == 148 and m["n_filled"] == 148 and not m["bad_values"] and not m["bad_codes"]),
                       "detail": f"заповнено {m['n_filled']} з {m['n_rows']}; невалідні значення {m['bad_values']}, коди {m['bad_codes']}"})
        checks.append({"hard": False, "check": f"Російськомовних неявних не менше {m['min_implicit']} (умова H1)", "passed": bool(m["enough_ru"]["base"]),
                       "detail": f"базово {m['implicit']['ru']}, вузьке {m['implicit_narrow']['ru']}, широке {m['implicit_wide']['ru']}"})

    # ---- навчальна частина (потрібен S4)
    idx_path = s04 / "labeled_index.csv" if (s04 / "labeled_index.csv").exists() else s04 / "labeled_index.csv.gz"
    need = {"index": idx_path, "exclude": s04 / "exclude_labeled_keys_cos985.csv", "pairs": s04 / "gold_copies_pairs.csv"}
    have = all(p.exists() for p in need.values())
    sizes = []
    if have:
        idx = pd.read_csv(need["index"])
        excl = pd.read_csv(need["exclude"])
        pairs = pd.read_csv(need["pairs"])
        gh = idx.loc[idx["in_gold"].astype(bool)].drop_duplicates("key").set_index("key")["text_hash"]
        gold["text_hash"] = gold["key"].map(gh)
        pool, excluded = sp.build_pool(idx, set(gold["key"]), set(gold["text_hash"].dropna()), set(excl["key"]))
        ex_sum = sp.exclusion_summary(int(idx["in_labeled"].astype(bool).sum()), excluded)
        summary["pool"] = {**ex_sum, "n_pool_narrative": int(pool["weak_narrative"].sum()), "n_pool_channels": int(pool["channel_id"].nunique()),
                           "weak_narrative_share_pool": float(pool["weak_narrative"].mean()),
                           "date_min": str(pool["date_utc"].min()), "date_max": str(pool["date_utc"].max()),
                           "n_exclude_keys_s04": int(len(excl)), "n_gold_with_text_hash": int(gold["text_hash"].notna().sum())}
        protos = {"P1": [(None, sp.split_p1(pool, gold))], "P2": [(None, sp.split_p2(pool, gold, args.cutoff))],
                  "P3": [(f, d) for f, d in sp.split_p3(pool, gold, folds).items()]}
        for name, runs in protos.items():
            for f, d in runs:
                tr, te = d["train"], d["test"]
                d_dir = spl / name
                d_dir.mkdir(parents=True, exist_ok=True)
                tag = "" if f is None else f"fold{f}_"
                tr[["channel_id", "message_id", "weak_narrative"]].to_csv(d_dir / f"{tag}train.csv.gz", index=False)
                te[["channel_id", "message_id", "y", "date_utc"]].to_csv(d_dir / f"{tag}test.csv", index=False)
                sizes.append(sp.size_row(name, f, tr, te))
                for c in sp.leakage_checks(tr, te, pairs, by_channel=(name == "P3")):
                    checks.append({"hard": True, "check": f"{name}{'' if f is None else f' група {f}'}: {c['check']}", "passed": bool(c["passed"]), "detail": c["detail"]})
        excluded.to_csv(spl / "excluded_from_train.csv.gz", index=False)
        sz = pd.DataFrame(sizes)
        tables["split_sizes"] = sz
        tables["exclusion_by_reason"] = pd.DataFrame([{"reason": r, "n_any": ex_sum[f"reason_{r}"], "n_first": ex_sum[f"first_reason_{r}"]} for r in sp.REASONS])
        checks += [
            {"hard": True, "check": "Пул і виключені разом дають усю розмічену партію", "passed": bool(ex_sum["n_pool"] + ex_sum["n_excluded"] == ex_sum["n_labeled"]),
             "detail": f"{th(ex_sum['n_pool'])} + {th(ex_sum['n_excluded'])} = {th(ex_sum['n_labeled'])}"},
            {"hard": True, "check": "У кожному train є обидва класи слабких міток, у кожному test обидва класи золотих",
             "passed": bool((sz["n_train_narrative"] > 0).all() and (sz["n_train_narrative"] < sz["n_train"]).all() and (sz["n_test_narrative"] > 0).all() and (sz["n_test_narrative"] < sz["n_test"]).all()),
             "detail": f"train наративних {sz['n_train_narrative'].min()} до {sz['n_train_narrative'].max()}"},
            {"hard": True, "check": "Для всіх 412 золотих знайдено хеш тексту в індексі S4", "passed": bool(gold["text_hash"].notna().all()), "detail": f"{int(gold['text_hash'].notna().sum())} з {len(gold)}"},
            {"hard": False, "check": "Усі ключі списку виключення S4 є в розміченій партії", "passed": bool(excl["key"].isin(set(idx.loc[idx['in_labeled'].astype(bool), 'key'])).all()), "detail": f"{len(excl)} ключів"},
        ]
        summary["train_side"] = "done"
    else:
        summary["train_side"] = "pending_S4: " + ", ".join(k for k, p in need.items() if not p.exists())

    eda.save_all(tables, summary, checks, out)
    write_run_json(out, "scripts/s05_splits.py", {"cutoff": args.cutoff, "folds": args.folds, "seed": seed, "s04": str(s04), "seconds": round(time.time() - t0)})
    bad = [c for c in checks if c["hard"] and not c["passed"]]
    print(f"S5: перевірок {len(checks)}, жорстких не пройдено {len(bad)}; навчальна частина: {summary['train_side']}")
    for c in checks:
        print(("ok  " if c["passed"] else "FAIL") + " " + c["check"] + " | " + c["detail"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
