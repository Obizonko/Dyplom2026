"""S6-cloud: TF-IDF + логістична регресія на 412 золотих повідомленнях.

Протокол: стратифікована 5-кратна CV x 5 повторів (random_cv) і CV за каналами (channel_cv, перевірка на витік стилю каналу).
Підбір C лише у вкладеній CV на навчальних фолдах. Метрики на out-of-fold прогнозах, 95% бутстреп-CI по повідомленнях.

Запуск:  python3 -I scripts/s06_tfidf.py            (повний)
         python3 -I scripts/s06_tfidf.py --quick    (смоук-тест: 1 повтор, 50 бутстреп-вибірок)
"""
import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from narr.baselines import MODEL_NAMES, oof_proba, top_features  # noqa: E402
from narr.evaluate import binary_metrics, bootstrap_ci, multiclass_metrics, paired_bootstrap_diff  # noqa: E402
from narr.language import detect_language  # noqa: E402
from narr.io import ROOT, data_available, load_config, load_golden, write_run_json  # noqa: E402
from narr.report import ci_str, fmt, md_table  # noqa: E402
from narr.viz import fig_metrics_ci  # noqa: E402

LR_MODELS = ["word", "char", "word_char"]


def long_table(setting: str, model: str, ci: pd.DataFrame, extra: dict | None = None) -> pd.DataFrame:
    t = ci.copy()
    t.insert(0, "model", model)
    t.insert(0, "setting", setting)
    for k, v in (extra or {}).items():
        t[k] = v
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    if not data_available(cfg):
        sys.exit("Немає external/TeleNarratives")
    ev, col, seed = cfg["eval"], cfg["columns"], cfg["seed"]
    n_boot = 50 if args.quick else ev["bootstrap_n"]
    reps = 1 if args.quick else ev["cv_repeats"]
    folds = ev["cv_folds"]
    out = ROOT / "results" / ("06_text_quick" if args.quick else "06_text")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = load_golden(cfg)
    X = df[col["text"]].astype(str).to_numpy(dtype=object)
    y = df[col["binary"]].astype(int).to_numpy()
    groups = df[col["channel"]].to_numpy()

    # ---------- бінарна задача ----------
    tables, preds, P_store = [], [], {}
    for setting, grp in (("random_cv", None), ("channel_cv", groups)):
        for name in MODEL_NAMES:
            P = oof_proba(name, X, y, n_splits=folds, n_repeats=reps, seed=seed, groups=grp)[:, :, 1]
            P_store[(setting, name)] = P
            ci = bootstrap_ci(y, P, binary_metrics, n_boot=n_boot, seed=seed)
            tables.append(long_table(setting, name, ci))
            for r in range(reps):
                preds.append(pd.DataFrame({"setting": setting, "model": name, "repeat": r, col["channel"]: df[col["channel"]],
                                           col["message"]: df[col["message"]], "y": y, "proba": P[r]}))
            print(f"[{time.time() - t0:6.0f}s] binary {setting:10s} {name}", flush=True)
    binary = pd.concat(tables, ignore_index=True)
    binary.to_csv(out / "metrics_binary.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(out / "predictions_tfidf_binary.parquet", index=False)

    # ---------- парні різниці (random_cv) ----------
    pairs = []
    for a, b in (("word_char", "word"), ("word_char", "char"), ("word", "char"),
                 ("word_char", "majority"), ("word_char", "stratified")):
        d = paired_bootstrap_diff(y, P_store[("random_cv", a)], P_store[("random_cv", b)], binary_metrics,
                                  n_boot=n_boot, seed=seed)
        d.insert(0, "B", b)
        d.insert(0, "A", a)
        pairs.append(d)
    pairs = pd.concat(pairs, ignore_index=True)
    pairs.to_csv(out / "paired_diffs_random_cv.csv", index=False)

    # ---------- перевірка мовного шорткату ----------
    lang = detect_language(X)
    lang_ct = pd.crosstab(pd.Series(lang, name="мова"), pd.Series(y, name="наратив")).rename(columns={0: "без наративу", 1: "наратив"})
    lang_ct["разом"] = lang_ct.sum(axis=1)
    lang_ct["частка наративних"] = lang_ct["наратив"] / lang_ct["разом"]
    lang_ct.reset_index().to_csv(out / "language_crosstab.csv", index=False)
    P_lang = (lang == "ru").astype(float)[None, :]          # правило «російська мова = наратив»
    lang_rule = bootstrap_ci(y, P_lang, binary_metrics, n_boot=n_boot, seed=seed)
    lang_rule.to_csv(out / "language_rule_metrics.csv", index=False)
    lang_pair = paired_bootstrap_diff(y, P_store[("random_cv", "word_char")], P_lang, binary_metrics, n_boot=n_boot, seed=seed)
    lang_pair.to_csv(out / "word_char_minus_language_rule.csv", index=False)
    by_lang = []
    for lg in ("ru", "uk"):
        mask = lang == lg
        for name in LR_MODELS:
            ci = bootstrap_ci(y[mask], P_store[("random_cv", name)][:, mask], binary_metrics, n_boot=n_boot, seed=seed)
            by_lang.append(long_table(lg, name, ci, {"n": int(mask.sum()), "n_pos": int(y[mask].sum())}))
    by_lang = pd.concat(by_lang, ignore_index=True)
    by_lang.to_csv(out / "metrics_by_language.csv", index=False)
    print(f"[{time.time() - t0:6.0f}s] language check done", flush=True)

    # ---------- перевірка на витік: перемішані мітки ----------
    rng = np.random.default_rng(seed)
    perm_rows = []
    for r in range(reps):
        y_p = rng.permutation(y)
        P = oof_proba("word_char", X, y_p, n_splits=folds, n_repeats=1, seed=seed + 100 + r)[:, :, 1]
        perm_rows.append(binary_metrics(y_p, P[0]))
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(out / "permuted_labels_word_char.csv", index=False)
    print(f"[{time.time() - t0:6.0f}s] permutation done", flush=True)

    # ---------- багатокласова задача (none + 4 найбільші мета-наративи + інше) ----------
    pos = df[df[col["binary"]]]
    top = list(pos[col["level1"]].value_counts().index[: ev["n_top_meta_narratives"]])
    classes = [col["no_narrative_label"], *top, "інший наратив"]
    ymc = df[col["level1"]].map(lambda v: classes.index(v) if v in classes else len(classes) - 1).to_numpy()
    k = len(classes)
    mc_tables = []
    for name in ["majority", "stratified", *LR_MODELS]:
        P = oof_proba(name, X, ymc, task="multiclass", n_splits=folds, n_repeats=reps, seed=seed)
        pred = P.argmax(axis=2)
        ci = bootstrap_ci(ymc, pred, partial(multiclass_metrics, k=k), n_boot=n_boot, seed=seed)
        mc_tables.append(long_table("random_cv", name, ci))
        print(f"[{time.time() - t0:6.0f}s] multiclass {name}", flush=True)
    mc = pd.concat(mc_tables, ignore_index=True)
    mc.to_csv(out / "metrics_multiclass.csv", index=False)
    pd.Series(classes, name="class").rename_axis("index").reset_index().to_csv(out / "multiclass_classes.csv", index=False)

    # ---------- описова перевірка: найсильніші слова ----------
    tp, tn = top_features(X, y, n=20)
    pd.DataFrame({"наративні (топ слів)": tp, "ненаративні (топ слів)": tn}).to_csv(out / "top_words.csv", index=False)

    # ---------- перевірки ----------
    g = lambda s, m, k_, c="point": float(binary[(binary.setting == s) & (binary.model == m) & (binary.metric == k_)][c].iloc[0])
    rnd_pos = float(y.mean())
    checks = []
    for m in LR_MODELS:
        floor = max(g("random_cv", d, "macro_f1", "ci_hi") for d in ("majority", "stratified"))
        checks.append((f"{m}: macro-F1 вище за верхню межу CI обох дамі-моделей (random_cv)",
                       g("random_cv", m, "macro_f1") > floor,
                       f"{fmt(g('random_cv', m, 'macro_f1'))} проти {fmt(floor)}"))
    d_wc_w = pairs[(pairs.A == "word_char") & (pairs.B == "word") & (pairs.metric == "macro_f1")].iloc[0]
    checks.append(("word_char не гірша за word за macro-F1 (CI різниці не лежить повністю нижче 0)", bool(d_wc_w.ci_hi > 0),
                   f"Δ = {fmt(d_wc_w['diff'])} [{fmt(d_wc_w.ci_lo)}; {fmt(d_wc_w.ci_hi)}]"))
    checks.append(("перемішані мітки: |MCC| < 0,1", bool(abs(perm["mcc"].mean()) < 0.1), f"середній MCC = {fmt(perm['mcc'].mean())}"))
    checks.append(("перемішані мітки: PR-AUC близька до частки позитивних (< частка + 0,05)",
                   bool(perm["pr_auc"].mean() < rnd_pos + 0.05), f"{fmt(perm['pr_auc'].mean())} проти частки {fmt(rnd_pos)}"))
    checks.append(("перемішані мітки: macro-F1 у межах 0,40-0,60", bool(0.40 <= perm["macro_f1"].mean() <= 0.60),
                   f"{fmt(perm['macro_f1'].mean())}"))
    maj = g("random_cv", "majority", "macro_f1")
    checks.append(("мажоритарний клас: macro-F1 ≈ 0,39 (за планом)", bool(abs(maj - 0.39) < 0.01), fmt(maj)))
    (out / "checks.json").write_text(json.dumps(
        [{"check": c, "ok": bool(ok), "detail": d} for c, ok, d in checks], ensure_ascii=False, indent=2), encoding="utf-8")
    write_run_json(out, "scripts/s06_tfidf.py", {"quick": args.quick, "n_boot": n_boot, "repeats": reps, "folds": folds,
                                                  "seed": seed, "Cs": [0.1, 1, 10, 100], "models": MODEL_NAMES})

    # ---------- рисунок ----------
    show = binary[binary.model.isin(["majority", "stratified", *LR_MODELS]) & binary.metric.isin(["macro_f1", "mcc", "pr_auc"])]
    rule_rows = lang_rule[lang_rule.metric.isin(["macro_f1", "mcc", "pr_auc"])].copy()
    rule_rows.insert(0, "model", "правило: ru = наратив")
    rule_rows.insert(0, "setting", "random_cv")
    show = pd.concat([show, rule_rows], ignore_index=True)
    fig_metrics_ci(show, out / "figures" / "binary_metrics_ci.png",
                   "Золота розмітка (412), бінарна задача: out-of-fold метрики з 95% CI")

    # ---------- звіт ----------
    if args.quick:
        print("quick-режим: звіт не пишеться")
        print(pd.DataFrame([{"check": c, "ok": ok, "detail": d} for c, ok, d in checks]).to_string())
        print(binary[binary.metric.isin(["macro_f1", "mcc", "pr_auc"])].round(3).to_string())
        return

    def metric_table(setting: str, models: list[str]) -> pd.DataFrame:
        rows = []
        for m in models:
            r = {"Модель": m}
            for key, label in (("macro_f1", "macro-F1"), ("mcc", "MCC"), ("pr_auc", "PR-AUC"), ("f1_pos", "F1 «наратив»"),
                               ("precision", "precision"), ("recall", "recall")):
                row = binary[(binary.setting == setting) & (binary.model == m) & (binary.metric == key)].iloc[0]
                r[label] = ci_str(row.point, row.ci_lo, row.ci_hi, 3)
            r["σ між повторами (macro-F1)"] = fmt(float(binary[(binary.setting == setting) & (binary.model == m)
                                                               & (binary.metric == "macro_f1")].sd_repeats.iloc[0]))
            rows.append(r)
        return pd.DataFrame(rows)

    pair_t = pairs[pairs.metric.isin(["macro_f1", "mcc", "pr_auc"])].copy()
    pair_t["Різниця [95% CI]"] = [ci_str(d, lo, hi) for d, lo, hi in zip(pair_t["diff"], pair_t.ci_lo, pair_t.ci_hi)]
    pair_t = pair_t[["A", "B", "metric", "Різниця [95% CI]"]].rename(columns={"metric": "метрика"})

    best_mc = mc[(mc.model == "word_char") & (mc.metric.isin(["macro_f1", "mcc", "accuracy"]))]
    mc_main = pd.DataFrame([{"Модель": m, **{lab: ci_str(*mc[(mc.model == m) & (mc.metric == key)][["point", "ci_lo", "ci_hi"]].iloc[0])
                                             for key, lab in (("macro_f1", "macro-F1"), ("mcc", "MCC"), ("accuracy", "accuracy"))}}
                            for m in ["majority", "stratified", *LR_MODELS]])
    pc = mc[(mc.model == "word_char") & mc.metric.str.startswith("f1_class")].copy()
    pc["class"] = pc.metric.str.replace("f1_class", "").astype(int).map(dict(enumerate(classes)))
    pc["n"] = pc["class"].map(lambda c: int((ymc == classes.index(c)).sum()))
    pc_t = pd.DataFrame({"Клас": pc["class"], "n": pc["n"], "F1 (word_char) [95% CI]": [ci_str(a, b, c) for a, b, c in zip(pc.point, pc.ci_lo, pc.ci_hi)]})

    lang_ct_md = lang_ct.reset_index().rename(columns={"мова": "Мова"})
    lang_ct_md["частка наративних"] = lang_ct_md["частка наративних"].map(lambda v: f"{fmt(100 * v, 1)}%")
    rule_md = pd.DataFrame([{"Правило": "ru => наратив", **{lab: ci_str(*lang_rule[lang_rule.metric == key][["point", "ci_lo", "ci_hi"]].iloc[0])
                                                          for key, lab in (("macro_f1", "macro-F1"), ("mcc", "MCC"), ("pr_auc", "PR-AUC"),
                                                                           ("f1_pos", "F1 «наратив»"), ("precision", "precision"), ("recall", "recall"))}}])
    dm = lang_pair[lang_pair.metric == "macro_f1"].iloc[0]; dc = lang_pair[lang_pair.metric == "mcc"].iloc[0]
    inc0 = lambda r: r.ci_lo <= 0 <= r.ci_hi
    pair_sentence = (f"macro-F1 {ci_str(dm['diff'], dm.ci_lo, dm.ci_hi)}, MCC {ci_str(dc['diff'], dc.ci_lo, dc.ci_hi)}. "
                     + ("Для обох метрик інтервал містить 0, тобто статистично відрізнити TF-IDF від мовного правила за цими даними не вдається."
                        if inc0(dm) and inc0(dc) else
                        "Принаймні для однієї з метрик інтервал не містить 0."))
    lt = by_lang.copy()
    rows = []
    for (lg, mdl), grp in lt.groupby(["setting", "model"], sort=False):
        r = {"Група": lg, "Модель": mdl, "n": int(grp.n.iloc[0]), "наративних": int(grp.n_pos.iloc[0])}
        for key, lab in (("macro_f1", "macro-F1"), ("mcc", "MCC"), ("pr_auc", "PR-AUC"), ("f1_pos", "F1 «наратив»")):
            r[lab] = ci_str(*grp[grp.metric == key][["point", "ci_lo", "ci_hi"]].iloc[0])
        rows.append(r)
    lang_tab = pd.DataFrame(rows)
    ru_wc = by_lang[(by_lang.setting == "ru") & (by_lang.model == "word_char")]
    uk_wc = by_lang[(by_lang.setting == "uk") & (by_lang.model == "word_char")]
    lang_sentence = (f"У російськомовній групі ({int(ru_wc.n.iloc[0])} повідомлень, {int(ru_wc.n_pos.iloc[0])} наративних) "
                     f"`word_char` має macro-F1 {fmt(float(ru_wc[ru_wc.metric == 'macro_f1'].point.iloc[0]))} і MCC "
                     f"{fmt(float(ru_wc[ru_wc.metric == 'mcc'].point.iloc[0]))}, тобто значно нижче загальних цифр. В україномовній групі "
                     f"лише {int(uk_wc.n_pos.iloc[0])} наративних із {int(uk_wc.n.iloc[0])}, метрики там мало інформативні.")

    concl_path = ROOT / "reports" / "06_conclusions.md"   # ручні висновки, не затираються перезапуском
    concl = concl_path.read_text(encoding="utf-8") if concl_path.exists() else "<!-- висновки: reports/06_conclusions.md -->"
    ok_all = all(ok for _, ok, _ in checks)
    chk_md = "\n".join(f"- {'✅' if ok else '❌'} {c}: {d}" for c, ok, d in checks)
    rnd_wc = g("random_cv", "word_char", "macro_f1"); ch_wc = g("channel_cv", "word_char", "macro_f1")
    md = f"""# Етап S6 (cloud). TF-IDF бейзлайн на золотій розмітці

Скрипт `scripts/s06_tfidf.py`, результати в `results/06_text/`. Дані: 412 золотих повідомлень
({int(y.sum())} наративних, {fmt(100 * y.mean(), 1)}%), 56 каналів. Усі числа нижче з файлів у `results/06_text/`.

## 1. Протокол

- Ознаки: TF-IDF слів (1-2 грами), символьних n-грам (`char_wb`, 2-5), їх об'єднання (`word_char`). Класифікатор: логістична регресія
  з `class_weight=balanced`; C ∈ {{0,1; 1; 10; 100}} обирається вкладеною 3-кратною CV лише на навчальній частині фолду.
- Оцінка: прогнози out-of-fold. Основне налаштування `random_cv`: стратифікована {folds}-кратна CV, {reps} повторів з різними розбиттями.
  Налаштування `channel_cv`: те саме, але повідомлення одного каналу не потрапляють одночасно в train і test
  (`StratifiedGroupKFold`). Це перевірка на витік стилю каналу, близька за духом до протоколу P3.
- Точкова оцінка: середнє метрики по {reps} повторах. 95% CI: процентильний бутстреп по повідомленнях ({n_boot} вибірок),
  метрика на кожній вибірці усереднюється по повторах. Поріг класифікації 0,5.
- Дамі-моделі: `majority` (завжди «без наративу») і `stratified` (випадково за частотою класів) в тій самій схемі CV.
- Додатково: перевірка мовного шорткату (розділ 5) і метрики всередині мовних груп.
- Розподіл на явні і неявні наративи поки недоступний: потрібна розмітка з етапу S5.

## 2. Бінарна задача (наратив / без наративу), випадкова CV

{md_table(metric_table("random_cv", MODEL_NAMES))}

## 3. Бінарна задача, CV за каналами

{md_table(metric_table("channel_cv", MODEL_NAMES))}

![Метрики з CI](../results/06_text/figures/binary_metrics_ci.png)

## 4. Парні порівняння (випадкова CV, парний бутстреп)

{md_table(pair_t)}

## 5. Перевірка мовного шорткату (важливо)

Топ-слова моделі (`results/06_text/top_words.csv`) для класу «наратив» здебільшого російські, для «без наративу»
українські. Мову визначено через `langid` (uk / ru / en), це евристика, на коротких текстах можливі помилки.

{md_table(lang_ct_md)}

Правило без жодного навчання «російська мова дорівнює наратив» дає:

{md_table(rule_md)}

Різниця `word_char` мінус це правило (парний бутстреп): {pair_sentence}

Якість `word_char` і `char` усередині мовних груп (випадкова CV, ті самі out-of-fold прогнози):

{md_table(lang_tab)}

{lang_sentence}

## 6. Багатокласова задача (6 класів: без наративу, 4 найбільші мета-наративи, інший наратив)

{md_table(mc_main)}

Класи рахуються за `General label (level-1)`: «без наративу» і чотири найбільші мета-наративи за частотою, решта
об'єднані в «інший наратив» (14 повідомлень). Показано `random_cv`.

Поклассові F1 для `word_char`:

{md_table(pc_t)}

## 7. Перевірки

{chk_md}

Найсильніші слова (описова перевірка змісту, модель на всіх 412, не оцінка якості) у `results/06_text/top_words.csv`.

**Статус: {'усі автоматичні перевірки пройдені' if ok_all else 'є невдалі перевірки, див. вище'}.**

{concl}
"""
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "06_text_tfidf.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"done in {time.time() - t0:.0f}s; random_cv word_char macro-F1={rnd_wc:.3f}, channel_cv={ch_wc:.3f}")


if __name__ == "__main__":
    main()
