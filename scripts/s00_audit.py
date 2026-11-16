"""S0: аудит TeleNarratives. Усі числа у звіті перераховуються тут, не копіюються з плану.

Запуск з кореня репозиторію:  python3 -I scripts/s00_audit.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from narr import audit  # noqa: E402
from narr.io import (ROOT, data_available, load_config, load_golden, load_shared, path_of,  # noqa: E402
                     validate_golden, write_run_json)
from narr.report import fmt, md_table  # noqa: E402
from narr.viz import fig_gold_overview  # noqa: E402

OUT = ROOT / "results" / "00_audit"
REPORT = ROOT / "reports" / "00_audit.md"


def main() -> None:
    cfg = load_config()
    if not data_available(cfg):
        sys.exit("Немає external/TeleNarratives. Виконайте: git clone https://github.com/yuliavistak/TeleNarratives external/TeleNarratives")
    exp = cfg["expected"]
    OUT.mkdir(parents=True, exist_ok=True)

    df, shared = load_golden(cfg), load_shared(cfg)
    iaa_file = pd.read_csv(path_of(cfg, "iaa"))
    channels, narratives = pd.read_csv(path_of(cfg, "channels")), pd.read_csv(path_of(cfg, "narratives"))

    # --- золота вибірка
    problems = validate_golden(df, cfg)
    facts = audit.gold_facts(df, cfg)
    monthly, meta, ch = audit.monthly_table(df, cfg), audit.meta_distribution(df, cfg), audit.channel_counts(df, cfg)
    (OUT / "gold_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    monthly.to_csv(OUT / "gold_monthly.csv", index=False)
    meta.to_csv(OUT / "meta_narratives.csv", index=False)
    ch.to_csv(OUT / "gold_channels.csv", index=False)
    fig_gold_overview(monthly, meta, OUT / "figures" / "gold_overview.png")

    # --- IAA
    iaa = audit.iaa_recomputed(shared, iaa_file, cfg)
    iaa.to_csv(OUT / "iaa_recomputed.csv", index=False)

    # --- оригінальні моделі
    models, curves = audit.original_models(path_of(cfg, "gnn_results"))
    models.to_csv(OUT / "original_models.csv", index=False)
    curves.to_csv(OUT / "original_val_curves.csv", index=False)
    weak = audit.weak_label_quality(path_of(cfg, "weak_eval"))
    weak.to_csv(OUT / "weak_label_quality.csv", index=False)

    # --- перевірки
    base = models[models["dir"] == "text-baseline-10"].iloc[0]
    full_graph = models[models["dir"].isin(["gnn-posted-forwards", "gnn-posted-similar-to", "gnn-posted-forwards-similar-to"])]
    checks = [
        ("схема і вміст золотої вибірки (validate_golden)", not problems, "; ".join(problems) or "без зауважень"),
        (f"рядків у золотій вибірці = {exp['n_messages']}", facts["n_messages"] == exp["n_messages"], str(facts["n_messages"])),
        (f"наративних = {exp['n_narrative']}", facts["n_narrative"] == exp["n_narrative"], str(facts["n_narrative"])),
        (f"каналів у золотій вибірці = {exp['n_channels_in_gold']}", facts["n_channels"] == exp["n_channels_in_gold"], str(facts["n_channels"])),
        (f"спільна підвибірка = {exp['n_shared']}", len(shared) == exp["n_shared"], str(len(shared))),
        (f"каналів у config = {exp['n_channels_config']}", len(channels) == exp["n_channels_config"], str(len(channels))),
        (f"піднаративів у config = {exp['n_sub_narratives']}", len(narratives) == exp["n_sub_narratives"], str(len(narratives))),
        ("повторно пораховані κ Фляйса збігаються з файлом (|Δ| < 0,001)", bool((iaa["abs_diff"] < 1e-3).all()),
         "макс. |Δ| = " + fmt(float(iaa["abs_diff"].max()), 5)),
        ("general_f1 у метриках оригіналу = зважений F1 (|Δ| < 0,002)",
         bool(((models["general_f1_reported"] - models["weighted_f1_recomputed"]).abs() < 2e-3).all()),
         "макс. |Δ| = " + fmt(float((models["general_f1_reported"] - models["weighted_f1_recomputed"]).abs().max()), 5)),
        ("MCC з файлу збігається з перерахунком за матрицею помилок (|Δ| < 0,001)",
         bool(((models["mcc_reported"] - models["mcc_recomputed"]).abs() < 1e-3).all()),
         "макс. |Δ| = " + fmt(float((models["mcc_reported"] - models["mcc_recomputed"]).abs().max()), 5)),
        ("n_train + n_val + n_test однаковий для трьох GNN на повному графі",
         full_graph["n_total"].nunique() == 1, ", ".join(str(v) for v in full_graph["n_total"])),
    ]
    (OUT / "checks.json").write_text(json.dumps(
        [{"check": c, "ok": bool(ok), "detail": d} for c, ok, d in checks], ensure_ascii=False, indent=2), encoding="utf-8")
    write_run_json(OUT, "scripts/s00_audit.py", {"expected": exp})

    # --- звіт
    facts_tab = pd.DataFrame([
        ("Повідомлень", facts["n_messages"]),
        ("Наративних (Binary label = True)", f"{facts['n_narrative']} ({fmt(100 * facts['share_narrative'], 1)}%)"),
        ("Каналів у вибірці", f"{facts['n_channels']} з {len(channels)}"),
        ("Частка форвардів", f"{fmt(100 * facts['share_forwarded'], 1)}%"),
        ("Період", f"{facts['date_min']} - {facts['date_max']} ({facts['n_months']} міс.)"),
        ("Повідомлень на місяць (мін / медіана / макс)",
         f"{facts['per_month_min']} / {fmt(facts['per_month_median'], 1)} / {facts['per_month_max']}"),
        ("Довжина тексту, символів (медіана / середня / 90-й перцентиль)",
         f"{fmt(facts['text_len_median'], 0)} / {fmt(facts['text_len_mean'], 0)} / {fmt(facts['text_len_p90'], 0)}"),
        ("Частка найбільшого каналу / п'яти найбільших",
         f"{fmt(100 * facts['top1_channel_share'], 1)}% / {fmt(100 * facts['top5_channel_share'], 1)}%"),
        ("Каналів з 5 і більше повідомленнями", facts["channels_with_ge5_messages"]),
    ], columns=["Показник", "Значення"])

    meta_t = meta.rename(columns={"label": "Мета-наратив (рівень 1)", "n": "Повідомлень", "share_of_narrative": "Частка"})
    meta_t["Частка"] = meta_t["Частка"].map(lambda v: f"{fmt(100 * v, 1)}%")

    iaa_t = iaa.rename(columns={"case": "Випадок", "n_items": "Об'єктів", "fleiss_kappa_recomputed": "κ Фляйса (перераховано)",
                                "fleiss_kappa_in_file": "κ Фляйса (у файлі)", "abs_diff": "|Δ|"})

    orig = models[["model", "n_test", "general_f1_reported", "weighted_f1_recomputed", "macro_f1_recomputed", "f1_pos",
                   "mcc_recomputed", "pr_auc", "roc_auc"]].rename(columns={
        "model": "Модель", "n_test": "n (test)", "general_f1_reported": "F1 у файлі (general_f1)",
        "weighted_f1_recomputed": "зважений F1 (перераховано)", "macro_f1_recomputed": "macro-F1 (перераховано)",
        "f1_pos": "F1 класу «наратив»", "mcc_recomputed": "MCC", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"})
    split_t = models[["model", "n_train", "n_val", "n_test", "n_total", "share_pos_test"]].rename(columns={
        "model": "Модель", "n_train": "train", "n_val": "val", "n_test": "test", "n_total": "разом",
        "share_pos_test": "частка наративних у test"})
    curves_t = curves.rename(columns={"model": "Модель", "n_epochs_val": "епох", "val_f1_min": "мін. val F1",
                                      "val_f1_max": "макс. val F1", "val_f1_std": "σ", "max_abs_step": "макс. стрибок за епоху"})
    weak_t = weak.rename(columns={"method": "Метод", "file": "Файл", "support": "n", "precision": "precision",
                                  "recall": "recall", "f1_pos": "F1 «наратив»", "mcc": "MCC", "roc_auc": "ROC-AUC"})

    best_gnn = full_graph.sort_values("macro_f1_recomputed", ascending=False).iloc[0]
    verb = "не перевищує" if best_gnn["macro_f1_recomputed"] <= base["macro_f1_recomputed"] else "перевищує"
    ok_all = all(ok for _, ok, _ in checks)
    checks_md = "\n".join(f"- {'✅' if ok else '❌'} {c}: {d}" for c, ok, d in checks)

    md = f"""# Етап S0. Аудит даних TeleNarratives

Усі числа перераховані скриптом `scripts/s00_audit.py`, результати в `results/00_audit/`.
Джерело: клон https://github.com/yuliavistak/TeleNarratives (MIT).

## 1. Золота вибірка

{md_table(facts_tab)}

![Золота вибірка](../results/00_audit/figures/gold_overview.png)

Розподіл мета-наративів серед наративних повідомлень:

{md_table(meta_t)}

Висновок для оцінки: багатокласові метрики мають сенс лише для чотирьох найбільших класів, решта класів замалі. Основна
оцінка бінарна. Концентрація по каналах помітна (таблиця вище), тому окрім звичайної крос-валідації потрібна CV за каналами.

## 2. Міжанотаторська згода (спільна підвибірка)

{md_table(iaa_t, nd=4)}

Перерахунок за колонками рецензентів збігається з файлом авторів. У файлі колонка третього рецензента на рівні level-1
названа `Reviewer 2 (level-1).1`, у конфігу це враховано.

## 3. Результати оригінальних моделей

{md_table(orig)}

**Важливо про метрику.** Колонка `general_f1` у файлах авторів дорівнює зваженому F1, а не macro-F1. Для текстового
бейзлайну зважений F1 дорівнює {fmt(base['weighted_f1_recomputed'])}, macro-F1 дорівнює {fmt(base['macro_f1_recomputed'])},
F1 класу «наратив» дорівнює {fmt(base['f1_pos'])}. У нашій роботі основна метрика macro-F1, тому числа оригіналу
в таблиці вище не можна порівнювати з нашими без цього перерахунку.

Розміри вибірок за сплітами:

{md_table(split_t)}

Разом train, val і test для моделей на повному графі дають {fmt(int(full_graph['n_total'].iloc[0]))} повідомлень.
Отже, модель оцінювалась на слабких (LLM) мітках графа, золота розмітка у цих метриках не використана.
Текстовий бейзлайн і GNN мали різні тестові вибірки (n = {fmt(int(base['n_test']))} проти
{fmt(int(full_graph['n_test'].iloc[0]))}), тому пряме порівняння лише орієнтовне. Найкращий GNN на повному графі за macro-F1
({best_gnn['model']}: {fmt(best_gnn['macro_f1_recomputed'])}) {verb} текстовий бейзлайн ({fmt(base['macro_f1_recomputed'])}).

Стабільність навчання за валідаційними кривими (F1 на val по епохах, як у файлах авторів):

{md_table(curves_t)}

## 4. Якість слабких міток за файлами оцінки авторів

Бінарні метрики методів розмітки на золотій вибірці, як вони записані у файлах оцінки (колонка `n` показує,
на скількох повідомленнях рахували; частина прогонів на 100 повідомленнях).

{md_table(weak_t)}

Це оцінка джерела слабких міток, а не наших моделей. Вона показує межу шуму міток, на яких навчаються моделі за протоколом P1.

## 5. Перевірки етапу

{checks_md}

**Статус етапу S0: {'виконано, усі перевірки пройдені' if ok_all else 'є невдалі перевірки, див. вище'}.**

## 6. Що далі

- S6-cloud: TF-IDF бейзлайн на 412 золотих (`reports/06_text_tfidf.md`).
- S1: ноутбук `colab/01_fetch_osf.ipynb` для завантаження повних даних з OSF і маніфесту.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(md, encoding="utf-8")
    print(md)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
