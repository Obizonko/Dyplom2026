"""Збирає reports/05_protocol.md з results/05_protocol (усі числа з summary.json, checks.json, CSV).

Запуск:  python3 -I scripts/s05_render_report.py [--dir results/05_protocol] [--out reports/05_protocol.md]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narr.io import ROOT  # noqa: E402
from narr.report import fmt, md_table  # noqa: E402


def pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:.{nd}f}".replace(".", ",") + "%"


def render(d: Path) -> str:
    S = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    chk = json.loads((d / "checks.json").read_text(encoding="utf-8"))
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    G, GC, IB = S["gold"], S["gold_channels"], S.get("implicit_auto_b")
    folds = pd.read_csv(d / "p3_fold_sizes.csv")
    p2l = pd.read_csv(d / "p2_test_by_language.csv")
    done = S["train_side"] == "done"
    n_pass, n_hard = sum(c["passed"] for c in chk), sum(c["hard"] for c in chk)
    commit = (run.get("git_commit") or "")[:7]
    L = []
    add = L.append

    add("# S5. Протокол оцінювання і преєстрація\n")
    add(f"Код: `src/narr/splits.py`, `scripts/s05_splits.py`, `tests/test_splits.py`, `tests/test_leakage.py`, коміт `{commit}`. Преєстрація: `configs/prereg.md` (версія 1, ще не заморожена). "
        f"Числа з `results/05_protocol/` (`summary.json`, `checks.json`, CSV).\n")
    status = ("усе виконано, усі жорсткі перевірки пройдено" if done and all(c["passed"] for c in chk if c["hard"]) else
              "виконано тестову частину (золоті множини P1-P3, шаблон розмітки), навчальна частина чекає на результати S4 (список копій золотих і індекс розмічених)")
    add(f"**Статус:** {status}.\n")

    add("## 1. Підсумок\n")
    add(f"Три протоколи описано і закодовано. Тест P1 це всі {G['n']} золотих ({G['n_narrative']} наративних, {G['n_channels']} каналів). "
        f"Тест P2 це {G['n_p2_test']} золотих від {G['cutoff']} ({G['n_p2_test_narrative']} наративних, {G['n_p2_test_channels']} каналів), число збігається з планом. "
        f"P3 за планом виявився порожнім, тому його перевизначено на групи за каналами (знахідка 1). "
        f"Автоматичне визначення неявних наративів дає лише {IB['n_implicit'] if IB else 'н/д'} зі {IB['n_narrative'] if IB else 'н/д'} (знахідка 2), тож для гіпотези H1 потрібна ручна розмітка (знахідка 3).\n")

    add("## 2. Тестові множини\n")
    add(md_table(pd.DataFrame([
        {"Протокол": "P1", "Тест": G["n"], "Наративних": G["n_narrative"], "Каналів": G["n_channels"]},
        {"Протокол": "P2 (від " + G["cutoff"] + ")", "Тест": G["n_p2_test"], "Наративних": G["n_p2_test_narrative"], "Каналів": G["n_p2_test_channels"]},
        {"Протокол": f"P3 ({G['p3_folds']} груп каналів)", "Тест": int(folds["n_test"].sum()), "Наративних": int(folds["n_narrative"].sum()), "Каналів": int(folds["n_channels"].sum())}])))
    add("\nГрупи P3 (кожне золоте повідомлення тестується рівно в одній групі):\n")
    f2 = folds.rename(columns={"fold": "Група", "n_test": "Тест", "n_narrative": "Наративних", "n_channels": "Каналів", "n_ru": "ru", "n_uk": "uk"})
    add(md_table(f2))
    add("\nТест P2 за мовою:\n")
    p2 = p2l.assign(Частина=p2l["in_p2_test"].map({True: "тест P2 (від " + G["cutoff"] + ")", False: "до " + G["cutoff"]})).rename(
        columns={"lang": "Мова", "n": "Повідомлень", "n_narrative": "Наративних"})[["Частина", "Мова", "Повідомлень", "Наративних"]]
    add(md_table(p2))
    add("")

    add("## 3. Навчальний пул і спліти\n")
    if done:
        P = S["pool"]
        add(f"Розмічена партія: {fmt(P['n_labeled'])} повідомлень. До пулу потрапило {fmt(P['n_pool'])} ({fmt(P['n_pool_narrative'])} наративних слабких міток, "
            f"{pct(P['weak_narrative_share_pool'])}), з {fmt(P['n_pool_channels'])} каналів. Прибрано {fmt(P['n_excluded'])}:\n")
        add(md_table(pd.read_csv(d / "exclusion_by_reason.csv").assign(reason=lambda t: t["reason"].map({
            "is_gold": "саме золоте повідомлення", "same_text_as_gold": "той самий нормалізований текст, що в золотого",
            "semantic_copy_of_gold": "семантична копія золотого (cos ≥ 0,985)"})).rename(columns={"reason": "Причина", "n_any": "Усього з цією причиною", "n_first": "Перша причина"})))
        add("\nПричини перекриваються, тому «усього» не додається до підсумку.\n")
        sz = pd.read_csv(d / "split_sizes.csv")
        sz["fold"] = sz["fold"].map(lambda v: "" if pd.isna(v) else int(v))
        add(md_table(sz.rename(columns={"protocol": "Протокол", "fold": "Група", "n_train": "Train", "n_train_narrative": "Train наративних",
                                        "n_train_channels": "Train каналів", "n_test": "Тест", "n_test_narrative": "Тест наративних", "n_test_channels": "Тест каналів"})))
        add("\nСписки ключів: `splits/P1|P2|P3/` (train у `*.csv.gz`, тест у `*.csv`), виключені з причинами `splits/excluded_from_train.csv.gz`.\n")
    else:
        add(f"Очікує результатів S4 ({S['train_side']}). Код розбиття, тести і перевірки витоку готові й пройдені на синтетичному виході S4. "
            f"Після отримання `04_semantic.zip` скрипт `scripts/s05_splits.py` запускається без змін і дописує розміри train, список виключених і перевірки витоку.\n")

    add("## 4. Знахідки\n")
    add(f"1. **P3 за планом порожній.** План вимагав тест на золотих «з каналів, яких не було в навчанні», але всі {G['n_channels']} золотих каналів мають слабко розмічені повідомлення "
        f"(мінімум {fmt(GC['min_n_labeled'])}, медіана {fmt(GC['median_n_labeled'], 0)} на канал). Замінено на {G['p3_folds']} груп каналів: для тестової групи з навчання прибрано всі повідомлення її каналів. "
        f"Це відхилення записано в `configs/prereg.md`, розділ 10.")
    if IB:
        add(f"2. **Автоматичне визначення неявних вироджене.** За визначенням (б) з плану (TF-IDF дає ймовірність наративу < 0,5 у 5-кратній CV) неявних лише {IB['n_implicit']} зі {IB['n_narrative']} наративних: "
            f"{IB['n_implicit_ru']} російськомовних із {IB['n_narrative_ru']} і {IB['n_implicit_uk']} україномовних із {IB['n_narrative_uk']}. "
            f"У російськомовному зрізі це {IB['n_implicit_ru']} повідомлень, менше за мінімум 30 з преєстрації, а визначення змішане з мовою: "
            f"{IB['n_implicit_uk']} з {IB['n_narrative_uk']} україномовних наративних потрапили в «неявні». Тому основним лишається ручне визначення (а).")
    add(f"3. **Потрібна ручна розмітка «явний / неявний».** Шаблон: `results/05_protocol/implicit_labeling.csv` (148 рядків у випадковому порядку, інструкція в `implicit_labeling_README.md`), приблизно 1 година. "
        f"Золоті тексти в наборі нормалізовані (малі літери, без розділових знаків), оригінального форматування немає, це видно в шаблоні. Без розмітки H1 отримає статус «недостатньо даних».")
    add("")

    add("## 5. Перевірки\n")
    add(f"Пройдено {n_pass} з {len(chk)}, жорстких {n_hard}.\n")
    add(md_table(pd.DataFrame([{"Перевірка": c["check"], "Жорстка": c["hard"], "Результат": "пройдено" if c["passed"] else "НЕ пройдено", "Деталі": c["detail"]} for c in chk if not c["check"].startswith(("P1:", "P2:", "P3 "))])))
    leak = [c for c in chk if c["check"].startswith(("P1:", "P2:", "P3 "))]
    if leak:
        add(f"\nПеревірки витоку для кожного спліту (ключі, однакові тексти, копії cos ≥ 0,985, канали в P3): {sum(c['passed'] for c in leak)} з {len(leak)} пройдено.")
    add("")

    add("## 6. Обмеження\n")
    add("- Золоті тексти нормалізовані, тому розмітка «явний / неявний» робиться за спрощеним текстом.\n"
        "- Групи P3 збалансовано за розміром, кількістю наративних і російськомовних, але канал майже визначає мову, тож групи не є незалежними вибірками за мовою.\n"
        "- Тест P2 має лише 37 наративних повідомлень, довірчі інтервали на ньому будуть широкими.\n"
        "- Преєстрація ще не заморожена: чекає на S4 і відповіді користувача (розділ 10 преєстрації).\n")

    add("## 7. Що далі\n")
    add("Після відповіді користувача щодо розмітки і результатів S4: перезапуск `s05_splits.py`, перегляд звіту, заморожування `configs/prereg.md` окремим комітом. Далі S6 (Colab): ембединги + LogReg і XLM-R за P1.")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "05_protocol"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "05_protocol.md"))
    a = ap.parse_args(argv)
    txt = render(Path(a.dir))
    Path(a.out).write_text(txt, encoding="utf-8")
    print(f"записано {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
