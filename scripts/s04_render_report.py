"""Збирає reports/04_semantic.md з results/04_semantic (усі числа з summary.json, checks.json, run.json, CSV).

Запуск:  python3 -I scripts/s04_render_report.py [--dir results/04_semantic] [--out reports/04_semantic.md]
Текст висновків пише людина (тут), числа підставляються зі змінних.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narr.io import ROOT  # noqa: E402
from narr.report import fmt, md_table  # noqa: E402

CODE_COMMIT = "33e2963"  # код S4 (d6c4f7f додав labeled_index)


def pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:.{nd}f}".replace(".", ",") + "%"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def render(d: Path) -> str:
    S = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    chk = json.loads((d / "checks.json").read_text(encoding="utf-8"))
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    reach = pd.read_csv(d / "reachability.csv")
    gcs = pd.read_csv(d / "gold_copies_summary.csv")
    pj = pd.read_csv(d / "pair_judgement.csv")
    sample = pd.read_csv(d / "threshold_pairs_sample.csv")
    E, ENC, R, OE, PG = S["embedding_set"], S["encoder"], S["reproduction"], S["own_edges"], S["pair_groups"]
    FF = S["forward_vs_channel_flow"]
    S5 = json.loads((d.parent / "05_protocol" / "summary.json").read_text(encoding="utf-8"))["pool"]
    n_hard_fail = sum(c["hard"] and not c["passed"] for c in chk)
    soft_fail = [c["check"] for c in chk if not c["hard"] and not c["passed"]]
    n_pass, n_all = sum(c["passed"] for c in chk), len(chk)
    n_hard = sum(c["hard"] for c in chk)

    # ---- перевірки
    ct = pd.DataFrame([{"Перевірка": c["check"], "Пройдено": "так" if c["passed"] else "ні", "Тип": "жорстка" if c["hard"] else "м'яка",
                        "Значення": c["detail"]} for c in chk])

    # ---- відтворення оцінок
    rep = pd.DataFrame([
        {"Підмножина пар": f"усі ({fmt(R['n'])})", "Пірсон": fmt(R["pearson"]), "Спірмен": fmt(R["spearman"]), "Сер. модуль різниці": fmt(R["mean_abs_diff_direct"], 4)},
        {"Підмножина пар": f"нетривіальні, оцінка < 0,995 ({fmt(R['n_nontrivial'])})", "Пірсон": fmt(R["pearson_nontrivial"]), "Спірмен": fmt(R["spearman_nontrivial"]),
         "Сер. модуль різниці": fmt(R["mean_abs_diff_direct_nontrivial"], 4)}])

    # ---- власні ребра за порогами
    rows = []
    OV = S["overlap_with_orig_all_embedded"]
    for th in ("0.95", "0.975", "0.985", "0.99"):
        b, o = OE["by_threshold"][th], OV[th]
        rows.append({"Поріг": th.replace(".", ","), "Пар": fmt(b["n_pairs"]), "Вузлів з сусідом": fmt(b["n_nodes_with_neighbor"]),
                     "Вузлів із вичерпаним k": fmt(b["n_nodes_saturated"]), "Знайдено ребер набору": pct(o["recall_of_orig"]), "Своїх ребер у наборі": pct(o["share_of_own_in_orig"])})
    edges = pd.DataFrame(rows)

    # ---- 50 пар
    n_copy, n_tpl = int((pj["verdict"] == "copy").sum()), int((pj["verdict"] == "template").sum())
    lo, hi = wilson(n_copy, len(pj))
    both = pj.merge(sample, on=["src_channel_id", "src_message_id", "dst_channel_id", "dst_message_id"], suffixes=("", "_s"))
    tpl_txt = both[both["verdict"] == "template"]["text_src"].astype(str)
    n_births = int(tpl_txt.str.contains("малыш|малышк").sum())

    # ---- копії золотих
    g = gcs.copy()
    g.columns = ["Поріг", "Золотих", "З сусідом", "З сусідом у розміченій партії", "Повідомлень-сусідів", "З них у розміченій партії", "Пар", "Пар з однаковим текстом"]
    g["Поріг"] = g["Поріг"].map(lambda x: fmt(float(x)))
    g985 = gcs[gcs["threshold"] == 0.985].iloc[0]

    # ---- досяжність
    k3 = reach[reach["k"] == 3].copy()
    lay = {"forward": "форварди", "semantic_orig": "семантичні ребра набору", "semantic_own_0985": "власні семантичні, 0,985", "semantic_own_0975": "власні семантичні, 0,975",
           "forward+semantic_orig": "форварди + семантика набору"}
    dire = {"n_to_nn": "n до nn", "nn_to_n": "nn до n", "all": "усі"}
    k3["Шар"] = k3["layer"].map(lay)
    k3["Напрям"] = k3["direction"].map(dire)
    k3["Досягають іншої групи"] = [f"{fmt(int(a))} з {fmt(int(b))} ({pct(c, 2)})" for a, b, c in zip(k3["n_reach"], k3["n_start"], k3["share"])]
    k3["Нульова модель, середнє [p05; p95]"] = [("н/д" if pd.isna(m) else f"{pct(m)} [{pct(a)}; {pct(b)}]") for m, a, b in zip(k3["null_mean"], k3["null_p05"], k3["null_p95"])]
    rt = k3[["Шар", "Напрям", "Досягають іншої групи", "Нульова модель, середнє [p05; p95]"]]
    f3 = k3[(k3["layer"] == "forward") & (k3["direction"] == "n_to_nn")].iloc[0]
    s3 = k3[(k3["layer"] == "semantic_orig") & (k3["direction"] == "all")].iloc[0]
    # на скільки кроків насичується
    a = reach[(reach["layer"] == "semantic_orig") & (reach["direction"] == "all")].set_index("k")["share"]

    return f"""# S4. Семантичний шар

Джерело чисел: `results/04_semantic/` (`summary.json`, `checks.json`, `run.json`, `reachability.csv`, `gold_copies_summary.csv`, `pair_judgement.csv`, `figures/`). Ембединги, пошук сусідів і UMAP рахував Colab на GPU (`colab/04_semantic.ipynb`, код з коміту {CODE_COMMIT}, seed 42, {fmt(int(run['params']['seconds']))} с), підсумкові таблиці й перевірки в хмарній сесії. Текст звіту збирає `scripts/s04_render_report.py`.

Файли плану названо інакше: `summary.json` це `edges_stats.json`, `figures/03_umap.png` це `umap.png`.

## 1. Метод

- Вхід: {fmt(E['n_embedded'])} повідомлень: {fmt(E['n_labeled'])} слабко розмічених, 412 золотих і кінці семантичних ребер набору ({fmt(E['n_sem_endpoints'])} вузлів). Порожніх текстів немає.
- Енкодер `{ENC['model']}` (fp16, {fmt(ENC['dim'])} вимірів, max_len {fmt(ENC['max_len'])}). Медіана довжини {fmt(int(ENC['tokens_median']))} токенів, p95 {fmt(int(ENC['tokens_p95']))}, обрізано {pct(ENC['share_truncated'], 2)} текстів. Норми векторів одиничні, макс. відхилення {str(f"{ENC['max_abs_norm_dev']:.1e}").replace('.', ',')}.
- Власні ребра: hnswlib на косинусі, k = {OE['k']}, пороги 0,95, 0,975, 0,985, 0,99. Повноту hnswlib перевірено точним пошуком на 500 випадкових вузлів: {fmt(OE['hnsw_recall']['0.985']['recall'], 4)} за порога 0,985.

## 2. Перевірки

Пройдено {n_pass} з {n_all}, жорстких {n_hard}, з них не пройдено {n_hard_fail}. М'яких не пройдено {len(soft_fail)}.

{md_table(ct)}

Не пройдена м'яка перевірка описує відтворення оцінок набору за рангами (розділ 3).

## 3. Чи відтворюються оцінки схожості набору

Для {fmt(R['n_pairs_total'])} ребер набору порівняно колонку `similarity_score` з нашим косинусом.

{md_table(rep)}

Оцінки відтворюються за значенням (середній модуль різниці {fmt(R['mean_abs_diff_direct'], 4)}), але не за рангами: {pct(R['share_score_ge_0995'])} пар мають оцінку не нижче 0,995, і серед решти кореляція слабка (спірмен {fmt(R['spearman_nontrivial'])}). Отже, набір рахував схожість іншим енкодером чи іншою обробкою тексту, і наші ребра не копія його ребер. Знайдено лише {pct(OV['0.985']['recall_of_orig'])} ребер набору за порога 0,985, а свої ребра в наборі присутні лише на {pct(OV['0.985']['share_of_own_in_orig'])}.
Практичний наслідок: у моделях семантичні ребра будуємо власним пошуком, а ребра набору лишаємо для порівняння.

## 4. Власні семантичні ребра

{md_table(edges)}

- Ребра набору всі між різними каналами ({pct(PG['orig']['same_channel_share'])} внутрішньоканальних). У власних ребрах за порога 0,985 внутрішньоканальних пар {pct(PG['own_0.985']['same_channel_share'])}, за 0,975 {pct(PG['own_0.975']['same_channel_share'])}. Імовірно, це серійні повтори в межах одного каналу (не перевіряли). Для моделей це ризик: канал майже визначає мітку (S2). Тому в преєстрації основний шар семантичних ребер містить лише міжканальні пари, а чутливість рахується з усіма (`configs/prereg.md`, п. 6 розділу 10).
- Пар між наративними й ненаративними групами каналів (cross) у власних ребрах {pct(PG['own_0.985']['cross_share'], 2)}, у ребрах набору {pct(PG['orig']['cross_share'], 2)}.

### Перегляд 50 пар біля порогу

Вимога плану: переглянути 50 випадкових пар біля порогу. Узято 50 пар ребер набору з оцінками 0,975 до 0,985, вердикти виставила Claude за початком текстів (`pair_judgement_README.md`).
Копій (те саме повідомлення чи подія з тими самими фактами) {n_copy} з {len(pj)} ({pct(n_copy / len(pj), 0)}, 95% ДІ Вілсона {pct(lo, 0)} до {pct(hi, 0)}), серійних шаблонів з різними фактами {n_tpl}. З шаблонів {n_births} це щоденні повідомлення Запорізького мінздраву про кількість народжених (різні числа), решта серійні зведення (МО РФ, ДТП, обстріли) і оголошення. Висновок: поріг на косинусі добре ловить копії, але разом з ними бере серійні формати, тож «семантичне ребро» не гарантує «той самий наратив чи факт». Для ознак моделей це допустимо (шаблонність сама по собі інформативна). Для виключення витоку ми обираємо консервативно: з навчання йдуть і шаблони, тому межа 0,985 разом зі збігом нормалізованого тексту.

## 5. Копії золотих повідомлень

{md_table(g)}

За порога 0,985 {int(g985['gold_with_neighbor'])} золотих мають близького сусіда, всього {int(g985['neighbor_messages'])} повідомлень-сусідів, з них {int(g985['neighbor_messages_in_labeled'])} у розміченій партії. Їх (та ще золоті, збіги за текстом) виключає S5: в пулі виключено {fmt(S5['n_excluded'])} з {fmt(E['n_labeled'])} (`results/05_protocol/summary.json`). Виключення мале, тобто золоті майже не мають копій у навчальній партії, але перевірка потрібна саме для цього.

## 6. Досяжність між групами каналів

Частка повідомлень, з яких за 3 кроки по ребрах можна дійти до повідомлення каналу іншої групи (n наративні, nn ненаративні, групи за `table_label` з S3). Нульова модель: 20 перемішувань міток груп між каналами із збереженням розмірів груп.

{md_table(rt)}

- Спостережена досяжність набагато нижча за нульову модель: для форвардів n до nn {pct(f3['share'], 2)} проти {pct(f3['null_mean'])} (p05 {pct(f3['null_p05'])}). Групи в мережі розділені сильніше, ніж за випадкового розподілу міток, що узгоджується з 0,346% міжгрупових форвардів у S3.
- Досяжність насичується за 1 до 2 кроки: для семантичних ребер набору (усі) {pct(a[1], 2)} за 1 крок, {pct(a[2], 2)} за 2, {pct(a[3], 2)} за 3 і {pct(a[4], 2)} за 4.
- Значення приблизно 0,54 за 3 кроки з плану не відтворено: найближче {pct(s3['share'], 2)} (семантика набору, усі напрями). Невідомо, як рахувалася та цифра в плані, тому порівнювати прямо не варто.

![Досяжність](../results/04_semantic/figures/02_reachability.png)

## 7. Узгодження кількості форвардів

У таблиці форвардів {fmt(FF['n_forward_edges'])} ребер, з них {fmt(FF['n_self_forward'])} самопересилань (канал пересилає сам себе). Без них сума ваг {fmt(FF['weight_sum_without_self'])}, і вона збігається з сумою ваг `channel_flow_edges` у S3 ({fmt(FF['channel_flow_weight_sum'])}) пара за парою. Тобто розбіжність у 730 пояснена самопересиланнями.

## 8. UMAP

UMAP по {fmt(S['umap']['n_points'])} точках (20 000 з графа і 412 золотих).

![UMAP](../results/04_semantic/figures/03_umap.png)

Зліва за мета-наративом, у центрі за мовою, справа золоті на тлі вибірки. Мета-наративи в основній масі сильно перекриваються, окремі компактні острівці дає лише частина класів (зокрема «визвольна» роль Росії). Російська і українська мови в основній масі змішані, тобто bge-m3 тримає їх у спільному просторі, а походження окремих острівців ru-текстів не перевіряли. Золоті наративні повідомлення розкидані по всій хмарі, а не в окремій області. Це розвідувальна картинка без числової оцінки.

![Відтворення оцінок](../results/04_semantic/figures/01_repro_scores.png)

## 9. Обмеження і що далі

- Цифри в розділах 3 і 6 описові. Вони не доводять, що наші ребра кращі за ребра набору, лише що вони інші.
- Вердикти по 50 парах належать одній оцінювачці й тільки за початком текстів.
- Групи n і nn агреговані зі слабких міток, тож досяжність описує мережу каналів, а не наративи як такі.
- Далі: S6 (ембединги bge-m3 з S4 плюс LogReg, дотренування XLM-R за P1) і S7 (ознаки з ребер, де основний шар семантики міжканальний).
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "04_semantic"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "04_semantic.md"))
    a = ap.parse_args(argv)
    txt = render(Path(a.dir))
    bad = [c for c in ("—", "–") if c in txt]
    if bad:
        raise SystemExit(f"У звіті є довгі тире: {bad}")
    Path(a.out).write_text(txt, encoding="utf-8")
    print(f"Записано {a.out} ({len(txt.splitlines())} рядків)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
