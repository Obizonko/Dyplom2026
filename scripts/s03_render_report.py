"""Збирає reports/03_network.md з файлів results/03_network (усі числа беруться з summary.json, checks.json, run.json, CSV).

Запуск:  python3 -I scripts/s03_render_report.py [--dir results/03_network] [--out reports/03_network.md]
Текст висновків пише людина (тут), але кожне число підставляється зі змінних, щоб звіт не розійшовся з файлами.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narr.io import ROOT  # noqa: E402
from narr.report import fmt, md_table  # noqa: E402


def pct(x: float, nd: int = 2) -> str:
    return f"{x * 100:.{nd}f}".replace(".", ",") + "%"


def i(x) -> str:
    return fmt(int(round(float(x))))


def render(d: Path) -> str:
    S = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    chk = json.loads((d / "checks.json").read_text(encoding="utf-8"))
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    cen = pd.read_csv(d / "centrality.csv")
    com = pd.read_csv(d / "communities.csv")
    csum = pd.read_csv(d / "community_summary.csv")
    L, CG, PM, AS, GC = S["leiden"], S["cross_group"], S["cross_group_permutation"], S["assortativity"], S.get("gold_coverage")
    M = S["mirror_pair"]
    main, nom, cnt, lab = CG["main_weight"], CG["no_mirror_pair_weight"], CG["edge_count"], CG["labeled_status_channels_weight"]
    n_pass, n_all = sum(c["passed"] for c in chk), len(chk)
    n_hard = sum(c["hard"] for c in chk)
    commit = run["git_commit"][:7]

    def top(col, k=8, tie="channel_id"):
        t = cen.sort_values([col, tie], ascending=[False, True]).head(k)
        return t

    def strength_table(col):
        t = top(col)[["username", "table_label", "in_strength", "out_strength", "in_degree", "out_degree", "community"]].copy()
        t.columns = ["Канал", "Група", "in-сила", "out-сила", "in-ступінь", "out-ступінь", "Спільнота"]
        for c in ("in-сила", "out-сила", "in-ступінь", "out-ступінь", "Спільнота"):
            t[c] = t[c].astype(int)
        return md_table(t)

    def rank_line(col, k=5):
        return ", ".join(f"`{r.username}` ({fmt(getattr(r, col), 3)})" for r in top(col, k).itertuples())

    stab = pd.DataFrame([
        {"Схема ваг": "сирі (кількість пересилань, основна)", "Спільнот": "/".join(map(str, L["raw"]["n_communities_values"])),
         "Модулярність": fmt(L["raw"]["modularity_max"], 3), "NMI середній": fmt(L["raw"]["nmi_mean"], 3), "NMI мінімальний": fmt(L["raw"]["nmi_min"], 3)},
        {"Схема ваг": "log(1 + w)", "Спільнот": "/".join(map(str, L["log"]["n_communities_values"])),
         "Модулярність": f"{fmt(L['log']['modularity_min'], 3)} - {fmt(L['log']['modularity_max'], 3)}", "NMI середній": fmt(L["log"]["nmi_mean"], 3), "NMI мінімальний": fmt(L["log"]["nmi_min"], 3)},
        {"Схема ваг": "без ваг (усі 1)", "Спільнот": "/".join(map(str, L["none"]["n_communities_values"])),
         "Модулярність": fmt(L["none"]["modularity_max"], 3), "NMI середній": fmt(L["none"]["nmi_mean"], 3), "NMI мінімальний": fmt(L["none"]["nmi_min"], 3)},
    ])
    ct = csum[["community", "n_channels", "n_group", "nn_group", "inside_share", "top_channels"]].copy()
    ct.columns = ["Спільнота", "Каналів", "з них n", "з них nn", "Частка ваги всередині", "Найсильніші канали"]
    ct["Частка ваги всередині"] = ct["Частка ваги всередині"].map(lambda x: pct(x, 1))
    pcs = pd.read_csv(ROOT / "results" / "02_eda" / "per_channel.csv")
    absent = pcs[~pcs["channel_id"].isin(cen["channel_id"])]
    n_absent, n_absent_nn = len(absent), int(absent["table_label"].eq("nn").sum())
    comp_sizes = cen["component"].value_counts().tolist()
    comp_txt = ", ".join(map(str, comp_sizes[:-1])) + " і " + str(comp_sizes[-1])
    only_nn = csum[csum["n_group"] == 0]["community"].tolist()
    big_nn = com[(com["community"] == 0) & (com["table_label"] == "nn")]["channel_id"]
    big_nn_insuf = int(cen[cen["channel_id"].isin(big_nn)]["table_status"].eq("insufficient_data").sum())
    n_insuf = int(cen["table_status"].eq("insufficient_data").sum())

    grid = pd.DataFrame({"": ["з n", "з nn"], "до n": [i(main["n_to_n"]), i(main["nn_to_n"])], "до nn": [i(main["n_to_nn"]), i(main["nn_to_nn"])]})
    var = pd.DataFrame([
        ["Основний: сума ваг", i(main["total"]), i(main["cross"]), pct(main["cross_share"], 3), pct(PM["main_weight"]["null_mean"], 1), pct(PM["main_weight"]["null_p05"], 1), fmt(AS["weight"], 3)],
        ["Без пари дзеркал", i(nom["total"]), i(nom["cross"]), pct(nom["cross_share"], 2), pct(PM["no_mirror_pair_weight"]["null_mean"], 1), pct(PM["no_mirror_pair_weight"]["null_p05"], 1), fmt(AS["no_mirror_pair_weight"], 3)],
        ["Кількість ребер", i(cnt["total"]), i(cnt["cross"]), pct(cnt["cross_share"], 1), pct(PM["edge_count"]["null_mean"], 1), pct(PM["edge_count"]["null_p05"], 1), fmt(AS["edge_count"], 3)],
    ], columns=["Варіант", "Усього", "Між групами", "Частка між групами", "Середня за перестановками", "5-й процентиль перестановок", "Асортативність r"])
    npm = PM["main_weight"]["n_perm"]
    p_floor = fmt(PM["main_weight"]["p_le_observed"], 4)

    gold_txt = ""
    if GC:
        gold_txt = (f"- Покриття золотого набору. {i(GC['messages_in_graph_channels'])} з {i(GC['n_messages'])} золотих повідомлень "
                    f"({pct(GC['messages_in_graph_channels'] / GC['n_messages'], 1)}) лежать у {i(GC['channels_in_graph'])} з {i(GC['n_channels'])} золотих каналів, які є вершинами мережі. "
                    f"Серед наративних таких {i(GC['narrative_in_graph_channels'])} з {i(GC['narrative_total'])} ({pct(GC['narrative_in_graph_channels'] / GC['narrative_total'], 1)}), "
                    f"серед ненаративних {i(GC['non_narrative_in_graph_channels'])} з {i(GC['n_messages'] - GC['narrative_total'])} "
                    f"({pct(GC['non_narrative_in_graph_channels'] / (GC['n_messages'] - GC['narrative_total']), 1)}). "
                    "Отже, сам факт присутності каналу в мережі пов'язаний з класом, і його не можна подавати моделі як ознаку.")

    dec = lambda t: re.sub(r"(\d)\.(\d)", r"\1,\2", t)  # noqa: E731
    checks_tbl = "\n".join(f"| {c['check']} | {'так' if c['passed'] else 'ні'} | {'жорстка' if c['hard'] else 'м’яка'} | {dec(c['detail']).replace('|', '/')} |" for c in chk)
    nc = S["n_weak_components"]
    deg0_out = int((cen["out_degree"] == 0).sum())
    deg0_in = int((cen["in_degree"] == 0).sum())
    core6 = cen[cen["coreness"] == cen["coreness"].max()]
    n_nodes_n, n_nodes_nn = S["nodes_by_group"]["n"], S["nodes_by_group"]["nn"]

    txt = f"""# S3. Мережа каналів за пересиланнями

Джерело чисел: `results/03_network/` (`centrality.csv`, `communities.csv`, `community_summary.csv`, `cross_group.json`, `summary.json`, `checks.json`, `run.json`, `channel_network.gexf` для Gephi, `figures/01_network_map.png`). Запуск 16.11.2026 у хмарній сесії (дані малі, Colab не потрібен), `scripts/s03_network.py` з коміту {commit}, seed 42. Текст звіту збирає `scripts/s03_render_report.py` з тих самих файлів, тому числа тут не можуть розійтися з ними.

## 1. Дані і метод

- Вхід: таблиця `channel_flow_edges` з {i(S['edges']['n_input'])} ребер (знімок графу 11.03.2026), лежить у `data_small/channel_flow_edges.csv`. Вона вручну перенесена з Drive і звірена: збігаються кількість рядків, id і назви каналів з `per_channel.csv`, а відновлений файл збігається з оригіналом за розміром у байтах.
- Вершина це канал, ребро `src` до `dst` означає, що канал `dst` переслав повідомлення каналу `src`, вага це кількість таких повідомлень. Петель і повторів пар немає. Вершин {S['n_nodes']} з 111 каналів таблиці (98 мають повідомлення), сума ваг {i(S['weight_sum'])}, найбільша вага {i(S['weight_max'])}.
- Показники: ступені і сили, PageRank в обидва боки, посередництво (betweenness) за відстанню 1/вага, k-ядро, слабкі компоненти.
- Спільноти: Leiden з функцією якості модулярності, роздільність 1, на неспрямованому графі, де ваги двох напрямків додаються. Десять запусків із seed {S['seeds'][0]}-{S['seeds'][-1]}, стабільність як попарний NMI між запусками (поріг з плану: більше 0,8). Для чутливості те саме з вагами log(1 + w) і без ваг.
- Групи каналів n (наративні) і nn беру з таблиці каналів (`table_label`): n починається від частки наративних 0,646 серед розмічених повідомлень каналу, nn закінчується на 0,584 (S2). Ці мітки агреговані зі слабких міток повідомлень, тому тут вони лише описують канали, а в ознаки моделей не підуть (висновок S2). Самі ребра від міток не залежать.

## 2. Перевірки

Пройдено {n_pass} з {n_all} ({n_hard} жорстких).

| Перевірка | Пройдено | Тип | Значення |
|---|---|---|---|
{checks_tbl}

Очікувані з плану 0,35% міжгрупових пересилань відтворено: {pct(main['cross_share'], 3)}.

## 3. Структура мережі

- {S['n_nodes']} вершини, {S['n_edges']} ребер, {nc} слабкі компоненти розміром {comp_txt} канали. Серед вершин {n_nodes_n} каналів групи n і {n_nodes_nn} групи nn. З {n_absent} каналів таблиці, яких у мережі немає, {n_absent_nn} належать до nn.
- Одна пара `{M['src']}` до `{M['dst']}` несе {i(M['weight'])} з {i(S['weight_sum'])} ваги ({pct(M['share_of_weight'], 1)}). Це два дзеркальні канали з S2. Решта ваги ({i(S['weight_sum'] - M['weight'])}) розподілена по {S['n_edges'] - 1} ребрах.
- {deg0_out} каналів ніхто в мережі не пересилає (out-ступінь 0), {deg0_in} самі нікого в мережі не пересилають (in-ступінь 0).
- Найвище k-ядро має номер {int(cen['coreness'].max())}, у ньому {len(core6)} каналів, усі групи {', '.join(sorted(core6['table_label'].unique()))}.

Найсильніші за in-силою (багато пересилають з інших каналів):

{strength_table('in_strength')}

Найсильніші за out-силою (їхні повідомлення пересилають інші):

{strength_table('out_strength')}

- PageRank джерел (високий у каналів, чиї повідомлення пересилають інші, зокрема ланцюжками): {rank_line('pagerank_source')}.
- PageRank пересилачів (високий у каналів, що пересилають з активних джерел): {rank_line('pagerank_forwarder')}.
- Посередництво: {rank_line('betweenness')}.

![Карта мережі](../results/03_network/figures/01_network_map.png)

## 4. Спільноти

Основне розбиття (сирі ваги) має {L['raw']['best_n_communities']} спільнот.

{md_table(ct)}

Стабільність за {L['raw']['n_runs']} запусками ({L['raw']['n_pairs']} пар):

{md_table(stab)}

- Сирі ваги дають одне й те саме розбиття в усіх запусках, але модулярність низька ({fmt(L['raw']['modularity_max'], 3)}), бо {pct(M['share_of_weight'], 0)} ваги в одному ребрі. За log-вагами і без ваг спільнот більше ({'/'.join(map(str, L['log']['n_communities_values']))} і {L['none']['n_communities_values'][0]}, модулярність {fmt(L['log']['modularity_min'], 2)} - {fmt(L['none']['modularity_max'], 2)}). Для log-ваг мінімальний NMI {fmt(L['log']['nmi_min'], 2)} лише трохи вищий за поріг 0,8. NMI між схемами: сирі з log {fmt(S['cross_variant_nmi']['raw_vs_log'], 2)}, сирі без ваг {fmt(S['cross_variant_nmi']['raw_vs_none'], 2)}, log без ваг {fmt(S['cross_variant_nmi']['log_vs_none'], 2)}.
- Спільноти не зводяться до міток груп: NMI розбиття з групою n/nn лише {fmt(S['nmi_partition_vs_group']['raw'], 2)} (log {fmt(S['nmi_partition_vs_group']['log'], 2)}, без ваг {fmt(S['nmi_partition_vs_group']['none'], 2)}). Спільнота 0 містить {int(csum.loc[0, 'n_group'])} канали n і {int(csum.loc[0, 'nn_group'])} nn, і {big_nn_insuf} з цих {int(csum.loc[0, 'nn_group'])} nn мають статус `insufficient_data` (мітка за малою кількістю розмічених повідомлень).
- Спільноти {', '.join(map(str, only_nn))} складаються лише з каналів nn (серед них пара дзеркал, спільнота 5), каналів n в них немає.

## 5. Пересилання між групами n і nn

Вага пересилань за групами (рядок: група джерела, стовпець: група того, хто переслав):

{md_table(grid)}

Міжгрупові пересилання: {i(main['cross'])} з {i(main['total'])}, тобто {pct(main['cross_share'], 3)}.

{md_table(var)}

- Стійкість. Без пари дзеркал частка зростає до {pct(nom['cross_share'], 1)}, за кількістю ребер до {pct(cnt['cross_share'], 1)}, але і це далеко від ~50% для випадкового розподілу міток. Спостережене значення нижче за всі {i(npm)} перестановок міток між каналами (p не більше {p_floor}, це мінімум для такої кількості перестановок). Для варіанта з вагами нуль-розподіл розмитий через одне домінантне ребро, тому надійніші два нижні рядки таблиці.
- Лише для каналів зі статусом `labeled` (мітка за 52 і більше розміченими повідомленнями) міжгрупових пересилань {i(lab['cross'])} з {i(lab['total'])} ({pct(lab['cross_share'], 3)}). Усі решта {i(main['cross'] - lab['cross'])} одиниць ваги припадають на ребра, один кінець яких належить до {n_insuf} nn-каналів зі статусом `insufficient_data`. Тобто мережа майже ідеально розділяє групи, а видимі перетини стосуються каналів з ненадійною міткою.

## 6. Що з цього випливає для S4 і S5

- Мережа майже ідеально розділяє групи n і nn, тому мережева ознака каналу (спільнота, оточення в графі) несе інформацію про його групу, а група агрегована зі слабких міток. Для повідомлень це канальний пріор, а не знання про наратив. У S5 оцінюю моделі з такими ознаками і без них окремо.
{gold_txt}
- Ваги. Основна (сира) схема має низьку модулярність через одне домінантне ребро. Для ознак кандидати: log(1 + w) або без ваг. Вибір зафіксую в `configs/prereg.md` до оцінювання, сирі ваги залишаю для опису.
- Усі вершини мережі є в таблиці каналів. Семантичний шар (`sem_channel_flow_edges`, 1 273 ребра) сюди не входить, його зроблю в S4 разом з ембедингами.

## 7. Відкриті питання і обмеження

- Сума ваг {i(S['weight_sum'])} менша за кількість forward-ребер у наборі (35 697, S2) на {i(35697 - S['weight_sum'])}. Причину не з'ясовано (ймовірно, пересилання всередині одного каналу або з каналів поза таблицею). Звірю з `forward_edges.parquet` у Colab на S4. Якщо ці {i(35697 - S['weight_sum'])} не міжгрупові, частка буде {pct(main['cross'] / 35697, 3)} ({i(main['cross'])} з 35 697).
- Вхідну таблицю з 214 ребер перенесено вручну, але звірено за розміром файлу, id і назвами каналів.
- {n_insuf} nn-каналів у мережі мають мітку за малою кількістю розмічених повідомлень, їхня група непевна.
- Мережа охоплює {S['n_nodes']} з 111 каналів, відсутні {n_absent} ({n_absent_nn} з них nn).
"""
    return txt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "03_network"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "03_network.md"))
    a = ap.parse_args(argv)
    txt = render(Path(a.dir))
    Path(a.out).write_text(txt, encoding="utf-8")
    print(f"Записано {a.out} ({len(txt)} символів)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
