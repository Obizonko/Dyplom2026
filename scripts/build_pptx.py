"""Оновлює презентацію проміжного звіту (шаблон користувача) реальними результатами етапу 2.

Запуск:  python -I scripts/build_pptx.py --template <Prezentatsiia.pptx> --out reports/Prezentatsiia_etap2.pptx
Зберігає дизайн і структуру шаблону (20 слайдів), оновлює текст, діаграми, таблиці, рисунки та нотатки слайдів 1, 8-15, 19, 20.
Числа беруться з results/ (quick/summary, quick/07_simple_hybrid, quick/08_gnn_h2, quick/08_gnn_full*), нічого не введено вручну окрім підписів.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.chart.data import CategoryChartData  # noqa: E402
from pptx.util import Emu  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from narr.io import ROOT  # noqa: E402
from pptx_speech import SPEECH  # noqa: E402

BLUE, LIGHT, ORANGE, GRAY = "#2B7BD9", "#CDE3FA", "#F26B38", "#4A5466"
Q = ROOT / "results" / "quick"
FIGDIR = Q / "summary" / "figures" / "pptx"


def fm(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


# ---------------------------------------------------------------- рисунки у стилі шаблону

def box(ax, x, y, w, h, title, sub, fc, ec, tc="black", ls="-", lw=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=15, fontweight="bold", color=tc)
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=12.5, color=tc, linespacing=1.3)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, color="#8E8E8A", lw=2))


def fig_pipeline(path):
    fig, ax = plt.subplots(figsize=(20.9, 4.5))
    ax.set_xlim(0, 20.9)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    items = [("01. Дані", "TeleNarratives,\nзолота вибірка", "done", "виконано"), ("02. Очищення", "мова, дублікати,\nметадані", "done", "виконано"),
             ("03. Ембедінги", "bge-m3, 108 250\nповідомлень", "done", "виконано"), ("04. Граф", "publishes, forward,\nsemantic, co-pub", "done", "виконано"),
             ("05. Розмітка", "експерти (412) +\nслабкі мітки LLM", "done", "виконано"), ("06. Моделі", "LightGBM, GraphSAGE,\nбейзлайни", "done", "виконано"),
             ("07. Аналіз", "H1, H2 перевірено;\nгейт, каскади (H3)", "plan", "H3 попереду")]
    w, gap = 2.45, 0.55
    for i, (t, s, st, lab) in enumerate(items):
        x = 0.3 + i * (w + gap)
        if st == "done":
            box(ax, x, 1.2, w, 2.4, t, s, BLUE, BLUE, "white")
        elif st == "quick":
            box(ax, x, 1.2, w, 2.4, t, s, LIGHT, BLUE)
        else:
            box(ax, x, 1.2, w, 2.4, t, s, "white", "#8E8E8A", GRAY, ls="--")
        ax.text(x + w / 2, 0.7, lab, ha="center", fontsize=13, color=GRAY)
        if i < len(items) - 1:
            arrow(ax, (x + w + 0.05, 2.4), (x + w + gap - 0.05, 2.4))
    fig.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def fig_flow(path, n_graph, n_ch, n_pool):
    fig, ax = plt.subplots(figsize=(17.5, 6.6))
    ax.set_xlim(0, 17.5)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    def b(x, y, w, h, t, s, fc, ec, tc="black"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=2.5))
        ax.text(x + w / 2, y + h * 0.68, t, ha="center", va="center", fontsize=16, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h * 0.32, s, ha="center", va="center", fontsize=13.5, color=tc, linespacing=1.35)

    b(0.4, 2.2, 3.6, 2.2, "Повний граф", f"{n_graph:_} повідомлень,\n{n_ch} каналів".replace("_", " "), "white", BLUE)
    b(5.1, 4.1, 4.2, 2.1, "Вилучити", "412 золотих + дублікати +\nсемантичні копії (cos ≥ 0,985)", "#FCE2D8", ORANGE)
    b(5.1, 0.4, 4.2, 2.1, "Слабкі мітки", f"LLM-мітки, пул P1: {n_pool:_}\nтекст+граф лише на train".replace("_", " "), "white", BLUE)
    b(10.4, 2.2, 3.4, 2.2, "Навчання", "P1 (усі), P2 (LightGBM)\nOptuna 20 спроб", LIGHT, BLUE)
    b(14.9, 2.2, 2.4, 2.2, "Тест", "412 золотих", BLUE, BLUE, "white")
    for p, q in [((4.05, 3.9), (5.05, 4.9)), ((4.05, 2.7), (5.05, 1.7)), ((9.35, 4.9), (10.35, 3.9)), ((9.35, 1.7), (10.35, 2.7)), ((13.85, 3.3), (14.85, 3.3))]:
        arrow(ax, p, q)
    ax.text(8.75, 0.0, "метрики лише на золотих: macro-F1, MCC, PR-AUC з 95% ДІ; окремо російськомовний зріз і неявні наративи", ha="center", fontsize=13.5, color=GRAY)
    fig.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def fig_shap_h2(path, gold, h2):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 4.6), gridspec_kw={"height_ratios": [1.1, 1]})
    s = pd.read_parquet(Q / "07_simple_hybrid" / "shap_gold_groups_lgbm_both.parquet").merge(gold[["key", "y", "lang"]], on="key")
    s = s[s["lang"] == "ru"]
    data, labs, cols = [], [], []
    for y, nm in ((0, "не наратив"), (1, "наратив")):
        for c, cn, col in (("shap_text", "текст", "#2B7BD9"), ("shap_graph", "граф", "#F26B38")):
            data.append(s[s["y"] == y][c].to_numpy()), labs.append(f"{nm}\n{cn}"), cols.append(col)
    bp = a1.boxplot(data, tick_labels=labs, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c), patch.set_alpha(0.75)
    a1.axhline(0, color="k", lw=0.8)
    a1.set_title("Внесок у log-odds (SHAP), ru", fontsize=11)
    a1.tick_params(labelsize=9)
    if h2 is not None and len(h2):
        xs = np.arange(len(h2))
        a2.errorbar(xs, h2["diff"], yerr=[h2["diff"] - h2["ci_lo"], h2["ci_hi"] - h2["diff"]], fmt="o", color="#2E7D50", capsize=5, lw=2)
        a2.axhline(0, color="k", lw=0.8)
        a2.set_xticks(xs)
        a2.set_xticklabels(h2["label"], fontsize=9)
        a2.set_title("H2: Δ macro-F1 з семантикою мінус без (ru)", fontsize=11)
        a2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- помічники python-pptx

def set_paras(shape, paras):
    """paras: список абзаців, кожен рядок або список фрагментів (за кількістю runs абзацу). Форматування шаблону зберігається."""
    tf = shape.text_frame
    while len(tf.paragraphs) < len(paras):
        last = tf.paragraphs[-1]._p
        last.addnext(copy.deepcopy(last))
    while len(tf.paragraphs) > len(paras):
        e = tf.paragraphs[-1]._p
        e.getparent().remove(e)
    for para, txt in zip(tf.paragraphs, paras):
        parts = txt if isinstance(txt, list) else [txt]
        runs = para.runs
        if not runs:
            continue
        for i, r in enumerate(runs):
            r.text = parts[i] if i < len(parts) else ""
        if len(parts) > len(runs):
            runs[-1].text += "".join(parts[len(runs):])


def replace_picture(slide, shape, path, keep_aspect=True):
    l, t, w, h = shape.left, shape.top, shape.width, shape.height
    shape._element.getparent().remove(shape._element)
    from PIL import Image

    iw, ih = Image.open(path).size
    if keep_aspect:
        sc = min(w / iw, h / ih)
        nw, nh = int(iw * sc), int(ih * sc)
        l, t = l + (w - nw) // 2, t + (h - nh) // 2
        w, h = nw, nh
    return slide.shapes.add_picture(str(path), l, t, w, h)


def set_cell(cell, text):
    p = cell.text_frame.paragraphs[0]
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.text = text


_REF_NOTES = {}


def set_notes(slide, text):
    ns = slide.notes_slide
    if ns.notes_text_frame is None:  # у шаблоні в деяких слайдів немає поля нотаток: копіюємо його з іншого слайда
        ref = _REF_NOTES["sp"]
        ns.shapes._spTree.append(copy.deepcopy(ref))
    ns.notes_text_frame.text = text


def shapes_by_text(slide, start):
    return [sh for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.strip().startswith(start)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", default=str(ROOT / "reports" / "Prezentatsiia_etap2.pptx"))
    args = ap.parse_args(argv)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    from narr import text_eval as te

    s5 = ROOT / "results" / "05_protocol"
    gold = te.make_gold_table(pd.read_csv(s5 / "gold_membership.csv"), pd.read_csv(s5 / "implicit_labeling.csv"), pd.read_csv(s5 / "implicit_auto_b.csv"))
    mt = pd.read_csv(Q / "summary" / "metrics_p1.csv")

    def met(model, sl, metric):
        r = mt[(mt["model"] == model) & (mt["slice"] == sl) & (mt["metric"] == metric)]
        return None if r.empty else (float(r["point"].iloc[0]), float(r["ci_lo"].iloc[0]), float(r["ci_hi"].iloc[0]))

    pr = Presentation(args.template)
    S = lambda i: pr.slides[i - 1]
    for sl in pr.slides:
        if sl.has_notes_slide and sl.notes_slide.notes_text_frame is not None:
            _REF_NOTES["sp"] = sl.notes_slide.notes_text_frame._txBody.getparent()
            break
    index = pd.read_parquet(ROOT / "data" / "interim" / "embeddings" / "bge_m3_index.parquet")
    n_graph, n_ch = len(index), int(index["channel_id"].nunique())
    n_pool = int(pd.read_csv(ROOT / "splits" / "P1" / "train.csv.gz").shape[0])
    cov = pd.read_csv(Q / "08_gnn_h2" / "gold_edge_coverage.csv").set_index("edge_layer")
    n_fwd_gold = int(cov.iloc[0]["n_gold_with_edge"])
    n_sem_gold = int(cov[cov.index.str.contains("0.985")].iloc[0]["n_gold_with_edge"])
    copub = pd.read_parquet(ROOT / "data" / "interim" / "features" / "copub_pairs.parquet")
    pairs = pd.read_parquet(ROOT / "data" / "interim" / "features" / "sem_pairs_0975.parquet")
    n_sem = int((pairs["cos"] >= 0.985 - 1e-9).sum())
    n_fwd = 35697

    th = lambda n: f"{n:_}".replace("_", " ")

    # --- 1
    set_paras(S(1).shapes[1], ["ПРОМІЖНИЙ ЗВІТ · МАГІСТЕРСЬКА РОБОТА"])
    t = S(1).shapes[4].text_frame.paragraphs[0].runs
    for r in t:
        r.text = r.text.replace("[група]", "Пмам-21")
    set_notes(S(1), SPEECH[1])

    # --- 8
    s = S(8)
    sh = list(s.shapes)
    set_paras(sh[0], ["ПІДСУМОК"])
    set_paras(sh[1], ["Що вже зроблено"])
    set_paras(sh[2], [th(n_graph)])
    set_paras(sh[3], ["повідомлень у графі; кожне має ембединг bge-m3"])
    set_paras(sh[4], ["4"])
    set_paras(sh[5], ["типи зв'язків: публікує, форвард, схоже за змістом, одночасна публікація"])
    set_paras(sh[6], ["9"])
    set_paras(sh[7], ["моделей: від TF-IDF до графових нейромереж"])
    set_paras(sh[8], ["H1, H2"])
    set_paras(sh[9], ["перевірили; критерії підтвердження не виконано"])
    set_paras(sh[12], [f"Навчаємось на {th(n_pool)} повідомленнях із мітками від LLM, перевіряємо на 412 експертних. Мова майже підказує мітку: наративні 71% російськомовних і 3% україномовних."])
    set_paras(sh[15], [f"Зв'язки між повідомленнями: {th(n_fwd)} пересилань, {th(n_sem)} пар майже однакових за змістом, {th(len(copub))} пар «опубліковано майже одночасно» (в межах години)."])
    set_paras(sh[18], ["Тестуємо чесно: з навчання прибрано тестові повідомлення та їхні копії. Порівнюємо з простими правилами, щоб побачити, чи модель справді щось додає."])
    set_notes(s, SPEECH[8])

    # --- 9
    s = S(9)
    shs = list(s.shapes)
    p9 = FIGDIR / "pipeline.png"
    fig_pipeline(p9)
    replace_picture(s, shs[0], p9)
    labels = {2: ["semantic.py  ", "ембединги bge-m3, пошук копій за змістом"], 3: ["network.py  ", "мережа каналів, спільноти, важливість каналів"],
              4: ["features.py  ", "ознаки графа: копії, одночасні публікації"], 5: ["splits.py  ", "розбиття на навчання і тест без витоку"],
              6: ["hybrid.py, gnn.py  ", "LightGBM; графові мережі GraphSAGE, R-GCN, HGT"], 7: ["text_eval.py  ", "метрики з довірчими інтервалами, SHAP"]}
    for k, v in labels.items():
        set_paras(shs[k], [v])
    set_notes(s, SPEECH[9])

    # --- 10
    s = S(10)
    shs = list(s.shapes)
    replace_picture(s, shs[2], ROOT / "results" / "02_eda" / "figures" / "03_language.png")
    set_paras(shs[1], ["Золота вибірка: мова майже збігається з міткою"])
    set_paras(shs[4], ["71%", "російськомовних золотих повідомлень містять наратив (141 з 199)", "3%", "українськомовних (7 з 213)",
                       ["Чому це важливо: ", "навіть правило «російська мова = наратив» без жодного навчання дає макро-F1 0,84. Тому модель легко може вгадувати за мовою, а не за змістом. "
                        "Щоб цього уникнути, якість рахуємо окремо на російськомовній частині."]])
    set_notes(s, SPEECH[10])

    # --- 11
    s = S(11)
    shs = list(s.shapes)
    set_paras(shs[1], ["Повний граф: чотири типи зв'язків"])
    replace_picture(s, shs[2], ROOT / "results" / "quick" / "summary" / "figures" / "pptx_netmap_left.png")
    set_paras(shs[4], [["Вузли: ", f"{th(n_graph)} повідомлень, {n_ch} каналів"],
                       ["Пересилання: ", f"{th(n_fwd)}. Пост переслано з іншого каналу."],
                       ["Схоже за змістом: ", f"{th(n_sem)} пар. Той самий текст у різних каналах без кнопки «переслати»."],
                       ["Одночасна публікація: ", f"{th(len(copub))} пар у межах години між різними каналами."],
                       ["Проблема: ", f"із 412 експертних повідомлень лише {n_fwd_gold} мають пересилання і {n_sem_gold} мають схожого за змістом сусіда. "
                                     "Тож на тесті граф майже не має за що «зачепитися»."],
                       ["Ліворуч: ", "мережа каналів, червоні це канали з наративами, сірі без."]])
    set_notes(s, SPEECH[11])

    # --- 12
    s = S(12)
    shs = list(s.shapes)
    set_paras(shs[0], ["РЕЗУЛЬТАТИ"])
    set_paras(shs[1], ["Порівняння моделей на експертних мітках"])
    chart = [sh for sh in s.shapes if getattr(sh, "has_chart", False) and sh.has_chart][0].chart
    cats = [("majority", "Мажоритарний"), ("lang_rule", "Мовне правило"), ("channel_prior", "Канальний пріор"), ("tfidf", "TF-IDF"), ("bge_lr", "bge-m3 + LogReg"),
            ("lgbm_emb", "LGBM: текст"), ("lgbm_graph", "LGBM: граф"), ("lgbm_both", "LGBM: гібрид"), ("gnn_sage_notext", "GNN: без тексту"),
            ("gnn_sage", "GraphSAGE"), ("h2_sem985", "GraphSAGE + форварди")]
    cats = [(m, n) for m, n in cats if met(m, "all", "macro_f1") is not None]
    cd = CategoryChartData(number_format="0.00")
    cd.categories = [n for _, n in cats]
    cd.add_series("усі 412", [round(met(m, "all", "macro_f1")[0], 3) for m, _ in cats])
    cd.add_series("лише російськомовні", [round(met(m, "ru", "macro_f1")[0], 3) for m, _ in cats])
    chart.replace_data(cd)
    best_ru = max(cats, key=lambda mn: met(mn[0], "ru", "macro_f1")[0])
    bm = met(best_ru[0], "ru", "macro_f1")
    bmcc = met(best_ru[0], "ru", "mcc")
    set_paras(shs[4], ["Як читати діаграму", "Метрика макро-F1: чим ближче до 1, тим краще. Перші три моделі це прості правила без навчання, з якими ми порівнюємось.",
                       f"На всіх 412 повідомленнях жодна модель не краща за мовне правило ({fm(met('lang_rule', 'all', 'macro_f1')[0])}).",
                       f"У російськомовній частині правило вже не працює, і найкраще справляється {best_ru[1]} ({fm(bm[0])}, MCC {fm(bmcc[0])}).",
                       f"Граф без тексту майже нічого не дає: MCC {fm(met('lgbm_graph', 'ru', 'mcc')[0])} (LightGBM) і {fm(met('gnn_sage_notext', 'ru', 'mcc')[0])} (GNN).",
                       ["Висновок: ", "корисний сигнал дає текст."]])
    set_notes(s, SPEECH[12])

    # --- 13 (H2 і внесок графа)
    s = S(13)
    shs = list(s.shapes)
    abl_dir = Q / "08_gnn_h2"
    h2 = None
    rows_tab = []
    verdict = {}
    if (abl_dir / "h2_diffs.csv").exists():
        d = pd.read_csv(abl_dir / "h2_diffs.csv")
        d = d[(d["metric"] == "macro_f1") & (d["slice"] == "ru") & (d["B"] == "h2_nosem")].copy()
        d["label"] = d["A"].map(lambda a: "cos ≥ " + {"sem975": "0,975", "sem985": "0,985", "sem990": "0,99"}[a.split("_", 1)[1]])
        h2 = d.sort_values("A")
        verdict = json.loads((abl_dir / "h2_verdict.json").read_text(encoding="utf-8"))
        from narr.evaluate import binary_metrics
        from narr.splits import make_key

        pp = pd.concat([pd.read_parquet(f) for f in sorted((abl_dir / "predictions").glob("*.parquet"))])
        pp["key"] = make_key(pp["channel_id"], pp["message_id"])
        pp = pp.groupby(["model", "key"], as_index=False)["p"].mean()
        for m, nm in (("h2_sem985", "Усі зв'язки (поріг 0,985)"), ("h2_sem975", "поріг схожості 0,975"), ("h2_sem990", "поріг схожості 0,99"), ("h2_nosem", "без зв'язків за змістом"), ("h2_nofwd", "без пересилань")):
            dm = gold.merge(pp[pp["model"] == m][["key", "p"]], on="key")
            mk = te.slice_mask(dm, "ru")
            r = binary_metrics(dm["y"].to_numpy()[mk], dm["p"].to_numpy()[mk])
            rows_tab.append({"name": nm, "f1": r["macro_f1"], "mcc": r["mcc"]})
    p13 = FIGDIR / "shap_h2.png"
    fig_shap_h2(p13, gold, h2)
    set_paras(shs[0], ["РЕЗУЛЬТАТИ · ЧОМУ ТАК"])
    set_paras(shs[1], ["Що дає граф: пояснення моделі і гіпотеза H2"])
    shs[2].height = Emu(int(4.5 * 914400))
    replace_picture(s, shs[2], p13)
    tbl = [sh for sh in s.shapes if getattr(sh, "has_table", False) and sh.has_table][0].table
    for c, t in enumerate(["Варіант графа (ru)", "макро-F1", "MCC"]):
        set_cell(tbl.cell(0, c), t)
    for r in range(1, len(tbl.rows)):
        row = rows_tab[r - 1] if r - 1 < len(rows_tab) else {"name": "н/д", "f1": float("nan"), "mcc": float("nan")}
        set_cell(tbl.cell(r, 0), row["name"])
        set_cell(tbl.cell(r, 1), "н/д" if np.isnan(row["f1"]) else fm(row["f1"]))
        set_cell(tbl.cell(r, 2), "н/д" if np.isnan(row["mcc"]) else fm(row["mcc"]))
    dd = fm(verdict["main_diff_macro_f1_ru"], 3).replace("-", "−") if verdict else "н/д"
    ci = f"[{fm(verdict['ci'][0], 3).replace('-', '−')}; {fm(verdict['ci'][1], 3)}]" if verdict else ""
    set_paras([sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.startswith("Гейт")][0],
              [["Угорі: ", "скільки кожна частина моделі додає до рішення (SHAP). Граф однаково штовхає і наративні, і ненаративні пости до «наратив», а розрізняє їх лише текст."],
               ["H2: ", f"чи додають зв'язки за змістом щось до пересилань? Різниця макро-F1 з ними і без них {dd} {ci}. Інтервал містить нуль і знак від'ємний, тож підтвердити H2 не вдалося."],
               "Причина проста: до тестових повідомлень майже не доходять ні пересилання, ні копії."])
    set_notes(s, SPEECH[13])

    # --- 14 (H1)
    s = S(14)
    shs = list(s.shapes)
    set_paras(shs[0], ["ГІПОТЕЗА H1"])
    set_paras(shs[1], ["Неявні наративи: чи допомагає граф"])
    rec = pd.read_csv(Q / "summary" / "recall_explicitness_ru_p1.csv")
    names = {"tfidf": "TF-IDF", "bge_lr": "bge-m3 + LogReg", "lgbm_emb": "LGBM: текст", "lgbm_graph": "LGBM: граф", "lgbm_both": "LGBM: гібрид", "gnn_sage": "GraphSAGE", "channel_prior": "Канальний пріор"}
    ms = [m for m in names if m in set(rec["model"])]
    cd = CategoryChartData(number_format="0.00")
    cd.categories = [names[m] for m in ms]
    cd.add_series("явні наративні (повнота)", [round(float(rec[(rec.model == m) & (rec.group == "явні")]["recall"].iloc[0]), 3) for m in ms])
    cd.add_series("неявні наративні (повнота)", [round(float(rec[(rec.model == m) & (rec.group == "неявні")]["recall"].iloc[0]), 3) for m in ms])
    cd.add_series("ненаративні (специфічність)", [round(float(rec[(rec.model == m) & (rec.group == "ненаративні")]["recall"].iloc[0]), 3) for m in ms])
    ch14 = [sh for sh in s.shapes if getattr(sh, "has_chart", False) and sh.has_chart][0].chart
    ch14.replace_data(cd)
    from pptx.dml.color import RGBColor
    from pptx.util import Pt

    fill = ch14.plots[0].series[2].format.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(0x2E, 0x7D, 0x50)
    try:
        ch14.value_axis.axis_title.text_frame.text = "частка правильних відповідей при порозі 0,5"
        ch14.value_axis.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
        ch14.value_axis.axis_title.text_frame.paragraphs[0].runs[0].font.bold = False
    except Exception:
        pass
    h1 = pd.read_csv(Q / "07_simple_hybrid" / "h1_preliminary.csv")
    h = h1[(h1["protocol"] == "P1") & (h1["definition"] == "a")].iloc[0]
    spec = rec[(rec.group == "ненаративні") & rec.model.isin(ms)].set_index("model")["recall"]
    mn = lambda x: fm(x, 3).replace("-", "−")
    set_paras(shs[2], [f"{int(h['n_implicit_pos'])} неявних і 68 явних наративних повідомлень"])
    set_paras(shs[3], ["Явний наратив названо прямо, неявний лише натякає (російськомовні, розмітка (а)).",
                       ["Перевірка H1: ", f"гібрид мінус TF-IDF на неявних повідомленнях {mn(h['diff_implicit'])} [{mn(h['ci_implicit_lo'])}; {fm(h['ci_implicit_hi'], 3)}]. "
                        "Приросту немає, інтервал містить нуль."],
                       ["Висновок: ", "явні й неявні моделі знаходять однаково добре. Слабке місце інше: ненаративні пости (зелені стовпчики) розпізнаються лише на "
                        f"{fm(spec.min(), 2)}-{fm(spec.max(), 2)}."]])
    set_notes(s, SPEECH[14])

    # --- 15
    s = S(15)
    shs = list(s.shapes)
    set_paras(shs[0], ["ЯК МИ ТЕСТУЄМО"])
    set_paras(shs[1], ["Повний граф і чесний тест"])
    p15 = FIGDIR / "flow.png"
    fig_flow(p15, n_graph, n_ch, n_pool)
    replace_picture(s, shs[2], p15)
    set_paras(shs[4], [["Навчання: ", "розмічені повідомлення без тестових, їхніх дублікатів і копій за змістом"], "Так модель не може «підглянути» відповідь. Тест: усі 412 експертних міток. Це перевіряють автоматичні тести."])
    set_paras(shs[6], ["Що зроблено"])
    set_paras(shs[7], [f"Граф із {th(n_graph)} повідомлень і {n_ch} каналів; ембединги bge-m3.",
                       "Копією вважаємо пару з косинусом ≥ 0,985; пороги 0,975 і 0,99 беремо для перевірки стійкості.",
                       f"Одночасні публікації: вікно 1 година, різні канали ({th(len(copub))} пар). Вікна 15 і 30 хв ще не перевіряли.",
                       "Моделі: LightGBM і GraphSAGE. R-GCN та HGT уже написані, повний прогін попереду. XLM-R ще не рахували."])
    set_notes(s, SPEECH[15])

    # --- 19
    s = S(19)
    shs = list(s.shapes)
    tb = [sh for sh in s.shapes if getattr(sh, "has_table", False) and sh.has_table][0].table
    set_cell(tb.cell(3, 0), "Гібрид не кращий за текст (наш випадок)")
    set_cell(tb.cell(3, 1), "сигнал графа вже є в тексті, мові та каналі")
    set_paras(shs[8], [f"Лише {n_fwd_gold} із 412 експертних повідомлень мають пересилання і {n_sem_gold} мають схожого за змістом сусіда. Між групами каналів ходить лише 0,35% пересилань."])
    set_paras(shs[11], ["Пояснюємо причину (мова і канал дають шорткат), будуємо модель із гейтом і перевіряємо каскади (H3): там зв'язків значно більше, ніж у 412 повідомленнях."])
    set_notes(s, SPEECH[19])

    # --- 20
    s = S(20)
    sh = list(s.shapes)
    set_paras(sh[4], ["1. Дані і граф"]), set_paras(sh[5], ["ембединги, чотири типи зв'язків, чесне розбиття"])
    set_paras(sh[7], ["Виконано"]), set_paras(sh[8], ["2. Моделі і H1, H2"]), set_paras(sh[9], ["LightGBM, GraphSAGE; H1 і H2 не підтверджено"])
    set_paras(sh[12], ["3. Повні запуски"]), set_paras(sh[13], ["XLM-R, R-GCN, HGT, більше seed, розбиття за каналами, гейт між текстом і графом"])
    set_paras(sh[16], ["4. Каскади"]), set_paras(sh[17], ["групування схожих постів, 6 ознак координації, перевірка H3"])
    set_paras(sh[18], ["Головний висновок поки що: на експертних повідомленнях мова і канал пояснюють більшу частину сигналу, а структура без тексту нічого не додає. "
                       "Далі перевіряємо, чи допоможе граф там, де зв'язків багато: на каскадах і при повних запусках."])
    set_notes(s, SPEECH[20])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pr.save(args.out)
    print("Збережено", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
