"""S2: похідні числа для reports/02_eda.md.

Читає summary.json і tables_all.json (вихід scripts/s02_eda.py + блок пакування в colab/02_eda.ipynb),
рахує частки і суми, які згадуються у звіті, і пише derived_numbers.json. Нових даних не читає.

Запуск: python scripts/s02_report_numbers.py --dir results/02_eda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Максимальна вага ребра channel_flow_edges (u_now -> u_now_ua), звіт S1 `reports/01_data_manifest.md`, розділ 7.
MAX_FLOW_S1 = 30_810


def _by(rows: list[dict], key: str) -> dict:
    return {r[key]: r for r in rows}


def derive(summary: dict, tables: dict) -> dict:
    out: dict = {}
    lab = summary["labels"]
    n_msg, n_lab, n_nar = lab["n_messages"], lab["n_labeled"], lab["n_narrative_any"]
    out["labeled_share"] = n_lab / n_msg
    out["narrative_share_of_corpus"] = n_nar / n_msg
    out["narrative_share_of_labeled"] = n_nar / n_lab

    # label_confidence
    conf = {float(r["label_confidence"]): r for r in tables["label_confidence"]}
    groups = {"0.0": [0.0], "0.3-0.65": [0.3, 0.4, 0.5, 0.6, 0.65], "0.7-0.85": [0.7, 0.75, 0.8, 0.85],
              "0.9": [0.9], "0.95": [0.95], "1.0": [1.0]}
    out["confidence_groups"] = {k: int(sum(conf[v]["n"] for v in vs)) for k, vs in groups.items()}
    n_not_nar = n_lab - n_nar
    out["confidence_zero_share_of_non_narrative"] = conf[0.0]["n_not_narrative"] / n_not_nar
    out["confidence_non_narrative_nonzero"] = int(n_not_nar - conf[0.0]["n_not_narrative"])
    out["confidence_0_9_share_of_narrative"] = conf[0.9]["n_narrative"] / n_nar
    out["confidence_ge_0_9_share_of_narrative"] = sum(conf[v]["n_narrative"] for v in (0.9, 0.95, 1.0)) / n_nar
    out["confidence_below_0_7_narrative"] = int(sum(conf[v]["n_narrative"] for v in (0.3, 0.4, 0.5, 0.6, 0.65)))

    # binary_label
    bc = tables["binary_label_crosstab"]
    def tot(**kw):
        return sum(r["n"] for r in bc if all(r[k] == v for k, v in kw.items()))
    out["binary_label"] = {
        "narrative_empty": tot(has_narrative_id=True, binary_label="(порожньо)"),
        "narrative_false": tot(has_narrative_id=True, binary_label="False"),
        "narrative_true": tot(has_narrative_id=True, binary_label="True"),
        "unlabeled_true": tot(is_labeled_batch=False, binary_label="True"),
        "true_total": tot(binary_label="True"),
        "false_total": tot(binary_label="False"),
    }
    out["binary_label"]["narrative_empty_share"] = out["binary_label"]["narrative_empty"] / n_nar

    # графова належність
    gm = _by(tables["graph_membership"], "group")
    fwd = gm["forward і semantic"]["n_labeled"] + gm["лише forward"]["n_labeled"]
    out["forward_labeled"] = int(fwd)
    out["forward_labeled_share_of_labeled"] = fwd / n_lab
    out["outside_graph_labeled_share"] = gm["поза графом"]["labeled_share"]

    # канали-дзеркала
    pc = _by(tables["per_channel"], "username")
    a, b = pc["u_now"], pc["u_now_ua"]
    pair_l, pair_n = a["n_labeled"] + b["n_labeled"], a["n_narrative"] + b["n_narrative"]
    out["mirror_pair"] = {
        "labeled": int(pair_l), "labeled_share": pair_l / n_lab,
        "narrative": int(pair_n), "narrative_share_of_all_narrative": pair_n / n_nar,
        "narrative_share_inside": pair_n / pair_l,
        "narrative_share_rest": (n_nar - pair_n) / (n_lab - pair_l),
        "max_flow_share_of_forward_edges": MAX_FLOW_S1 / summary["edges"]["forward_edges"],
    }

    # мова
    lx = {(r["source"], r["lang"]): r for r in tables["language_x_label"]}
    w_ru, w_uk = lx[("слабкі мітки (розмічені)", "ru")], lx[("слабкі мітки (розмічені)", "uk")]
    g_ru, g_uk = lx[("золота розмітка", "ru")], lx[("золота розмітка", "uk")]
    out["language"] = {
        "weak_ru_share": w_ru["n"] / (w_ru["n"] + w_uk["n"]),
        "gold_ru_share": g_ru["n"] / (g_ru["n"] + g_uk["n"]),
        "weak_narrative_ru": w_ru["narrative_share"], "weak_narrative_uk": w_uk["narrative_share"],
        "gold_narrative_ru": g_ru["narrative_share"], "gold_narrative_uk": g_uk["narrative_share"],
    }
    gold_ru = out["language"]["gold_ru_share"]
    out["language"]["weak_rates_at_gold_language_mix"] = (
        gold_ru * w_ru["narrative_share"] + (1 - gold_ru) * w_uk["narrative_share"])

    # час
    mo = tables["monthly"]
    def pooled(rows):
        return sum(r["n_narrative"] for r in rows) / sum(r["n_labeled"] for r in rows)
    early = [r for r in mo if r["month"] <= "2025-05"]
    late = [r for r in mo if r["month"] >= "2025-06"]
    out["monthly"] = {
        "early_months": f'{early[0]["month"]}..{early[-1]["month"]}', "early_narrative_share": pooled(early),
        "late_months": f'{late[0]["month"]}..{late[-1]["month"]}', "late_narrative_share": pooled(late),
        "labeled_share_by_month": {r["month"]: r["n_labeled"] / r["n"] for r in mo},
    }

    # мета-наративи: слабкі проти золотих
    gold_meta = _by(tables["gold_meta_counts"], "level_1")
    rows = []
    for r in tables["meta_weak"]:
        g = gold_meta.get(r["level_1"], {"n_gold": 0, "share_gold": 0.0})
        rows.append({"level_1_id": r["level_1_id"], "n_weak": r["n_weak"], "share_weak": r["share_weak"],
                     "n_gold": g["n_gold"], "share_gold": g["share_gold"]})
    out["meta_weak_vs_gold"] = rows
    out["meta_sums"] = {"weak": int(sum(r["n_weak"] for r in rows)), "gold": int(sum(r["n_gold"] for r in rows))}

    # золотий набір
    g = summary["gold"]
    out["gold"] = {
        "graph_coverage_upper_bound": (g["n_gold_in_forward_edges"] + g["n_gold_in_semantic_edges"]) / g["n_gold"],
        "labeled_overlap_share": g["n_gold_labeled_weak"] / g["n_gold"],
        "with_corpus_copy_share": g["n_gold_with_corpus_copy"] / g["n_gold"],
        "narrative_in_unlabeled": round(g["gold_narrative_share_present_not_labeled"] * g["n_gold_present_not_labeled"]),
        "duplicates_share_all": g["duplicates"]["share_all"],
    }

    # довжина
    ls = _by(tables["length_stats"], "group")
    out["length"] = {
        "median_narrative": ls["розмічені, наративні"]["median"], "median_non_narrative": ls["розмічені, ненаративні"]["median"],
        "median_all": ls["усі"]["median"], "gold_median_raw_text": next(
            r["медіана довжини"] for r in tables["text_variants"] if r["варіант тексту"].startswith("messages")),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/02_eda", type=Path)
    a = ap.parse_args()
    summary = json.loads((a.dir / "summary.json").read_text(encoding="utf-8"))
    tables = json.loads((a.dir / "tables_all.json").read_text(encoding="utf-8"))
    d = derive(summary, tables)
    (a.dir / "derived_numbers.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(d, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
