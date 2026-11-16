"""Тести S2 (EDA) на малому синтетичному наборі зі відомими кількостями."""
import numpy as np
import pandas as pd
import pytest

from narr import eda


def lang_stub(texts):
    return np.where(pd.Series(texts).str.contains("ы"), "ru", "uk")


@pytest.fixture()
def inp():
    rows = []
    for ch in (1, 2, 3):
        for mid in range(1, 21):
            lab = mid <= 10
            narr = lab and mid % 2 == 0
            sub = mid % 6 + 1 if narr else None
            text = f"текст {ch} {mid}" + (" ты" if ch == 2 else "")
            rows.append({"channel_id": ch, "message_id": mid, "date_utc": f"2025-0{1 + (mid - 1) // 7}-1{mid % 9}T10:00:00+00:00",
                         "channel_username": f"ch{ch}", "text": text, "views": 10, "forwards": 1, "has_media": False,
                         "binary_label": ("True" if narr else "False") if lab else None,
                         "is_labeled_batch": True if lab else None,
                         "narrative_id": float((sub + 1) // 2) if narr else np.nan,
                         "sub_narrative_id": float(sub) if narr else np.nan,
                         "label_confidence": (0.9 if narr else 0.0) if lab else np.nan})
    msgs = pd.DataFrame(rows)
    # копія золотого тексту в іншому каналі
    msgs.loc[(msgs.channel_id == 3) & (msgs.message_id == 15), "text"] = msgs.loc[
        (msgs.channel_id == 1) & (msgs.message_id == 2), "text"].iloc[0]

    tg = msgs[["channel_id", "message_id", "date_utc"]].copy()
    tg["date_utc"] = pd.to_datetime(tg["date_utc"], utc=True)
    tg["is_reply"] = tg["message_id"] % 5 == 0
    tg["reply_to_msg_id"] = np.nan
    tg["is_forwarded"] = tg["message_id"] % 4 == 0
    tg["fwd_from_channel_id"] = np.where(tg["is_forwarded"], np.where(tg["message_id"] % 8 == 0, 1.0, 999.0), np.nan)
    tg["fwd_from_message_id"] = np.where(tg["is_forwarded"], 3.0, np.nan)
    for c in ("views", "forwards", "replies_count"):
        tg[c] = 1.0
    tg["has_media"] = False
    tg = pd.concat([tg, tg.iloc[:1].assign(channel_id=3, message_id=999)], ignore_index=True)  # ключ лише в tg

    channels = pd.DataFrame({
        "channel_id": [1, 2, 3, 4], "username": ["ch1", "ch2", "ch3", "ch4"],
        "narrative_presence": [True, False, True, False],
        "narrative_presence_status": ["labeled"] * 3 + ["no_labeled_messages"],
        "channel_label": ["n", "nn", "n", "nn"],
        "labeled_messages": [10, 10, 10, 0], "narrative_messages": [5, 5, 5, 0],
        "narrative_share": [0.5, 0.5, 0.5, np.nan]})
    coding = pd.DataFrame({"sub_narrative_id": range(1, 7), "narrative_id": [1, 1, 2, 2, 3, 3],
                           "level_1_id": [1, 1, 1, 1, 2, 2], "level_1": ["A", "A", "A", "A", "B", "B"]})
    fwd = pd.DataFrame({"logical_src_channel_id": [1, 2], "logical_src_message_id": [2, 4],
                        "logical_dst_channel_id": [2, 3], "logical_dst_message_id": [2, 4]})
    sem = pd.DataFrame({"src_channel_id": [1, 3], "src_message_id": [2, 20], "dst_channel_id": [3, 3],
                        "dst_message_id": [15, 19], "similarity_score": [0.99, 0.98]})
    gold = pd.DataFrame({"channel_id": [1, 1, 2, 2, 3, 3], "message_id": [2, 3, 4, 5, 6, 7]})
    g = msgs.merge(gold, on=["channel_id", "message_id"])
    gold = gold.merge(g[["channel_id", "message_id", "text"]], on=["channel_id", "message_id"])
    gold["Binary label"] = [True, False, True, False, False, False]
    gold["General label (level-1)"] = np.where(gold["Binary label"], "A", "Не містить наративу")
    tg_text = gold[["channel_id", "message_id"]].assign(text=eda.normalize_text(gold["text"]))
    return {"msgs": msgs, "tg": tg, "tg_gold_text": tg_text, "gold": gold, "fwd": fwd, "sem": sem,
            "channels": channels, "coding": coding}


def test_normalize_text():
    s = pd.Series(["Привіт, Світе!!  Ok", None])
    assert list(eda.normalize_text(s)) == ["привіт світе ok", ""]


def test_binary_scores_known_values():
    y = np.array([1, 1, 0, 0, 1], bool)
    p = np.array([1, 0, 0, 1, 1], bool)
    sc = eda._binary_scores(y, p)
    assert (sc["tp"], sc["fp"], sc["fn"], sc["tn"]) == (2, 1, 1, 1)
    assert sc["accuracy"] == pytest.approx(3 / 5)
    assert sc["mcc"] == pytest.approx((2 * 1 - 1 * 1) / np.sqrt(3 * 3 * 2 * 2))


def test_integrity(inp):
    r = eda.check_integrity(inp["msgs"], inp["tg"])
    assert r["n_msgs"] == 60 and r["n_tg"] == 61
    assert r["dup_keys_msgs"] == 0 and r["dup_keys_tg"] == 0
    assert r["keys_both"] == 60 and r["keys_only_tg"] == 1 and r["keys_only_msgs"] == 0
    assert r["unparsed_dates_msgs"] == 0


def test_integrity_detects_duplicates_and_range(inp):
    msgs = pd.concat([inp["msgs"], inp["msgs"].iloc[:2]], ignore_index=True)
    r = eda.check_integrity(msgs, inp["tg"], date_min="2025-01-01", date_max="2025-01-31")
    assert r["dup_keys_msgs"] == 2
    assert r["out_of_range_msgs"] > 0


def test_monthly_table(inp):
    t = eda.monthly_table(inp["msgs"])
    assert t["n"].sum() == 60 and t["n_labeled"].sum() == 30 and t["n_narrative"].sum() == 15
    assert t["partial_month"].sum() == 2


def test_per_channel(inp):
    pc = eda.per_channel(inp["msgs"], inp["tg"], inp["channels"])
    assert len(pc) == 4
    assert pc.set_index("channel_id").loc[4, "n_messages"] == 0
    assert pc[pc["n_messages"] > 0]["counts_match"].all()
    assert pc.set_index("channel_id").loc[1, "share_recomputed"] == pytest.approx(0.5)
    bad = inp["channels"].copy()
    bad.loc[0, "labeled_messages"] = 11
    pc2 = eda.per_channel(inp["msgs"], inp["tg"], bad)
    assert not pc2.set_index("channel_id").loc[1, "counts_match"]


def test_label_tables(inp):
    s, t = eda.label_tables(inp["msgs"], inp["coding"])
    assert s["n_labeled"] == 30 and s["n_narrative_labeled"] == 15 and s["narrative_not_labeled"] == 0
    assert s["narrative_share_of_labeled"] == pytest.approx(0.5)
    assert s["sub_to_narrative_match_share"] == 1.0 and s["sub_ids_not_in_coding"] == 0
    assert t["meta_weak"]["n_weak"].sum() == 15
    assert set(t["meta_weak"]["level_1"]) <= {"A", "B"}
    assert t["label_confidence"]["n_narrative"].sum() == 15
    assert t["binary_label_crosstab"]["n"].sum() == 60
    ls = t["length_stats"].set_index("group")
    assert ls.loc["усі", "n"] == 60 and ls.loc["нерозмічені", "n"] == 30


def test_label_tables_detects_inconsistent_sub(inp):
    msgs = inp["msgs"].copy()
    i = msgs.index[msgs["narrative_id"].notna()][0]
    msgs.loc[i, "narrative_id"] = 32.0
    s, _ = eda.label_tables(msgs, inp["coding"])
    assert s["sub_to_narrative_match_share"] < 1.0


def test_language_tables(inp):
    t = eda.language_tables(inp["msgs"], inp["gold"], lang_stub, n_unlabeled=10, seed=1)
    g = t["language_by_group"]
    assert g[g["group"] == "розмічені (усі)"]["n"].sum() == 30
    assert g[g["group"] == "нерозмічені (вибірка)"]["n"].sum() == 10
    x = t["language_x_label"]
    weak = x[x["source"] == "слабкі мітки (розмічені)"].set_index("lang")
    assert weak["n"].sum() == 30
    assert weak.loc["ru", "n"] == 10 and weak.loc["ru", "n_narrative"] == 5
    assert x[x["source"] == "золота розмітка"]["n"].sum() == 6


def test_language_sampling_is_seeded(inp):
    a = eda.language_tables(inp["msgs"], inp["gold"], lang_stub, 10, seed=7)["language_by_channel"]
    b = eda.language_tables(inp["msgs"], inp["gold"], lang_stub, 10, seed=7)["language_by_channel"]
    pd.testing.assert_frame_equal(a, b)


def test_graph_membership(inp):
    t = eda.graph_membership(inp["msgs"], inp["fwd"], inp["sem"]).set_index("group")
    assert t["n"].sum() == 60
    # (1,2) у forward і semantic; (2,2) (2,4) (3,4) лише forward; (3,15) (3,20) (3,19) лише semantic
    assert t.loc["forward і semantic", "n"] == 1
    assert t.loc["лише forward", "n"] == 3 and t.loc["лише semantic", "n"] == 3
    assert t.loc["поза графом", "n"] == 53


def test_forwards_summary(inp):
    f = eda.forwards_summary(inp["tg"], inp["msgs"], inp["channels"], len(inp["fwd"]))
    # is_forwarded: message_id кратне 4 -> 5 на канал (4,8,12,16,20) * 3 + рядок 999 (копія першого, message_id=1) не пересилання
    assert f["n_forwarded"] == 15
    assert f["n_forwarded_from_corpus_channel"] == 6   # message_id кратне 8: 8, 16 у трьох каналах
    assert f["n_forward_edges_file"] == 2


def test_gold_overlap(inp):
    s, t = eda.gold_overlap(inp["gold"], inp["msgs"], inp["coding"], inp["tg_gold_text"], inp["fwd"], inp["sem"])
    assert s["n_gold"] == 6 and s["n_gold_in_graph_messages"] == 6 and s["n_gold_labeled_weak"] == 6
    w = s["weak_vs_gold"]
    assert w["n"] == 6
    assert (w["tp"], w["fp"], w["fn"], w["tn"]) == (2, 1, 0, 3)   # (3,6) слабко наративне, золоте ні
    assert t["text_variants"].set_index("варіант тексту").iloc[0]["точно збігається"] == 6
    assert s["n_corpus_copies_of_gold_text"] == 1 and s["n_gold_with_corpus_copy"] == 1
    assert s["duplicates"]["n_rows_in_duplicate_groups"] == 2
    assert s["n_gold_in_forward_edges"] == 2 and s["n_gold_in_semantic_edges"] == 1


def test_run_all_save_and_figures(inp, tmp_path):
    tables, summary, checks = eda.run_all(inp, lang_stub, lang_sample_unlabeled=10, seed=3)
    hard = [c for c in checks if c["hard"]]
    assert hard and all(c["passed"] for c in hard), [c for c in hard if not c["passed"]]
    assert not [c for c in checks if c["check"].startswith("Кількість повідомлень")][0]["passed"]  # 60 != 1 052 563
    paths = eda.save_all(tables, summary, checks, tmp_path / "out")
    assert (tmp_path / "out" / "summary.json").exists() and (tmp_path / "out" / "per_channel.csv").exists()
    assert len(paths) == len(tables) + 2
    figs = eda.make_figures(tables, summary, tmp_path / "fig")
    assert len(figs) == 6 and all(p.stat().st_size > 2000 for p in figs)
