import numpy as np
import pandas as pd
import pytest

from narr import text_eval as te
from narr.splits import make_key


def gold_frame():
    """10 золотих: 4 ненаративних, 6 наративних (2 явні, 2 неявні, 1 неявне сумнівне, 1 неясне)."""
    ch = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    mid = list(range(10))
    y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    expl = ["", "", "", "", "explicit", "explicit", "implicit", "implicit", "implicit", "unclear"]
    comment = [""] * 8 + ["framing;borderline", "mixed"]
    mem = pd.DataFrame({"key": make_key(ch, mid), "channel_id": ch, "message_id": mid, "y": y, "lang": ["ru"] * 6 + ["uk"] * 4,
                        "fold": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5], "in_p2_test": [False] * 5 + [True] * 5})
    lab = pd.DataFrame({"channel_id": ch[4:], "message_id": mid[4:], "explicitness": expl[4:], "comment": comment[4:]})
    auto = pd.DataFrame({"key": mem["key"], "implicit_auto": [False] * 6 + [True, False, False, False]})
    return te.make_gold_table(mem, lab, auto)


def test_gold_table_requires_labels_for_narrative():
    g = gold_frame()
    assert g["expl"].tolist()[:4] == [""] * 4
    mem = g[["key", "channel_id", "message_id", "y", "lang", "fold", "in_p2_test"]]
    lab = pd.DataFrame({"channel_id": [3], "message_id": [4], "explicitness": ["explicit"], "comment": [""]})
    with pytest.raises(ValueError):
        te.make_gold_table(mem, lab, pd.DataFrame({"key": mem["key"], "implicit_auto": False}))


def test_explicitness_masks_definitions():
    g = gold_frame()
    m = te.explicitness_masks(g)
    imp, exp = m["a"]
    assert imp.sum() == 3 and exp.sum() == 2                       # неясне не входить нікуди
    imp_n, exp_n = m["narrow"]
    assert imp_n.sum() == 2 and exp_n.sum() == 2                   # сумнівне неявне виключено з неявних
    imp_w, exp_w = m["wide"]
    assert imp_w.sum() == 3 and exp_w.sum() == 2                   # у цьому наборі немає сумнівних явних
    imp_b, exp_b = m["b"]
    assert imp_b.sum() == 1 and exp_b.sum() == 5                   # (б) охоплює всі 6 наративних


def test_subsets_always_include_all_negatives():
    g = gold_frame()
    s = te.subset_masks(g)
    neg = (g["y"] == 0).to_numpy()
    for name, mk in s.items():
        assert mk[neg].all(), name
    assert s["implicit_a"].sum() == 4 + 3 and s["explicit_a"].sum() == 4 + 2
    assert s["all"].all()


def test_slice_mask():
    g = gold_frame()
    assert te.slice_mask(g, "ru").sum() == 6 and te.slice_mask(g, "uk").sum() == 4 and te.slice_mask(g, "all").all()


def _splits_for(g):
    tr = pd.DataFrame({"channel_id": [1, 1, 1, 2, 2, 9], "message_id": range(100, 106), "weak_narrative": [True, True, False, False, False, False]})
    tr["key"] = make_key(tr["channel_id"], tr["message_id"])
    te_ = g[["channel_id", "message_id", "y", "key"]].copy()
    return [{"protocol": "P1", "fold": 0, "train": tr, "test": te_}]


def test_comparators_from_train_only():
    g = gold_frame()
    comp = te.comparator_frame(g, _splits_for(g))
    maj = comp[comp["model"] == "majority"]
    assert (maj["p"] == 0.0).all()                                  # у train наративних 2 з 6
    lr = comp[comp["model"] == "lang_rule"].set_index("key")["p"]
    assert lr.loc[g.loc[g["lang"] == "ru", "key"]].eq(1).all() and lr.loc[g.loc[g["lang"] == "uk", "key"]].eq(0).all()
    cp = comp[comp["model"] == "channel_prior"].set_index("key")["p"]
    prev = 2 / 6
    expect1 = (2 + prev) / (3 + 1)                                  # канал 1: 2 з 3, згладжування alpha=1
    k1 = g.loc[g["channel_id"] == 1, "key"].iloc[0]
    assert cp.loc[k1] == pytest.approx(expect1)
    k5 = g.loc[g["channel_id"] == 5, "key"].iloc[0]                 # каналу 5 немає в train: глобальна частка
    assert cp.loc[k5] == pytest.approx(prev)


def test_comparators_do_not_use_gold_labels():
    g = gold_frame()
    a = te.comparator_frame(g, _splits_for(g))
    g2 = g.assign(y=1 - g["y"])
    b = te.comparator_frame(g2, _splits_for(g2))
    assert a["p"].tolist() == b["p"].tolist()


def _pred(rows):
    return pd.DataFrame(rows, columns=["protocol", "fold", "model", "seed", "channel_id", "message_id", "p"])


def test_check_predictions_detects_problems():
    g = gold_frame()
    rows = [("P1", 0, "m", 0, r.channel_id, r.message_id, 0.5) for r in g.itertuples()]
    ok = te.check_predictions(_pred(rows), g)
    assert ok["n_duplicates"] == 0 and ok["all_covered"] and ok["p_in_range"]
    dup = te.check_predictions(_pred(rows + [rows[0]]), g)
    assert dup["n_duplicates"] == 1
    miss = te.check_predictions(_pred(rows[:-1]), g)
    assert not miss["all_covered"] and miss["not_covered"] == ["m/P1/s0"]
    bad = te.check_predictions(_pred(rows[:-1] + [("P1", 0, "m", 0, 99, 99, 0.5)]), g)
    assert bad["n_unknown_keys"] == 1
    out = te.check_predictions(_pred([(*r[:-1], 1.2) for r in rows]), g)
    assert not out["p_in_range"]
    # P2 має покривати лише золоті з in_p2_test
    p2 = [("P2", 0, "m", 0, r.channel_id, r.message_id, 0.5) for r in g[g["in_p2_test"]].itertuples()]
    assert te.check_predictions(_pred(p2), g)["all_covered"]


def test_mean_over_seeds():
    g = gold_frame()
    rows = [("P1", 0, "xlmr", s, r.channel_id, r.message_id, v) for s, v in ((0, 0.2), (1, 0.6)) for r in g.itertuples()]
    pm = te.mean_over_seeds(_pred(rows))
    assert len(pm) == len(g) and np.allclose(pm["p"], 0.4) and (pm["n_seeds"] == 2).all()


def test_select_best_uses_validation_only():
    units = [{"protocol": "P1", "model": "tfidf", "seed": 0, "valid_macro_f1": 0.70, "splits": [{}]},
             {"protocol": "P1", "model": "xlmr", "seed": 0, "valid_macro_f1": 0.80, "splits": [{"best_epoch": 2}]},
             {"protocol": "P1", "model": "xlmr", "seed": 1, "valid_macro_f1": 0.82, "splits": [{"best_epoch": 3}]},
             {"protocol": "P1", "model": "bge_lr_shuf", "seed": 0, "valid_macro_f1": 0.99, "splits": [{}]}]
    vt = te.valid_table(units)
    sel = te.select_best(vt, "P1")
    assert sel.loc[sel["selected"], "model"].tolist() == ["xlmr"]
    assert "bge_lr_shuf" not in sel["model"].tolist()
    assert sel.loc[sel["model"] == "xlmr", "valid_macro_f1"].iloc[0] == pytest.approx(0.81)


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    ch = rng.integers(0, 20, n)
    return y, ch, rng


def test_cluster_bootstrap_wider_than_message_bootstrap_with_clustered_signal():
    from narr.evaluate import binary_metrics, bootstrap_ci
    rng = np.random.default_rng(1)
    ch = np.repeat(np.arange(25), 16)
    y = (rng.random(25)[ch] < 0.5).astype(int)                     # мітка визначається каналом
    p = np.where(rng.random(25)[ch] < 0.5, y, 1 - y).astype(float) # модель вгадує мітку за каналом цілими каналами
    c = te.cluster_bootstrap(y, p, ch, n_boot=300, seed=0).set_index("metric")
    m = bootstrap_ci(y, p[None], binary_metrics, n_boot=300, seed=0).set_index("metric")
    wc, wm = c.loc["macro_f1", "ci_hi"] - c.loc["macro_f1", "ci_lo"], m.loc["macro_f1", "ci_hi"] - m.loc["macro_f1", "ci_lo"]
    assert wc > wm
    assert c.loc["macro_f1", "point"] == pytest.approx(m.loc["macro_f1", "point"])


def test_recall_by_explicitness_perfect_and_blind_model():
    g = gold_frame()
    pm = pd.concat([pd.DataFrame({"protocol": "P1", "model": "ora", "key": g["key"], "p": g["y"].astype(float), "n_seeds": 1}),
                    pd.DataFrame({"protocol": "P1", "model": "zero", "key": g["key"], "p": 0.0, "n_seeds": 1})])
    r = te.recall_by_explicitness(pm, g, "P1", ["ora", "zero"], ["all"], n_boot=50)
    assert (r.loc[r["model"] == "ora", ["recall_implicit", "recall_explicit"]] == 1).all().all()
    assert (r.loc[r["model"] == "zero", ["recall_implicit", "recall_explicit"]] == 0).all().all()
    assert set(r["definition"]) == {"a", "narrow", "wide", "b"}


def test_paired_table_sign():
    rng = np.random.default_rng(0)
    n = 60
    g = pd.DataFrame({"key": np.arange(n), "channel_id": rng.integers(0, 6, n), "y": rng.integers(0, 2, n), "lang": "ru"})
    pm = pd.concat([pd.DataFrame({"protocol": "P1", "model": "good", "key": g["key"], "p": g["y"].astype(float), "n_seeds": 1}),
                    pd.DataFrame({"protocol": "P1", "model": "bad", "key": g["key"], "p": 1.0 - g["y"], "n_seeds": 1})])
    t = te.paired_table(pm, g, "P1", [("good", "bad")], ["all"], n_boot=50)
    d = t.set_index("metric").loc["macro_f1"]
    assert d["diff"] == pytest.approx(1.0) and d["ci_lo"] > 0
    # замало повідомлень одного класу: різниця не рахується
    tiny = te.paired_table(pm[pm["key"] < 6], g[g["key"] < 6], "P1", [("good", "bad")], ["all"], n_boot=20)
    assert tiny.empty


def test_run_checks_flags_failures():
    # модель з перемішаними мітками, що «знає» відповіді, має провалити жорстку перевірку
    rows = []
    for m, mcc_lo, mcc_hi, f1 in (("majority", 0, 0, 0.4), ("tfidf", 0.4, 0.6, 0.8), ("tfidf_shuf", 0.3, 0.5, 0.7)):
        rows.append({"protocol": "P1", "model": m, "slice": "all", "subset": "all", "metric": "macro_f1", "point": f1, "ci_lo": f1 - .05, "ci_hi": f1 + .05})
        rows.append({"protocol": "P1", "model": m, "slice": "all", "subset": "all", "metric": "mcc", "point": (mcc_lo + mcc_hi) / 2, "ci_lo": mcc_lo, "ci_hi": mcc_hi})
    main = pd.DataFrame(rows)
    pcheck = {"n_duplicates": 0, "n_unknown_keys": 0, "p_in_range": True, "p_finite": True, "all_covered": True, "not_covered": []}
    vt = pd.DataFrame({"protocol": ["P1"], "model": ["tfidf"], "seed": [0], "valid_macro_f1": [0.8], "C": [8.0], "best_epoch": [np.nan], "seconds": [1.0]})
    checks = te.run_checks(main, pcheck, vt, pd.DataFrame(columns=["model", "protocol", "slice", "macro_f1"]), {"ok": True, "n_keys": 5, "n_mismatch": 0},
                           {"majority", "tfidf", "tfidf_shuf"})
    failed = {c["check"] for c in checks if not c["passed"]}
    assert any("tfidf_shuf" in f and "MCC" in f for f in failed)
    assert not any("мажоритарний" in f for f in failed)
