import numpy as np
import pandas as pd
import pytest

from narr import splits as sp


def _make(seed=0, n_ch=12, per=40, n_gold_ch=8):
    """Синтетичний labeled_index: канали 1..n_ch, у кожному per розмічених; золоті беруться з перших n_gold_ch каналів."""
    rng = np.random.default_rng(seed)
    rows = []
    for ch in range(1, n_ch + 1):
        for m in range(1, per + 1):
            d = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=int(rng.integers(0, 330)))
            rows.append({"channel_id": ch, "message_id": m, "date_utc": d, "text_hash": f"{ch:04x}{m:04x}", "in_labeled": True,
                         "weak_narrative": bool(rng.random() < 0.4)})
    idx = pd.DataFrame(rows)
    idx["key"] = sp.make_key(idx["channel_id"], idx["message_id"])
    idx["in_gold"] = False
    gold_rows = []
    for ch in range(1, n_gold_ch + 1):
        for m in range(1, 8):
            d = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=int(rng.integers(0, 340)))
            gold_rows.append({"channel_id": ch, "message_id": 1000 + m, "date_utc": d, "Binary label": bool(rng.random() < 0.35),
                              "text_hash": f"g{ch:04x}{m:04x}"})
    gold = pd.DataFrame(gold_rows)
    # один золотий лежить і в розміченій партії; один розмічений має текст золотого; один семантична копія
    gk = sp.make_key(gold["channel_id"], gold["message_id"])
    idx.loc[0, ["channel_id", "message_id"]] = [int(gold.loc[0, "channel_id"]), int(gold.loc[0, "message_id"])]
    idx["key"] = sp.make_key(idx["channel_id"], idx["message_id"])
    idx.loc[0, "in_gold"] = True
    idx.loc[1, "text_hash"] = gold.loc[1, "text_hash"]
    exclude = {int(idx.loc[2, "key"])}
    gold_h = pd.Series(gold["text_hash"].to_numpy(), index=gk)
    return idx, gold, gold_h, exclude


def test_pool_excludes_three_reasons_and_counts():
    idx, gold, gh, ex = _make()
    gf = sp.gold_frame(gold, gh)
    pool, excluded = sp.build_pool(idx, set(gf["key"]), set(gf["text_hash"]), ex)
    assert len(pool) == len(idx) - 3 and len(excluded) == 3
    s = sp.exclusion_summary(len(idx), excluded)
    assert s["reason_is_gold"] == 1 and s["reason_same_text_as_gold"] == 1 and s["reason_semantic_copy_of_gold"] == 1
    assert s["n_pool"] == len(pool)
    assert not pool["key"].isin(set(gf["key"])).any() and not pool["text_hash"].isin(set(gf["text_hash"])).any()


def test_p1_p2_sizes_and_cutoff():
    idx, gold, gh, ex = _make()
    gf = sp.gold_frame(gold, gh)
    pool, _ = sp.build_pool(idx, set(gf["key"]), set(gf["text_hash"]), ex)
    p1 = sp.split_p1(pool, gf)
    assert len(p1["test"]) == len(gold)
    p2 = sp.split_p2(pool, gf, "2025-09-01")
    t = pd.Timestamp("2025-09-01", tz="UTC")
    assert (p2["train"]["date_utc"] < t).all() and (p2["test"]["date_utc"] >= t).all()
    assert len(p2["test"]) == int((gf["date_utc"] >= t).sum())


def test_channel_folds_partition_and_p3_has_no_channel_overlap():
    idx, gold, gh, ex = _make()
    gf = sp.gold_frame(gold, gh)
    pool, _ = sp.build_pool(idx, set(gf["key"]), set(gf["text_hash"]), ex)
    folds = sp.channel_folds(gf, k=4, seed=1)
    assert sorted(folds["channel_id"]) == sorted(gf["channel_id"].unique()) and set(folds["fold"]) == {1, 2, 3, 4}
    assert folds["n"].sum() == len(gf)
    p3 = sp.split_p3(pool, gf, folds)
    assert sum(len(v["test"]) for v in p3.values()) == len(gf)  # кожен золотий тестується рівно один раз
    for f, d in p3.items():
        assert not (set(d["train"]["channel_id"]) & set(d["test"]["channel_id"]))
        assert all(c["passed"] for c in sp.leakage_checks(d["train"], d["test"], by_channel=True))


def test_channel_folds_deterministic_and_balanced():
    idx, gold, gh, ex = _make(per=5, n_gold_ch=8)
    gf = sp.gold_frame(gold, gh)
    a, b = sp.channel_folds(gf, k=3, seed=7), sp.channel_folds(gf, k=3, seed=7)
    pd.testing.assert_frame_equal(a, b)
    sizes = a.groupby("fold")["n"].sum()
    assert sizes.max() - sizes.min() <= 7  # жадібне розбиття не гірше за один канал


def test_group_valid_split_keeps_duplicates_together():
    tr = pd.DataFrame({"text_hash": ["a", "a", "b", "c", "c", "c", "d", "e", "f", "g"] * 10})
    mtr, mva = sp.group_valid_split(tr, frac=0.3, seed=3)
    assert not (set(tr.loc[mtr, "text_hash"]) & set(tr.loc[mva, "text_hash"])) and mva.sum() > 0 and (mtr | mva).all()


def test_channel_prior_uses_only_given_train():
    tr = pd.DataFrame({"channel_id": [1, 1, 1, 2, 2], "weak_narrative": [True, True, False, False, False]})
    pr = sp.channel_prior(tr, alpha=1.0)
    p = 2 / 5
    assert pr[1] == pytest.approx((2 + p) / 4) and pr[2] == pytest.approx((0 + p) / 3)
    assert 3 not in pr.index  # канал, якого немає в train, значення не отримує
