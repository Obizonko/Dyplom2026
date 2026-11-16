"""Перевірки витоку з плану S5: перетин ключів, семантичні копії, ознаки каналів лише з train."""
import numpy as np
import pandas as pd

from narr import splits as sp
from test_splits import _make


def _setup():
    idx, gold, gh, ex = _make()
    gf = sp.gold_frame(gold, gh)
    pool, _ = sp.build_pool(idx, set(gf["key"]), set(gf["text_hash"]), ex)
    return idx, gf, pool, ex


def test_key_intersection_empty_for_all_protocols():
    idx, gf, pool, ex = _setup()
    sets = [sp.split_p1(pool, gf), sp.split_p2(pool, gf)] + list(sp.split_p3(pool, gf, sp.channel_folds(gf, 4)).values())
    for d in sets:
        res = sp.leakage_checks(d["train"], d["test"])
        assert all(c["passed"] for c in res), res


def test_semantic_copy_check_detects_leak_and_passes_after_exclusion():
    idx, gf, pool, ex = _setup()
    leaky_key = int(pool["key"].iloc[5])
    pairs = pd.DataFrame({"gold_key": [int(gf["key"].iloc[0])], "neighbor_key": [leaky_key], "cos": [0.99]})
    res = sp.leakage_checks(pool, gf, pairs)
    assert not all(c["passed"] for c in res)  # копія золотого лишилась у train: перевірка має впасти
    clean = pool[pool["key"] != leaky_key]
    assert all(c["passed"] for c in sp.leakage_checks(clean, gf, pairs))
    below = pairs.assign(cos=0.97)  # нижче порога: не вважається копією
    assert all(c["passed"] for c in sp.leakage_checks(pool, gf, below))


def test_same_text_check_detects_leak():
    idx, gf, pool, ex = _setup()
    leaky = pool.copy()
    leaky.loc[leaky.index[0], "text_hash"] = gf["text_hash"].iloc[3]
    assert not all(c["passed"] for c in sp.leakage_checks(leaky, gf))


def test_channel_features_do_not_change_when_test_rows_change():
    idx, gf, pool, ex = _setup()
    p3 = sp.split_p3(pool, gf, sp.channel_folds(gf, 4))
    d = p3[1]
    a = sp.channel_prior(d["train"])
    flipped = d["test"].assign(y=1 - d["test"]["y"])  # зміна тестових міток не може впливати на ознаки
    b = sp.channel_prior(d["train"])
    pd.testing.assert_series_equal(a, b)
    assert not (set(a.index) & set(flipped["channel_id"]))  # тестові канали взагалі не мають пріора з train (P3)


def test_shuffled_train_labels_break_channel_prior_signal():
    """Негативний контроль: пріор каналу на справжніх мітках добре розділяє класи, на перемішаних дає рівень випадку."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    n = 4000
    ch = rng.integers(0, 20, n)
    y = rng.random(n) < np.where(ch < 10, 0.9, 0.1)  # мітка майже повністю визначається каналом
    tr = pd.DataFrame({"channel_id": ch, "weak_narrative": y})
    auc_real = roc_auc_score(y, tr["channel_id"].map(sp.channel_prior(tr)))
    sh = tr.assign(weak_narrative=rng.permutation(y))
    auc_shuf = roc_auc_score(sh["weak_narrative"], sh["channel_id"].map(sp.channel_prior(sh)))
    assert auc_real > 0.85 and abs(auc_shuf - 0.5) < 0.05
