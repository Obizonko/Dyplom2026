import numpy as np
import pandas as pd

from narr import weights as wt
from narr.evaluate import binary_metrics, weighted_macro_f1


def _train():
    return pd.DataFrame({"key": [f"k{i}" for i in range(8)], "channel_id": [1, 1, 1, 1, 1, 2, 2, 3],
                         "text_hash": ["a", "a", "b", "c", "d", "e", "f", "g"], "weak_narrative": [True] * 4 + [False] * 4})


def test_copy_components_join_hash_and_pairs():
    tr = _train()
    pairs = pd.DataFrame({"key_i": ["k2", "k5", "zz"], "key_j": ["k3", "k6", "k0"]})  # zz немає в train: ігнорується
    comp = wt.copy_components(tr, pairs)
    assert comp[0] == comp[1]  # однаковий текст
    assert comp[2] == comp[3] and comp[5] == comp[6]  # пари копій
    assert len(np.unique(comp)) == 5


def test_pool_weights_match_corpus_shares_without_clip():
    tr = _train()
    comp = wt.copy_components(tr, None)
    cc = pd.DataFrame({"channel_id": [1, 2, 3, 99], "n_corpus": [10, 30, 60, 500]})  # канал 99 відсутній у train
    w, info = wt.pool_weights(tr, comp, cc, clip=(1e-9, 1e9))
    assert np.isclose(w.mean(), 1.0)
    share = pd.Series(w).groupby(tr["channel_id"].to_numpy()).sum() / w.sum()
    assert np.allclose(share.loc[[1, 2, 3]].to_numpy(), [0.1, 0.3, 0.6])
    assert np.isclose(w[0], w[1]) and np.isclose(w[0] * 2, w[2])  # компонента розміру 2 ділить вагу
    assert info["kish_ess"] <= len(w)


def test_pool_weights_clip():
    tr = _train()
    cc = pd.DataFrame({"channel_id": [1, 2, 3], "n_corpus": [1, 1, 10_000]})
    _, info = wt.pool_weights(tr, wt.copy_components(tr, None), cc, clip=(0.5, 2.0))
    assert info["n_channels_clipped_low"] >= 1 and info["n_channels_clipped_high"] >= 1


def test_group_split_keeps_groups_together():
    g = np.repeat(np.arange(50), 3)
    v = wt.group_split(g, 0.1, 42)
    assert v.sum() == 15
    assert all(len(set(v[g == k])) == 1 for k in range(50))


def test_weighted_macro_f1_equals_unweighted():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 200), rng.random(200)
    assert np.isclose(weighted_macro_f1(y, p), binary_metrics(y, p)["macro_f1"])
    assert np.isclose(weighted_macro_f1(y, p, np.full(200, 3.0)), binary_metrics(y, p)["macro_f1"])
