import numpy as np
import pandas as pd

from narr import features as ft


def _edges():
    return pd.DataFrame({"src_channel_id": [1, 2, 3, 4, 1, 5], "dst_channel_id": [2, 3, 4, 5, 3, 1], "weight": [5, 3, 8, 1, 2, 4]})


def test_rewire_preserves_degrees_and_weights():
    e = _edges()
    r = ft.rewire_edges(e, seed=1)
    assert len(r) == len(e)
    assert sorted(r["weight"]) == sorted(e["weight"].astype(float))
    for col in ("src_channel_id", "dst_channel_id"):
        assert sorted(r[col]) == sorted(e[col])  # вхідні і вихідні ступені збережено


def test_channel_features_columns():
    comm = pd.DataFrame({"channel_id": [1, 2, 3, 4, 5], "community_log": [0, 0, 1, 1, 1]})
    f = ft.channel_network_features(_edges(), comm)
    assert set(ft.NET_COLS) <= set(f.columns) and len(f) == 5


def test_semantic_copy_features_cross_channel_only():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(4, 16))
    emb = np.vstack([base[0], base[0], base[1], base[2]]).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ch = np.array([10, 20, 10, 10])  # 0 і 1 копії з різних каналів
    f = ft.semantic_copy_features(emb, ch, pd.Series(pd.to_datetime(["2025-01-01 00:00", "2025-01-01 02:00", "2025-01-02 00:00", "2025-01-03 00:00"], utc=True)), device="cpu")
    assert f["sem_n_copies_xch"].tolist() == [1, 1, 0, 0]
    assert abs(f["sem_first_copy_delta_h"].iloc[0] - 2.0) < 1e-6 and abs(f["sem_first_copy_delta_h"].iloc[1] + 2.0) < 1e-6


def test_oof_prior_does_not_see_own_label():
    tr = pd.DataFrame({"channel_id": [1] * 40, "weak_narrative": [True] * 20 + [False] * 20, "text_hash": [f"h{i}" for i in range(40)]})
    p = ft.oof_channel_prior(tr, n_folds=5)
    assert np.isfinite(p).all() and p.min() > 0 and p.max() < 1
    # у фолді з однаковою часткою пріор не збігається з пріором усього train так, як якби мітки були в ознаці
    full = ft.prior_for(tr, tr)
    assert not np.allclose(p, full)
