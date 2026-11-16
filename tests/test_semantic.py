"""Тести S4: ручні малі приклади з відомою відповіддю (без torch і без великих даних)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hnswlib")

from narr import semantic as sm  # noqa: E402
from narr.eda import KEY_MULT  # noqa: E402


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _planted(n=300, d=32, seed=0):
    """n випадкових векторів, а вектори 1 і 2 це майже копії вектора 0, вектор 4 майже копія 3."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    x[1] = x[0] + 0.02 * rng.normal(size=d)
    x[2] = x[0] + 0.05 * rng.normal(size=d)
    x[4] = x[3] + 0.02 * rng.normal(size=d)
    return _unit(x).astype(np.float32)


def test_build_embedding_set_roundtrip():
    k = lambda c, m: c * KEY_MULT + m  # noqa: E731
    msgs = pd.DataFrame({"channel_id": [1, 1, 2, 2], "message_id": [10, 11, 20, 21], "text": ["Raw A!", "Raw B", "Raw C", "Raw D"],
                         "is_labeled_batch": [True, False, True, None], "narrative_id": [3.0, np.nan, np.nan, np.nan]})
    tg = pd.DataFrame({"channel_id": [1, 1, 2], "message_id": [10, 11, 20], "text": ["raw a", "raw b", "   "]})
    gold = pd.DataFrame({"channel_id": [1], "message_id": [10], "text": ["raw a"]})
    sem = pd.DataFrame({"src_channel_id": [1], "src_message_id": [11], "dst_channel_id": [2], "dst_message_id": [21], "similarity_score": [0.99]})
    s, info = sm.build_embedding_set(msgs, tg, gold, sem)
    s = s.set_index("key")
    # (2,20) має порожній текст у tg: відкидається; (2,21) немає в tg: береться нормалізований сирий текст
    assert k(2, 20) not in s.index and info["n_empty_dropped"] == 1
    assert s.loc[k(2, 21), "text"] == "raw d" and info["n_text_from_raw_fallback"] == 1
    assert s.loc[k(1, 10), "in_gold"] and s.loc[k(1, 10), "in_labeled"] and not s.loc[k(1, 11), "in_labeled"]
    assert s.loc[k(1, 11), "in_sem"] and s.loc[k(2, 21), "in_sem"]
    assert info["n_gold"] == 1 and info["n_embedded"] == len(s) == 3


def test_embed_sorted_restores_order_and_normalizes():
    texts = ["ccc", "a", "bb", "dddd"]
    enc = lambda batch: np.array([[len(t), 1.0, 0.0] for t in batch])  # noqa: E731
    e = sm.embed_sorted(texts, enc, batch_size=2)
    assert np.allclose(np.linalg.norm(e, axis=1), 1)
    assert [round(float(v)) for v in (e[:, 0] / e[:, 1]) * 1] == [3, 1, 2, 4]  # порядок початковий


def test_knn_pairs_finds_planted_and_dedups():
    x = _planted()
    pairs, info = sm.knn_pairs(x, k=10, threshold=0.9)
    got = {(int(a), int(b)) for a, b in zip(pairs["i"], pairs["j"])}
    assert {(0, 1), (0, 2), (3, 4)} <= got
    assert all(a < b for a, b in got) and len(got) == len(pairs)
    assert info["n_nodes_saturated"] == 0
    rec = sm.hnsw_recall(x, pairs, 0.9, n_check=300, k=10)
    assert rec["recall"] >= 0.99


def test_exact_neighbors_matches_bruteforce_and_excludes_self():
    x = _planted()
    ex = sm.exact_neighbors(x, np.array([0, 3]), 0.9)
    assert set(zip(ex["q"], ex["i"])) == {(0, 1), (0, 2), (3, 4)}
    assert (ex["cos"] >= 0.9).all()


def test_compare_scores_direct_and_mapped():
    rng = np.random.default_rng(1)
    cos = rng.uniform(0.90, 1.0, 500).astype(np.float32)
    d = sm.compare_scores(cos.copy(), cos)
    assert d["mean_abs_diff_direct"] < 1e-6 and d["spearman"] > 0.999999 and abs(d["fit_slope"] - 1) < 1e-4
    m = sm.compare_scores((cos + 1) / 2, cos)
    assert m["mean_abs_diff_mapped_half"] < 1e-6 and m["spearman"] > 0.999999 and abs(m["fit_slope"] - 0.5) < 1e-4
    assert m["mean_abs_diff_direct"] > 0.01  # пряме порівняння виявляє іншу шкалу


def test_pair_cosines_and_missing_keys():
    x = _planted(n=10)
    key2idx = {100: 0, 101: 1, 102: 2}
    c = sm.pair_cosines(x, key2idx, np.array([100, 100, 999]), np.array([101, 102, 100]))
    assert c[0] > 0.95 and c[1] > 0.9 and np.isnan(c[2])


def test_overlap_and_pair_set():
    a = sm.pair_set(np.array([1, 2, 3]), np.array([2, 1, 4]))  # (1,2) двічі як (2,1)
    assert a == {(1, 2), (3, 4)}
    o = sm.overlap({(1, 2), (5, 6)}, {(1, 2), (3, 4)})
    assert o["n_intersection"] == 1 and o["recall_of_orig"] == 0.5 and o["share_of_own_in_orig"] == 0.5


def test_gold_copies_counts():
    x = _planted(n=20)
    es = pd.DataFrame({"key": np.arange(20) + 1000, "text": [f"t{i}" for i in range(20)],
                       "in_gold": [True, False, False, True] + [False] * 16, "in_labeled": [False, True, False, False, True] + [False] * 15})
    es.loc[2, "text"] = "t0"  # точна копія тексту
    pairs, summ = sm.gold_copies(x, es, thresholds=(0.9, 0.99))
    s = summ.set_index("threshold")
    assert s.loc[0.9, "n_gold"] == 2 and s.loc[0.9, "gold_with_neighbor"] == 2
    assert s.loc[0.9, "gold_with_labeled_neighbor"] == 2  # сусіди 1 і 4 розмічені
    assert s.loc[0.9, "pairs_exact_text"] == 1
    p = pairs[(pairs["gold_key"] == 1000)]
    assert set(p["neighbor_key"]) == {1001, 1002}


def test_reachability_path_graph():
    # шлях 0(n) - 1(n) - 2(nn) - 3(nn) - 4(n)
    a, b = np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4])
    g = np.array([True, True, False, False, True])
    r = sm.reachability(5, a, b, g, kmax=3).set_index(["direction", "k"])
    # з n до nn: вузол 0 за 2 кроки, вузол 1 за 1, вузол 4 за 1: k=1 -> 2/3, k=2 -> 3/3
    assert np.isclose(r.loc[("n_to_nn", 1), "share"], 2 / 3) and np.isclose(r.loc[("n_to_nn", 2), "share"], 1.0)
    # з nn до n: вузол 2 за 1 крок (до 1), вузол 3 за 1 крок (до 4): k=1 -> 1
    assert np.isclose(r.loc[("nn_to_n", 1), "share"], 1.0)
    assert r.loc[("n_to_nn", 1), "n_start"] == 3


def test_reachability_ignores_isolated_nodes():
    r = sm.reachability(4, np.array([0]), np.array([1]), np.array([True, False, True, False]), kmax=1).set_index(["direction", "k"])
    assert r.loc[("n_to_nn", 1), "n_start"] == 1 and r.loc[("n_to_nn", 1), "share"] == 1.0


def test_reachability_null_deterministic():
    a, b = np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4])
    ch = np.array([1, 1, 2, 2, 3])
    cg = {1: True, 2: False, 3: True}
    n1 = sm.reachability_null(5, a, b, ch, cg, kmax=2, n_perm=10, seed=3)
    n2 = sm.reachability_null(5, a, b, ch, cg, kmax=2, n_perm=10, seed=3)
    pd.testing.assert_frame_equal(n1, n2)
    assert len(n1) == 4


def test_forward_vs_channel_flow():
    fwd = pd.DataFrame({"logical_src_channel_id": [1, 1, 2, 3], "logical_src_message_id": [1, 2, 3, 4],
                        "logical_dst_channel_id": [2, 2, 3, 3], "logical_dst_message_id": [5, 6, 7, 8]})
    flow = pd.DataFrame({"src_channel_id": [1, 2], "dst_channel_id": [2, 3], "weight": [2, 1]})
    r = sm.forward_vs_channel_flow(fwd, flow)
    assert r["n_self_forward"] == 1 and r["weight_sum_without_self"] == 3 and r["equal_to_channel_flow"]
    flow.loc[0, "weight"] = 5
    assert not sm.forward_vs_channel_flow(fwd, flow)["equal_to_channel_flow"]


def test_hash_encoder_similar_texts_close():
    enc = sm.hash_encoder()
    e = sm.embed_sorted(["оперативна інформація про обстріл", "оперативна інформація про обстріл!", "зовсім інше речення про погоду"], enc)
    assert e[0] @ e[1] > 0.9 > e[0] @ e[2]


def test_quantize_fp16_unit_and_close():
    x = _planted(n=10)
    q = sm.quantize_fp16(x)
    assert np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-6) and np.abs(q - x).max() < 1e-3


def test_pair_group_stats():
    r = sm.pair_group_stats(np.array([1, 1, 2, 9]), np.array([1, 2, 3, 1]), {1: True, 2: False, 3: False})
    assert r["n_pairs"] == 4 and r["n_unknown_group"] == 1 and r["same_channel_share"] == 0.25
    assert r["n_n"] == 1 and r["n_nn"] == 1 and r["nn_nn"] == 1 and np.isclose(r["cross_share"], 1 / 3)
