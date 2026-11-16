"""Тести S3 (мережа каналів): ручні малі графи з відомою відповіддю, детермінізм, наскрізний запуск скрипта."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("igraph")
pytest.importorskip("leidenalg")

from narr import network as nw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _edges(rows):
    return pd.DataFrame(rows, columns=["src_channel_id", "dst_channel_id", "weight"])


def _toy():
    """Дві трійки каналів (1,2,3 наративні; 4,5,6 ні), всередині трійок щільні ребра, між трійками одне слабке ребро 3 -> 4."""
    rows = [(1, 2, 10), (2, 3, 10), (3, 1, 10), (1, 3, 4), (4, 5, 10), (5, 6, 10), (6, 4, 10), (4, 6, 4), (3, 4, 1)]
    ch = pd.DataFrame({"channel_id": [1, 2, 3, 4, 5, 6], "username": list("abcdef"),
                       "table_narrative_presence": [True] * 3 + [False] * 3,
                       "table_label": ["n"] * 3 + ["nn"] * 3, "table_status": ["labeled"] * 6})
    return _edges(rows), ch


def test_clean_edges_drops_loops_and_merges_duplicates():
    e = _edges([(1, 1, 5), (1, 2, 3), (1, 2, 4), (2, 1, 1)])
    ce, info = nw.clean_edges(e)
    assert info == {"n_input": 4, "n_loops_removed": 1, "n_duplicate_pairs_merged": 1, "n_edges": 2}
    assert ce.set_index(["src_channel_id", "dst_channel_id"])["weight"].to_dict() == {(1, 2): 7, (2, 1): 1}


def test_centrality_strengths_and_pagerank_sum():
    e, _ = _toy()
    g = nw.build_graph(nw.clean_edges(e)[0])
    c = nw.centrality(g).set_index("channel_id")
    assert c["in_strength"].sum() == c["out_strength"].sum() == e["weight"].sum()
    assert c.loc[1, "out_strength"] == 14 and c.loc[1, "in_strength"] == 10
    assert c.loc[3, "out_degree"] == 2 and c.loc[3, "in_degree"] == 2
    assert np.isclose(c["pagerank_forwarder"].sum(), 1) and np.isclose(c["pagerank_source"].sum(), 1)
    assert c["component"].nunique() == 1


def test_direction_source_to_forwarder():
    """Ребро src -> dst: dst переслав повідомлення з src. Сток отримує in-силу, джерело out-силу."""
    g = nw.build_graph(_edges([(1, 2, 7)]))
    c = nw.centrality(g).set_index("channel_id")
    assert c.loc[1, "out_strength"] == 7 and c.loc[2, "in_strength"] == 7
    assert c.loc[2, "pagerank_forwarder"] > c.loc[1, "pagerank_forwarder"]
    assert c.loc[1, "pagerank_source"] > c.loc[2, "pagerank_source"]


def test_cross_group_exact():
    e, ch = _toy()
    grp = dict(zip(ch["channel_id"], ch["table_narrative_presence"]))
    r = nw.cross_group(e, grp)
    assert r["total"] == 69 and r["cross"] == 1 and r["n_to_nn"] == 1 and r["nn_to_n"] == 0
    assert np.isclose(r["cross_share"], 1 / 69)
    rc = nw.cross_group(e, grp, count=True)
    assert rc["total"] == 9 and rc["cross"] == 1


def test_cross_group_unknown_channel_raises():
    e, ch = _toy()
    with pytest.raises(ValueError):
        nw.cross_group(e, {1: True})


def test_assortativity_perfect_and_neutral():
    ce = _edges([(1, 2, 5), (3, 4, 5)])
    assert np.isclose(nw.assortativity(ce, {1: True, 2: True, 3: False, 4: False}), 1.0)
    # усі чотири клітинки рівні: 0
    ce = _edges([(1, 2, 1), (1, 3, 1), (3, 2, 1), (3, 4, 1)])
    assert np.isclose(nw.assortativity(ce, {1: True, 2: True, 3: False, 4: False}), 0.0)


def test_permutation_is_deterministic_and_bounded():
    e, ch = _toy()
    grp = dict(zip(ch["channel_id"], ch["table_narrative_presence"]))
    a = nw.permutation_cross(e, grp, 500, 7)
    b = nw.permutation_cross(e, grp, 500, 7)
    assert a == b
    assert 1 / 501 <= a["p_le_observed"] <= 1.0
    assert a["observed"] < a["null_mean"]


def test_leiden_finds_two_triples_and_is_stable():
    e, _ = _toy()
    g = nw.build_graph(nw.clean_edges(e)[0])
    ug = nw.undirected_weights(g, "raw")
    runs = nw.leiden_runs(ug, list(range(5)))
    best = nw.best_run(runs)
    ids = g.vs["channel_id"]
    mem = dict(zip(ids, best["membership"]))
    assert len({mem[1], mem[2], mem[3]}) == 1 and len({mem[4], mem[5], mem[6]}) == 1 and mem[1] != mem[4]
    st = nw.stability(runs)
    assert st["nmi_min"] > 0.99 and st["n_pairs"] == 10


def test_undirected_weights_sum_both_directions():
    g = nw.build_graph(_edges([(1, 2, 3), (2, 1, 4)]))
    ug = nw.undirected_weights(g, "raw")
    assert ug.ecount() == 1 and ug.es["weight"] == [7.0]
    assert nw.undirected_weights(g, "none").es["weight"] == [2.0]
    with pytest.raises(ValueError):
        nw.undirected_weights(g, "bad")


def test_run_all_toy_is_deterministic_and_checks_pass():
    e, ch = _toy()
    t1, s1, c1, _ = nw.run_all(e, ch, n_perm=200, seed=1, strict=False)
    t2, s2, c2, _ = nw.run_all(e, ch, n_perm=200, seed=1, strict=False)
    pd.testing.assert_frame_equal(t1["centrality"], t2["centrality"])
    pd.testing.assert_frame_equal(t1["communities"], t2["communities"])
    assert s1 == s2
    assert all(c["passed"] for c in c1 if c["hard"]), [c for c in c1 if not c["passed"]]
    assert s1["n_nodes"] == 6 and s1["cross_group"]["main_weight"]["cross"] == 1
    assert t1["community_summary"]["n_channels"].sum() == 6


def test_run_all_reports_channel_missing_from_table():
    e, ch = _toy()
    with pytest.raises(ValueError, match="відсутні"):
        nw.run_all(e, ch[ch["channel_id"] != 6], n_perm=10, strict=False)


def test_gold_coverage():
    gold = pd.DataFrame({"channel_id": [1, 1, 2, 9], "Binary label": [True, False, True, True]})
    r = nw.gold_coverage(gold, {1, 2})
    assert r == {"n_messages": 4, "n_channels": 3, "messages_in_graph_channels": 3, "channels_in_graph": 2,
                 "narrative_total": 3, "narrative_in_graph_channels": 2, "non_narrative_in_graph_channels": 1}


def test_script_end_to_end(tmp_path):
    e, ch = _toy()
    e.rename(columns={}).assign(src_channel=lambda d: d["src_channel_id"], dst_channel=lambda d: d["dst_channel_id"]).to_csv(tmp_path / "edges.csv", index=False)
    ch.to_csv(tmp_path / "ch.csv", index=False)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "s03_network.py"), "--edges", str(tmp_path / "edges.csv"),
                        "--channels", str(tmp_path / "ch.csv"), "--out", str(out), "--n-perm", "100", "--lenient",
                        "--golden", str(tmp_path / "no_gold.csv")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    for f in ("centrality.csv", "communities.csv", "community_summary.csv", "cross_group.json", "summary.json", "checks.json",
              "run.json", "channel_network.gexf", "figures/01_network_map.png"):
        assert (out / f).exists(), f
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["n_nodes"] == 6
