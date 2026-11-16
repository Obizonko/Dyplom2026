"""Наскрізний запуск scripts/s04_semantic.py на малому синтетичному наборі (хеш-ембединги, без GPU і без UMAP)."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("hnswlib")

from test_eda import inp, lang_stub  # noqa: E402,F401  (фікстура inp)
from test_eda_pipeline import _write_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("s04_semantic", ROOT / "scripts" / "s04_semantic.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s04_semantic"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_script_end_to_end(inp, tmp_path):
    gold_csv = _write_dataset(inp, tmp_path)
    pd.DataFrame({"channel_id": [1, 2, 3, 4], "table_narrative_presence": [True, False, True, False]}).to_csv(tmp_path / "pc.csv", index=False)
    pd.DataFrame({"src_channel_id": [1, 2], "dst_channel_id": [2, 3], "weight": [1, 1]}).to_csv(tmp_path / "flow.csv", index=False)
    out = tmp_path / "out"
    code = _load_script().main(["--raw-dir", str(tmp_path), "--golden", str(gold_csv), "--out", str(out), "--encoder", "hash", "--skip-umap",
                                "--per-channel", str(tmp_path / "pc.csv"), "--channel-flow", str(tmp_path / "flow.csv"), "--n-perm", "3", "--k", "10"])
    assert code == 0
    for f in ("summary.json", "checks.json", "run.json", "reachability.csv", "gold_copies_summary.csv", "gold_copies_pairs.csv",
              "exclude_labeled_keys_cos985.csv", "threshold_pairs_sample.csv", "repro_sample.csv",
              "figures/01_repro_scores.png", "figures/02_reachability.png"):
        assert (out / f).exists(), f
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["embedding_set"]["n_gold"] == 6 and s["encoder"]["dim"] == 256
    r = pd.read_csv(out / "reachability.csv")
    assert set(r["layer"]) == {"forward", "semantic_orig", "semantic_own_0985", "semantic_own_0975", "forward+semantic_orig"}
    assert set(r["direction"]) == {"n_to_nn", "nn_to_n", "all"} and r["k"].max() == 4
    li = pd.read_csv(out / "labeled_index.csv")
    assert set(li.columns) == {"key", "channel_id", "message_id", "date_utc", "text_hash", "in_gold", "in_labeled", "weak_narrative"}
    assert li["in_gold"].sum() == 6 and li["date_utc"].notna().all() and li["key"].is_unique
    assert not (li["weak_narrative"] & ~li["in_labeled"]).any() and li["text_hash"].str.len().eq(16).all()
    assert s["labeled_index"]["n_rows"] == len(li) and s["labeled_index"]["n_missing_date"] == 0
    ex = pd.read_csv(out / "exclude_labeled_keys_cos985.csv")
    assert set(ex.columns) >= {"key", "channel_id", "message_id", "max_cos", "n_gold_matched"}
    assert not ex["key"].isin(pd.read_csv(gold_csv).eval("channel_id * 10000000 + message_id")).any()  # золоті не потрапляють у список виключення
