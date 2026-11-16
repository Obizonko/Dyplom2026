"""Наскрізний запуск scripts/s05_splits.py: справжні золоті з external/, синтетичний вихід S4."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from narr import splits as sp
from narr.io import data_available, load_golden

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not data_available(), reason="немає external/TeleNarratives")


def _load():
    spec = importlib.util.spec_from_file_location("s05_splits", ROOT / "scripts" / "s05_splits.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s05_splits"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_s04(tmp, n_extra=3000, seed=0):
    g = load_golden()
    rng = np.random.default_rng(seed)
    gk = sp.make_key(g["channel_id"], g["message_id"])
    chans = g["channel_id"].unique()
    rows = []
    for i in range(n_extra):
        ch = int(rng.choice(chans))
        rows.append({"channel_id": ch, "message_id": 5_000_000 + i, "date_utc": (pd.Timestamp("2024-12-20", tz="UTC") + pd.Timedelta(days=int(rng.integers(0, 420)))).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "text_hash": f"t{i:08x}", "in_gold": False, "in_labeled": True, "weak_narrative": bool(rng.random() < 0.4)})
    lab = pd.DataFrame(rows)
    gold = pd.DataFrame({"channel_id": g["channel_id"], "message_id": g["message_id"], "date_utc": g["date_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "text_hash": [f"g{k:x}" for k in gk], "in_gold": True, "in_labeled": False, "weak_narrative": False})
    both = gold.iloc[:5].assign(in_labeled=True)            # золоті, що лежать і в розміченій партії
    gold = pd.concat([gold.iloc[5:], both])
    lab.loc[0, "text_hash"] = gold["text_hash"].iloc[10]    # розмічений з текстом золотого
    idx = pd.concat([gold, lab], ignore_index=True)
    idx["key"] = sp.make_key(idx["channel_id"], idx["message_id"])
    d = tmp / "s04"
    d.mkdir()
    idx.to_csv(d / "labeled_index.csv", index=False)
    copy_key = int(lab["channel_id"].iloc[1]) * sp.KEY_MULT + int(lab["message_id"].iloc[1])
    pd.DataFrame({"key": [copy_key], "channel_id": [copy_key // sp.KEY_MULT], "message_id": [copy_key % sp.KEY_MULT], "max_cos": [0.99], "n_gold_matched": [1]}).to_csv(d / "exclude_labeled_keys_cos985.csv", index=False)
    pd.DataFrame({"gold_key": [int(gk[20])], "neighbor_key": [copy_key], "cos": [0.99], "neighbor_in_labeled": [True], "neighbor_in_gold": [False]}).to_csv(d / "gold_copies_pairs.csv", index=False)
    return d, len(idx), copy_key


def test_end_to_end_with_fake_s04(tmp_path):
    s04, _, copy_key = _fake_s04(tmp_path)
    out, spl = tmp_path / "out", tmp_path / "splits"
    code = _load().main(["--s04", str(s04), "--out", str(out), "--splits", str(spl)])
    assert code == 0
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["train_side"] == "done" and s["gold"]["n_p2_test"] == 116
    pool = s["pool"]
    # причини перекриваються: 5 золотих із розміченої партії мають і текст золотого, тому same_text = 5 + 1
    assert pool["reason_is_gold"] == 5 and pool["reason_same_text_as_gold"] == 6 and pool["reason_semantic_copy_of_gold"] == 1
    assert pool["first_reason_is_gold"] == 5 and pool["first_reason_same_text_as_gold"] == 1 and pool["first_reason_semantic_copy_of_gold"] == 1
    assert pool["n_pool"] + pool["n_excluded"] == pool["n_labeled"]
    p1 = pd.read_csv(spl / "P1" / "train.csv.gz")
    assert len(p1) == pool["n_pool"] and len(pd.read_csv(spl / "P1" / "test.csv")) == 412
    assert len(pd.read_csv(spl / "P2" / "test.csv")) == 116
    assert sorted(p.name for p in (spl / "P3").glob("fold*_test.csv")) == [f"fold{i}_test.csv" for i in range(1, 6)]
    assert copy_key not in set((p1["channel_id"].astype("int64") * sp.KEY_MULT + p1["message_id"].astype("int64")))
    ck = json.loads((out / "checks.json").read_text(encoding="utf-8"))
    assert all(c["passed"] for c in ck if c["hard"])
    assert (out / "implicit_labeling.csv").exists() and len(pd.read_csv(out / "implicit_labeling.csv")) == 148


def test_template_not_overwritten_when_answers_exist(tmp_path):
    m = _load()
    s04, _, _ = _fake_s04(tmp_path)
    out = tmp_path / "out"
    args = ["--s04", str(s04), "--out", str(out), "--splits", str(tmp_path / "splits")]
    m.main(args)
    t = pd.read_csv(out / "implicit_labeling.csv", dtype=str)
    t.loc[0, "explicitness"] = "implicit"
    t.to_csv(out / "implicit_labeling.csv", index=False)
    m.main(args)
    assert pd.read_csv(out / "implicit_labeling.csv").loc[0, "explicitness"] == "implicit"
    assert not (out / "implicit_labeling.new.csv").exists()          # ті самі ключі: нічого не пишемо поруч
    t = pd.read_csv(out / "implicit_labeling.csv", dtype=str)
    t.loc[1, "message_id"] = "999999999"                                 # інший набір ключів: новий шаблон лише поруч
    t.to_csv(out / "implicit_labeling.csv", index=False)
    m.main(args)
    assert (out / "implicit_labeling.new.csv").exists()
    assert pd.read_csv(out / "implicit_labeling.csv").loc[0, "explicitness"] == "implicit"


def test_leak_detected_if_copy_not_excluded(tmp_path):
    s04, _, _ = _fake_s04(tmp_path)
    pd.read_csv(s04 / "exclude_labeled_keys_cos985.csv").iloc[0:0].to_csv(s04 / "exclude_labeled_keys_cos985.csv", index=False)  # список виключення порожній
    code = _load().main(["--s04", str(s04), "--out", str(tmp_path / "out"), "--splits", str(tmp_path / "splits")])
    assert code == 1  # перевірка витоку має впасти
