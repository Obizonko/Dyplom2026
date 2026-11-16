"""Наскрізний тест S2: синтетичні parquet-файли з тією самою структурою, що й у наборі OSF, запуск скрипта."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("langid")

from narr import eda  # noqa: E402
from test_eda import inp, lang_stub  # noqa: E402,F401  (фікстура inp)

ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(inp, raw: Path):
    gd = raw / "extracted" / "parquet_set" / "sub"
    gd.mkdir(parents=True)
    m = inp["msgs"].copy()
    m["binary_label"] = m["binary_label"].astype(object)
    m["is_labeled_batch"] = m["is_labeled_batch"].astype(object)
    m.to_parquet(gd / "messages.parquet")
    inp["channels"].to_parquet(gd / "channels.parquet")
    inp["fwd"].to_parquet(gd / "forward_edges.parquet")
    inp["sem"].to_parquet(gd / "semantic_edges.parquet")
    inp["coding"].to_csv(gd / "narrative_subnarrative_coding_en.csv", index=False)
    tg = inp["tg"].copy()
    tg["text"] = "x"
    tg.loc[tg.index[:len(inp["tg_gold_text"])], "text"] = "x"
    gt = inp["tg_gold_text"].set_index(["channel_id", "message_id"])["text"]
    tg["text"] = [gt.get((c, i), "x") for c, i in zip(tg["channel_id"], tg["message_id"])]
    tg.to_parquet(raw / "telegram_messages.parquet")
    g = inp["gold"]
    g.to_csv(raw / "gold.csv", index=False)
    return raw / "gold.csv"


def test_load_inputs_reads_expected_columns(inp, tmp_path):
    gold_csv = _write_dataset(inp, tmp_path)
    got = eda.load_inputs(tmp_path, gold_csv)
    assert len(got["msgs"]) == 60 and len(got["tg"]) == 61 and len(got["fwd"]) == 2 and len(got["sem"]) == 2
    assert set(got["tg_gold_text"]["message_id"]) == {2, 3, 4, 5, 6, 7}
    assert len(got["gold"]) == 6 and len(got["channels"]) == 4 and len(got["coding"]) == 6


def test_script_end_to_end(inp, tmp_path):
    gold_csv = _write_dataset(inp, tmp_path)
    spec = importlib.util.spec_from_file_location("s02_eda", ROOT / "scripts" / "s02_eda.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s02_eda"] = mod
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    code = mod.main(["--raw-dir", str(tmp_path), "--golden", str(gold_csv), "--out", str(out),
                     "--lang-sample", "10", "--jobs", "1", "--no-validate-gold"])
    assert code == 0
    for name in ("summary.json", "checks.json", "run.json", "monthly.csv", "per_channel.csv", "language_x_label.csv"):
        assert (out / name).exists(), name
    assert len(list((out / "figures").glob("*.png"))) == 6
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert s["labels"]["n_labeled"] == 30 and s["gold"]["n_gold_in_graph_messages"] == 6
    pc = pd.read_csv(out / "per_channel.csv")
    assert len(pc) == 4
