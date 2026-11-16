"""Перевірка золотої вибірки і конфігів TeleNarratives (пропускається, якщо external/ не клоновано)."""
import pandas as pd
import pytest

from narr.io import data_available, load_config, load_golden, load_shared, path_of, validate_golden

CFG = load_config()
pytestmark = pytest.mark.skipif(not data_available(CFG), reason="external/TeleNarratives не клоновано")


def test_golden_schema_and_counts():
    df = load_golden(CFG)
    assert validate_golden(df, CFG) == []


def test_golden_numbers_from_plan():
    df = load_golden(CFG)
    exp = CFG["expected"]
    assert len(df) == exp["n_messages"] == 412
    assert int(df[CFG["columns"]["binary"]].sum()) == exp["n_narrative"] == 148
    assert df[CFG["columns"]["channel"]].nunique() == exp["n_channels_in_gold"]


def test_shared_subset_size():
    assert len(load_shared(CFG)) == CFG["expected"]["n_shared"] == 103


def test_config_tables():
    assert len(pd.read_csv(path_of(CFG, "channels"))) == CFG["expected"]["n_channels_config"]
    assert len(pd.read_csv(path_of(CFG, "narratives"))) == CFG["expected"]["n_sub_narratives"]


def test_validate_golden_detects_problems():
    df = load_golden(CFG)
    broken = df.iloc[:-1].copy()
    assert any("рядків" in p for p in validate_golden(broken, CFG))
    flipped = df.copy()
    flipped.loc[flipped.index[0], CFG["columns"]["binary"]] = not flipped.loc[flipped.index[0], CFG["columns"]["binary"]]
    assert validate_golden(flipped, CFG)
