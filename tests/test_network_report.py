"""Звіт S3 збирається з results/03_network і не містить довгих тире; ключові числа збігаються з summary.json."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "03_network"
pytestmark = pytest.mark.skipif(not (RES / "summary.json").exists(), reason="немає results/03_network")


def test_render_matches_summary_and_style():
    sys.path.insert(0, str(ROOT / "scripts"))
    import s03_render_report as r

    txt = r.render(RES)
    s = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    assert "—" not in txt and "–" not in txt
    cs = s["cross_group"]["main_weight"]["cross_share"]
    assert f"{cs * 100:.3f}".replace(".", ",") + "%" in txt
    assert f"{int(s['weight_sum']):,}".replace(",", " ") in txt
    assert txt.startswith("# S3.")
