"""Преєстрація v1.1: підготовка ваг пулу. Рахує (1) кількість повідомлень очищеного корпусу за каналами, (2) семантичні пари cos >= 0,985
з будь-яких каналів на всьому наборі ембедингів S4 (точний пошук блоками, CPU або GPU), (3) зведення ваг і валідації для кожного спліту.

Запуск:  python -I scripts/v11_prepare.py
Виходи: data_small/corpus_channel_counts.csv (малий, комітиться), data/interim/features/sem_pairs_any_0985.parquet (не комітиться),
results/v11/weights_summary.csv, results/v11/run.json.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from narr import textmodels as tm, weights as wt  # noqa: E402
from narr.io import ROOT, write_run_json  # noqa: E402


def main() -> int:
    t0 = time.time()
    cc, pk, th = wt.load_v11_inputs(ROOT)
    splits = tm.load_splits(ROOT / "splits", ("P1", "P2", "P3"))
    summ = pd.DataFrame(wt.attach_weights(splits, pk, cc, th))
    out = ROOT / "results" / "v11"
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / "weights_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ[["protocol", "fold", "train_n_rows", "train_n_components", "train_kish_ess", "train_top_channel_share_before", "train_top_channel_share_after",
                "train_n_channels_clipped_low", "train_n_channels_clipped_high", "valid_n_rows", "valid_kish_ess"]].round(3).to_string(index=False))
    write_run_json(out, "scripts/v11_prepare.py", {"n_corpus": int(cc["n_corpus"].sum()), "n_corpus_channels": int(len(cc)), "n_pairs_any_0985": int(len(pk)),
                                                   "clip": list(wt.CLIP), "seconds": round(time.time() - t0)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
