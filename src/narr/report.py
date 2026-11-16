"""Допоміжні функції для markdown-звітів (українські числа: десяткова кома)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fmt(x, nd: int = 3) -> str:
    if x is None:
        return "н/д"
    if isinstance(x, (bool, np.bool_)):
        return "так" if x else "ні"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}".replace(",", " ")
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "н/д"
        return f"{x:.{nd}f}".replace(".", ",")
    return str(x)


def md_table(df: pd.DataFrame, nd: int = 3, index: bool = False) -> str:
    d = df.reset_index() if index else df
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(fmt(v, nd) for v in r) + " |" for r in d.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


def ci_str(point: float, lo: float, hi: float, nd: int = 3) -> str:
    return f"{fmt(point, nd)} [{fmt(lo, nd)}; {fmt(hi, nd)}]"
