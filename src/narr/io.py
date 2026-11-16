"""Конфігурація, завантаження даних, валідація схем, службова інформація про запуск."""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "base.yaml"


def load_config(path: Path | str | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_of(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"][key]


def data_available(cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    return path_of(cfg, "golden").exists()


def load_golden(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = pd.read_csv(path_of(cfg, "golden"))
    df[cfg["columns"]["date"]] = pd.to_datetime(df[cfg["columns"]["date"]], utc=True)
    return df


def load_shared(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(path_of(cfg, "shared"))


def validate_golden(df: pd.DataFrame, cfg: dict | None = None) -> list[str]:
    """Повертає список проблем. Порожній список означає, що схема і вміст відповідають очікуванням."""
    cfg = cfg or load_config()
    col, exp = cfg["columns"], cfg["expected"]
    no_narr = col["no_narrative_label"]
    problems: list[str] = []

    required = [col["text"], col["channel"], col["message"], col["date"], col["forwarded"],
                col["binary"], col["general"], col["level1"], *col["reviewers"], *col["reviewers_level1"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"відсутні колонки: {missing}"]

    if len(df) != exp["n_messages"]:
        problems.append(f"рядків {len(df)}, очікувалось {exp['n_messages']}")
    n_dup = int(df.duplicated([col["channel"], col["message"]]).sum())
    if n_dup:
        problems.append(f"дублікатів (channel_id, message_id): {n_dup}")
    n_pos = int(df[col["binary"]].sum())
    if n_pos != exp["n_narrative"]:
        problems.append(f"наративних {n_pos}, очікувалось {exp['n_narrative']}")
    if int(df[col["channel"]].nunique()) != exp["n_channels_in_gold"]:
        problems.append(f"каналів у вибірці {df[col['channel']].nunique()}, очікувалось {exp['n_channels_in_gold']}")
    if df[col["text"]].isna().any() or (df[col["text"]].astype(str).str.strip() == "").any():
        problems.append("є порожні тексти")
    if df[col["date"]].isna().any():
        problems.append("є непарсовані дати")
    if ((df[col["general"]] != no_narr) != df[col["binary"]]).any():
        problems.append("Binary label не узгоджена з General label")
    if ((df[col["level1"]] != no_narr) != df[col["binary"]]).any():
        problems.append("Binary label не узгоджена з General label (level-1)")
    return problems


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def write_run_json(out_dir: Path | str, script: str, params: dict) -> Path:
    """Пише run.json: дата, git-хеш, параметри, версії бібліотек."""
    import numpy
    import sklearn

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "script": script,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no", "--", "src", "scripts", "configs")),  # зміни коду відносно коміту
        "params": params,
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    path = out_dir / "run.json"
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
