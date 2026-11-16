"""Генерує colab/02_eda.ipynb (самодостатній: код репозиторію вбудований у ноутбук).
Запуск: python3 colab/build_02_eda.py"""
import hashlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
EMBED = ["src/narr/__init__.py", "src/narr/eda.py", "src/narr/language.py", "src/narr/io.py",
         "configs/base.yaml", "scripts/s02_eda.py"]
VERSION = "v1"

MD_INTRO = """# S2. Описова аналітика повних даних TeleNarratives

**Що робить ноутбук:** читає з вашого Google Drive файли, які завантажив S1 (`MyDrive/narratives-thesis/data/raw/`), рахує таблиці
і графіки EDA (цілісність, канали, місяці, мова, слабкі мітки, перетин із золотим набором, пересилання, дублікати) і зберігає
їх у `MyDrive/narratives-thesis/results/02_eda/`. Код репозиторію вбудований у ноутбук, окремо нічого ставити не потрібно.
Золотий набір (412 повідомлень) завантажується з GitHub-репозиторію TeleNarratives.

**Що зробити вам:** Runtime → Run all (GPU не потрібен, достатньо звичайного середовища). Дозвольте доступ до Drive.
Орієнтовний час 10-20 хвилин (найдовше визначення мови). Потрібно приблизно 6-8 ГБ оперативної пам'яті.
Результати я прочитаю з Drive сам. Архів `02_eda.zip` у тій самій папці можна додати до репозиторію.
"""

CELL_SETUP = '''# 1. Drive, залежності, шляхи
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

try:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive/narratives-thesis")
    REPO = Path("/content/repo")
except ImportError:  # запуск поза Colab (тестовий)
    BASE = Path("./narratives-thesis").resolve()
    REPO = Path("./repo").resolve()

RAW = BASE / "data" / "raw"
OUT = BASE / "results" / "02_eda"
assert RAW.exists(), f"Немає {RAW}. Спершу запустіть ноутбук S1."
OUT.mkdir(parents=True, exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "langid"], check=True)
print("Дані:", RAW)
print("Результати:", OUT)
'''

CELL_WRITE = '''# 2. Код репозиторію (вбудований у ноутбук)
FILES = __FILES__

for rel, text in FILES.items():
    p = REPO / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print(f"{hashlib.sha256(text.encode()).hexdigest()[:12]}  {rel}")
'''

CELL_GOLD = '''# 3. Золотий набір: клон TeleNarratives (потрібен лише golden set CSV)
EXT = REPO / "external" / "TeleNarratives"
if not EXT.exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/yuliavistak/TeleNarratives", str(EXT)], check=True)
head = subprocess.run(["git", "-C", str(EXT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("TeleNarratives commit:", head)
gold_csv = next(EXT.glob("labeling results/strong labeling/golden set (labeled).csv"))
print("Золотий набір:", gold_csv)
'''

CELL_CHECK = '''# 4. Перевірка, що потрібні файли на місці
need = ["telegram_messages.parquet", "narratives_hierarchy.csv"]
for n in need:
    assert (RAW / n).exists(), f"немає {n}"
msgs_p = sorted(RAW.rglob("messages.parquet"))
assert msgs_p, "немає messages.parquet у розпакованому parquet-наборі"
print("messages.parquet:", msgs_p[0])
try:
    import psutil
    print(f"Пам'ять: {psutil.virtual_memory().total / 1e9:.1f} ГБ усього, {psutil.virtual_memory().available / 1e9:.1f} ГБ вільно")
except ImportError:
    pass
'''

CELL_RUN = '''# 5. Запуск S2 (скрипт scripts/s02_eda.py з репозиторію)
cmd = [sys.executable, "-I", str(REPO / "scripts" / "s02_eda.py"), "--raw-dir", str(RAW), "--out", str(OUT),
       "--lang-sample", "150000", "--jobs", "2"]
t0 = time.time()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(REPO))
for line in proc.stdout:
    print(line, end="")
code = proc.wait()
print(f"\\nКод завершення {code}, {time.time() - t0:.0f} с")
'''

CELL_PACK = '''# 6. Підсумок, один JSON з усіма таблицями і архів
import shutil
import pandas as pd

summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
checks = json.loads((OUT / "checks.json").read_text(encoding="utf-8"))

tables = {}
for p in sorted(OUT.glob("*.csv")):
    df = pd.read_csv(p)
    tables[p.stem] = df.to_dict(orient="records")
(OUT / "tables_all.json").write_text(json.dumps(tables, ensure_ascii=False, default=str), encoding="utf-8")

shutil.make_archive(str(OUT.parent / "02_eda"), "zip", OUT)
print("Архів:", OUT.parent / "02_eda.zip")

print("\\nПеревірки:")
for c in checks:
    mark = "ok" if c["passed"] else ("ПОМИЛКА" if c["hard"] else "розбіжність")
    print(f"  [{mark}] {c['check']}: {c['detail']}")
print("\\nФайли в", OUT)
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUT)}  {p.stat().st_size / 1e3:.1f} КБ")
assert code == 0, "S2: жорсткі перевірки не пройдено, див. вивід вище"
'''


def build(path: Path) -> str:
    files = {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in EMBED}
    cell_write = CELL_WRITE.replace("__FILES__", repr(files))
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(MD_INTRO), nbf.v4.new_code_cell(CELL_SETUP), nbf.v4.new_code_cell(cell_write),
                nbf.v4.new_code_cell(CELL_GOLD), nbf.v4.new_code_cell(CELL_CHECK), nbf.v4.new_code_cell(CELL_RUN),
                nbf.v4.new_code_cell(CELL_PACK)]
    nb.metadata = {"colab": {"name": f"02_eda_{VERSION}.ipynb"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
    nbf.validate(nb)
    nbf.write(nb, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "02_eda.ipynb"
    print("written", out, build(out))
