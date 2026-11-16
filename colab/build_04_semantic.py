"""Генерує colab/04_semantic.ipynb (самодостатній: код репозиторію і малі вхідні таблиці вбудовано в ноутбук).
Запуск: python3 colab/build_04_semantic.py"""
import hashlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
EMBED = ["src/narr/__init__.py", "src/narr/eda.py", "src/narr/language.py", "src/narr/io.py", "src/narr/viz.py",
         "src/narr/semantic.py", "src/narr/semantic_viz.py", "configs/base.yaml", "scripts/s04_semantic.py",
         "results/02_eda/per_channel.csv", "data_small/channel_flow_edges.csv"]
VERSION = "v1"

MD_INTRO = """# S4. Семантичний шар: ембединги bge-m3, семантичні ребра, копії золотих, досяжність

**Що робить ноутбук.** Бере з вашого Google Drive файли S1 (`MyDrive/narratives-thesis/data/raw/`), кодує моделлю `BAAI/bge-m3`
приблизно 108 тисяч повідомлень (412 золотих, уся розмічена партія 81 584 і кінці семантичних ребер набору), порівнює наш косинус з оцінками
ребер набору, будує власні семантичні ребра (hnswlib), шукає точні копії золотих повідомлень, рахує досяжність між групами каналів
для форвардів і семантики, малює UMAP і звіряє `forward_edges` з таблицею `channel_flow_edges` (залишок S3).
Результати зберігаються в `MyDrive/narratives-thesis/results/04_semantic/`, ембединги (float16, близько 220 МБ) у
`MyDrive/narratives-thesis/data/interim/embeddings/`. Код репозиторію вбудований у ноутбук, окремо нічого ставити не потрібно.

**Що зробити вам.**
1. Runtime → Change runtime type → **T4 GPU** (без GPU ноутбук зупиниться на початку з поясненням).
2. Runtime → Run all, дозволити доступ до Drive.
3. Коли все завершиться, надішліть мені файл `MyDrive/narratives-thesis/results/04_semantic.zip` (як минулого разу архів).

Орієнтовний час 30-45 хвилин (модель завантажується близько 2 ГБ, кодування 10-15 хвилин, UMAP кілька хвилин). Потрібно приблизно 8-10 ГБ оперативної
пам'яті. Якщо сеанс обірветься, просто запустіть знову: ембединги кешуються на Drive.
"""

CELL_SETUP = '''# 1. Drive, залежності, GPU, шляхи
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
OUT = BASE / "results" / "04_semantic"
INTERIM = BASE / "data" / "interim" / "embeddings"
assert RAW.exists(), f"Немає {RAW}. Спершу запустіть ноутбук S1."
OUT.mkdir(parents=True, exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers", "hnswlib", "umap-learn", "langid", "pyyaml"], check=True)

import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "GPU не підключено. Runtime -> Change runtime type -> T4 GPU, потім запустіть знову."
print("Дані:", RAW)
print("Результати:", OUT)
print("Кеш ембедингів:", INTERIM)
'''

CELL_WRITE = '''# 2. Код репозиторію і малі вхідні таблиці (вбудовані в ноутбук)
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
for n in ["telegram_messages.parquet", "narratives_hierarchy.csv"]:
    assert (RAW / n).exists(), f"немає {n}"
msgs_p = sorted(RAW.rglob("messages.parquet"))
assert msgs_p, "немає messages.parquet у розпакованому parquet-наборі"
gd = msgs_p[0].parent
for n in ["forward_edges.parquet", "semantic_edges.parquet", "narrative_subnarrative_coding_en.csv"]:
    assert (gd / n).exists(), f"немає {n} поруч із messages.parquet"
print("Набір:", gd)
try:
    import psutil
    print(f"Пам'ять: {psutil.virtual_memory().total / 1e9:.1f} ГБ усього, {psutil.virtual_memory().available / 1e9:.1f} ГБ вільно")
except ImportError:
    pass
'''

CELL_RUN = '''# 5. Запуск S4 (скрипт scripts/s04_semantic.py з репозиторію)
cmd = [sys.executable, "-I", str(REPO / "scripts" / "s04_semantic.py"), "--raw-dir", str(RAW), "--out", str(OUT),
       "--interim", str(INTERIM), "--golden", str(gold_csv), "--batch-size", "64", "--max-len", "512", "--k", "80",
       "--umap-n", "20000"]
t0 = time.time()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(REPO))
for line in proc.stdout:
    print(line, end="")
code = proc.wait()
print(f"\\nКод завершення {code}, {time.time() - t0:.0f} с")
'''

CELL_PACK = '''# 6. Підсумок і архів
import shutil

checks = json.loads((OUT / "checks.json").read_text(encoding="utf-8"))
shutil.make_archive(str(OUT.parent / "04_semantic"), "zip", OUT)
print("Архів:", OUT.parent / "04_semantic.zip")

print("\\nПеревірки:")
for c in checks:
    mark = "ok" if c["passed"] else ("ПОМИЛКА" if c["hard"] else "розбіжність")
    print(f"  [{mark}] {c['check']}: {c['detail']}")
print("\\nФайли в", OUT)
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUT)}  {p.stat().st_size / 1e3:.1f} КБ")
assert code == 0, "S4: жорсткі перевірки не пройдено, див. вивід вище"
'''


def build(path: Path) -> str:
    files = {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in EMBED}
    cell_write = CELL_WRITE.replace("__FILES__", repr(files))
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(MD_INTRO), nbf.v4.new_code_cell(CELL_SETUP), nbf.v4.new_code_cell(cell_write),
                nbf.v4.new_code_cell(CELL_GOLD), nbf.v4.new_code_cell(CELL_CHECK), nbf.v4.new_code_cell(CELL_RUN),
                nbf.v4.new_code_cell(CELL_PACK)]
    nb.metadata = {"colab": {"name": f"04_semantic_{VERSION}.ipynb", "gpuType": "T4"}, "accelerator": "GPU",
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}
    nbf.validate(nb)
    nbf.write(nb, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "04_semantic.ipynb"
    print("written", out, build(out))
