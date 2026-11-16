"""Генерує colab/06_text_models.ipynb (самодостатній: код репозиторію і спліти S5 вбудовано в ноутбук).
Запуск: python3 colab/build_06_text_models.py"""
import base64
import hashlib
import io
import zipfile
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
EMBED = ["src/narr/__init__.py", "src/narr/baselines.py", "src/narr/evaluate.py", "src/narr/splits.py", "src/narr/textmodels.py",
         "src/narr/xlmr.py", "scripts/s06_train.py"]
VERSION = "v1"

MD_INTRO = """# S6. Текстові моделі на слабких мітках: TF-IDF, ембединги bge-m3, XLM-R

**Що робить ноутбук.** Навчає текстові моделі на пулі зі слабкими (LLM) мітками і зберігає їхні ймовірності для золотих повідомлень за протоколами P1, P2, P3
(спліти S5 вбудовано в ноутбук). Золоті мітки тут не використовуються взагалі, метрики рахуються потім у хмарній сесії.

Моделі: TF-IDF (слова і символьні n-грами) + LogReg; ембединги bge-m3 з кешу S4 + LogReg; дотренування `xlm-roberta-base` (3 seed; P1 і P2).
Контролі з перемішаними мітками (P1) для кожної моделі. C у LogReg підбирається на валідації слабких міток (10% пулу), епоха XLM-R обирається так само.
Перед початком ноутбук звіряє хеші текстів з індексом ембедингів S4 і проганяє самоперевірку циклу XLM-R на крихітній випадковій моделі.

**Що зробити вам.**
1. Runtime → Change runtime type → **T4 GPU**.
2. Runtime → Run all, дозволити доступ до Drive.
3. Коли завершиться, надішліть мені `MyDrive/narratives-thesis/results/06_text_models.zip`.

Орієнтовний час 1,5-2,5 години (TF-IDF і ембединги близько 20-30 хвилин на CPU, XLM-R близько 1-1,5 години на T4). Потрібен Drive з S1 і S4
(`data/raw/telegram_messages.parquet`, `data/interim/embeddings/`). Якщо сеанс обірветься, просто запустіть знову: завершені одиниці пропускаються.
Якщо часу мало, у клітинці запуску змініть `XLMR_PROTOCOLS = "P1"` (без P2) або `SEEDS = 2`; це треба буде записати у звіт.
"""

CELL_SETUP = '''# 1. Drive, залежності, GPU, шляхи
import base64, hashlib, io, json, subprocess, sys, time, zipfile
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
EMB = BASE / "data" / "interim" / "embeddings"
OUT = BASE / "results" / "06_text_models"
OUT.mkdir(parents=True, exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "pyyaml"], check=True)
import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "GPU не підключено. Runtime -> Change runtime type -> T4 GPU, потім запустіть знову."
print("Дані:", RAW, "| ембединги:", EMB, "| результати:", OUT)
'''

CELL_WRITE = '''# 2. Код репозиторію і спліти S5 (вбудовані в ноутбук)
FILES = __FILES__
SPLITS_ZIP_B64 = "__SPLITS__"

for rel, text in FILES.items():
    p = REPO / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print(f"{hashlib.sha256(text.encode()).hexdigest()[:12]}  {rel}")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SPLITS_ZIP_B64))) as z:
    z.extractall(REPO / "splits")
print("Спліти:", sorted(str(p.relative_to(REPO / "splits")) for p in (REPO / "splits").rglob("*") if p.is_file()))
'''

CELL_CHECK = '''# 3. Перевірка, що потрібні файли на місці
for p in [RAW / "telegram_messages.parquet", EMB / "bge_m3_keys.npy", EMB / "bge_m3_fp16.npy", EMB / "bge_m3_index.parquet"]:
    assert p.exists(), f"немає {p}"
    print("ок", p, f"{p.stat().st_size / 1e6:.0f} МБ")
try:
    import psutil
    print(f"Пам'ять: {psutil.virtual_memory().total / 1e9:.1f} ГБ усього, {psutil.virtual_memory().available / 1e9:.1f} ГБ вільно")
except ImportError:
    pass
'''

CELL_SELFTEST = '''# 4. Самоперевірка циклу XLM-R (крихітна випадкова модель, без завантаження ваг, секунди)
r = subprocess.run([sys.executable, "-I", str(REPO / "scripts" / "s06_train.py"), "--selftest"], capture_output=True, text=True, cwd=str(REPO))
print(r.stdout[-2000:], r.stderr[-3000:])
assert r.returncode == 0, "самоперевірка не пройшла: див. вивід вище, основний запуск не починаємо"
'''

CELL_RUN = '''# 5. Основний запуск (scripts/s06_train.py). Завершені одиниці пропускаються, тож запуск можна повторювати.
XLMR_PROTOCOLS = "P1,P2"   # P3 для XLM-R не запускаємо (5 фолдів x 3 seed надто довго); за браком часу можна лишити "P1"
SEEDS = 3
cmd = [sys.executable, "-I", str(REPO / "scripts" / "s06_train.py"), "--raw-dir", str(RAW), "--splits-dir", str(REPO / "splits"),
       "--emb-dir", str(EMB), "--out", str(OUT), "--models", "tfidf,bge_lr,xlmr", "--protocols", "P1,P2,P3",
       "--xlmr-protocols", XLMR_PROTOCOLS, "--seeds", str(SEEDS), "--epochs", "3", "--max-len", "128", "--batch-size", "32", "--lr", "2e-5"]
t0 = time.time()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(REPO))
for line in proc.stdout:
    print(line, end="")
code = proc.wait()
print(f"\\nКод завершення {code}, {time.time() - t0:.0f} с")
'''

CELL_PACK = '''# 6. Підсумок і архів
import shutil

units = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "units").glob("*.json"))]
print(f"Одиниць: {len(units)}")
for u in units:
    print(f"  {u['model']:12s} {u['protocol']} seed {u['seed']}  valid macro-F1 {u['valid_macro_f1']:.4f}  {u.get('seconds', 0):.0f} с")
shutil.make_archive(str(OUT.parent / "06_text_models"), "zip", OUT)
print("Архів:", OUT.parent / "06_text_models.zip")
assert code == 0, "S6: запуск завершився з помилкою, див. вивід вище (завершені одиниці збережено, запустіть знову)"
'''


def splits_zip_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / "splits").rglob("*")):
            if p.is_file() and p.parent.name in ("P1", "P2", "P3"):
                z.write(p, p.relative_to(ROOT / "splits").as_posix())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build(path: Path) -> str:
    files = {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in EMBED}
    cell_write = CELL_WRITE.replace("__FILES__", repr(files)).replace("__SPLITS__", splits_zip_b64())
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(MD_INTRO), nbf.v4.new_code_cell(CELL_SETUP), nbf.v4.new_code_cell(cell_write),
                nbf.v4.new_code_cell(CELL_CHECK), nbf.v4.new_code_cell(CELL_SELFTEST), nbf.v4.new_code_cell(CELL_RUN),
                nbf.v4.new_code_cell(CELL_PACK)]
    nb.metadata = {"colab": {"name": f"06_text_models_{VERSION}.ipynb", "gpuType": "T4"}, "accelerator": "GPU",
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}
    nbf.validate(nb)
    nbf.write(nb, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "06_text_models.ipynb"
    print("written", out, build(out))
