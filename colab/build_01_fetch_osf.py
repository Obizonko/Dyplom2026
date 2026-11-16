"""Генерує colab/01_fetch_osf.ipynb. Запуск: python3 colab/build_01_fetch_osf.py"""
from pathlib import Path

import nbformat as nbf

MD_INTRO = """# S1. Завантаження повних даних TeleNarratives з OSF і маніфест

**Що робить ноутбук:** завантажує файли з OSF на ваш Google Drive (`MyDrive/narratives-thesis/data/raw/`),
рахує для кожного файлу розмір, sha256, кількість рядків, колонки, 3 приклади і профіль колонок, зберігає `data_manifest.json`.

**Що зробити вам:** Runtime → Run all (GPU не потрібен). Дозвольте доступ до Drive у першій комірці.
Наприкінці надішліть мені файл `data_manifest.json` (він також завантажиться у браузер автоматично і лежить у
`MyDrive/narratives-thesis/data_manifest.json`).

**Якщо OSF не дає завантажити програмно** (view-only посилання часто обмежене): завантажте файли вручну з
https://osf.io/mjcs9/overview?view_only=07148b5ce53b412998ec179587ad8240 у папку `MyDrive/narratives-thesis/data/raw/`
і просто запустіть ноутбук: комірки завантаження пропустять відсутнє, а маніфест буде побудовано з того, що є в папці.
"""

CELL_SETUP = '''# 1. Підключення Drive і налаштування
import hashlib, json, os, re, time
from pathlib import Path

try:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive/narratives-thesis")
except ImportError:  # запуск поза Colab
    BASE = Path("./narratives-thesis")

OSF_NODE = "mjcs9"
VIEW_ONLY = "07148b5ce53b412998ec179587ad8240"
TARGET = BASE / "data" / "raw"
TARGET.mkdir(parents=True, exist_ok=True)
print("Папка даних:", TARGET)
'''

CELL_LIST = '''# 2. Перелік файлів на OSF (через публічний API з view_only ключем)
import requests

API = "https://api.osf.io/v2"

def add_view(url):
    if "view_only=" in url:
        return url
    return url + ("&" if "?" in url else "?") + "view_only=" + VIEW_ONLY

def osf_list(url, prefix=""):
    files = []
    while url:
        r = requests.get(add_view(url), timeout=60)
        r.raise_for_status()
        j = r.json()
        for it in j["data"]:
            a = it["attributes"]
            if a.get("kind") == "folder":
                child = it["relationships"]["files"]["links"]["related"]["href"]
                files += osf_list(child, prefix + a["name"] + "/")
            else:
                files.append({"path": prefix + a["name"], "size": a.get("size"), "url": it["links"]["download"]})
        url = (j.get("links") or {}).get("next")
    return files

OSF_FILES = []
try:
    OSF_FILES = osf_list(f"{API}/nodes/{OSF_NODE}/files/osfstorage/?page[size]=100")
    total = sum(f["size"] or 0 for f in OSF_FILES)
    print(f"Знайдено файлів: {len(OSF_FILES)}, разом {total / 1e9:.2f} ГБ")
    for f in OSF_FILES:
        print(f"  {f['path']:70s} {((f['size'] or 0) / 1e6):10.1f} МБ")
except Exception as e:
    print("Не вдалося отримати перелік з OSF:", repr(e))
    print("Завантажте файли вручну (див. інструкцію вище) і запустіть наступні комірки.")
'''

CELL_DOWNLOAD = '''# 3. Завантаження (пропускає вже завантажені файли з тим самим розміром)
def download(f, retries=3):
    dst = TARGET / f["path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and f["size"] and dst.stat().st_size == f["size"]:
        print("  вже є:", f["path"]); return True
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(add_view(f["url"]), stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as out:
                    for chunk in r.iter_content(8 << 20):
                        out.write(chunk)
            tmp.rename(dst)
            print(f"  ок: {f['path']} ({dst.stat().st_size / 1e6:.1f} МБ)")
            return True
        except Exception as e:
            print(f"  спроба {attempt} не вдалась для {f['path']}: {e!r}")
            time.sleep(5 * attempt)
    return False

failed = [f["path"] for f in OSF_FILES if not download(f)]
print("Не завантажено:", failed or "нічого")
'''

CELL_MANIFEST = '''# 4. Маніфест: розмір, sha256, рядки, колонки, приклади, профіль колонок
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LONG_TEXT_CHARS = 100   # колонки з довшими рядками вважаємо текстом і не рахуємо унікальні значення

def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()

def profile_series(s):
    out = {"nulls": int(s.isna().sum())}
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        nn = s.dropna()
        if len(nn):
            out["min"], out["max"] = str(nn.min()), str(nn.max())
    if s.dtype == object or str(s.dtype).startswith("string") or str(s.dtype) == "str":
        sample = s.dropna().head(1000).astype(str)
        if len(sample) and sample.str.len().mean() > LONG_TEXT_CHARS:
            out["kind"] = "long_text"
            return out
    try:
        nun = int(s.nunique(dropna=True))
        out["n_unique"] = nun
        if nun <= 20:
            out["value_counts"] = {str(k): int(v) for k, v in s.value_counts(dropna=False).head(20).items()}
    except TypeError:
        out["kind"] = "unhashable"
    return out

def sample_rows(df, n=3):
    return [{k: (str(v)[:120]) for k, v in row.items()} for row in df.head(n).to_dict("records")]

def info_parquet(path):
    pf = pq.ParquetFile(path)
    info = {"rows": pf.metadata.num_rows,
            "columns": [{"name": f.name, "type": str(f.type)} for f in pf.schema_arrow]}
    info["sample"] = sample_rows(next(pf.iter_batches(batch_size=3)).to_pandas())
    prof = {}
    for f in pf.schema_arrow:   # по одній колонці, щоб не тримати все в пам'яті; списки (ембеддинги) пропускаємо
        if pa.types.is_list(f.type) or pa.types.is_large_list(f.type) or pa.types.is_fixed_size_list(f.type):
            first = pf.read_row_group(0, columns=[f.name]).column(0)[0].as_py()
            prof[f.name] = {"kind": "vector", "length_first_row": len(first) if first is not None else None}
            continue
        prof[f.name] = profile_series(pq.read_table(path, columns=[f.name]).column(0).to_pandas())
    info["column_profile"] = prof
    return info

def info_csv(path):
    n = 0
    prof_src = []
    for chunk in pd.read_csv(path, chunksize=200_000):
        n += len(chunk)
        if len(prof_src) < 1:
            prof_src.append(chunk)
    head = prof_src[0]
    return {"rows": n, "columns": [{"name": c, "type": str(t)} for c, t in head.dtypes.items()],
            "sample": sample_rows(head), "column_profile": {c: profile_series(head[c]) for c in head.columns},
            "profile_note": "профіль колонок за першим блоком до 200 000 рядків"}

def info_graphml(path):
    nodes = edges = 0
    with open(path, "rb") as f:
        for line in f:
            nodes += line.count(b"<node ")
            edges += line.count(b"<edge ")
    return {"nodes": nodes, "edges": edges}

def file_info(path, root):
    info = {"file": str(path.relative_to(root)), "size_bytes": path.stat().st_size}
    t0 = time.time()
    suf = path.suffix.lower()
    try:
        if suf == ".parquet":
            info.update(info_parquet(path))
        elif suf == ".csv":
            info.update(info_csv(path))
        elif suf == ".graphml":
            info.update(info_graphml(path))
    except Exception as e:
        info["error"] = repr(e)
    info["sha256"] = sha256(path)
    info["seconds"] = round(time.time() - t0, 1)
    return info

def build_manifest(root):
    files = sorted(p for p in Path(root).rglob("*") if p.is_file() and not p.name.endswith(".part"))
    items = []
    for p in files:
        print("профілюю:", p.relative_to(root), flush=True)
        items.append(file_info(p, root))
    return {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "root": str(root), "files": items}
'''

CELL_RUN = '''# 5. Побудова і збереження маніфесту
manifest = build_manifest(TARGET)
out_path = BASE / "data_manifest.json"
out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("Збережено:", out_path)

summary = pd.DataFrame([{"файл": f["file"], "МБ": round(f["size_bytes"] / 1e6, 1),
                         "рядків/вузлів": f.get("rows", f.get("nodes")), "колонок": len(f.get("columns", [])) or None,
                         "помилка": f.get("error")} for f in manifest["files"]])
try:
    display(summary)
except NameError:
    print(summary.to_string())

try:
    from google.colab import files
    files.download(str(out_path))
except ImportError:
    pass
'''


def build(path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(MD_INTRO), nbf.v4.new_code_cell(CELL_SETUP), nbf.v4.new_code_cell(CELL_LIST),
                nbf.v4.new_code_cell(CELL_DOWNLOAD), nbf.v4.new_code_cell(CELL_MANIFEST), nbf.v4.new_code_cell(CELL_RUN)]
    nb.metadata = {"colab": {"name": "01_fetch_osf.ipynb"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
    nbf.validate(nb)
    nbf.write(nb, path)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "01_fetch_osf.ipynb"
    build(out)
    print("written", out)
