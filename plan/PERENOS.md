# Перенесення проєкту на інший комп'ютер

Оновлено 28.11.2026. Замінює `README_PERENOS.md` від 23.11.2026.

Код, результати (`results/`), звіти, преєстрація й інструкції лежать у git. Окремим архівом `Dyplom2026_data_2026-11-28.zip` переносяться лише папки, яких у git немає: `data/` і `logs/`.

## Кроки

1. Клонувати репозиторій (або `git pull`, якщо він уже є):

   ```bash
   git clone https://github.com/Obizonko/Dyplom2026.git
   ```

2. Розпакувати архів у корінь репозиторію, щоб з'явились `data/` і `logs/`:

   ```bash
   tar -xf Dyplom2026_data_2026-11-28.zip -C Dyplom2026
   ```

   Перевірити цілісність можна за `SHA256SUMS.txt` поруч з архівом (там же список файлів з контрольними сумами всередині архіву: `data/MANIFEST_SHA256.txt`).

3. Клонувати зовнішній репозиторій із золотою розміткою (потрібен для тестів і оцінки):

   ```bash
   git clone --depth 1 https://github.com/yuliavistak/TeleNarratives external/TeleNarratives
   ```

4. Створити середовище.
   - **Комп'ютер з NVIDIA:** див. `plan/GPU_ETAPY.md`, крок 3 (torch cu128, pyg-lib, torch-sparse).
   - **Без GPU** (так працює Dell Latitude 7390, Python 3.12):

     ```bash
     py -3.12 -m venv .venv
     .venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
     .venv\Scripts\python.exe -m pip install -r requirements.txt
     .venv\Scripts\python.exe -m pip install pyg-lib -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
     ```

     Рядки `torch==...+cu128`, `pyg-lib`, `torch-sparse`, `hnswlib` з `requirements.txt` на CPU-машині без MSVC не ставляться; їх прибрати перед установкою (hnswlib не потрібен).

5. Перевірка:

   ```bash
   .venv\Scripts\python.exe -m pytest -q
   ```

   Очікується 105 passed, 2 skipped.

## Що в архіві

| Шлях | Розмір | Для чого |
|---|---|---|
| `data/raw/telegram_messages.parquet` | 386 МБ | вихідні повідомлення (OSF) |
| `data/raw/extracted/parquet_set/forward_edges.parquet` | 0,9 МБ | форвард-ребра |
| `data/interim/embeddings/` | 216 МБ | ембединги bge-m3 (fp16), ключі, індекс |
| `data/interim/features/` | 9 МБ | семантичні пари, пари копій для ваг v1.1 (`sem_pairs_any_0985.parquet`, нове 28.11), co_publication, ознаки |
| `data/processed/prototype/` | 0,7 МБ | дані для Streamlit-прототипу |
| `logs/` | малий | логи всіх прогонів, зокрема v1.1 (24-28.11) |

Не включено:
- `.venv`: відновлюється за кроком 4;
- `external/`: клонується за кроком 3;
- `results/dev/`: смоук-тести, не потрібні;
- згенерована сторінка `reports/progress.html`: будується командою `python -I scripts/progress_page.py`;
- вихідні архіви OSF: за потреби качаються з OSF;
- ваги моделей: їх у проєкті не зберігаємо.

## Де продовжувати

- Стан і чекліст: `STATUS.md`.
- Що запускати на GPU: `plan/GPU_ETAPY.md`, скрипт `scripts/gpu_run.ps1`.
- Прогрес: `.venv\Scripts\python.exe -I scripts\progress_page.py --watch 30`, далі відкрити `reports\progress.html`.
