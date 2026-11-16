# Dyplom2026: програмна частина магістерської роботи

Тема: «Гібридні моделі машинного навчання на основі графової топології та семантичного аналізу тексту для виявлення
дезінформаційних наративів». Дані й базова розмітка: [TeleNarratives](https://github.com/yuliavistak/TeleNarratives).
План виконання: `PLAN_vykonannya.md` (у Project «Dyplom»). Поточний стан: [STATUS.md](STATUS.md).

## Як відтворити

```bash
git clone https://github.com/Obizonko/Dyplom2026
cd Dyplom2026
git clone --depth 1 https://github.com/yuliavistak/TeleNarratives external/TeleNarratives
pip install -r requirements.txt

python -m pytest -q                 # тести
python -I scripts/s00_audit.py      # S0: аудит даних, reports/00_audit.md
python -I scripts/s06_tfidf.py      # S6 (cloud): TF-IDF бейзлайн, reports/06_text_tfidf.md
```

GPU-етапи (завантаження з OSF, ембеддинги, XLM-R, GNN) виконуються в Google Colab з папки `colab/`.
Повні дані (1,35 млн повідомлень) лежать на Google Drive, у репозиторій не комітяться.

## Структура

```text
configs/        base.yaml (шляхи, seed, пороги); prereg.md з'явиться на етапі S5
src/narr/       io, audit, evaluate, baselines, report, viz (інші модулі за планом додаються по етапах)
scripts/        по одному CLI на етап: s00_audit.py, s06_tfidf.py, ...
colab/          ноутбуки для GPU-етапів (01_fetch_osf.ipynb)
tests/          pytest: схеми, метрики, бейзлайни
results/NN_*/   metrics.csv, predictions.parquet, figures/*.png, run.json (дата, git-хеш, параметри, версії)
reports/NN_*.md звіт етапу; усі числа породжені скриптами
external/       клон TeleNarratives (не комітиться, не змінюється)
data/           raw / interim / processed (не комітиться)
```

## Правила

- Кожне число у звіті має джерело: файл у `results/` і скрипт, що його створив.
- Фінальні метрики лише на золотій розмітці (412 повідомлень); слабкі (LLM) мітки тільки для навчання.
- Метрики завжди з 95% CI (бутстреп). Основна метрика macro-F1 разом з MCC і PR-AUC.
- Увага: колонка `general_f1` у метриках TeleNarratives дорівнює accuracy, а не F1 (див. `reports/00_audit.md`).
