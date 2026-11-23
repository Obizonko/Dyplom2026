"""Перетворює reports/quick_report.md на самодостатній HTML (рисунки вбудовано в base64), щоб файл можна було відкрити де завгодно.
Запуск: python -I scripts/export_report_html.py [reports/quick_report.md]"""
import base64
import re
import sys
from pathlib import Path

import markdown

src = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "reports" / "quick_report.md")
text = src.read_text(encoding="utf-8")


def embed(m):
    alt, rel = m.group(1), m.group(2)
    p = (src.parent / rel).resolve()
    if not p.exists():
        return m.group(0)
    return f'<img alt="{alt}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" style="max-width:100%">'


text = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", embed, text)
body = markdown.markdown(text, extensions=["tables", "fenced_code"])
css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:2em auto;line-height:1.5;padding:0 1em}"
       "table{border-collapse:collapse;margin:1em 0;font-size:0.92em}th,td{border:1px solid #bbb;padding:4px 8px}th{background:#f0f0f0}"
       "code{background:#f4f4f4;padding:1px 4px}pre{background:#f4f4f4;padding:8px;overflow:auto}em{color:#444}")
out = src.with_suffix(".html")
out.write_text(f"<!doctype html><html lang='uk'><head><meta charset='utf-8'><title>Попередній звіт</title><style>{css}</style></head><body>{body}</body></html>", encoding="utf-8")
print("Збережено", out, f"{out.stat().st_size / 1e6:.1f} МБ")
