"""Сторінка прогресу роботи: зчитує готові одиниці в results/, логи в logs/ і стан етапів, пише reports/progress.html (оновлюється сама кожні 30 с).

Запуск один раз:      python -I scripts/progress_page.py
Стежити під час довгих прогонів:  python -I scripts/progress_page.py --watch 30
Відкрити: reports/progress.html у браузері (файл локальний, нікуди не надсилається).
"""
import argparse
import html
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Очікувані одиниці поточних прогонів (преєстрація 1.1): модель -> кількість файлів units/*.json
JOBS = [
    {"id": "s06", "name": "S6: текстові моделі (v1.1)", "dir": "results/v11/06_text", "log": "logs/v11_s06.log",
     "expect": {"tfidf": 3, "bge_lr": 3, "tfidf_shuf": 20, "bge_lr_shuf": 20}},
    {"id": "s07", "name": "S7: LightGBM (v1.1)", "dir": "results/v11/07_lgbm", "log": "logs/v11_s07.log",
     "expect": {"lgbm_graph": 15, "lgbm_graph_noprior": 15, "lgbm_emb": 15, "lgbm_both": 15, "lgbm_graph_shuf": 20, "lgbm_both_shuf": 20,
                "lgbm_both_rewired": 1}},
    {"id": "s08", "name": "S8: GNN (v1.1, GPU)", "dir": "results/v11/08_gnn", "log": "logs/v11_s08.log",
     "expect": {**{v: 5 for v in ("sage_full", "sage_nosem", "sage_sem975", "sage_sem990", "sage_nofwd", "sage_nocopub", "sage_notext",
                                   "rgcn_full", "rgcn_notext", "hgt_full", "hgt_notext")}, "sage_full_shuf": 20}},
    {"id": "xlmr", "name": "XLM-R (v1.1, GPU)", "dir": "results/v11/06_text", "log": "logs/v11_xlmr.log", "expect": {"xlmr": 6, "xlmr_shuf": 20}},
    {"id": "eval", "name": "Оцінка S6-S7 на золотих (v1.1)", "dir": "results/v11/eval", "log": None, "expect": {"run.json": 1}},
]
LABELS = {"sage_full": "GraphSAGE, повна модель", "sage_nosem": "без семантичних ребер", "sage_sem975": "семантика, поріг 0,975",
          "sage_sem990": "семантика, поріг 0,99", "sage_nofwd": "без форвардів", "sage_nocopub": "без co_publication", "sage_notext": "GraphSAGE без тексту",
          "rgcn_full": "R-GCN", "rgcn_notext": "R-GCN без тексту", "hgt_full": "HGT", "hgt_notext": "HGT без тексту",
          "sage_full_shuf": "GraphSAGE, перестановки", "xlmr": "XLM-R (P1, P2 × 3 seed)", "xlmr_shuf": "XLM-R, перестановки",
          "tfidf": "TF-IDF + LogReg", "bge_lr": "bge-m3 + LogReg", "tfidf_shuf": "TF-IDF, перестановки міток", "bge_lr_shuf": "bge-m3, перестановки міток",
          "lgbm_graph": "лише граф", "lgbm_graph_noprior": "граф без пріору", "lgbm_emb": "ембединги", "lgbm_both": "ембединги + граф",
          "lgbm_graph_shuf": "граф, перестановки", "lgbm_both_shuf": "ембединги + граф, перестановки", "lgbm_both_rewired": "перезв'язана мережа",
          "run.json": "метрики, H1, контролі"}

# Етапи плану: (етап, стан, примітка). Стан: done, active, partial, todo
STAGES = [
    ("S0-S5: аудит, дані, EDA, мережа, семантика, протокол", "done", "звіти 00-05, преєстрація v1 (1d5fa0f)"),
    ("Критичний аналіз і преєстрація v1.1", "done", "reports/review_2026-11-24.md, коміт cca2100"),
    ("S6: TF-IDF і bge-m3 + LogReg за v1.1", "done", "reports/v11_s06_s07.md"),
    ("S7: LightGBM за v1.1", "done", "H1: критерій не виконано"),
    ("S6: XLM-R", "todo", "GPU: scripts/gpu_run.ps1 -Stage xlmr, plan/GPU_ETAPY.md"),
    ("S8: GNN з co_publication, H2", "todo", "GPU: scripts/gpu_run.ps1 -Stage s08"),
    ("LLM-компаратор zero-shot", "todo", "потрібне рішення щодо API"),
    ("S9: гейтований гібрид і абляції", "todo", ""),
    ("S10: кластери і каскади, H3", "todo", ""),
    ("S11: зведені таблиці, розбір помилок", "todo", ""),
    ("S12: прототип на фінальних прогнозах", "partial", "є версія на швидкому режимі"),
    ("S14: текст диплома", "todo", ""),
]


def count_units(job):
    d = ROOT / job["dir"]
    if job["id"] == "eval":
        f = d / "run.json"
        return {"run.json": (1 if f.exists() else 0, [f.stat().st_mtime] if f.exists() else [])}
    out = {m: (0, []) for m in job["expect"]}
    u = d / "units"
    if u.exists():
        for f in u.glob("*.json"):
            m = f.name.split("__")[0]
            if m in out:
                n, ts = out[m]
                out[m] = (n + 1, ts + [f.stat().st_mtime])
    return out


def tail(path, n=8):
    if not path:
        return []
    p = ROOT / path
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return [x for x in lines if x.strip()][-n:]


def eta(done, total, stamps, start):
    """Оцінка часу до кінця за темпом останніх одиниць (грубо: одиниці різні за ціною)."""
    if done >= total or done == 0:
        return None
    stamps = sorted(stamps)
    recent = stamps[-8:]
    if len(recent) >= 2:
        rate = (recent[-1] - recent[0]) / (len(recent) - 1)
    else:
        rate = (recent[-1] - start) if start else None
    return None if not rate else rate * (total - done)


def fmt_dur(s):
    if s is None:
        return "оцінка з'явиться після кількох одиниць"
    m = int(s // 60)
    return f"≈ {m // 60} год {m % 60} хв" if m >= 60 else f"≈ {max(m, 1)} хв"


def build():
    now = time.time()
    jobs = []
    for j in JOBS:
        c = count_units(j)
        done = sum(v[0] for v in c.values())
        total = sum(j["expect"].values())
        stamps = [t for v in c.values() for t in v[1]]
        logp = ROOT / j["log"] if j["log"] else None
        start = logp.stat().st_ctime if logp and logp.exists() else None
        last = max(stamps) if stamps else None
        running = bool(logp and logp.exists() and now - logp.stat().st_mtime < 900 and done < total)
        state = "done" if done >= total else ("active" if running or (last and now - last < 1800) else ("todo" if done == 0 else "stalled"))
        jobs.append({"name": j["name"], "done": done, "total": total, "state": state, "eta": eta(done, total, stamps, start),
                     "rows": [(LABELS.get(m, m), c[m][0], j["expect"][m]) for m in j["expect"]], "log": tail(j["log"]),
                     "last": datetime.fromtimestamp(last).strftime("%d.%m %H:%M") if last else "ще немає"})
    done_all = sum(j["done"] for j in jobs)
    total_all = sum(j["total"] for j in jobs)
    return {"jobs": jobs, "done": done_all, "total": total_all, "updated": datetime.now().strftime("%d.%m.%Y %H:%M:%S")}


STATE_TXT = {"done": "готово", "active": "виконується", "partial": "частково", "todo": "попереду", "stalled": "немає руху понад 30 хв"}


def render(s, refresh):
    def bar(done, total):
        pct = 0 if total == 0 else round(100 * done / total)
        return f'<div class="bar" role="progressbar" aria-valuenow="{pct}" aria-valuemin="0" aria-valuemax="100"><span style="width:{pct}%"></span></div>'

    jobs_html = []
    for j in s["jobs"]:
        rows = "".join(f'<tr><td>{html.escape(n)}</td><td class="num">{d} / {t}</td><td>{bar(d, t)}</td></tr>' for n, d, t in j["rows"])
        log = "\n".join(html.escape(x) for x in j["log"]) or "лог ще порожній"
        jobs_html.append(f"""
<section class="job" data-state="{j['state']}">
  <header><h2>{html.escape(j['name'])}</h2><span class="pill {j['state']}">{STATE_TXT[j['state']]}</span></header>
  <div class="summary"><strong class="num">{j['done']} / {j['total']}</strong> одиниць · остання {j['last']} · до кінця: {fmt_dur(j['eta']) if j['state'] != 'done' else 'завершено'}</div>
  {bar(j['done'], j['total'])}
  <div class="scroll"><table><thead><tr><th>Модель або контроль</th><th>Готово</th><th class="w">Прогрес</th></tr></thead><tbody>{rows}</tbody></table></div>
  <details><summary>Останні рядки логу</summary><pre>{log}</pre></details>
</section>""")
    stages = "".join(f'<li data-state="{st}"><span class="pill {st}">{STATE_TXT[st]}</span><span class="st">{html.escape(n)}</span>'
                     f'<span class="note">{html.escape(note)}</span></li>' for n, st, note in STAGES)
    pct = 0 if s["total"] == 0 else round(100 * s["done"] / s["total"])
    return f"""<!doctype html>
<html lang="uk"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{refresh}">
<title>Прогрес Dyplom2026</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Пульт лабораторії: зверху загальний стан, нижче картки прогонів, унизу карта етапів плану */
:root {{ --bg:#f4f6f8; --panel:#ffffff; --fg:#17212b; --muted:#5b6977; --line:#d8dee5; --accent:#1f6f8b;
  --ok:#2f7d4f; --warn:#b7791f; --bad:#b4443a; --todo:#8a96a3;
  --sans:"IBM Plex Sans", "Segoe UI", system-ui, sans-serif; --mono:"IBM Plex Mono", Consolas, monospace; color-scheme: light }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#11171d; --panel:#18212a; --fg:#e4eaf0; --muted:#95a3b1; --line:#2a3743; --accent:#5fb3d0;
  --ok:#5cbf83; --warn:#e0a84a; --bad:#e2766b; --todo:#6f7c89; color-scheme: dark }} }}
* {{ box-sizing: border-box }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 var(--sans);
  padding-block:24px 48px; padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)) }}
main {{ max-width:960px; margin:0 auto; display:grid; gap:20px }}
h1 {{ font-size:1.5rem; margin:0; text-wrap:balance }}
h2 {{ font-size:1.05rem; margin:0 }}
.top {{ display:flex; flex-wrap:wrap; gap:8px 24px; align-items:baseline; justify-content:space-between }}
.meta {{ color:var(--muted); font-size:.875rem }}
.num {{ font-family:var(--mono); font-variant-numeric:tabular-nums }}
.overall {{ display:grid; gap:8px }}
.overall strong {{ font-size:2rem; font-family:var(--mono) }}
.bar {{ height:8px; background:var(--line); border-radius:4px; overflow:hidden; min-width:80px }}
.bar span {{ display:block; height:100%; background:var(--accent) }}
.job {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; display:grid; gap:12px; min-width:0 }}
.job[data-state="done"] .bar span {{ background:var(--ok) }}
.job header {{ display:flex; gap:12px; align-items:center; justify-content:space-between; flex-wrap:wrap }}
.summary {{ color:var(--muted); font-size:.9rem }}
.summary strong {{ color:var(--fg) }}
.pill {{ font-size:.75rem; letter-spacing:.03em; padding:2px 8px; border-radius:999px; border:1px solid currentColor; white-space:nowrap }}
.pill.done {{ color:var(--ok) }} .pill.active {{ color:var(--accent) }} .pill.partial {{ color:var(--warn) }}
.pill.todo {{ color:var(--todo) }} .pill.stalled {{ color:var(--bad) }}
.scroll {{ overflow-x:auto }}
table {{ width:100%; border-collapse:collapse; font-size:.9rem }}
th {{ text-align:left; color:var(--muted); font-weight:600; font-size:.75rem; letter-spacing:.04em; text-transform:uppercase; padding:4px 8px 4px 0 }}
td {{ padding:6px 8px 6px 0; border-top:1px solid var(--line); vertical-align:middle }}
th.w {{ width:40% }}
details summary {{ cursor:pointer; color:var(--muted); font-size:.875rem }}
details summary:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px }}
pre {{ font:12px/1.5 var(--mono); background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:10px; overflow-x:auto; white-space:pre-wrap; word-break:break-word }}
.stages {{ list-style:none; margin:0; padding:0; display:grid; gap:0; background:var(--panel); border:1px solid var(--line); border-radius:8px }}
.stages li {{ display:grid; grid-template-columns:auto 1fr; gap:4px 12px; padding:10px 16px; border-top:1px solid var(--line); align-items:baseline }}
.stages li:first-child {{ border-top:0 }}
.stages .note {{ grid-column:2; color:var(--muted); font-size:.85rem }}
.stages .note:empty {{ display:none }}
.stages li .pill {{ grid-row:span 2 }}
</style></head>
<body><main>
<div class="top"><h1>Прогрес Dyplom2026</h1><span class="meta">оновлено {s['updated']} · сторінка оновлюється кожні {refresh} с</span></div>
<div class="overall"><div class="meta">Поточні прогони за преєстрацією 1.1</div><strong>{pct}%</strong>{bar(s['done'], s['total'])}
<div class="meta num">{s['done']} з {s['total']} одиниць</div></div>
{''.join(jobs_html)}
<h2>Етапи плану</h2>
<ul class="stages">{stages}</ul>
<p class="meta">Одиниця це один прогін моделі (модель, протокол, seed) або одна перестановка контролю. Оцінка часу груба: одиниці різні за ціною.</p>
</main></body></html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="перебудовувати кожні N секунд, поки не завершаться всі прогони")
    ap.add_argument("--out", default=str(ROOT / "reports" / "progress.html"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    while True:
        s = build()
        out.write_text(render(s, args.watch or 30), encoding="utf-8")
        (out.with_suffix(".json")).write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        if not args.watch or s["done"] >= s["total"]:
            break
        time.sleep(args.watch)
    print(f"{out}: {s['done']} з {s['total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
