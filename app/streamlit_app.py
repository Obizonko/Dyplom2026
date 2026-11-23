"""S12: прототип. Перегляд золотих повідомлень: ймовірність наративу від текстової, графової та гібридної моделей,
внесок тексту і графа (знакові суми SHAP гібрида), 5 найсхожіших повідомлень набору.
Працює на збережених прогнозах, без GPU. Дані готує `python scripts/s12_prepare.py` (data/processed/prototype/).

Запуск:  streamlit run app/streamlit_app.py
Увага: усі показані ймовірності це прогнози на золотих повідомленнях за протоколом P1 (моделі їх не бачили в навчанні).
Попередній (швидкий) режим моделей: результати для презентації, не підсумкові.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "prototype"

st.set_page_config(page_title="Наративи: прототип", layout="wide")


@st.cache_data
def load():
    return pd.read_parquet(DATA / "messages.parquet"), pd.read_parquet(DATA / "neighbors.parquet"), pd.read_json(DATA / "meta.json", typ="series")


if not (DATA / "messages.parquet").exists():
    st.error("Немає даних прототипу. Запустіть `python scripts/s12_prepare.py` і оновіть сторінку.")
    st.stop()

msgs, nbrs, meta = load()
st.title("Виявлення дезінформаційних наративів: гібрид граф + текст")
st.caption(f"Режим результатів: {meta.get('mode', '?')}. Текстова модель: {meta.get('text_model', '?')}. "
           "Дані: золота розмітка (412 повідомлень), прогнози за протоколом P1.")

with st.sidebar:
    st.header("Вибір повідомлення")
    lang = st.selectbox("Мова", ["усі", "ru", "uk"])
    label = st.selectbox("Золота мітка", ["усі", "наратив", "не наратив"])
    err = st.checkbox("Лише помилки гібрида (поріг 0,5)")
    d = msgs
    if lang != "усі":
        d = d[d["lang"] == lang]
    if label != "усі":
        d = d[d["y"] == (1 if label == "наратив" else 0)]
    if err:
        d = d[(d["p_hybrid"] >= 0.5) != (d["y"] == 1)]
    st.write(f"Підходить повідомлень: {len(d)}")
    if d.empty:
        st.stop()
    idx = st.slider("Номер у вибірці", 1, len(d), 1)
    r = d.iloc[idx - 1]

left, right = st.columns([3, 2])
with left:
    st.subheader("Повідомлення")
    st.write(r["text"])
    st.caption(f"Канал {r['channel']} | мова {r['lang']} | золота мітка: {'наратив' if r['y'] == 1 else 'не наратив'}")
with right:
    st.subheader("Ймовірність наративу")
    t = pd.DataFrame({"модель": ["лише текст", "лише граф (ознаки каналу, копії)", "гібрид (текст + граф)"],
                      "p": [r["p_text"], r["p_graph"], r["p_hybrid"]]}).set_index("модель")
    st.bar_chart(t, horizontal=True)
    st.write(t.style.format("{:.3f}"))

st.subheader("Внесок тексту і графа в гібридну модель")
c1, c2, c3 = st.columns(3)
c1.metric("Внесок тексту (сума SHAP, log-odds)", f"{r['shap_text']:+.2f}")
c2.metric("Внесок графа (сума SHAP, log-odds)", f"{r['shap_graph']:+.2f}")
c3.metric("Базове значення моделі", f"{r['base_value']:+.2f}")
tot = abs(r["shap_text"]) + abs(r["shap_graph"])
if tot > 0:
    st.progress(float(abs(r["shap_text"]) / tot), text=f"частка модуля внеску тексту {abs(r['shap_text']) / tot:.0%}, графа {abs(r['shap_graph']) / tot:.0%}")
st.caption("SHAP підсумовує внески ознак LightGBM: колонки ембединга (текст) і ознаки каналу/копій/форварда (граф). Це не причинне пояснення.")

st.subheader("5 найсхожіших повідомлень набору (косинус bge-m3)")
n = nbrs[nbrs["key"] == r["key"]].sort_values("rank")
if n.empty:
    st.info("Сусідів немає.")
else:
    st.dataframe(n[["cos", "channel", "weak_label", "text"]].rename(columns={"cos": "косинус", "channel": "канал", "weak_label": "слабка мітка", "text": "текст"}),
                 hide_index=True, use_container_width=True)
    st.caption("Слабка мітка (LLM) показана лише для довідки. Сусіди шукаються серед 108 тис. повідомлень набору, золоті з тестового боку не змішуються з навчанням.")
