"""Наскрізний тест S6 на синтетичних даних: тренування (TF-IDF, ембединги, контролі) -> прогнози -> оцінка на золотих."""
import json

import numpy as np
import pandas as pd
import pytest

from narr import semantic as sm
from narr import text_eval as te
from narr import textmodels as tm
from narr.splits import make_key

SIGNAL = [f"сигнал{i}" for i in range(8)]
NEUTRAL = [f"слово{i}" for i in range(80)]


def _text(rng, narrative: bool, strong: bool, tag: int) -> str:
    w = list(rng.choice(NEUTRAL, int(rng.integers(8, 14))))
    if narrative:
        w += list(rng.choice(SIGNAL, 3 if strong else 1))
    rng.shuffle(w)
    return " ".join(w) + f" ід{tag}"


def make_world(tmp_path, seed=0):
    """Пул 14 каналів, золоті з 10 каналів. Повертає (splits_dir, s5_dir, texts Series)."""
    rng = np.random.default_rng(seed)
    rows, texts = [], {}
    for ch in range(1, 15):
        base = 0.15 if ch % 2 else 0.6
        for m in range(1, 81):
            nar = rng.random() < base
            weak = bool(nar ^ (rng.random() < 0.08))
            d = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=int(rng.integers(0, 330)))
            rows.append({"channel_id": ch, "message_id": m, "date_utc": d, "weak_narrative": weak})
            texts[int(make_key(ch, m))] = _text(rng, nar, True, ch * 1000 + m)
    pool = pd.DataFrame(rows)
    grows, expl = [], []
    for ch in range(1, 11):
        for m in range(1, 13):
            y = int(rng.random() < 0.4)
            strong = bool(rng.random() < 0.5)
            d = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=int(rng.integers(0, 340)))
            mid = 5000 + m
            grows.append({"channel_id": ch, "message_id": mid, "y": y, "date_utc": d, "lang": "ru" if ch % 2 == 0 else "uk"})
            texts[int(make_key(ch, mid))] = _text(rng, bool(y), strong, ch * 1000 + mid)
            if y:
                expl.append({"channel_id": ch, "message_id": mid, "lang": grows[-1]["lang"],
                             "explicitness": "explicit" if strong else ("implicit" if rng.random() < 0.9 else "unclear"),
                             "comment": "direct" if strong else ("framing;borderline" if rng.random() < 0.3 else "framing")})
    gold = pd.DataFrame(grows)
    gold["key"] = make_key(gold["channel_id"], gold["message_id"])
    folds = {ch: (ch - 1) // 2 + 1 for ch in range(1, 11)}
    gold["fold"] = gold["channel_id"].map(folds)
    gold["in_p2_test"] = gold["date_utc"] >= pd.Timestamp("2025-09-01", tz="UTC")

    sp = tmp_path / "splits"
    for pr in ("P1", "P2", "P3"):
        (sp / pr).mkdir(parents=True, exist_ok=True)

    def wtrain(path, df):
        df[["channel_id", "message_id", "weak_narrative"]].to_csv(path, index=False)

    def wtest(path, df):
        df[["channel_id", "message_id", "y", "date_utc"]].to_csv(path, index=False)

    wtrain(sp / "P1" / "train.csv.gz", pool)
    wtest(sp / "P1" / "test.csv", gold)
    wtrain(sp / "P2" / "train.csv.gz", pool[pool["date_utc"] < pd.Timestamp("2025-09-01", tz="UTC")])
    wtest(sp / "P2" / "test.csv", gold[gold["in_p2_test"]])
    for k in range(1, 6):
        test_ch = [c for c, f in folds.items() if f == k]
        wtrain(sp / "P3" / f"fold{k}_train.csv.gz", pool[~pool["channel_id"].isin(test_ch)])
        wtest(sp / "P3" / f"fold{k}_test.csv", gold[gold["channel_id"].isin(test_ch)])

    s5 = tmp_path / "s5"
    s5.mkdir()
    gold[["key", "channel_id", "message_id", "date_utc", "y", "lang", "fold", "in_p2_test"]].to_csv(s5 / "gold_membership.csv", index=False)
    lab = pd.DataFrame(expl)
    lab.insert(0, "n", range(1, len(lab) + 1))
    lab.to_csv(s5 / "implicit_labeling.csv", index=False)
    auto = gold[["key", "channel_id", "message_id", "y", "lang"]].assign(implicit_auto=lambda d: (d["y"] == 1) & (np.arange(len(d)) % 4 == 0))
    auto.to_csv(s5 / "implicit_auto_b.csv", index=False)
    return sp, s5, pd.Series(texts)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return make_world(tmp_path_factory.mktemp("s06"))


@pytest.fixture(scope="module")
def trained(world, tmp_path_factory):
    sp, s5, texts = world
    splits = tm.load_splits(sp)
    keys = tm.needed_keys(splits)
    tm.attach_text(splits, texts.loc[keys])
    enc = sm.hash_encoder(128)
    E = sm.embed_sorted(texts.loc[keys].tolist(), enc)
    row = pd.Series(np.arange(len(keys)), index=keys)
    out = tmp_path_factory.mktemp("s06out")
    logs = []
    infos = tm.run_all(splits, out, models=("tfidf", "bge_lr"), emb_lookup=lambda k: E[row.loc[k].to_numpy()], log=logs.append)
    return {"splits": splits, "out": out, "infos": infos, "logs": logs, "texts": texts, "s5": s5, "sp": sp, "keys": keys, "E": E, "row": row}


def test_hash_matches_s4(world):
    """Хеш тексту збігається з тим, що S4 записав у індекс (інакше звірка ембединг-текст нічого б не перевіряла)."""
    _, _, texts = world
    t = texts.iloc[:6]
    ks = t.index.to_numpy()
    es, _ = sm.build_embedding_set(
        pd.DataFrame({"channel_id": ks // sm.KEY_MULT, "message_id": ks % sm.KEY_MULT, "is_labeled_batch": True, "narrative_id": np.nan, "text": "x",
                      "date_utc": pd.Timestamp("2025-01-01", tz="UTC")}),
        pd.DataFrame({"channel_id": ks // sm.KEY_MULT, "message_id": ks % sm.KEY_MULT, "text": t.to_numpy()}),
        pd.DataFrame({"channel_id": [1], "message_id": [1], "text": ["x"], "date_utc": [pd.Timestamp("2025-01-01", tz="UTC")]}),
        pd.DataFrame({"src_channel_id": [1], "src_message_id": [1], "dst_channel_id": [2], "dst_message_id": [2], "similarity_score": [0.9]}))
    got = tm.check_hash_against_index(t, es)
    assert got["ok"] and got["n_in_index"] == 6
    bad = t.copy()
    bad.iloc[0] = "інший текст"
    assert tm.check_hash_against_index(bad, es)["n_mismatch"] == 1


def test_splits_no_overlap_and_texts_attached(trained):
    for s in trained["splits"]:
        assert not set(s["train"]["key"]) & set(s["test"]["key"])
        assert s["train"]["text"].notna().all() and s["test"]["text"].notna().all()
    p3 = [s for s in trained["splits"] if s["protocol"] == "P3"]
    assert len(p3) == 5
    for s in p3:
        assert not set(s["train"]["channel_id"]) & set(s["test"]["channel_id"])


def test_units_written_and_C_shared(trained):
    out = trained["out"]
    names = {p.stem for p in (out / "predictions").glob("*.parquet")}
    for m in ("tfidf", "bge_lr"):
        for pr in ("P1", "P2", "P3"):
            assert f"{m}__{pr}__s0" in names
        assert f"{m}_shuf__P1__s0" in names
    info = {(i["model"], i["protocol"]): i for i in trained["infos"]}
    for m in ("tfidf", "bge_lr"):
        c1 = info[(m, "P1")]["C"]
        assert info[(m, "P2")]["C"] == c1 and info[(m, "P3")]["C"] == c1 and info[(m + "_shuf", "P1")]["C"] == c1
        assert len(info[(m, "P3")]["splits"]) == 5


def test_resume_skips_existing(trained):
    out, splits = trained["out"], trained["splits"]
    logs = []
    enc_E, row = trained["E"], trained["row"]
    tm.run_all(splits, out, models=("tfidf", "bge_lr"), emb_lookup=lambda k: enc_E[row.loc[k].to_numpy()], log=logs.append)
    assert sum("[пропуск]" in str(x) for x in logs) == 8  # 2 моделі x (3 протоколи + контроль)


def test_models_learn_and_shuffled_controls_do_not(trained):
    pred = pd.concat([pd.read_parquet(p) for p in (trained["out"] / "predictions").glob("*.parquet")], ignore_index=True)
    key = make_key(pred["channel_id"], pred["message_id"])
    gold = pd.read_csv(trained["s5"] / "gold_membership.csv").set_index("key")
    y = gold.loc[key, "y"].to_numpy()
    from narr.evaluate import binary_metrics
    for (m, pr), d in pred.assign(y=y).groupby(["model", "protocol"]):
        r = binary_metrics(d["y"].to_numpy(), d["p"].to_numpy())
        if m.endswith("_shuf"):
            assert abs(r["mcc"]) < 0.3 and r["pr_auc"] < 0.6, (m, pr, r)
        else:
            assert r["macro_f1"] > 0.7 and r["pr_auc"] > 0.7, (m, pr, r)


def test_valid_f1_uses_weak_labels_only(trained):
    for i in trained["infos"]:
        if not i["model"].endswith("_shuf"):
            assert 0.6 < i["valid_macro_f1"] <= 1.0
        else:
            assert i["valid_macro_f1"] < 0.65


def test_eval_end_to_end(trained):
    from importlib import util
    from pathlib import Path

    spec = util.spec_from_file_location("s06_eval", Path(__file__).resolve().parents[1] / "scripts" / "s06_eval.py")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main(["--results", str(trained["out"]), "--splits", str(trained["sp"]), "--s5", str(trained["s5"]), "--n-boot", "40"])
    out = trained["out"]
    for f in ("metrics_main.csv", "metrics_subsets.csv", "metrics_cluster.csv", "paired_diffs.csv", "recall_by_explicitness.csv", "seeds.csv",
              "model_selection.csv", "valid_units.csv", "predictions_mean.parquet", "checks.json", "run.json"):
        assert (out / f).exists(), f
    checks = json.loads((out / "checks.json").read_text(encoding="utf-8"))
    hard_failed = [c for c in checks if c["hard"] and not c["passed"]]
    assert not hard_failed, hard_failed
    assert rc == 0
    main_t = pd.read_csv(out / "metrics_main.csv")
    assert {"majority", "lang_rule", "channel_prior", "tfidf", "bge_lr", "tfidf_shuf"} <= set(main_t["model"])
    assert set(main_t["protocol"]) == {"P1", "P2", "P3"} and set(main_t["slice"]) == {"all", "ru", "uk"}
    sel = pd.read_csv(out / "model_selection.csv")
    assert sel.groupby("protocol")["selected"].sum().eq(1).all()
    sub = pd.read_csv(out / "metrics_subsets.csv")
    assert any(s.startswith("implicit_") for s in sub["subset"]) and any(s.startswith("explicit_") for s in sub["subset"])


def test_train_script_main_with_cache_and_parquet(world, tmp_path):
    """Скрипт для Colab: читає telegram_messages.parquet, звіряє хеші з індексом ембедингів, бере ембединги з кешу, пише colab_run.json."""
    from importlib import util
    from pathlib import Path

    sp, _, texts = world
    raw, emb = tmp_path / "raw", tmp_path / "emb"
    raw.mkdir(), emb.mkdir()
    keys = texts.index.to_numpy()
    extra = pd.Series({int(make_key(99, 1)): "зайвий текст поза спліт"})
    allt = pd.concat([texts, extra])
    pd.DataFrame({"channel_id": allt.index.to_numpy() // 10_000_000, "message_id": allt.index.to_numpy() % 10_000_000, "text": allt.to_numpy()}
                 ).to_parquet(raw / "telegram_messages.parquet")
    E = sm.embed_sorted(texts.tolist(), sm.hash_encoder(64))
    np.save(emb / "bge_m3_keys.npy", keys)
    np.save(emb / "bge_m3_fp16.npy", E.astype(np.float16))
    pd.DataFrame({"key": keys, "text_hash": tm.hash_texts(texts).to_numpy()}).to_parquet(emb / "bge_m3_index.parquet")

    spec = util.spec_from_file_location("s06_train", Path(__file__).resolve().parents[1] / "scripts" / "s06_train.py")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    rc = mod.main(["--raw-dir", str(raw), "--splits-dir", str(sp), "--emb-dir", str(emb), "--out", str(out), "--models", "tfidf,bge_lr",
                   "--protocols", "P1,P2", "--no-controls"])
    assert rc == 0
    meta = json.loads((out / "colab_run.json").read_text(encoding="utf-8"))
    assert meta["integrity"]["ok"] and meta["n_units"] == 4
    assert {p.stem for p in (out / "predictions").glob("*.parquet")} == {"tfidf__P1__s0", "tfidf__P2__s0", "bge_lr__P1__s0", "bge_lr__P2__s0"}

    # зіпсований індекс: тексти не збігаються з тими, що кодувалися, скрипт має зупинитися
    bad = pd.read_parquet(emb / "bge_m3_index.parquet")
    bad.loc[0, "text_hash"] = "0" * 16
    bad.to_parquet(emb / "bge_m3_index.parquet")
    with pytest.raises(SystemExit):
        mod.main(["--raw-dir", str(raw), "--splits-dir", str(sp), "--emb-dir", str(emb), "--out", str(tmp_path / "out2"), "--models", "tfidf",
                  "--protocols", "P1", "--no-controls"])


def test_missing_text_raises(world, tmp_path):
    sp, _, texts = world
    raw = tmp_path / "raw"
    raw.mkdir()
    part = texts.iloc[:-1]
    pd.DataFrame({"channel_id": part.index.to_numpy() // 10_000_000, "message_id": part.index.to_numpy() % 10_000_000, "text": part.to_numpy()}
                 ).to_parquet(raw / "telegram_messages.parquet")
    with pytest.raises(KeyError):
        tm.load_texts(raw, texts.index.to_numpy())


def test_xlmr_units_with_tiny_model(world, tmp_path):
    """Оркестрація XLM-R: seed, протоколи, контроль з перемішаними мітками, усереднення по seed в оцінці (крихітна модель, без завантаження ваг)."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import zlib
    from importlib import util
    from pathlib import Path

    sp, s5, texts = world
    splits = tm.load_splits(sp)
    tm.attach_text(splits, texts.loc[tm.needed_keys(splits)])

    class Tok:
        pad_token_id = 1

        def __call__(self, t, truncation=True, max_length=64):
            return {"input_ids": [[0] + [zlib.crc32(w.encode()) % 480 + 5 for w in x.split()][:max_length - 2] + [2] for x in t]}

    def factory():
        return transformers.XLMRobertaForSequenceClassification(transformers.XLMRobertaConfig(
            vocab_size=500, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, max_position_embeddings=140,
            num_labels=2, pad_token_id=1))

    out = tmp_path / "out"
    infos = tm.run_all(splits, out, models=("xlmr",), protocols=("P1", "P2", "P3"), xlmr_protocols=("P1", "P2"), seeds=(0, 1),
                       xlmr_kwargs=dict(epochs=2, lr=1e-3, batch_size=32, max_len=32, device="cpu", tokenizer=Tok(), model_factory=factory),
                       log=lambda *_: None)
    names = sorted(i["model"] + "/" + i["protocol"] + f"/s{i['seed']}" for i in infos)
    assert names == ["xlmr/P1/s0", "xlmr/P1/s1", "xlmr/P2/s0", "xlmr/P2/s1", "xlmr_shuf/P1/s0"]
    assert all(len(i["splits"][0]["history"]) in (1, 2) for i in infos)
    assert next(i for i in infos if i["model"] == "xlmr_shuf")["splits"][0]["history"][-1]["epoch"] == 1

    spec = util.spec_from_file_location("s06_eval", Path(__file__).resolve().parents[1] / "scripts" / "s06_eval.py")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main(["--results", str(out), "--splits", str(sp), "--s5", str(s5), "--n-boot", "30"])
    seeds = pd.read_csv(out / "seeds.csv")
    assert set(seeds.loc[seeds["model"] == "xlmr", "seed"]) == {0, 1}
    pm = pd.read_parquet(out / "predictions_mean.parquet")
    assert pm.loc[(pm["model"] == "xlmr") & (pm["protocol"] == "P1"), "n_seeds"].eq(2).all()
    assert (out / "model_selection.csv").exists() and rc in (0, 1)
