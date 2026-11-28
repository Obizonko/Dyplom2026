"""S6 (Colab): навчання текстових моделей на пулі зі слабкими мітками, прогнози для золотих повідомлень за P1-P3.

Моделі: TF-IDF + LogReg, ембединги bge-m3 (кеш S4) + LogReg, XLM-R base (дотренування), контролі з перемішаними мітками.
Метрики тут не рахуються: прогнози і валідація зберігаються, оцінка на золотих у `scripts/s06_eval.py` (хмарна сесія).

Запуск (Colab): див. colab/06_text_models.ipynb. Приклад:
  python3 -I scripts/s06_train.py --raw-dir <data/raw> --splits-dir <splits> --emb-dir <data/interim/embeddings> --out <results/06_text_models>
Самоперевірка XLM-R на крихітній випадковій моделі без завантаження ваг (секунди):  python3 -I scripts/s06_train.py --selftest
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from narr import textmodels as tm  # noqa: E402


def selftest() -> None:
    """Проганяє справжній цикл `fine_tune` (torch + transformers) на синтетичних даних з крихітною випадковою моделлю."""
    from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification

    from narr.xlmr import fine_tune

    class Tok:
        pad_token_id = 1

        def __call__(self, texts, truncation=True, max_length=64):
            return {"input_ids": [[0] + [zlib.crc32(w.encode()) % 480 + 5 for w in t.split()][:max_length - 2] + [2] for t in texts]}

    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(40)]

    def make(n):
        y = rng.integers(0, 2, n)
        texts = []
        for v in y:
            w = rng.choice(words, 6).tolist()
            if v:
                w.insert(int(rng.integers(0, 7)), "ключ")  # сигнал на випадковій позиції
            texts.append(" ".join(w))
        return texts, y

    xtr, ytr = make(480)
    xva, yva = make(100)
    xte, _ = make(20)

    def factory():
        cfg = XLMRobertaConfig(vocab_size=500, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                               max_position_embeddings=140, num_labels=2, pad_token_id=1)
        return XLMRobertaForSequenceClassification(cfg)

    r = fine_tune(xtr, ytr, xva, yva, {"gold": xte}, seed=0, epochs=6, lr=1e-3, batch_size=16, max_len=32, tokenizer=Tok(), model_factory=factory)
    assert len(r["probs"]["gold"]) == 20 and np.isfinite(r["probs"]["gold"]).all()
    assert r["valid_macro_f1"] > 0.8, f"самоперевірка: крихітна модель не навчилася (valid macro-F1 {r['valid_macro_f1']:.3f})"
    print(f"САМОПЕРЕВІРКА ПРОЙДЕНА: valid macro-F1 {r['valid_macro_f1']:.3f}, епоха {r['best_epoch']}")


def make_lookup(emb_dir: Path):
    keys = np.load(emb_dir / "bge_m3_keys.npy")
    emb = np.load(emb_dir / "bge_m3_fp16.npy", mmap_mode="r")
    row = pd.Series(np.arange(len(keys)), index=keys)

    def lookup(k: np.ndarray) -> np.ndarray:
        v = np.asarray(emb[row.loc[k].to_numpy()], dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    return lookup


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--raw-dir")
    ap.add_argument("--splits-dir")
    ap.add_argument("--emb-dir")
    ap.add_argument("--out")
    ap.add_argument("--models", default="tfidf,bge_lr,xlmr")
    ap.add_argument("--protocols", default="P1,P2,P3")
    ap.add_argument("--xlmr-protocols", default="P1,P2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--model-name", default="xlm-roberta-base")
    ap.add_argument("--unfreeze-embeddings", action="store_true", help="XLM-R: навчати й словникові ембединги (на 6 ГБ VRAM не вміщається)")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="смоук-тест: залишити не більше N рядків train у кожному спліті")
    ap.add_argument("--v11", action="store_true", help="преєстрація 1.1 (розділ 11.1): ваги пулу, валідація за компонентами копій, без балансування класів")
    ap.add_argument("--n-perm", type=int, default=1, help="кількість перестановок міток для контролю (версія 1.1: 20)")
    ap.add_argument("--xlmr-perm", type=int, default=20, help="версія 1.1: перестановок для XLM-R, по 1 епосі (преєстрація 11.1 п. 6: 20)")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = tuple(args.models.split(","))
    protocols = tuple(args.protocols.split(","))

    splits = tm.load_splits(args.splits_dir, protocols)
    keys = tm.needed_keys(splits)
    print(f"Спліти: {len(splits)}, потрібних ключів {len(keys)}", flush=True)
    texts = tm.load_texts(args.raw_dir, keys)
    tm.attach_text(splits, texts)
    index = pd.read_parquet(Path(args.emb_dir) / "bge_m3_index.parquet", columns=["key", "text_hash"])
    integrity = tm.check_hash_against_index(texts, index)
    print("Звірка хешів текстів з індексом ембедингів S4:", integrity, flush=True)
    if not integrity["ok"]:
        raise SystemExit("Тексти не збігаються з тими, що кодувалися в S4: зупиняюсь.")
    for s in splits:  # золоті не в train (страхувальна перевірка, основна в S5)
        assert not set(s["train"]["key"]) & set(s["test"]["key"]), f"перетин train і test у {s['protocol']} fold {s['fold']}"
        if args.limit:
            s["train"] = s["train"].sample(n=min(args.limit, len(s["train"])), random_state=42).reset_index(drop=True)

    if args.v11:
        from narr import weights as wt
        from narr.io import ROOT

        cc, pk, th = wt.load_v11_inputs(ROOT)
        wsum = wt.attach_weights(splits, pk, cc, th)
        pd.DataFrame(wsum).to_csv(out / "weights_summary.csv", index=False)
        print("Версія 1.1: ваги пулу і валідація за компонентами копій додано", flush=True)
    xk = dict(epochs=args.epochs, max_len=args.max_len, batch_size=args.batch_size, lr=args.lr, model_name=args.model_name,
              freeze_word_embeddings=not args.unfreeze_embeddings)
    lookup = make_lookup(Path(args.emb_dir)) if "bge_lr" in models else None
    infos = tm.run_all(splits, out, models=models, protocols=protocols, xlmr_protocols=tuple(args.xlmr_protocols.split(",")),
                       seeds=tuple(range(args.seeds)), controls=not args.no_controls, emb_lookup=lookup, xlmr_kwargs=xk, n_perm=args.n_perm, xlmr_perm=args.xlmr_perm)

    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        tv = torch.__version__
    except ImportError:
        gpu, tv = None, None
    meta = {"args": vars(args), "integrity": integrity, "n_units": len(infos), "seconds_total": round(time.time() - t0, 1), "gpu": gpu, "torch": tv,
            "split_sizes": [{"protocol": s["protocol"], "fold": s["fold"], "n_train": int(len(s["train"])), "n_test": int(len(s["test"]))} for s in splits]}
    (out / "colab_run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Готово: {len(infos)} одиниць за {meta['seconds_total']} с", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
