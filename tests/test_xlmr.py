import numpy as np
import pytest

from narr import xlmr


def test_length_bucket_batches_cover_all_once():
    rng = np.random.default_rng(0)
    lengths = rng.integers(3, 100, 1003)
    bs = xlmr.length_bucket_batches(lengths, 32, np.random.default_rng(1))
    flat = np.concatenate(bs)
    assert sorted(flat.tolist()) == list(range(1003))
    assert all(len(b) <= 32 for b in bs)
    # всередині блоку довжини близькі: розкид у партії менший, ніж у випадковій партії
    spread = np.mean([lengths[b].max() - lengths[b].min() for b in bs])
    assert spread < np.mean([np.ptp(lengths[rng.choice(1003, 32, replace=False)]) for _ in range(50)])


def test_length_bucket_is_seeded():
    L = np.arange(200)
    a = xlmr.length_bucket_batches(L, 16, np.random.default_rng(5))
    b = xlmr.length_bucket_batches(L, 16, np.random.default_rng(5))
    assert all((x == y).all() for x, y in zip(a, b))


def test_pad_batch():
    ids, mask = xlmr.pad_batch([[5, 6, 7], [8]], pad_id=1)
    assert ids.tolist() == [[5, 6, 7], [8, 1, 1]]
    assert mask.tolist() == [[1, 1, 1], [1, 0, 0]]


def test_class_weights_balanced():
    w = xlmr.class_weights(np.array([0] * 75 + [1] * 25))
    assert np.allclose(w, [100 / (2 * 75), 100 / (2 * 25)])
    with pytest.raises(ValueError):
        xlmr.class_weights(np.zeros(10, dtype=int))


def test_lr_factor_shape():
    total, warm = 100, 10
    f = [xlmr.lr_factor(s, total, warm) for s in range(total + 1)]
    assert f[0] == 0 and abs(f[warm] - 1.0) < 1e-12 and f[total] == 0
    assert all(f[i] <= f[i + 1] for i in range(warm)) and all(f[i] >= f[i + 1] for i in range(warm, total))


def test_encode_texts_chunks():
    class Tok:
        def __call__(self, texts, truncation, max_length):
            assert truncation and max_length == 4
            return {"input_ids": [[len(t)] for t in texts]}

    out = xlmr.encode_texts(Tok(), ["a", "bb", "ccc"], 4, chunk=2)
    assert out == [[1], [2], [3]]


def test_fine_tune_tiny_model_if_torch_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import zlib

    from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification

    class Tok:
        pad_token_id = 1

        def __call__(self, texts, truncation=True, max_length=64):
            return {"input_ids": [[0] + [zlib.crc32(w.encode()) % 480 + 5 for w in t.split()][:max_length - 2] + [2] for t in texts]}

    rng = np.random.default_rng(0)

    def make(n):
        y = rng.integers(0, 2, n)
        texts = []
        for v in y:
            w = rng.choice([f"w{i}" for i in range(40)], 6).tolist()
            if v:
                w.insert(int(rng.integers(0, 7)), "ключ")
            texts.append(" ".join(w))
        return texts, y

    xtr, ytr = make(480)
    xva, yva = make(100)
    xte, _ = make(10)

    def factory():
        return XLMRobertaForSequenceClassification(XLMRobertaConfig(
            vocab_size=500, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
            max_position_embeddings=140, num_labels=2, pad_token_id=1))

    r = xlmr.fine_tune(xtr, ytr, xva, yva, {"gold": xte}, seed=0, epochs=6, lr=1e-3, batch_size=16, max_len=32, tokenizer=Tok(),
                       model_factory=factory, device="cpu", log=lambda *_: None)
    assert r["valid_macro_f1"] > 0.8 and len(r["probs"]["gold"]) == 10 and len(r["history"]) == 6
    assert torch.isfinite(torch.tensor(r["probs"]["gold"])).all()
    # версія 1.1: ваги пулу в навчанні й валідації, без балансування класів
    w_tr, w_va = rng.uniform(0.05, 5.0, len(ytr)), rng.uniform(0.05, 5.0, len(yva))
    r2 = xlmr.fine_tune(xtr, ytr, xva, yva, {"gold": xte}, seed=0, epochs=6, lr=1e-3, batch_size=16, max_len=32, tokenizer=Tok(),
                        model_factory=factory, device="cpu", log=lambda *_: None, w_train=w_tr, w_valid=w_va)
    assert r2["weighted"] and not r["weighted"]
    assert r2["valid_macro_f1"] > 0.8 and np.isfinite(r2["probs"]["gold"]).all()
