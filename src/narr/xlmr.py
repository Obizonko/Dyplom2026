"""S6: дотренування XLM-R на пулі зі слабкими мітками (бінарна задача «наратив / ні»).

torch і transformers імпортуються лише всередині функцій, тож модуль підключається і в середовищі без них (хмарна сесія).
Логіка без torch (пакування за довжиною, вагові коефіцієнти класів, розклад швидкості навчання) винесена в окремі функції і тестується.

Протокол: тренуємо на частині пулу `train`, після кожної епохи рахуємо macro-F1 на валідації зі слабкими мітками (поріг 0,5).
Прогнози для тестових текстів (золоті) беруться з епохи з найкращою валідацією. Золоті мітки в цьому модулі не з'являються взагалі:
функція отримує лише тексти золотих, без міток.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np

from narr.evaluate import binary_metrics


def length_bucket_batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator, chunk_mult: int = 50) -> list[np.ndarray]:
    """Перемішує приклади, у блоках по batch_size*chunk_mult сортує за довжиною (менше доповнень), ріже на партії і перемішує партії."""
    lengths = np.asarray(lengths)
    idx = rng.permutation(len(lengths))
    chunk = batch_size * chunk_mult
    batches: list[np.ndarray] = []
    for s in range(0, len(idx), chunk):
        c = idx[s:s + chunk]
        c = c[np.argsort(lengths[c], kind="stable")]
        batches += [c[i:i + batch_size] for i in range(0, len(c), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def pad_batch(seqs: list[list[int]], pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Доповнює послідовності до найдовшої в партії. Повертає (input_ids, attention_mask)."""
    width = max(len(s) for s in seqs)
    ids = np.full((len(seqs), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(seqs), width), dtype=np.int64)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = s
        mask[r, :len(s)] = 1
    return ids, mask


def class_weights(y: np.ndarray) -> np.ndarray:
    """Зважування класів n / (2 * n_c), як class_weight='balanced' у лінійних моделях."""
    y = np.asarray(y).astype(int)
    cnt = np.bincount(y, minlength=2).astype(float)
    if (cnt == 0).any():
        raise ValueError("в навчальній частині немає одного з класів")
    return len(y) / (2.0 * cnt)


def lr_factor(step: int, total: int, warmup: int) -> float:
    """Лінійний розігрів до 1, далі лінійне спадання до 0."""
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total - step) / max(1, total - warmup))


def encode_texts(tokenizer, texts: list[str], max_len: int, chunk: int = 5000) -> list[list[int]]:
    out: list[list[int]] = []
    for s in range(0, len(texts), chunk):
        out += tokenizer(list(texts[s:s + chunk]), truncation=True, max_length=max_len)["input_ids"]
    return out


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict_proba(model, ids: list[list[int]], pad_id: int, device, batch_size: int = 128, amp: bool = False) -> np.ndarray:
    """P(клас 1) для кожної послідовності, у початковому порядку (всередині сортуємо за довжиною)."""
    import torch

    order = np.argsort([len(s) for s in ids], kind="stable")
    out = np.zeros(len(ids), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(ids), batch_size):
            b = order[s:s + batch_size]
            x, m = pad_batch([ids[i] for i in b], pad_id)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(input_ids=torch.as_tensor(x, device=device), attention_mask=torch.as_tensor(m, device=device)).logits
            out[b] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
    return out


def fine_tune(train_texts: list[str], y_train: np.ndarray, valid_texts: list[str], y_valid: np.ndarray,
              eval_texts: dict[str, list[str]], *, seed: int, epochs: int = 3, lr: float = 2e-5, batch_size: int = 32,
              max_len: int = 128, warmup_frac: float = 0.06, weight_decay: float = 0.01, model_name: str = "xlm-roberta-base",
              device: str | None = None, fp16: bool = True, tokenizer=None, model_factory=None, log=print) -> dict:
    """Дотренування з вибором епохи за валідаційним macro-F1 (слабкі мітки).

    Повертає {"valid_macro_f1", "best_epoch", "history", "probs": {назва: ймовірності з найкращої епохи}, ...}.
    `tokenizer` і `model_factory` можна підставити (тести); за замовчуванням завантажується `model_name` з HuggingFace."""
    import torch
    from torch.optim.lr_scheduler import LambdaLR

    y_train, y_valid = np.asarray(y_train).astype(int), np.asarray(y_valid).astype(int)
    _seed_everything(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(fp16 and dev.type == "cuda")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_factory is None:
        from transformers import AutoModelForSequenceClassification

        def model_factory():
            return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    pad_id = int(tokenizer.pad_token_id)

    t0 = time.time()
    ids_tr = encode_texts(tokenizer, train_texts, max_len)
    ids_va = encode_texts(tokenizer, valid_texts, max_len)
    ids_ev = {k: encode_texts(tokenizer, v, max_len) for k, v in eval_texts.items()}
    len_tr = np.array([len(s) for s in ids_tr])
    info = {"n_train": int(len(ids_tr)), "n_valid": int(len(ids_va)), "max_len": int(max_len), "tokens_median": float(np.median(len_tr)),
            "share_truncated": float((len_tr >= max_len).mean()), "tokenize_seconds": round(time.time() - t0, 1)}
    log(f"  токенізація {info['tokenize_seconds']} с, медіана {info['tokens_median']:.0f} токенів, обрізано {info['share_truncated']:.1%}")

    model = model_factory().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(ids_tr) / batch_size)
    total, warm = steps_per_epoch * epochs, int(warmup_frac * steps_per_epoch * epochs)
    sched = LambdaLR(opt, lambda s: lr_factor(s, total, warm))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights(y_train), dtype=torch.float32, device=dev))
    rng = np.random.default_rng(seed)

    best = {"valid_macro_f1": -1.0, "best_epoch": -1, "probs": {}}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t_ep, losses = time.time(), []
        for b in length_bucket_batches(len_tr, batch_size, rng):
            x, m = pad_batch([ids_tr[i] for i in b], pad_id)
            xb, mb = torch.as_tensor(x, device=dev), torch.as_tensor(m, device=dev)
            yb = torch.as_tensor(y_train[b], device=dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                logits = model(input_ids=xb, attention_mask=mb).logits
            loss = loss_fn(logits.float(), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"нескінченна втрата на епосі {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
        pv = predict_proba(model, ids_va, pad_id, dev, amp=amp)
        f1 = binary_metrics(y_valid, pv)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_macro_f1": float(f1), "seconds": round(time.time() - t_ep, 1)})
        log(f"  епоха {epoch}: втрата {history[-1]['train_loss']:.4f}, valid macro-F1 {f1:.4f}, {history[-1]['seconds']} с")
        if f1 > best["valid_macro_f1"]:
            best = {"valid_macro_f1": float(f1), "best_epoch": epoch,
                    "probs": {k: predict_proba(model, v, pad_id, dev, amp=amp) for k, v in ids_ev.items()}}
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {**best, "history": history, **info}
