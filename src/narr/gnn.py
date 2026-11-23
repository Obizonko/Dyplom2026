"""S8: гетерогенний граф (повідомлення, канали) і GraphSAGE для класифікації повідомлень.

Вузли: `msg` (усі повідомлення набору S4) і `chan` (канали). Ребра: `msg-sem-msg` (семантичні копії між різними каналами, cos >= 0,985, у обидва боки),
`msg-pub-chan` і зворотні. Ознаки msg: ембединг bge-m3 (або нулі в варіанті «без тексту»); ознаки chan: структурні ознаки мережі S3 (log-ваги),
для каналів поза мережею нулі (членство в мережі як ознака не береться, S3 п. 12). Мітки (слабкі) лише в train-вузлів, золоті вузли в графі без міток.
Форвард-ребра на рівні повідомлень і co_publication у швидкому режимі не включено (див. STATUS).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from narr.evaluate import binary_metrics


def build_graph(emb: np.ndarray | None, msg_channel: np.ndarray, pairs: pd.DataFrame, chan_ids: np.ndarray, chan_feat: np.ndarray,
                text: bool = True):
    """HeteroData. msg_channel: channel_id кожного вузла msg; chan_ids: порядок вузлів chan; chan_feat: (n_chan, d) стандартизовані ознаки."""
    import torch
    from torch_geometric.data import HeteroData

    n = len(msg_channel)
    d = HeteroData()
    if text:
        d["msg"].x = torch.as_tensor(np.asarray(emb, dtype=np.float32) * float(np.sqrt(emb.shape[1])))  # одиничні вектори -> одиниця на координату
    else:
        d["msg"].x = torch.zeros((n, 1), dtype=torch.float32)
    d["chan"].x = torch.as_tensor(chan_feat.astype(np.float32))
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    ei = torch.as_tensor(np.vstack([np.concatenate([i, j]), np.concatenate([j, i])]), dtype=torch.long)
    d["msg", "sem", "msg"].edge_index = ei
    cmap = pd.Series(np.arange(len(chan_ids)), index=chan_ids)
    c = cmap.loc[msg_channel].to_numpy()
    pub = torch.as_tensor(np.vstack([np.arange(n), c]), dtype=torch.long)
    d["msg", "pub", "chan"].edge_index = pub
    d["chan", "rev_pub", "msg"].edge_index = pub.flip(0)
    return d


def make_model(in_msg: int, in_chan: int, hidden: int = 128, dropout: float = 0.3, text_dropout: float = 0.0):
    import torch
    from torch import nn
    from torch_geometric.nn import HeteroConv, SAGEConv

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin_m, self.lin_c = nn.Linear(in_msg, hidden), nn.Linear(in_chan, hidden)
            mk = lambda: HeteroConv({("msg", "sem", "msg"): SAGEConv(hidden, hidden), ("msg", "pub", "chan"): SAGEConv(hidden, hidden),
                                     ("chan", "rev_pub", "msg"): SAGEConv(hidden, hidden)}, aggr="sum")
            self.c1, self.c2 = mk(), mk()
            self.out = nn.Linear(hidden, 2)
            self.dp = nn.Dropout(dropout)
            self.text_dropout = text_dropout

        def forward(self, x_dict, ei_dict):
            xm = x_dict["msg"]
            if self.training and self.text_dropout > 0:  # modality dropout для ознак повідомлення
                xm = xm * (torch.rand(xm.shape[0], 1, device=xm.device) > self.text_dropout).float()
            h = {"msg": torch.relu(self.lin_m(xm)), "chan": torch.relu(self.lin_c(x_dict["chan"]))}
            for conv in (self.c1, self.c2):
                new = conv(h, ei_dict)
                h = {k: self.dp(torch.relu(new.get(k, h[k]) + h[k])) for k in h}  # залишкове з'єднання
            return self.out(h["msg"])

    return Net()


def class_weights(y: np.ndarray) -> np.ndarray:
    n1 = max(int(y.sum()), 1)
    n0 = max(int(len(y) - y.sum()), 1)
    return np.array([len(y) / (2 * n0), len(y) / (2 * n1)], dtype=np.float32)


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    ids, ps = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            logits = model(b.x_dict, b.edge_index_dict)[: b["msg"].batch_size]
            ps.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
            ids.append(b["msg"].n_id[: b["msg"].batch_size].cpu().numpy())
    return np.concatenate(ids), np.concatenate(ps)


def train_gnn(data, y_all: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray, eval_idx: dict[str, np.ndarray], *, seed: int, epochs: int = 12,
              hidden: int = 128, lr: float = 2e-3, batch_size: int = 2048, num_neighbors=(15, 10), patience: int = 4, text_dropout: float = 0.0,
              device: str | None = None, log=print) -> dict:
    """Навчання з вибором епохи за valid macro-F1 (слабкі мітки). Повертає ймовірності для eval_idx з найкращої епохи."""
    import torch
    from torch_geometric.loader import NeighborLoader

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    y_t = torch.as_tensor(y_all, dtype=torch.long)
    data["msg"].y = y_t
    mk = lambda idx, shuffle: NeighborLoader(data, num_neighbors=list(num_neighbors), input_nodes=("msg", torch.as_tensor(idx)), batch_size=batch_size,
                                             shuffle=shuffle, num_workers=0)
    tr_loader = mk(train_idx, True)
    va_loader = mk(valid_idx, False)
    ev_loaders = {k: mk(v, False) for k, v in eval_idx.items()}
    model = make_model(data["msg"].x.shape[1], data["chan"].x.shape[1], hidden, text_dropout=text_dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    w = torch.as_tensor(class_weights(y_all[train_idx]), device=dev)
    best = {"valid_macro_f1": -1.0, "epoch": -1, "probs": {}}
    hist, bad = [], 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for b in tr_loader:
            b = b.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(b.x_dict, b.edge_index_dict)[: b["msg"].batch_size]
            loss = torch.nn.functional.cross_entropy(logits, b["msg"].y[: b["msg"].batch_size], weight=w)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        ids, pv = predict(model, va_loader, dev)
        f1 = binary_metrics(y_all[ids], pv)["macro_f1"]
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)), "valid_macro_f1": float(f1), "seconds": round(time.time() - t0, 1)})
        log(f"    епоха {ep}: втрата {hist[-1]['train_loss']:.4f}, valid macro-F1 {f1:.4f}, {hist[-1]['seconds']} с")
        if f1 > best["valid_macro_f1"]:
            probs = {}
            for k, ld in ev_loaders.items():
                i2, p2 = predict(model, ld, dev)
                order = np.argsort(i2)
                probs[k] = (i2[order], p2[order])
            best = {"valid_macro_f1": float(f1), "epoch": ep, "probs": probs}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {**best, "history": hist}
