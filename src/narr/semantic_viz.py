"""Графіки S4: відтворення оцінок схожості, досяжність між групами, UMAP. Підписи українською, 150 dpi."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from narr.viz import C_ACC, C_NEG, C_POS, _style  # noqa: E402

LAYER_COLORS = {"forward": "#2c3e50", "semantic_orig": "#c0392b", "semantic_own_0985": "#e67e22", "semantic_own_0975": "#f1c40f",
                "forward+semantic_orig": "#16a085"}
LAYER_LABELS = {"forward": "форварди", "semantic_orig": "семантичні ребра набору (від 0,975)",
                "semantic_own_0985": "наші семантичні ребра (від 0,985)", "semantic_own_0975": "наші семантичні ребра (від 0,975)",
                "forward+semantic_orig": "форварди і семантичні ребра набору"}


def fig_repro(sample: pd.DataFrame, stats: dict, out: Path) -> None:
    """sample: колонки score (з набору) і cos (наш bge-m3)."""
    _style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].scatter(sample["cos"], sample["score"], s=4, alpha=0.25, color=C_ACC)
    lo, hi = float(min(sample["cos"].min(), sample["score"].min())), 1.0
    xs = np.linspace(lo, hi, 50)
    ax[0].plot(xs, xs, color=C_POS, lw=1.2, label="score = cos")
    ax[0].plot(xs, (xs + 1) / 2, color=C_NEG, lw=1.2, ls="--", label="score = (cos + 1) / 2")
    ax[0].set(xlabel="наш косинус (bge-m3)", ylabel="similarity_score з набору", title="Пари семантичних ребер набору")
    ax[0].legend(frameon=False)
    d = (sample["score"] - sample["cos"]).to_numpy()
    ax[1].hist(d, bins=60, color="#8aa9d6")
    ax[1].axvline(0, color=C_POS, lw=1)
    ax[1].set(xlabel="score - cos", ylabel="пар", title=("Різниця (спірмен " + f"{stats['spearman']:.3f}".replace(".", ",") + ", середній модуль " + f"{stats['mean_abs_diff_direct']:.3f}".replace(".", ",") + ")"))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_reach(reach: pd.DataFrame, out: Path) -> None:
    """reach: layer, direction (n_to_nn, nn_to_n, all), k, share, null_mean (може бути NaN). Вісь Y логарифмічна: справжні частки
    на порядки менші за нульову модель, у лінійному масштабі шари не розрізнити."""
    _style()
    dirs = [("n_to_nn", "від каналів n до груп nn"), ("nn_to_n", "від каналів nn до груп n"), ("all", "усі повідомлення шару")]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    handles = {}
    for ax, (d, title) in zip(axs, dirs):
        for layer, g in reach[reach["direction"] == d].groupby("layer"):
            g = g.sort_values("k")
            col = LAYER_COLORS.get(layer, "#7f8c8d")
            (h,) = ax.plot(g["k"], g["share"].where(g["share"] > 0), marker="o", color=col, label=LAYER_LABELS.get(layer, layer))
            handles[layer] = h
            if g["null_mean"].notna().any():
                ax.plot(g["k"], g["null_mean"], ls=":", color=col, alpha=0.7)
        ax.set(xlabel="кроків", title=title, xticks=sorted(reach["k"].unique()), yscale="log", ylim=(1e-3, 1.2))
    axs[0].set_ylabel("частка, що досягає іншої групи (лог. вісь)")
    fig.legend(handles=list(handles.values()), loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=3, frameon=False, fontsize=8)
    fig.text(0.5, 0.005, "Пунктир: нульова модель (групи випадково розкидані по каналах)", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_umap(c: pd.DataFrame, out: Path, top_meta: int = 6) -> None:
    """c: x, y, source (gold / graph), language, meta (текст або 'без наративу')."""
    _style()
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    metas = c.loc[c["meta"] != "без наративу", "meta"].value_counts()
    keep = list(metas.index[:top_meta])
    cmap = plt.get_cmap("tab10")
    mcol = {m: cmap(i) for i, m in enumerate(keep)}
    none = c["meta"] == "без наративу"
    axs[0].scatter(c.loc[none, "x"], c.loc[none, "y"], s=2, color="#d5d8dc", alpha=0.5, label="без наративу")
    for m in keep:
        s = c["meta"] == m
        axs[0].scatter(c.loc[s, "x"], c.loc[s, "y"], s=3, color=mcol[m], alpha=0.8, label=f"{m[:42]} ({int(s.sum())})")
    oth = ~none & ~c["meta"].isin(keep)
    axs[0].scatter(c.loc[oth, "x"], c.loc[oth, "y"], s=3, color="#7f8c8d", alpha=0.8, label=f"інші наративи ({int(oth.sum())})")
    axs[0].set_title("Мета-наратив (слабка мітка або золота)")
    axs[0].legend(frameon=False, fontsize=6, markerscale=3, loc="upper left")
    lcol = {"uk": "#2980b9", "ru": "#c0392b", "en": "#27ae60"}
    for lg, col in lcol.items():
        s = c["language"] == lg
        axs[1].scatter(c.loc[s, "x"], c.loc[s, "y"], s=2, color=col, alpha=0.6, label=f"{lg} ({int(s.sum())})")
    axs[1].set_title("Мова (langid)")
    axs[1].legend(frameon=False, fontsize=7, markerscale=4)
    g = c["source"] == "gold"
    axs[2].scatter(c.loc[~g, "x"], c.loc[~g, "y"], s=2, color="#d5d8dc", alpha=0.6, label=f"вибірка графа ({int((~g).sum())})")
    pos = g & (c["meta"] != "без наративу")
    axs[2].scatter(c.loc[g & ~pos, "x"], c.loc[g & ~pos, "y"], s=10, color=C_NEG, edgecolor="white", lw=0.3, label=f"золоті без наративу ({int((g & ~pos).sum())})")
    axs[2].scatter(c.loc[pos, "x"], c.loc[pos, "y"], s=10, color=C_POS, edgecolor="white", lw=0.3, label=f"золоті наративні ({int(pos.sum())})")
    axs[2].set_title("Золоті повідомлення на тлі вибірки графа")
    axs[2].legend(frameon=False, fontsize=7, markerscale=2)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
