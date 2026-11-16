"""Графіки: 150 dpi, підписи українською."""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 150
C_POS, C_NEG, C_ACC = "#c0392b", "#7f8c8d", "#2c3e50"


def _style():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": DPI, "savefig.dpi": DPI, "font.family": "DejaVu Sans"})


def wrap(s: str, w: int = 34) -> str:
    return "\n".join(textwrap.wrap(str(s), w))


def fig_gold_overview(monthly: pd.DataFrame, meta: pd.DataFrame, out: Path) -> None:
    """monthly: колонки month, n_narrative, n_other. meta: колонки label, n."""
    _style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(monthly))
    ax[0].bar(x, monthly["n_other"], color=C_NEG, label="без наративу")
    ax[0].bar(x, monthly["n_narrative"], bottom=monthly["n_other"], color=C_POS, label="наративні")
    ax[0].set_xticks(x, monthly["month"], rotation=60, ha="right")
    ax[0].set_ylabel("повідомлень")
    ax[0].set_title("Золота вибірка за місяцями")
    ax[0].set_ylim(0, float((monthly["n_other"] + monthly["n_narrative"]).max()) * 1.25)
    ax[0].legend(frameon=False, ncol=2, loc="upper left")
    m = meta.sort_values("n")
    ax[1].barh([wrap(s) for s in m["label"]], m["n"], color=C_ACC)
    for i, v in enumerate(m["n"]):
        ax[1].text(v + 0.5, i, str(v), va="center")
    ax[1].set_xlabel("наративних повідомлень")
    ax[1].set_title("Мета-наративи (рівень 1)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def fig_metrics_ci(table: pd.DataFrame, out: Path, title: str) -> None:
    """table: колонки setting, model, metric, point, ci_lo, ci_hi (long). Три панелі: macro_f1, mcc, pr_auc."""
    _style()
    metrics = [("macro_f1", "macro-F1"), ("mcc", "MCC"), ("pr_auc", "PR-AUC")]
    models = list(dict.fromkeys(table["model"]))
    settings = list(dict.fromkeys(table["setting"]))
    colors = {"random_cv": C_ACC, "channel_cv": C_POS}
    names = {"random_cv": "випадкова CV", "channel_cv": "CV за каналами"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (key, label) in zip(axes, metrics):
        for j, s in enumerate(settings):
            sub = table[(table["metric"] == key) & (table["setting"] == s)].set_index("model").reindex(models)
            y = np.arange(len(models)) + (j - (len(settings) - 1) / 2) * 0.28
            ax.errorbar(sub["point"], y, xerr=[sub["point"] - sub["ci_lo"], sub["ci_hi"] - sub["point"]],
                        fmt="o", color=colors.get(s, "k"), capsize=3, label=names.get(s, s))
        ax.set_yticks(np.arange(len(models)), models)
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
