"""S3: мережа каналів за пересиланнями (channel_flow_edges).

Напрямок ребра: канал-оригінал (src) до каналу, який переслав повідомлення (dst), тобто так, як у forward_edges
(оригінал раніше за копію). Тому out-сила каналу показує, наскільки його повідомлення пересилають інші, а in-сила,
наскільки він сам пересилає. Модуль не читає диск, крім `save_gexf`/`make_map`, що пишуть файли.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

EDGE_COLS = ["src_channel_id", "dst_channel_id", "weight"]


def clean_edges(edges: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Прибирає петлі й зводить дублікати пар; повертає чисті ребра і лічильники для checks."""
    e = edges[EDGE_COLS].copy()
    e["src_channel_id"] = e["src_channel_id"].astype("int64")
    e["dst_channel_id"] = e["dst_channel_id"].astype("int64")
    n_in = len(e)
    n_loops = int((e["src_channel_id"] == e["dst_channel_id"]).sum())
    e = e[e["src_channel_id"] != e["dst_channel_id"]]
    n_dup = int(e.duplicated(["src_channel_id", "dst_channel_id"]).sum())
    e = e.groupby(["src_channel_id", "dst_channel_id"], as_index=False)["weight"].sum()
    return e, {"n_input": n_in, "n_loops_removed": n_loops, "n_duplicate_pairs_merged": n_dup, "n_edges": len(e)}


def build_graph(edges: pd.DataFrame) -> ig.Graph:
    """Спрямований зважений граф; вершини лише ті канали, що мають хоч одне ребро."""
    ids = sorted(set(edges["src_channel_id"]) | set(edges["dst_channel_id"]))
    idx = {c: i for i, c in enumerate(ids)}
    g = ig.Graph(n=len(ids), directed=True)
    g.vs["channel_id"] = ids
    g.add_edges([(idx[s], idx[d]) for s, d in zip(edges["src_channel_id"], edges["dst_channel_id"])])
    g.es["weight"] = [float(w) for w in edges["weight"]]
    return g


def undirected_weights(g: ig.Graph, transform: str = "raw") -> ig.Graph:
    """Неспрямований граф: ваги двох напрямків додаються. transform: raw | log | none (усі ваги 1)."""
    ug = g.copy()
    if transform == "log":
        ug.es["weight"] = [float(np.log1p(w)) for w in g.es["weight"]]
    elif transform == "none":
        ug.es["weight"] = [1.0] * g.ecount()
    elif transform != "raw":
        raise ValueError(transform)
    ug.to_undirected(mode="collapse", combine_edges={"weight": "sum"})
    return ug


def centrality(g: ig.Graph) -> pd.DataFrame:
    """Ступені, сили, PageRank (forwarder: накопичується в каналах, що багато пересилають; source: у джерелах), betweenness, k-core, слабка компонента."""
    w = g.es["weight"]
    dist = [1.0 / x for x in w]
    n = g.vcount()
    norm = (n - 1) * (n - 2) if n > 2 else 1
    rev = g.copy()
    rev.reverse_edges(list(range(g.ecount())))
    und = g.copy()
    und.to_undirected(mode="collapse")
    comp = g.connected_components(mode="weak").membership
    return pd.DataFrame({
        "channel_id": g.vs["channel_id"],
        "in_degree": g.indegree(), "out_degree": g.outdegree(),
        "in_strength": g.strength(mode="in", weights="weight"),
        "out_strength": g.strength(mode="out", weights="weight"),
        "pagerank_forwarder": g.pagerank(directed=True, damping=0.85, weights="weight"),
        "pagerank_source": rev.pagerank(directed=True, damping=0.85, weights="weight"),
        "betweenness": [b / norm for b in g.betweenness(directed=True, weights=dist)],
        "coreness": und.coreness(mode="all"),
        "component": comp,
    })


def leiden_runs(ug: ig.Graph, seeds: list[int]) -> list[dict]:
    out = []
    for s in seeds:
        p = la.find_partition(ug, la.ModularityVertexPartition, weights="weight", seed=int(s), n_iterations=-1)
        out.append({"seed": int(s), "membership": list(p.membership), "modularity": float(p.modularity),
                    "n_communities": len(set(p.membership))})
    return out


def stability(runs: list[dict]) -> dict:
    nmis = [normalized_mutual_info_score(a["membership"], b["membership"]) for a, b in combinations(runs, 2)]
    return {"n_runs": len(runs), "n_pairs": len(nmis), "nmi_mean": float(np.mean(nmis)), "nmi_min": float(np.min(nmis)),
            "modularity_min": float(min(r["modularity"] for r in runs)),
            "modularity_max": float(max(r["modularity"] for r in runs)),
            "n_communities_values": sorted({r["n_communities"] for r in runs})}


def best_run(runs: list[dict]) -> dict:
    """Представницький розбиття: найвища модулярність, за рівності менший seed."""
    return sorted(runs, key=lambda r: (-r["modularity"], r["seed"]))[0]


def cross_group(edges: pd.DataFrame, group: dict, count: bool = False) -> dict:
    """Частка ваги (або кількості ребер) між групами. group: channel_id до bool (True = канал з наративами)."""
    s = edges["src_channel_id"].map(group)
    d = edges["dst_channel_id"].map(group)
    if s.isna().any() or d.isna().any():
        raise ValueError("є канали без групи")
    w = pd.Series(1.0, index=edges.index) if count else edges["weight"].astype(float)
    tot = float(w.sum())
    cell = {f"{'n' if a else 'nn'}_to_{'n' if b else 'nn'}": float(w[(s == a) & (d == b)].sum())
            for a in (True, False) for b in (True, False)}
    cross = cell["n_to_nn"] + cell["nn_to_n"]
    return {"total": tot, **cell, "cross": cross, "cross_share": cross / tot if tot else float("nan")}


def permutation_cross(edges: pd.DataFrame, group: dict, n_perm: int, seed: int, count: bool = False) -> dict:
    """Спостережена міжгрупова частка проти нульового розподілу (мітки груп перемішуються між каналами мережі).
    p = (k + 1) / (n + 1), де k число перестановок з часткою не більшою за спостережену."""
    rng = np.random.default_rng(seed)
    ids = sorted(set(edges["src_channel_id"]) | set(edges["dst_channel_id"]))
    labels = np.array([group[c] for c in ids])
    pos = {c: i for i, c in enumerate(ids)}
    si, di = edges["src_channel_id"].map(pos).to_numpy(), edges["dst_channel_id"].map(pos).to_numpy()
    w = np.ones(len(edges)) if count else edges["weight"].to_numpy(float)
    tot = w.sum()
    obs = float(w[labels[si] != labels[di]].sum() / tot)
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(labels)
        null[k] = w[p[si] != p[di]].sum() / tot
    return {"observed": obs, "null_mean": float(null.mean()), "null_p05": float(np.quantile(null, 0.05)),
            "p_le_observed": float(((null <= obs).sum() + 1) / (n_perm + 1)), "n_perm": n_perm}


def assortativity(edges: pd.DataFrame, group: dict, count: bool = False) -> float:
    """Номінальна асортативність Ньюмана для спрямованої мережі (r = 1: усі ребра всередині груп, 0: як випадково).

    Рахуємо самі, бо `igraph.assortativity_nominal` у встановленій версії падає на аргументі normalized."""
    s = edges["src_channel_id"].map(group).astype(int).to_numpy()
    d = edges["dst_channel_id"].map(group).astype(int).to_numpy()
    w = np.ones(len(edges)) if count else edges["weight"].to_numpy(float)
    e = np.zeros((2, 2))
    np.add.at(e, (s, d), w)
    e /= e.sum()
    a, b = e.sum(axis=1), e.sum(axis=0)
    tr, ab = float(np.trace(e)), float((a * b).sum())
    return (tr - ab) / (1 - ab) if ab < 1 else float("nan")


def save_gexf(g: ig.Graph, nodes: pd.DataFrame, path: Path) -> None:
    import networkx as nx
    G = nx.DiGraph()
    meta = nodes.set_index("channel_id")
    for v in g.vs:
        c = v["channel_id"]
        attrs = {k: (v_.item() if hasattr(v_, "item") else v_) for k, v_ in meta.loc[c].items()
                 if isinstance(v_, (int, float, str, bool, np.integer, np.floating, np.bool_)) and not (isinstance(v_, float) and np.isnan(v_))}
        G.add_node(int(c), label=str(meta.loc[c, "username"]), **attrs)
    for e in g.es:
        G.add_edge(int(g.vs[e.source]["channel_id"]), int(g.vs[e.target]["channel_id"]), weight=float(e["weight"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, str(path))


def _component_layout(G, seed: int) -> dict:
    """Розкладка по компонентах: найбільша компонента зліва, решта стовпчиком справа, щоб карта не розтягувалась."""
    import networkx as nx

    comps = sorted(nx.connected_components(G), key=lambda c: (-len(c), min(c)))
    pos = {}
    for i, comp in enumerate(comps):
        H = G.subgraph(comp)
        if len(comp) == 1:
            local = {next(iter(comp)): np.zeros(2)}
        else:
            local = nx.spring_layout(H, weight="weight", seed=seed, k=3.6 / np.sqrt(len(comp)), iterations=500)
        arr = np.array(list(local.values()))
        arr = arr - arr.mean(axis=0)
        scale = np.abs(arr).max() or 1.0
        if i == 0:
            radius, centre = 1.0, np.array([0.0, 0.0])
        else:
            radius = 0.10 + 0.05 * min(len(comp), 8)
            centre = np.array([1.9, 1.0 - 0.55 * (i - 1)])
        for n, xy in local.items():
            pos[n] = centre + (np.asarray(xy) - np.mean(list(local.values()), axis=0)) / scale * radius
    return pos


def make_map(g: ig.Graph, nodes: pd.DataFrame, out: Path, seed: int = 42, top_labels: int = 8) -> None:
    """Карта мережі: ліворуч колір за групою каналу (n/nn), праворуч за спільнотою Leiden. Розмір вузла: логарифм сумарної сили."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    meta = nodes.set_index("channel_id")
    G = nx.Graph()
    for v in g.vs:
        G.add_node(v["channel_id"])
    for e in g.es:
        a, b = g.vs[e.source]["channel_id"], g.vs[e.target]["channel_id"]
        w = np.log1p(e["weight"]) + (G[a][b]["weight"] if G.has_edge(a, b) else 0.0)
        G.add_edge(a, b, weight=w)
    pos = _component_layout(G, seed)
    strength = (meta.loc[list(G.nodes), "in_strength"] + meta.loc[list(G.nodes), "out_strength"]).astype(float)
    size = 25 + 28 * np.log1p(strength.to_numpy())
    comps = sorted(nx.connected_components(G), key=lambda c: (-len(c), min(c)))
    lab_ids = set(strength.loc[list(comps[0])].sort_values(ascending=False).head(top_labels).index)
    for c in comps[1:]:
        lab_ids.add(strength.loc[list(c)].sort_values(ascending=False).index[0])
    width = [0.3 + 0.35 * G[a][b]["weight"] for a, b in G.edges]
    plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "savefig.dpi": 150})
    fig, axs = plt.subplots(1, 2, figsize=(14, 7.2))
    group_col = ["#c0392b" if meta.loc[c, "narrative_presence"] else "#7f8c8d" for c in G.nodes]
    cmap = plt.get_cmap("tab10")
    comm_col = [cmap(int(meta.loc[c, "community"]) % 10) for c in G.nodes]
    for ax, cols, title in ((axs[0], group_col, "Група каналу: червоні n (наративні), сірі nn; розмір: log сили"),
                            (axs[1], comm_col, "Спільноти Leiden (основна схема ваг)")):
        nx.draw_networkx_edges(G, pos, ax=ax, width=width, alpha=0.3, edge_color="#34495e")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=size, node_color=cols, linewidths=0.4, edgecolors="white")
        nx.draw_networkx_labels(G, pos, ax=ax, labels={c: str(meta.loc[c, "username"]) for c in lab_ids}, font_size=6.5,
                                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.7})
        ax.set_title(title)
        ax.margins(0.08)
        ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def gold_coverage(gold: pd.DataFrame, node_ids: set, channel_col: str = "channel_id", binary_col: str = "Binary label") -> dict:
    """Скільки золотих повідомлень лежить у каналах, що є вершинами мережі (рівень каналу, не повідомлення)."""
    inn = gold[channel_col].isin(node_ids)
    pos = gold[binary_col].astype(bool)
    return {"n_messages": int(len(gold)), "n_channels": int(gold[channel_col].nunique()),
            "messages_in_graph_channels": int(inn.sum()), "channels_in_graph": int(gold.loc[inn, channel_col].nunique()),
            "narrative_total": int(pos.sum()), "narrative_in_graph_channels": int((inn & pos).sum()),
            "non_narrative_in_graph_channels": int((inn & ~pos).sum())}


# ---------------------------------------------------------------------------------------------------------------
# Збірний запуск S3
# ---------------------------------------------------------------------------------------------------------------
EXPECTED = {"n_rows": 214, "n_nodes": 73, "weight_sum": 34_967, "weight_max": 30_810, "cross_share_plan": 0.0035}
VARIANTS = ("raw", "log", "none")  # raw: ваги = кількість пересилань (основний), log: log(1+w), none: усі ваги 1


def _mem_by_id(g: ig.Graph, membership: list[int]) -> dict:
    return dict(zip(g.vs["channel_id"], membership))


def _renumber_by_size(mem: list[int]) -> list[int]:
    """Номери спільнот за спаданням розміру (0 найбільша), за рівності за найменшим індексом вершини."""
    s = pd.Series(mem)
    order = s.groupby(s).apply(lambda x: (-len(x), x.index.min())).sort_values().index.tolist()
    m = {old: new for new, old in enumerate(order)}
    return [m[x] for x in mem]


def run_all(edges_raw: pd.DataFrame, per_channel: pd.DataFrame, seeds: list[int] | None = None, n_perm: int = 10_000,
            seed: int = 42, strict: bool = True) -> tuple[dict, dict, list, ig.Graph]:
    """Повертає (tables, summary, checks, graph). strict=False вимикає перевірки на конкретні числа таблиці (для тестів)."""
    seeds = list(seeds) if seeds is not None else list(range(seed, seed + 10))
    ce, info = clean_edges(edges_raw)
    g = build_graph(ce)
    pc = per_channel.drop_duplicates("channel_id").set_index("channel_id")
    missing = sorted(set(g.vs["channel_id"]) - set(pc.index))
    if missing:
        raise ValueError(f"канали з ребер відсутні в таблиці каналів: {missing}")
    group = {c: bool(pc.loc[c, "table_narrative_presence"]) for c in g.vs["channel_id"] if c in pc.index}

    names = {c: str(pc.loc[c, "username"]) for c in group}
    ce = ce.assign(src=ce["src_channel_id"].map(names), dst=ce["dst_channel_id"].map(names))
    top = ce.sort_values("weight", ascending=False).iloc[0]
    mirror_ids = {int(top["src_channel_id"]), int(top["dst_channel_id"])}

    # центральності
    cen = centrality(g)
    cen.insert(1, "username", cen["channel_id"].map(names))
    cen["table_label"] = cen["channel_id"].map(pc["table_label"])
    cen["table_status"] = cen["channel_id"].map(pc["table_status"])

    # спільноти: три схеми ваг, по 10 запусків
    var_out, best = {}, {}
    for v in VARIANTS:
        ug = undirected_weights(g, v)
        runs = leiden_runs(ug, seeds)
        st = stability(runs)
        b = best_run(runs)
        var_out[v] = {**st, "best_seed": b["seed"], "best_modularity": b["modularity"], "best_n_communities": b["n_communities"]}
        best[v] = _renumber_by_size(b["membership"])
    cross_variant_nmi = {f"{a}_vs_{b}": float(normalized_mutual_info_score(best[a], best[b])) for a, b in combinations(VARIANTS, 2)}

    comm = pd.DataFrame({"channel_id": g.vs["channel_id"], "username": [names[c] for c in g.vs["channel_id"]],
                         "table_label": [pc.loc[c, "table_label"] for c in g.vs["channel_id"]],
                         "community": best["raw"], "community_log": best["log"], "community_unweighted": best["none"]})
    cen["community"] = cen["channel_id"].map(dict(zip(comm["channel_id"], comm["community"])))

    # зведення по спільнотах (основна схема)
    wsum = ce.assign(sc=ce["src_channel_id"].map(dict(zip(comm["channel_id"], comm["community"]))),
                     dc=ce["dst_channel_id"].map(dict(zip(comm["channel_id"], comm["community"]))))
    rows = []
    for k, sub in comm.groupby("community"):
        ids = set(sub["channel_id"])
        inside = float(wsum[(wsum["sc"] == k) & (wsum["dc"] == k)]["weight"].sum())
        touch = float(wsum[(wsum["sc"] == k) | (wsum["dc"] == k)]["weight"].sum())
        st = cen[cen["channel_id"].isin(ids)].assign(strength=lambda d: d["in_strength"] + d["out_strength"]).sort_values("strength", ascending=False)
        rows.append({"community": int(k), "n_channels": len(sub), "n_group": int((sub["table_label"] == "n").sum()),
                     "nn_group": int((sub["table_label"] == "nn").sum()), "weight_inside": inside,
                     "weight_touching": touch, "inside_share": inside / touch if touch else float("nan"),
                     "top_channels": ", ".join(st["username"].head(3))})
    comm_sum = pd.DataFrame(rows)

    # міжгрупові пересилання, варіанти
    status_ok = {c for c in group if pc.loc[c, "table_status"] == "labeled"}
    ce_ng = ce[~(ce["src_channel_id"].isin(mirror_ids) & ce["dst_channel_id"].isin(mirror_ids))]
    ce_lab = ce[ce["src_channel_id"].isin(status_ok) & ce["dst_channel_id"].isin(status_ok)]
    cg = {
        "main_weight": cross_group(ce, group),
        "no_mirror_pair_weight": cross_group(ce_ng, group),
        "edge_count": cross_group(ce, group, count=True),
        "labeled_status_channels_weight": cross_group(ce_lab, group) if len(ce_lab) else None,
    }
    perm = {
        "main_weight": permutation_cross(ce, group, n_perm, seed),
        "no_mirror_pair_weight": permutation_cross(ce_ng, group, n_perm, seed),
        "edge_count": permutation_cross(ce, group, n_perm, seed, count=True),
    }
    assort = {"weight": assortativity(ce, group), "no_mirror_pair_weight": assortativity(ce_ng, group),
              "edge_count": assortativity(ce, group, count=True)}
    nmi_group = {v: float(normalized_mutual_info_score([int(group[c]) for c in comm["channel_id"]], comm[col]))
                 for v, col in (("raw", "community"), ("log", "community_log"), ("none", "community_unweighted"))}

    n_nodes_group = {"n": int(sum(group.values())), "nn": int(len(group) - sum(group.values()))}
    summary = {
        "edges": info, "n_nodes": g.vcount(), "n_edges": g.ecount(), "weight_sum": float(ce["weight"].sum()),
        "weight_max": float(ce["weight"].max()), "mirror_pair": {"src": top["src"], "dst": top["dst"], "weight": float(top["weight"]),
                                                               "share_of_weight": float(top["weight"] / ce["weight"].sum())},
        "nodes_by_group": n_nodes_group, "n_weak_components": int(cen["component"].nunique()),
        "largest_component_nodes": int(cen["component"].value_counts().iloc[0]),
        "src_nodes": int(ce["src_channel_id"].nunique()), "dst_nodes": int(ce["dst_channel_id"].nunique()),
        "leiden": var_out, "cross_variant_nmi": cross_variant_nmi, "nmi_partition_vs_group": nmi_group,
        "cross_group": cg, "cross_group_permutation": perm, "assortativity": assort,
        "seeds": seeds,
    }

    ok_nmi = var_out["raw"]["nmi_min"] > 0.8
    checks = [
        {"hard": True, "check": "Усі вершини графа є в таблиці каналів", "passed": not missing, "detail": f"немає в per_channel: {missing}" if missing else f"{g.vcount()} вершин"},
        {"hard": True, "check": "Ваги ребер додатні цілі", "passed": bool((ce["weight"] > 0).all() and (ce["weight"] % 1 == 0).all()), "detail": f"min {ce['weight'].min():.0f}, max {ce['weight'].max():.0f}"},
        {"hard": True, "check": "Сума in-сил = сума out-сил = сума ваг", "passed": bool(np.isclose(cen["in_strength"].sum(), ce["weight"].sum()) and np.isclose(cen["out_strength"].sum(), ce["weight"].sum())),
         "detail": f"in {cen['in_strength'].sum():.0f}, out {cen['out_strength'].sum():.0f}, ваги {ce['weight'].sum():.0f}"},
        {"hard": True, "check": "PageRank сумується в 1", "passed": bool(np.isclose(cen["pagerank_forwarder"].sum(), 1) and np.isclose(cen["pagerank_source"].sum(), 1)),
         "detail": f"{cen['pagerank_forwarder'].sum():.6f} і {cen['pagerank_source'].sum():.6f}"},
        {"hard": True, "check": "Кожна вершина має спільноту, спільнот більше однієї", "passed": bool(comm["community"].notna().all() and comm["community"].nunique() > 1), "detail": f"{comm['community'].nunique()} спільнот"},
        {"hard": True, "check": "Стабільність Leiden: мінімальний попарний NMI більший за 0,8 (основна схема ваг)", "passed": ok_nmi,
         "detail": f"mean {var_out['raw']['nmi_mean']:.3f}, min {var_out['raw']['nmi_min']:.3f}, {len(seeds)} запусків"},
        {"hard": False, "check": "Стабільність Leiden: те саме для log-ваг і незважених", "passed": bool(var_out["log"]["nmi_min"] > 0.8 and var_out["none"]["nmi_min"] > 0.8),
         "detail": f"log min {var_out['log']['nmi_min']:.3f}, none min {var_out['none']['nmi_min']:.3f}"},
    ]
    if strict:
        checks += [
            {"hard": True, "check": "Таблиця ребер збігається з перевіреним перенесенням з Drive (214 рядків, 73 вершини, сума ваг 34 967, максимум 30 810)",
             "passed": bool(info["n_input"] == EXPECTED["n_rows"] and g.vcount() == EXPECTED["n_nodes"] and ce["weight"].sum() == EXPECTED["weight_sum"] and ce["weight"].max() == EXPECTED["weight_max"]),
             "detail": f"{info['n_input']} рядків, {g.vcount()} вершин, сума {ce['weight'].sum():.0f}, максимум {ce['weight'].max():.0f}"},
            {"hard": False, "check": "Частка міжгрупових пересилань близька до очікуваних 0,35% з плану (допуск 0,05 в.п.)",
             "passed": bool(abs(cg["main_weight"]["cross_share"] - EXPECTED["cross_share_plan"]) <= 0.0005), "detail": f"{cg['main_weight']['cross_share'] * 100:.3f}%"},
        ]
    tables = {"centrality": cen.sort_values("channel_id").reset_index(drop=True), "communities": comm, "community_summary": comm_sum}
    return tables, summary, checks, g
