import os
import sys
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import utils

# 以下の関数は別モジュール(strings.py など)で定義されていることを前提
from strings import (
    parse_line,
    build_edge_pairs_fixed,
    decompose,
    adjust_alternating_rotation
)

__all__ = [
    'draw_multiedge_with_labels',
    'visualize_and_decompose',
]

def draw_multiedge_with_labels(G, pos, edgeid_to_color=None):
    """
    NetworkX MultiGraph G と座標 dict pos を受け取り、
    multi-edge を曲線で描画、edge key のラベルを表示します。

    edgeid_to_color: {edge_key: color} の辞書
    """
    ax = plt.gca()
    # ノードペアごとにエッジをグループ化
    groups = defaultdict(list)
    for u, v, k in G.edges(keys=True):
        groups[tuple(sorted((u, v)))].append((u, v, k))

    for (u, v), es in groups.items():
        # エッジをキー順にソート
        es_sorted = sorted(es, key=lambda x: x[2])
        m = len(es_sorted)
        # 曲率符号: 2本なら[-1,+1], 3本なら[-2,0,+2]...
        signs = list(range(-(m-1), m, 2))

        # ノード間の距離を計算
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dist = math.hypot(x2-x1, y2-y1)

        base_curvature = 0.2
        scale = base_curvature / (dist + 1e-6)

        for (u2, v2, k), s in zip(es_sorted, signs):
            color = edgeid_to_color.get(k, 'black') if edgeid_to_color else 'black'
            rad = s * scale * 0.3
            patch = mpatches.FancyArrowPatch(
                pos[u2], pos[v2],
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle='-',
                mutation_scale=10,
                linewidth=2.0,
                color=color,
                shrinkA=0, shrinkB=0,
                zorder=1
            )
            ax.add_patch(patch)

            # ラベル位置を中点＋法線方向オフセット
            xm = (pos[u2][0] + pos[v2][0]) / 2
            ym = (pos[u2][1] + pos[v2][1]) / 2
            dx = pos[v2][1] - pos[u2][1]
            dy = -(pos[v2][0] - pos[u2][0])
            norm = math.hypot(dx, dy) or 1
            off = 0.01 if rad == 0 else rad * 0.5

            ax.text(
                xm + dx / norm * off,
                ym + dy / norm * off,
                str(k),
                fontsize=12,
                ha="center", va="center",
                clip_on=False,
                zorder=2
            )


def visualize_and_decompose(line: str, idx: int, cfg):
    """
    文字列 line をパースしてループ分解し、NetworkX グラフを生成して可視化。
    cfg.layout に応じたレイアウト、cfg.outdir, cfg.noshow で保存/表示制御。
    """
    # --- 1. パース & 分解 ---
    neigh = parse_line(line)
    e_pair, eid_of, eid2info = build_edge_pairs_fixed(neigh)

    # Dart configuration
    print(f"--- [{idx}] {line.strip()} ---")
    edge_id_lists = []
    v_pair = {}
    for u, nb in enumerate(neigh):
        darts = [u * 4 + i for i in range(4)]
        edge_ids = [eid_of[d] for d in darts]
        runs = []
        start = 0
        for i in range(1, 4):
            if nb[i] != nb[i-1]:
                if i - start > 1:
                    runs.append((start, i))
                start = i
        if 4 - start > 1:
            runs.append((start, 4))
        for s, e in runs:
            if u > nb[s]:
                edge_ids[s:e] = edge_ids[s:e][::-1]
                darts[s:e]   = darts[s:e][::-1]
        edge_id_lists.append(edge_ids)
        for i, d in enumerate(darts):
            v_pair[d] = darts[(i + 2) % 4]

    loops = decompose(e_pair, eid_of, eid2info, v_pair)

    # --- 2. over/under が交互になるよう順序調整 ---
    edge_id_lists = adjust_alternating_rotation(edge_id_lists, loops, eid2info)

    # --- 3. 分解結果を表示 ---
    print("== 全頂点のedges (cyclic) 一覧 ==")
    print(edge_id_lists)
    lengths = sorted([len(L) for L in loops], reverse=True)
    print(f"→ Found {len(loops)} loops: lengths = {lengths}")
    for i, L in enumerate(loops, 1):
        path = " → ".join(f"{eid2info[e][0]}-{eid2info[e][1]}[{e}]" for e in L)
        print(f"   Loop {i}: {path}")

    # 4. ループごとの頂点シーケンス
    def loop_to_vertex_seq(loop):
        verts = []
        for k, eid in enumerate(loop):
            u, v = eid2info[eid]
            if k == 0:
                verts.extend([u, v])
            else:
                verts.append(v if verts[-1] == u else u)
        if verts[-1] == verts[0]:
            verts = verts[:-1]
        return verts

    string_vertex_lists = [loop_to_vertex_seq(L) for L in loops]
    n = len(neigh)
    # 頂点ごとの string 所属情報とハッシュを計算
    string_nums_per_vertex = [[] for _ in range(n)]
    string_idx_per_vertex = [[] for _ in range(n)]
    for s_idx, verts in enumerate(string_vertex_lists):
        for v in verts:
            string_nums_per_vertex[v].append(f's{s_idx}')
            string_idx_per_vertex[v].append(s_idx)
    string_lengths = [len(v) for v in string_vertex_lists]

    G = nx.MultiGraph()
    G.add_nodes_from(range(n))
    for d, eid in eid_of.items():
        u = d // 4
        u2, v2 = eid2info[eid]
        v = v2 if u2 == u else u2
        G.add_edge(u, v, key=eid)

    # 2重辺の隣接数
    double_edge_neighbor_count = []
    for u in range(n):
        neighbors = set(v for v in G.neighbors(u) if G.number_of_edges(u, v) == 2)
        double_edge_neighbor_count.append(len(neighbors))

    # 1次/2次ハッシュ
    string_len_hashkey_per_vertex = []
    for idxs in string_idx_per_vertex:
        if len(idxs) == 2 and idxs[0] == idxs[1]:
            l = string_lengths[idxs[0]]
            string_len_hashkey_per_vertex.append((l*l,))
        else:
            string_len_hashkey_per_vertex.append(tuple(sorted(string_lengths[s] for s in idxs)))
    vertex_hash_list = [hash(key + (double_edge_neighbor_count[i],)) for i, key in enumerate(string_len_hashkey_per_vertex)]
    second_hash_list = []
    for v in range(n):
        hashes = [vertex_hash_list[nbr] for nbr in G.neighbors(v)] + [vertex_hash_list[v]]
        second_hash_list.append(hash(tuple(sorted(hashes))))

    print("=== 各頂点ごとのstring所属番号リスト, 2重辺数, 1次ハッシュ, 2次ハッシュ ===")
    for v in range(n):
        print(f"  頂点{v}: {string_nums_per_vertex[v]} {double_edge_neighbor_count[v]} {vertex_hash_list[v]} {second_hash_list[v]}")

    # ループ色＆ノード色を決定
    loop_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.BASE_COLORS.values())
    edgeid_to_color = {}
    for idx_loop, L in enumerate(loops):
        color = loop_palette[idx_loop % len(loop_palette)]
        for eid in L:
            edgeid_to_color[eid] = color

    node_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.BASE_COLORS.values())
    unique_hashes = sorted(set(second_hash_list))
    hash_to_color = {h: node_palette[i % len(node_palette)] for i, h in enumerate(unique_hashes)}
    node_colors = [hash_to_color[h] for h in second_hash_list]

    # レイアウトに応じて座標取得
    pos = nx.planar_layout(G) if cfg.layout == 'planar' else nx.spring_layout(G)

    fig = plt.figure(figsize=(5,5))
    plt.title(line.strip())
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, labels={v: str(v) for v in G.nodes()})

    # multi-edge を曲線で描画
    draw_multiedge_with_labels(G, pos, edgeid_to_color=edgeid_to_color)

    plt.axis('off')

    # 保存・表示制御
    if getattr(cfg, 'outdir', None):
        os.makedirs(cfg.outdir, exist_ok=True)
        fname = os.path.join(cfg.outdir, f'graph_{idx:03d}.png')
        fig.savefig(fname, dpi=150)
        print(f"[Saved] {fname}", file=sys.stderr)
    if not getattr(cfg, 'noshow', False):
        plt.show()
    plt.close(fig)
