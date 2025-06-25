#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import argparse
import math
import matplotlib
import collections
# PyQt5 backend を指定
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from typing import List, Dict, Tuple
import visualize
from utils import (
    parse_line,
    build_edge_pairs_fixed,
    decompose,
    adjust_alternating_rotation
)

def parse_args():
    p = argparse.ArgumentParser(
        description='Quartic planar graph の可視化＆ストリング分解'
    )
    p.add_argument('input', nargs='?', type=argparse.FileType('r'),
                   default=sys.stdin,
                   help='入力ファイル (デフォルト: 標準入力)')
    p.add_argument('-l', '--layout', choices=['planar','spring'],
                   default='planar',
                   help='レイアウト方式 (default: planar)')
    p.add_argument('-o', '--outdir', metavar='DIR',
                   help='図を保存するディレクトリ (指定なければ画面表示のみ)')
    p.add_argument('--noshow', action='store_true',
                   help='画面表示せずファイル保存のみ行う')
    return p.parse_args()

def draw_multiedge_with_labels(G, pos, edgeid_to_color=None):
    import matplotlib
    import math
    from collections import defaultdict

    ax = plt.gca()
    # group edges by unordered node‐pair
    groups = defaultdict(list)
    for u, v, k in G.edges(keys=True):
        groups[tuple(sorted((u, v)))].append((u, v, k))

    for (u, v), es in groups.items():
        es_sorted = sorted(es, key=lambda x: x[2])
        m = len(es_sorted)

        # multiedge のための「符号」列: 2 本なら [-1,+1], 3 本なら [-2,0,+2] ...
        signs = [i for i in range(-(m-1), m, 2)]

        # ノード間距離を取得
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dist = math.hypot(x2-x1, y2-y1)

        # 基本曲率係数 (距離 1 のときの曲がり具合)
        base_curvature = 0.2

        # scale は距離に応じて小さくなるように設定
        # （距離が 2 なら半分、距離が 0.5 なら 2 倍）
        scale = base_curvature / (dist + 1e-6)

        for (u2, v2, k), s in zip(es_sorted, signs):
            # 色
            color = edgeid_to_color.get(k, 'black') if edgeid_to_color else 'black'

            # 曲率は符号 × scale
            rad = s * scale * 0.3

            patch = matplotlib.patches.FancyArrowPatch(
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

            # compute label position: midpoint plus small offset perpendicular to edge
            xm = (pos[u2][0] + pos[v2][0]) / 2
            ym = (pos[u2][1] + pos[v2][1]) / 2
            dx = pos[v2][1] - pos[u2][1]
            dy = -(pos[v2][0] - pos[u2][0])
            norm = math.hypot(dx, dy) or 1
            # ensure even rad=0 (straight) edges get a little offset
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



"""
def visualize_and_decompose(line: str, idx: int, cfg):
    #もとの関数に「交互 over/under 回転へ調整」ステップを追加。
    #それ以外のロジックは変更していない。

    import matplotlib.colors as mcolors
    # --- 前半部は元のまま ---
    neigh = parse_line(line)
    e_pair, eid_of, eid2info = build_edge_pairs_fixed(neigh)

    v_pair = {}
    edge_id_lists = []
    print(f"--- [{idx}] {line.strip()} ---")
    print("=== Dart configuration per vertex ===")
    for u, nb in enumerate(neigh):
        darts    = [u * 4 + i for i in range(4)]
        edge_ids = [eid_of[d] for d in darts]
        runs = []
        start = 0
        for i in range(1,4):
            if nb[i] != nb[i-1]:
                if i - start > 1:
                    runs.append((start,i))
                start = i
        if 4 - start > 1:
            runs.append((start,4))
        for s,e in runs:
            if u > nb[s]:
                edge_ids[s:e] = edge_ids[s:e][::-1]
                darts   [s:e] = darts   [s:e][::-1]
        edge_id_lists.append(edge_ids)
        for i,d in enumerate(darts):
            v_pair[d] = darts[(i+2)%4]

    # string 分解
    loops = decompose(e_pair, eid_of, eid2info, v_pair)

    # === ここで交互 over/under になるよう回転順序を調整 ===
    edge_id_lists = adjust_alternating_rotation(edge_id_lists, loops, eid2info)

    # --- 以下，元の可視化・出力部はそのまま ---
    print("== 全頂点のedges (cyclic) 一覧 ==")
    print(edge_id_lists)

    lengths = sorted([len(L) for L in loops], reverse=True)
    print(f"→ Found {len(loops)} loops: lengths = {lengths}")
    for i, L in enumerate(loops, 1):
        path = " → ".join(f"{eid2info[e][0]}-{eid2info[e][1]}[{e}]" for e in L)
        print(f"   Loop {i}: {path}")

    def loop_to_vertex_seq(loop):
        verts = []
        for k, eid in enumerate(loop):
            u, v = eid2info[eid]
            if k == 0:
                verts.extend([u, v])
            else:
                if verts[-1] == u:
                    verts.append(v)
                else:
                    verts.append(u)
        if verts[-1] == verts[0]:
            verts = verts[:-1]
        return verts

    string_vertex_lists = [loop_to_vertex_seq(loop) for loop in loops]
    n = len(neigh)
    string_nums_per_vertex = [[] for _ in range(n)]
    string_idx_per_vertex = [[] for _ in range(n)]
    for s_idx, verts in enumerate(string_vertex_lists):
        for v in verts:
            string_nums_per_vertex[v].append(f's{s_idx}')
            string_idx_per_vertex[v].append(s_idx)
    string_lengths = [len(verts) for verts in string_vertex_lists]

    # グラフ作成
    G = nx.MultiGraph()
    G.add_nodes_from(range(n))
    for d, eid in eid_of.items():
        u = d // 4
        u2, v2 = eid2info[eid]
        v = v2 if u2 == u else u2
        G.add_edge(u, v, key=eid)

    # 2重辺カウント
    double_edge_neighbor_count = [0]*n
    for u in range(n):
        neighbors = set()
        for v in G.neighbors(u):
            if G.number_of_edges(u, v) == 2:
                neighbors.add(v)
        double_edge_neighbor_count[u] = len(neighbors)

    string_len_key = []
    for idxs in string_idx_per_vertex:
        uniq = set(idxs)
        if len(uniq) == 1:
            # 同じサイクルが２本分通っている頂点
            l = string_lengths[idxs[0]]
            key = (l, l)
        else:
            # 異なるサイクル
            lens = sorted(string_lengths[s] for s in uniq)
            key = tuple(lens)
        string_len_key.append(key)

    # 初期カラー（第0ラウンド）
    colors = [tuple([string_len_key[v][0]] if len(string_len_key[v])==1 else list(string_len_key[v]))
              for v in range(n)]

    # WLライクに k ラウンド反復
    # k は図の「直径」もしくはサイクル長の最大値より大きめに
    k = max(string_lengths) + 1
    for _ in range(k):
        new_colors = []
        for v in range(n):
            # 自分のカラー＋隣接ノードのカラーソート
            neigh_cols = sorted(colors[w] for w in G.neighbors(v))
            new_colors.append(tuple([colors[v]] + neigh_cols))
        colors = new_colors

    # 最終的な色を hash して数字化（任意）
    vertex_hash_list = [hash(c) for c in colors]

    # (従来の “第2ハッシュ” は不要になります)
    second_hash_list = vertex_hash_list.copy()
    # ─────── 書き換えここまで ───────

    print("=== 各頂点ごとのstring所属番号リスト, 2重辺数, 最終カラーキー（ハッシュ） ===")
    for v in range(n):
        print(f"  頂点{v}: {string_nums_per_vertex[v]} {double_edge_neighbor_count[v]} {second_hash_list[v]}")
    
    # --- string (ループ) ごとの色決定 ---
    loop_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.BASE_COLORS.values())
    edgeid_to_color = {}
    for idx, loop in enumerate(loops):
        color = loop_palette[idx % len(loop_palette)]
        for eid in loop:
            edgeid_to_color[eid] = color

    # 2次ハッシュでノード色分け
    node_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.BASE_COLORS.values())
    unique_hashes = list(sorted(set(second_hash_list)))
    hash_to_color = {h: node_palette[i % len(node_palette)] for i, h in enumerate(unique_hashes)}
    node_colors = [hash_to_color[h] for h in second_hash_list]

    labels = {v: str(v) for v in range(n)}

    pos = nx.planar_layout(G) if cfg.layout == 'planar' else nx.spring_layout(G)
    fig = plt.figure(figsize=(5,5))
    plt.title(line.strip())
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, labels)

    # 曲がった多重辺＋string色で描画
    draw_multiedge_with_labels(G, pos, edgeid_to_color=edgeid_to_color)

    plt.axis('off')

    if cfg.outdir:
        os.makedirs(cfg.outdir, exist_ok=True)
        fname = os.path.join(cfg.outdir, f'graph_{idx:03d}.png')
        fig.savefig(fname, dpi=150)
        print(f"[Saved] {fname}", file=sys.stderr)
    if not cfg.noshow:
        plt.show()
    plt.close(fig)
"""

def refine_by_edge_multiplicity(
        G: nx.MultiGraph,
        max_rounds: int = None,
        simple_mode: bool = False
) -> List[int]:
    """
    頂点を
       (a) 2重辺の隣人個数
       (b) そのとき繋がる 1 重 / 2 重 辺の本数分布
    で細分化し、安定化するまで反復する。

    戻り値: 各頂点 v に対応する整数クラス ID
    """
    n = G.number_of_nodes()
    # (0) 初期クラス = 2重辺を持つ隣人の数
    class_id = []
    for v in G.nodes():
        double_neighbors = sum(
            1 for u in G.neighbors(v) if G.number_of_edges(v, u) == 2
        )
        class_id.append(double_neighbors)

    # ラウンド上限
    if simple_mode:
        R = 3
    else:
        # 「頂点数+1」程度あれば確実に安定
        R = n + 1 if max_rounds is None else max_rounds
    # (1) 反復細分化
    for rnd in range(R):
        # 各頂点のシグネチャを作る
        sigs: List[Tuple] = []
        for v in G.nodes():
            single_cnt = collections.Counter()
            double_cnt = collections.Counter()
            for u in G.neighbors(v):
                mult = G.number_of_edges(v, u)
                g = class_id[u]
                if mult == 1:
                    single_cnt[g] += 1
                else:            # mult == 2
                    double_cnt[g] += 1
            # シグネチャ: (自分のクラス,
            #              sorted((class,本数) for single),
            #              '|' 区切り,
            #              sorted((class,本数) for double))
            sig = (class_id[v],)
            sig += tuple(sorted(single_cnt.items()))
            sig += ('|',)       # 区切りマーカー
            sig += tuple(sorted(double_cnt.items()))
            sigs.append(sig)

        # (2) 同じシグネチャ→同じ新 ID
        sig2id: Dict[Tuple, int] = {}
        new_class_id = []
        for s in sigs:
            if s not in sig2id:
                sig2id[s] = len(sig2id)
            new_class_id.append(sig2id[s])

        if new_class_id == class_id:
            print(f"  [refine] 安定: {rnd+1} ラウンドで終了")
            break
        class_id = new_class_id
    else:
        print(f"  [refine] 最大 {R} ラウンドまで実行")

    return class_id

def visualize_and_decompose(line: str, idx: int, cfg):
    """
    元の visualize_and_decompose に
      • adjust_alternating_rotation ステップ
      • WL風反復カラーリング（早期停止＋簡易モード）
    をまとめて組み込んだもの
    """
    import os
    import sys
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # --- 前半部（元のまま） ---
    neigh = parse_line(line)
    e_pair, eid_of, eid2info = build_edge_pairs_fixed(neigh)

    # v_pair と edge_id_lists の構築
    v_pair = {}
    edge_id_lists = []
    print(f"--- [{idx}] {line.strip()} ---")
    print("=== Dart configuration per vertex ===")
    for u, nb in enumerate(neigh):
        darts    = [u * 4 + i for i in range(4)]
        edge_ids = [eid_of[d] for d in darts]
        runs = []
        start = 0
        for i in range(1,4):
            if nb[i] != nb[i-1]:
                if i - start > 1:
                    runs.append((start,i))
                start = i
        if 4 - start > 1:
            runs.append((start,4))
        for s,e in runs:
            if u > nb[s]:
                edge_ids[s:e] = edge_ids[s:e][::-1]
                darts   [s:e] = darts   [s:e][::-1]
        edge_id_lists.append(edge_ids)
        for i,d in enumerate(darts):
            v_pair[d] = darts[(i+2)%4]

    # string 分解
    loops = decompose(e_pair, eid_of, eid2info, v_pair)

    # 回転順序調整
    edge_id_lists = adjust_alternating_rotation(edge_id_lists, loops, eid2info)

    # 可視化・出力部の準備
    print("== 全頂点のedges (cyclic) 一覧 ==")
    print(edge_id_lists)
    lengths = sorted([len(L) for L in loops], reverse=True)
    print(f"→ Found {len(loops)} loops: lengths = {lengths}")
    for i, L in enumerate(loops, 1):
        path = " → ".join(f"{eid2info[e][0]}-{eid2info[e][1]}[{e}]" for e in L)
        print(f"   Loop {i}: {path}")

    def loop_to_vertex_seq(loop):
        verts = []
        for k, eid in enumerate(loop):
            u, v = eid2info[eid]
            if k == 0:
                verts.extend([u, v])
            else:
                verts.append(v if verts[-1] == u else u)
        if verts[-1] == verts[0]:
            verts.pop()
        return verts

    string_vertex_lists = [loop_to_vertex_seq(loop) for loop in loops]
    n = len(neigh)
    string_nums_per_vertex = [[] for _ in range(n)]
    string_idx_per_vertex  = [[] for _ in range(n)]
    for s_idx, verts in enumerate(string_vertex_lists):
        for v in verts:
            string_nums_per_vertex[v].append(f's{s_idx}')
            string_idx_per_vertex[v].append(s_idx)
    string_lengths = [len(verts) for verts in string_vertex_lists]

    # MultiGraph 作成
    G = nx.MultiGraph()
    G.add_nodes_from(range(n))
    for d, eid in eid_of.items():
        u = d // 4
        u2, v2 = eid2info[eid]
        v = v2 if u2 == u else u2
        G.add_edge(u, v, key=eid)

    
    # 2重辺カウント
    double_edge_neighbor_count = [0]*n
    for u in range(n):
        neighbors = set()
        for v in G.neighbors(u):
            if G.number_of_edges(u, v) == 2:
                neighbors.add(v)
        double_edge_neighbor_count[u] = len(neighbors)

    """
    # ─── WLライク色分け: 初期キー構築 ───
    string_len_key = []
    for idxs in string_idx_per_vertex:
        uniq = set(idxs)
        if len(uniq) == 1:
            # 同一サイクル２本分 → (l,l)
            l = string_lengths[idxs[0]]
            string_len_key.append((l, l))
        else:
            # 異なるサイクル → ソート済みタプル
            lens = sorted(string_lengths[s] for s in uniq)
            string_len_key.append(tuple(lens))

    # 初期カラー
    colors = [key for key in string_len_key]

    # 繰り返し回数設定（簡易モード or フルモード）
    if getattr(cfg, 'simple_mode', False):
        max_iter = 3
    else:
        max_iter = max(string_lengths, default=0) + 1

    # WLライク反復 + 早期停止
    for i in range(max_iter):
        new_colors = []
        for v in range(n):
            neigh_cols = sorted(colors[w] for w in G.neighbors(v))
            new_colors.append(tuple([colors[v]] + neigh_cols))
        if new_colors == colors:
            print(f"カラー安定：{i+1} ラウンドで終了")
            break
        colors = new_colors
    else:
        print(f"最大 {max_iter} ラウンド実行")

    """
    # 最終ハッシュ化（描画用に数値化）
    second_hash_list = refine_by_edge_multiplicity(
        G,
        simple_mode=getattr(cfg, 'simple_mode', False)
)
    #second_hash_list = [hash(c) for c in colors]

    print("=== 各頂点ごとのstring所属番号, 2重辺数, カラーキー(ハッシュ) ===")
    for v in range(n):
        print(f"  頂点{v}: {string_nums_per_vertex[v]} "
              f"{double_edge_neighbor_count[v]} {second_hash_list[v]}")

    # --- 描画部 ---
    # Loopごとの色
    loop_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.BASE_COLORS.values())
    edgeid_to_color = {}
    for idx, loop in enumerate(loops):
        color = loop_palette[idx % len(loop_palette)]
        for eid in loop:
            edgeid_to_color[eid] = color

    # Node色
    node_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.BASE_COLORS.values())
    unique_hashes = sorted(set(second_hash_list))
    hash_to_color = {h: node_palette[i % len(node_palette)] for i, h in enumerate(unique_hashes)}
    node_colors = [hash_to_color[h] for h in second_hash_list]

    labels = {v: str(v) for v in range(n)}
    pos = nx.planar_layout(G) if cfg.layout == 'planar' else nx.spring_layout(G)

    fig = plt.figure(figsize=(5,5))
    plt.title(line.strip())
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, labels)
    draw_multiedge_with_labels(G, pos, edgeid_to_color=edgeid_to_color)
    plt.axis('off')

    if cfg.outdir:
        os.makedirs(cfg.outdir, exist_ok=True)
        fname = os.path.join(cfg.outdir, f'graph_{idx:03d}.png')
        fig.savefig(fname, dpi=150)
        print(f"[Saved] {fname}", file=sys.stderr)
    if not getattr(cfg, 'noshow', False):
        plt.show()
    plt.close(fig)



def main():
    cfg = parse_args()
    for i, raw in enumerate(cfg.input, 1):
        line = raw.strip()
        if not line: 
            continue
        visualize_and_decompose(line, i, cfg)

if __name__ == '__main__':
    main()
