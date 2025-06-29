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
import matplotlib.colors as mcolors
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
    import json, math
    import networkx as nx
    from utils import (
        parse_line,
        build_edge_pairs_fixed,
        decompose,
        adjust_alternating_rotation
    )

    # --- (A) 元の PD_CODE 相当を計算 ---
    neigh = parse_line(line)
    e_pair, eid_of, eid2info = build_edge_pairs_fixed(neigh)

    # --- (B) Dart configuration per vertex (v_pair, edge_id_lists) ---
    v_pair = {}
    edge_id_lists = []
    for u, nb in enumerate(neigh):
        darts    = [u * 4 + i for i in range(4)]
        edge_ids = [eid_of[d] for d in darts]
        # runs を検出して必要なら反転
        runs = []; start = 0
        for i in range(1,4):
            if nb[i] != nb[i-1]:
                if i - start > 1: runs.append((start,i))
                start = i
        if 4 - start > 1: runs.append((start,4))
        for s,e in runs:
            if u > nb[s]:
                edge_ids[s:e] = edge_ids[s:e][::-1]
                darts   [s:e] = darts   [s:e][::-1]
        edge_id_lists.append(edge_ids)
        for i,d in enumerate(darts):
            v_pair[d] = darts[(i+2)%4]

    

    # --- (C) ループ分解＆over/under 調整 ---
    loops = decompose(e_pair, eid_of, eid2info, v_pair)
    edge_id_lists = adjust_alternating_rotation(edge_id_lists, loops, eid2info)

    # --- (D) 交差用 z_flags 自動計算 (vertex, neighbor)ごと ---
    def loop_vertices(loop):
        verts = []
        for k, eid in enumerate(loop):
            u, v = eid2info[eid]
            if k == 0:
                verts.extend([u, v])
            else:
                verts.append(v if verts[-1] == u else u)
        return verts[:-1]

    temp_rot = [lst.copy() for lst in edge_id_lists]
    z_flags_dict = {}
    for loop in loops:
        verts = loop_vertices(loop)
        parity = 0  # 0→over(+1), 1→under(-1)
        for eid, v, v_next in zip(loop, verts, verts[1:] + [verts[0]]):
            rot = temp_rot[v]
            idx0 = rot.index(eid)
            for sh in range(4):
                if ((idx0 - sh) % 4) % 2 == parity:
                    temp_rot[v] = rot[sh:] + rot[:sh]
                    break
            # 頂点vからv_nextへ抜けるこのstringの通過で over/under を記録
            # キーは "v-v_next" 文字列や、(v, v_next)タプル等で
            key = f"{eid}-{v}"
            z_flags_dict[key] = +1 if parity == 0 else -1
            key_vnext = f"{eid}-{v_next}"
            z_flags_dict[key_vnext] = +1 if parity == 1 else -1
            parity ^= 1 
    # --- (E) レイアウト取得 ---
    G = nx.MultiGraph()
    G.add_nodes_from(range(len(neigh)))
    for d, eid in eid_of.items():
        u = d//4
        u2,v2 = eid2info[eid]
        v  = v2 if u2==u else u2
        G.add_edge(u, v, key=eid)
    pos = nx.planar_layout(G) if cfg.layout=='planar' else nx.spring_layout(G)

    n = len(neigh)
    # 2重辺カウント
    double_edge_neighbor_count = [0]*n
    for u in range(n):
        neighbors = set()
        for v in G.neighbors(u):
            if G.number_of_edges(u, v) == 2:
                neighbors.add(v)
        double_edge_neighbor_count[u] = len(neighbors)

    # --- (F) JSON 出力 ---
    # 1) メインループ（辺ID列）→頂点列に変換
    def loop_to_vertex_seq(loop):
        verts = []
        for k, eid in enumerate(loop):
            u, v = eid2info[eid]
            if k == 0:
                verts.extend([u, v])
            else:
                verts.append(v if verts[-1] == u else u)
        return verts[:-1]

    main_loop = max(loops, key=len)
    verts      = loop_to_vertex_seq(main_loop)

    # 2) 頂点⇄辺を交互に並べたループ列を作成
    vertex_edge_route = []
    for i, eid in enumerate(main_loop):
        vertex_edge_route.append(verts[i])
        vertex_edge_route.append(eid)
    #vertex_edge_route.append(verts[0])

    # 3) 辺ID→端点ペアを文字列キーで準備
    eid2verts = { str(eid): [u, v] for eid, (u, v) in eid2info.items() }

    preferred_signs = {}
    # 各（u, v）の全多重辺ごとに
    groups = collections.defaultdict(list)
    for u, v, k in G.edges(keys=True):
        key = tuple(sorted((u, v)))
        groups[key].append(k)
    for (u, v), eids in groups.items():
        # edge idごとに昇順でソート（描画順と揃える）
        es_sorted = sorted([(u, v, eid) for eid in eids], key=lambda x: x[2])
        m = len(es_sorted)
        signs = [i for i in range(-(m-1), m, 2)]
        # 枝番号ごとに符号を割り当て
        for idx, ((u2, v2, eid), sign) in enumerate(zip(es_sorted, signs)):
            # 小さい方から大きい方への向きで+1/-1/0を決定
            if u2 > v2:
                sign = -sign
            preferred_signs[str(eid)] = sign

    # 出力用データをまとめる
    data = {
      "coords":             { str(v): [float(x), float(y)] for v,(x,y) in pos.items() },
      "vertex_edge_route":  vertex_edge_route,
      "eid2verts":          eid2verts,
      "flat_vertex":        0,  # 必要に応じて変更
      "z_flags":            z_flags_dict,
      "preferred_signs":    preferred_signs
    }

    print(json.dumps(data, ensure_ascii=False, indent=2))


    string_vertex_lists = [loop_to_vertex_seq(loop) for loop in loops]
    
    string_nums_per_vertex = [[] for _ in range(n)]
    string_idx_per_vertex = [[] for _ in range(n)]
    for s_idx, verts in enumerate(string_vertex_lists):
        for v in verts:
            string_nums_per_vertex[v].append(f's{s_idx}')
            string_idx_per_vertex[v].append(s_idx)
    string_lengths = [len(verts) for verts in string_vertex_lists]


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
