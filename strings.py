#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import argparse
import math
import matplotlib
# PyQt5 backend を指定
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

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

def parse_line(line: str):
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f'不正な入力行: {line!r}')
    neigh = [[ord(c) - ord('a') for c in token]
             for token in parts[1].split(',')]
    if any(len(nb) != 4 for nb in neigh):
        raise ValueError(f'各頂点は 4 次正則である必要があります: {line!r}')
    return neigh

def build_edge_pairs_fixed(neigh):
    outgoing = defaultdict(list)
    for u, nb in enumerate(neigh):
        for i, v in enumerate(nb):
            outgoing[(u, v)].append(i)

    e_pair = {}
    eid_of = {}
    eid2info = {}
    eid = 0
    for (u, v), pos_u in outgoing.items():
        if u < v:
            pos_v = outgoing.get((v, u), [])
            if len(pos_u) != len(pos_v):
                raise RuntimeError(f"多重辺数不一致: {(u,v)}")
            for k in range(len(pos_u)):
                du = u * 4 + pos_u[k]
                dv = v * 4 + pos_v[k]
                e_pair[du] = dv
                e_pair[dv] = du
                eid_of[du] = eid_of[dv] = eid
                eid2info[eid] = (u, v)
                eid += 1
    return e_pair, eid_of, eid2info

def decompose(e_pair, eid_of, eid2info, v_pair):
    visited = set()
    loops = []
    for dart in eid_of.keys():
        if dart in visited:
            continue
        cur = dart
        loop = []
        while True:
            visited.add(cur)
            opp = v_pair[cur]
            visited.add(opp)
            loop.append(eid_of[opp])
            cur = e_pair[opp]
            if cur == dart or cur in visited:
                break
        if loop:
            loops.append(loop)
    return loops

def draw_multiedge_with_labels(G, pos):
    ax = plt.gca()
    groups = defaultdict(list)
    for u, v, k in G.edges(keys=True):
        groups[tuple(sorted((u, v)))].append((u, v, k))
    max_m = max(len(es) for es in groups.values())
    rad_list = [i * 0.2 for i in range(-(max_m-1), max_m, 2)]
    for (u, v), es in groups.items():
        es_sorted = sorted(es, key=lambda x: x[2])
        start = (max_m - len(es_sorted)) // 2
        for (u2, v2, k), rad in zip(es_sorted, rad_list[start:start+len(es_sorted)]):
            patch = matplotlib.patches.FancyArrowPatch(
                pos[u2], pos[v2],
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle='-',
                mutation_scale=10,
                linewidth=1.0,
                color='black',
                shrinkA=0, shrinkB=0
            )
            ax.add_patch(patch)
            # ラベル
            xm = (pos[u2][0] + pos[v2][0]) / 2
            ym = (pos[u2][1] + pos[v2][1]) / 2
            dx, dy = pos[v2][1] - pos[u2][1], -(pos[v2][0] - pos[u2][0])
            norm = math.hypot(dx, dy) or 1
            off = rad * 0.5
            plt.text(
                xm + dx/norm*off,
                ym + dy/norm*off,
                str(k),
                fontsize=8, ha="center", va="center"
            )

def draw_multiedge_with_labels(G, pos, edgeid_to_color=None):
    import matplotlib
    ax = plt.gca()
    from collections import defaultdict
    groups = defaultdict(list)
    for u, v, k in G.edges(keys=True):
        groups[tuple(sorted((u, v)))].append((u, v, k))
    max_m = max(len(es) for es in groups.values())
    rad_list = [i * 0.2 for i in range(-(max_m-1), max_m, 2)]
    for (u, v), es in groups.items():
        es_sorted = sorted(es, key=lambda x: x[2])
        start = (max_m - len(es_sorted)) // 2
        for (u2, v2, k), rad in zip(es_sorted, rad_list[start:start+len(es_sorted)]):
            color = 'black'
            if edgeid_to_color is not None:
                color = edgeid_to_color.get(k, 'gray')
            patch = matplotlib.patches.FancyArrowPatch(
                pos[u2], pos[v2],
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle='-',
                mutation_scale=10,
                linewidth=2.0,
                color=color,
                shrinkA=0, shrinkB=0
            )
            ax.add_patch(patch)
            # ラベル
            xm = (pos[u2][0] + pos[v2][0]) / 2
            ym = (pos[u2][1] + pos[v2][1]) / 2
            dx, dy = pos[v2][1] - pos[u2][1], -(pos[v2][0] - pos[u2][0])
            norm = math.hypot(dx, dy) or 1
            off = rad * 0.5
            plt.text(
                xm + dx/norm*off,
                ym + dy/norm*off,
                str(k),
                fontsize=8, ha="center", va="center"
            )

def visualize_and_decompose(line: str, idx: int, cfg):
    import matplotlib.colors as mcolors

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

    print("== 全頂点のedges (cyclic) 一覧 ==")
    print(edge_id_lists)

    # string分解
    loops = decompose(e_pair, eid_of, eid2info, v_pair)
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

    # string番号を用いて: 同じstring2回→[長さ*長さ]、異なるstring→[len1, len2]
    string_len_hashkey_per_vertex = []
    for idxs in string_idx_per_vertex:
        if len(idxs) == 2 and idxs[0] == idxs[1]:
            l = string_lengths[idxs[0]]
            string_len_hashkey_per_vertex.append( (l*l,) )
        else:
            lens = tuple(sorted(string_lengths[sidx] for sidx in idxs))
            string_len_hashkey_per_vertex.append(lens)

    # 1次ハッシュ
    vertex_hash_list = []
    for v in range(n):
        h = hash(string_len_hashkey_per_vertex[v] + (double_edge_neighbor_count[v],))
        vertex_hash_list.append(h)

    # 2次ハッシュ（隣接＋自分自身の1次ハッシュ）
    second_hash_list = []
    for v in range(n):
        neighbors = list(G.neighbors(v))
        hashes = [vertex_hash_list[nbr] for nbr in neighbors] + [vertex_hash_list[v]]
        second_hash = hash(tuple(sorted(hashes)))
        second_hash_list.append(second_hash)

    print("=== 各頂点ごとのstring所属番号リスト, 2重辺数, 1次ハッシュ, 2次ハッシュ ===")
    for v in range(n):
        print(f"  頂点{v}: {string_nums_per_vertex[v]} {double_edge_neighbor_count[v]} {vertex_hash_list[v]} {second_hash_list[v]}")

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



def main():
    cfg = parse_args()
    for i, raw in enumerate(cfg.input, 1):
        line = raw.strip()
        if not line: 
            continue
        visualize_and_decompose(line, i, cfg)

if __name__ == '__main__':
    main()
