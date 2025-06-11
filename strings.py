#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

# サンプルデータ（plantri形式の文字列表現）
SAMPLES = [
    "6 bbcc,adda,aeea,bffb,cffc,deed",
    "6 bbcd,adca,abee,affb,cffc,deed", 
    "6 bccd,aeef,affa,afee,bddb,bdcc",
    "6 bbcd,adca,abef,afeb,cdff,ceed",
    "6 bbcd,adea,affd,aceb,bdff,ceec",
    "6 bbcd,adea,aeff,afeb,bdfc,cedc",
    "6 bbcd,adea,aeef,affb,bfcc,cedd",
    "6 bbcd,aefa,affd,acee,bddf,becc",
    "6 bcde,aefc,abfd,acfe,adfb,bedc"
]

# 隣接リスト文字列 -> Pythonリスト

def parse_line(line: str):
    parts = line.strip().split()
    neigh = [[ord(c) - ord('a') for c in token]
             for token in parts[1].split(',')]
    return neigh

# 多重辺を含むダートペアリング

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
                raise RuntimeError(f"多重辺数が一致しません: {(u,v)}")
            for k in range(len(pos_u)):
                du = u*4 + pos_u[k]
                dv = v*4 + pos_v[k]
                e_pair[du] = dv
                e_pair[dv] = du
                eid_of[du] = eid_of[dv] = eid
                eid2info[eid] = (u, v)
                eid += 1
    return e_pair, eid_of, eid2info

# ループ分解

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

# 多重辺対応描画

def draw_multiedge_with_labels(G, pos):
    groups = defaultdict(list)
    for u, v, k in G.edges(keys=True):
        groups[tuple(sorted((u, v)))].append((u, v, k))
    max_m = max(len(es) for es in groups.values())
    rad_list = [i*0.2 for i in range(-(max_m-1), max_m, 2)]
    for (u, v), es in groups.items():
        es_sorted = sorted(es, key=lambda x: x[2])
        start = (max_m - len(es_sorted)) // 2
        for (u2, v2, k), rad in zip(es_sorted, rad_list[start:start+len(es_sorted)]):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u2, v2)],
                connectionstyle=f"arc3,rad={rad}"
            )
            xm = (pos[u2][0] + pos[v2][0]) / 2
            ym = (pos[u2][1] + pos[v2][1]) / 2
            dx, dy = pos[v2][1] - pos[u2][1], -(pos[v2][0] - pos[u2][0])
            norm = math.hypot(dx, dy) or 1
            off = rad * 0.5
            plt.text(
                xm + dx/norm*off,
                ym + dy/norm*off,
                str(k), fontsize=8, ha="center", va="center"
            )

# 可視化＋分解

def visualize_and_decompose(line: str):
    neigh = parse_line(line)
    e_pair, eid_of, eid2info = build_edge_pairs_fixed(neigh)

    # cyclic ログおよび対向マップ構築
    v_pair = {}
    print("=== Dart configuration per vertex (plantri order with parallel run conditional reversal) ===")
    for u, nb in enumerate(neigh):
        darts = [u*4 + i for i in range(len(nb))]
        edge_ids = [eid_of[d] for d in darts]
        # 同じ隣接先が連続する区間を条件付きで逆順
        runs = []
        start = 0
        for i in range(1, len(nb)):
            if nb[i] != nb[i-1]:
                if i - start > 1:
                    runs.append((start, i))
                start = i
        if len(nb) - start > 1:
            runs.append((start, len(nb)))
        # u > v の場合のみ reversal
        for s, e in runs:
            v = nb[s]
            if u > v:
                edge_ids[s:e] = list(reversed(edge_ids[s:e]))
                darts[s:e] = list(reversed(darts[s:e]))
        print(f"Vertex {u}: edges (cyclic) = {edge_ids}")
        # 対向ダート (index+2 mod m)
        m = len(darts)
        for idx, d in enumerate(darts):
            v_pair[d] = darts[(idx + 2) % m]

    # グラフ構築とレイアウト
    G = nx.MultiGraph()
    G.add_nodes_from(range(len(neigh)))
    for d, eid in eid_of.items():
        u = d // 4
        u2, v2 = eid2info[eid]
        v = v2 if u2 == u else u2
        G.add_edge(u, v, key=eid)
    try:
        pos = nx.planar_layout(G)
    except:
        pos = nx.spring_layout(G)

    # 描画
    plt.figure(figsize=(4,4))
    nx.draw_networkx_nodes(G, pos, node_color="lightgray")
    nx.draw_networkx_labels(G, pos)
    draw_multiedge_with_labels(G, pos)
    plt.axis('off')
    plt.show()

    # 分解と結果表示
    loops = decompose(e_pair, eid_of, eid2info, v_pair)
    lengths = sorted([len(L) for L in loops], reverse=True)
    print(f"→ Found {len(loops)} loops: lengths = {lengths}")
    for i, L in enumerate(loops, 1):
        path = " → ".join(f"{eid2info[e][0]}-{eid2info[e][1]}[{e}]" for e in L)
        print(f"   Loop {i}: {path}")

if __name__ == '__main__':
    for line in SAMPLES:
        print(f"--- {line} ---")
        visualize_and_decompose(line)

