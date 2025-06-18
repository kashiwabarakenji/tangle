#%%writefile utils.py
from collections import defaultdict
import networkx as nx

def parse_line(line: str):
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f'不正な入力行: {line!r}')
    neigh = [[ord(c) - ord('a') for c in token]
             for token in parts[1].split(',')]
    if any(len(nb) != 4 for nb in neigh):
        raise ValueError(f'各頂点は 4 次正則である必要があります: {line!r}')
    return neigh

def build_graph_from_pd(diagram):
    G = nx.MultiGraph()
    G.add_nodes_from(range(len(diagram)))
    label_map = {}
    for idx, e in enumerate(diagram):
        for label in e['pd_code']:
            label_map.setdefault(label, []).append(idx)
    for nodes in label_map.values():
        if len(nodes) == 2:
            u, v = nodes
            G.add_edge(u, v)
    return G

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

def adjust_alternating_rotation(edge_id_lists, loops, eid2info):
    """
    edge_id_lists を、各 string（loops）を進むたびに
    『次に進む辺』の偶奇が over/under で交互になるよう回転させる。
    """

    # 頂点ごとに「確定済みか」「現在のシフト量」を持つ
    fixed  = [False] * len(edge_id_lists)
    shift  = [0] * len(edge_id_lists)      # 記録用（デバッグにも便利）

    # ユーティリティ: 辺列と eid2info から「出発頂点」リストを得る
    def loop_vertices(loop):
        verts = []
        for k, eid in enumerate(loop):
            u, v = eid2info[eid]
            if k == 0:
                verts.extend([u, v])
            else:
                verts.append(v if verts[-1] == u else u)
        return verts[:-1]  # 長さ = len(loop)

    # 各 string について処理
    for loop in loops:
        verts = loop_vertices(loop)
        parity_needed = 0                   # 0: even(over), 1: odd(under)

        for eid, v in zip(loop, verts):
            rot = edge_id_lists[v]
            idx = rot.index(eid)            # 現在 eid が何番目か

            if not fixed[v]:
                # eid が偶奇 parity_needed になるよう最小シフトを取る
                for sh in range(4):
                    new_idx = (idx - sh) % 4
                    if new_idx % 2 == parity_needed:
                        # 左に sh シフト
                        edge_id_lists[v] = rot[sh:] + rot[:sh]
                        shift[v] = (shift[v] + sh) % 4
                        break
                fixed[v] = True
            else:
                # すでに確定している → 偶奇が合わなければ 180° 反転だけ試す
                new_idx = edge_id_lists[v].index(eid)
                if new_idx % 2 != parity_needed:
                    edge_id_lists[v] = edge_id_lists[v][2:] + edge_id_lists[v][:2]
                    shift[v] = (shift[v] + 2) % 4

            parity_needed ^= 1              # 交互に反転

    return edge_id_lists