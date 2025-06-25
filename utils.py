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
    各 string（loops）を進むたびに『次に進む辺』の偶奇が over/under で交互になるよう、
    各頂点の辺リストを最小シフトで毎回調整するアプローチ。

    - edge_id_lists: 各頂点ごとの cyclic edge-list のリスト
    - loops: string ごとに探索された edge-ID のループ一覧
    - eid2info: edge-ID → (u,v) のマッピング
    """
    # ユーティリティ: ループから頂点シーケンスを作成
    def loop_vertices(loop):
        verts = []
        for k, eid in enumerate(loop):
            u, v = eid2info[eid]
            if k == 0:
                verts.extend([u, v])
            else:
                verts.append(v if verts[-1] == u else u)
        return verts[:-1]

    # 各ループを独立に処理
    for loop in loops:
        verts = loop_vertices(loop)
        parity_needed = 0  # 0: even(over), 1: odd(under)

        for eid, v in zip(loop, verts):
            rot = edge_id_lists[v]
            idx = rot.index(eid)
            # 0〜3 のシフトを試し、目的の偶奇に最小シフトを適用
            for sh in range(4):
                new_idx = (idx - sh) % 4
                if new_idx % 2 == parity_needed:
                    edge_id_lists[v] = rot[sh:] + rot[:sh]
                    break
            parity_needed ^= 1

    return edge_id_lists

          # 交互に反転

    return edge_id_lists