import sys
import argparse
import ast
import json
import networkx as nx
import math
from collections import defaultdict

def parse_pd_code(pd_code, flat_vertex):
    diag = []
    for v, row in enumerate(pd_code):
        typ = 'flat' if v == flat_vertex else 'crossing'
        diag.append({'type': typ, 'pd_code': list(row)})
    return diag

def build_edge_info_direct(diag):
    occ = defaultdict(list)
    for v, info in enumerate(diag):
        for pos, lbl in enumerate(info['pd_code']):
            occ[lbl].append((v, pos))
    e_pair   = {}
    eid_of   = {}
    eid2info = {}
    for lbl, pair in occ.items():
        if len(pair) != 2:
            raise RuntimeError(f"pd_code {lbl} の対応が {len(pair)} 個あります")
        (v1, p1), (v2, p2) = pair
        du = v1*4 + p1
        dv = v2*4 + p2
        e_pair[du] = dv
        e_pair[dv] = du
        eid_of[du] = lbl
        eid_of[dv] = lbl
        eid2info[lbl] = (v1, v2)
    return e_pair, eid_of, eid2info

def decompose(e_pair, eid_of, eid2info, v_pair, diag):
    visited, loops = set(), []
    for dart in eid_of:
        if dart in visited:
            continue
        cur, loop = dart, []
        while True:
            visited.add(cur)
            opp = v_pair[cur]
            visited.add(opp)
            loop.append(eid_of[opp])
            cur = e_pair[opp]
            if cur in visited:
                break
        loops.append(loop)
    return loops

def make_vertex_edge_routes(diag):
    e_pair, eid_of, eid2info = build_edge_info_direct(diag)
    v_pair = {v*4+i: v*4+((i+2)%4) for v in range(len(diag)) for i in range(4)}
    loops = decompose(e_pair, eid_of, eid2info, v_pair, diag)
    vertex_edge_routes = []
    for loop in loops:
        verts = []
        eid0 = loop[0]
        u, v = eid2info[eid0]
        verts.append(u)
        prev_v = u
        for eid in loop:
            verts.append(eid)
            a, b = eid2info[eid]
            next_v = b if a == prev_v else a
            verts.append(next_v)
            prev_v = next_v
        if verts[0] != verts[-1]:
            raise RuntimeError("始点と終点が一致しません。")
        vertex_edge_routes.append(verts[:-1])
    return vertex_edge_routes, e_pair, eid_of, eid2info

def build_graph_from_pd(diag):
    G = nx.MultiGraph()
    G.add_nodes_from(range(len(diag)))
    label_map = defaultdict(list)
    for idx, e in enumerate(diag):
        for lbl in e['pd_code']:
            label_map[lbl].append(idx)
    for lbl, nodes in label_map.items():
        if len(nodes) == 2:
            u, v = nodes
            G.add_edge(u, v, key=lbl, label=lbl)
    return G

def compute_z_flags(diag, eid2info, flat_vertex):
    z_flags = {}
    for v, info in enumerate(diag):
        if v == flat_vertex:
            continue
        for i, eid in enumerate(info['pd_code']):
            sign = +1 if i % 2 == 0 else -1
            key = f"{eid}-{v}"
            z_flags[key] = sign
    return z_flags

def compute_preferred_signs(diag, eid2info):
    G = build_graph_from_pd(diag)
    pos = nx.planar_layout(G)
    groups = defaultdict(list)
    for eid, (u, v) in eid2info.items():
        groups[tuple(sorted((u, v)))].append(eid)
    preferred_signs = {}
    for (u, v), eids in groups.items():
        m = len(eids)
        es_sorted = sorted(eids, key=lambda x: x)
        signs = list(range(-(m - 1), m, 2))
        for eid, sign in zip(es_sorted, signs):
            preferred_signs[str(eid)] = sign
    return preferred_signs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pd_code', type=str, help='PD_CODE（Pythonリスト形式）')
    parser.add_argument('flat_vertex', type=int, help='flat vertexの番号')
    parser.add_argument('flip_vertices', nargs='?', type=str, default=None,
                        help='符号反転したい頂点リスト 例: [2,3,5]')
    parser.add_argument('-o', '--output', type=str, default='data.json', help='出力ファイル名')
    parser.add_argument('-L', '--label', type=str, default=None, help='ラベル名')
    parser.add_argument('-V', '--vertex_coords', type=str, default=None,
                        help='頂点座標ファイル(JSON)を指定するとcoordsとして読み込み')
    args = parser.parse_args()

    pd_code = ast.literal_eval(args.pd_code)
    flat_vertex = args.flat_vertex
    flip_vertices = []
    if args.flip_vertices is not None:
        flip_vertices = ast.literal_eval(args.flip_vertices)
    print(f"[INFO] flip_vertices = {flip_vertices}")

    diag = parse_pd_code(pd_code, flat_vertex)
    vertex_edge_routes, e_pair, eid_of, eid2info = make_vertex_edge_routes(diag)

    G = build_graph_from_pd(diag)

    if args.vertex_coords is not None:
        with open(args.vertex_coords, 'r', encoding='utf-8') as f:
            location_data = json.load(f)
        coords = location_data["coords"]
    else:
        pos = nx.planar_layout(G)
        coords = {str(v): [float(x), float(y)] for v, (x, y) in pos.items()}

    eid2verts = {str(eid): list(pair) for eid, pair in eid2info.items()}
    z_flags = compute_z_flags(diag, eid2info, flat_vertex)

    if flip_vertices:
        for v in flip_vertices:
            v_str = str(v)
            for k in list(z_flags.keys()):
                if k.endswith(f"-{v_str}"):
                    z_flags[k] *= -1
                    print(f"[INFO] z_flag {k} を反転 → {z_flags[k]}")

    preferred_signs = compute_preferred_signs(diag, eid2info)

    data = {
        "coords": coords,
        "vertex_edge_routes": vertex_edge_routes,
        "eid2verts": eid2verts,
        "flat_vertex": flat_vertex,
        "z_flags": z_flags,
        "preferred_signs": preferred_signs
    }
    if args.label is not None:
        data["label"] = args.label

    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print(json_str)

if __name__ == '__main__':
    main()
