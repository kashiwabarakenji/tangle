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
        # loop: [eid, eid, ...] の列
        # 頂点列を構築（始点はどこでもOK）
        verts = []
        # 最初の辺で両端を取得
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

        # 始点終点一致チェック
        if verts[0] != verts[-1]:
            raise RuntimeError("始点と終点が一致しません。")
        # 最後の頂点（flat_vertex）省略
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
        # 交差点: 0,2 over(+1), 1,3 under(-1)
        for i, eid in enumerate(info['pd_code']):
            sign = +1 if i % 2 == 0 else -1
            key = f"{eid}-{v}"
            z_flags[key] = sign
    return z_flags

def diagram_to_neigh(diag):
    occ = defaultdict(list)
    for v, info in enumerate(diag):
        for pos, lbl in enumerate(info['pd_code']):
            occ[lbl].append((v, pos))
    neigh = [[None]*4 for _ in diag]
    for lbl, pairs in occ.items():
        (v1,p1),(v2,p2) = pairs
        neigh[v1][p1], neigh[v2][p2] = v2, v1
    return neigh

def is_counter_clockwise(center_pos, neighbor_positions):
    angles = [math.atan2(y - center_pos[1], x - center_pos[0]) for x, y in neighbor_positions]
    angles = [(a + 2*math.pi) % (2*math.pi) for a in angles]
    diffs = [(angles[(i+1)%4] - angles[i]) % (2*math.pi) for i in range(4)]
    total = sum(diffs)
    return total > 2 * math.pi



def check_vertex_order(diag, pos):
    neigh = diagram_to_neigh(diag)
    ccw_list = [None] * len(diag)

    # 初期評価
    for v in range(len(diag)):
        if (diag[v]['type'] != 'crossing' and len(neigh[v]) < 4) or len(set(neigh[v])) <= 2:
            continue #flat vertexでも次数が4以上であれば関係あるのでは。

        center = pos[v]
        neighbors = [pos[neigh[v][i]] for i in range(4) if neigh[v][i] is not None]
        ccw = is_counter_clockwise(center, neighbors)
        ccw_list[v] = ccw

    # 補完処理（近傍の ccw を伝播）
    changed = True
    while changed:
        changed = False
        for v in range(len(diag)):
            if ccw_list[v] is not None or diag[v]['type'] != 'crossing':
                continue
            # 隣接する頂点の ccw を探す
            neighbor_ccws = [ccw_list[neigh[v][i]] for i in range(4) if neigh[v][i] is not None]
            neighbor_ccws = [x for x in neighbor_ccws if x is not None]
            if neighbor_ccws:
                # 多数決（または最初の値を採用）
                ccw_list[v] = neighbor_ccws[0]
                changed = True

    # 表示
    print("Visual Vertex Rotation:",end='')
    for v in range(len(diag)):
        if diag[v]['type'] != 'crossing':
            continue
        ccw = ccw_list[v]
        if ccw is None:
            print(f"N",end = '')
        else:
            print(f" {'L' if ccw else 'R'}",end = '')
    print()

    # 判定不能なら CW 扱いで補完
    for v in range(len(diag)):
        if ccw_list[v] is None:
            ccw_list[v] = False


    return ccw_list

def flip_x(pos):
    return {v: (-x, y) for v, (x, y) in pos.items()}

def compute_preferred_signs(diag, eid2info):
    # 多重辺対応: 並び順をpd_codeの位置とccw情報で決める。現状でccw情報は見てない。
    # まずグラフ作成
    G = build_graph_from_pd(diag)
    pos = nx.planar_layout(G)
    ccw = check_vertex_order(diag, pos)
    # edgeグループ（同一端点ペアの多重辺グループ化）
    groups = defaultdict(list) #端点ペアごとに辺を記録。
    for eid, (u, v) in eid2info.items(): # 辺の端点ペアを取得
        groups[tuple(sorted((u, v)))].append(eid)
    # signs: -(m-1), ..., m-1 まで2飛び
    preferred_signs = {}
    for (u, v), eids in groups.items(): #頂点ペアごとにループ
        #es_sorted = sorted(eids, key=lambda x: x)  # 辺番号の単純な昇順で仮並べ
        m = len(eids)
        center = v if u < v else u #大きい方の頂点中心に考える。

        # ── pd_code の cyclic リストから位置マップ ──
        seq = list(diag[center]['pd_code'])
        N = len(seq)
        pos_map = { lbl: idx for idx, lbl in enumerate(seq) }

        # ── edges を cyclic に並び替え ──
        if m == 2:
            # ２本だけなら、隣接距離を比べて順序を決定
            i0, i1 = pos_map[eids[0]], pos_map[eids[1]]
            d01 = (i1 - i0) % N
            d10 = (i0 - i1) % N
            #print(u,v,eids[0],eids[1],d01,d10,pos_map,seq,center)
            if (d01 == 1) ^ ccw[u]:
                es_sorted = eids
            elif (d10 == 1) ^ ccw[v]:
                es_sorted = reversed(eids)
            else:
              print("parallel data error",eids,pos_map,d01,d10)
        else:
            # ３本以上は輪をそのまま線形化してソート
            es_sorted = sorted(eids, key=lambda x: x)

        signs = list(range(-(m - 1), m, 2)) #1つの場合は[0],2つの場合は [-1,1]
        for eid, sign in zip(es_sorted, signs):
            # 小さい頂点番号から大きい頂点番号への向きで符号を決めている。これは間違っている。
            # 辺番号や頂点番号では符号は決まらない。PD_CODEの並び順で決まる。PD_CODEにおける隣接関係を見る必要。
            #if u > v:
            #    sign = -sign
            preferred_signs[str(eid)] = sign
    return preferred_signs

def main():
    import traceback
    parser = argparse.ArgumentParser()
    parser.add_argument('pd_code', type=str, help='PD_CODE（Pythonリスト形式）')
    parser.add_argument('flat_vertex', type=int, help='flat vertexの番号')
    parser.add_argument('flip_vertices', nargs='?', type=str, default=None,
                        help='符号反転したい頂点リスト 例: [2,3,5]')
    parser.add_argument('-o', '--output', type=str, default='data.json', help='出力ファイル名')
    parser.add_argument('-L', '--label', type=str, default=None, help='ラベル名')
    args = parser.parse_args()

    pd_code = ast.literal_eval(args.pd_code)
    flat_vertex = args.flat_vertex
    flip_vertices = []
    if args.flip_vertices is not None:
        flip_vertices = ast.literal_eval(args.flip_vertices)
    # ログ表示
    print(f"[INFO] flip_vertices = {flip_vertices}")

    diag = parse_pd_code(pd_code, flat_vertex)
    vertex_edge_routes, e_pair, eid_of, eid2info = make_vertex_edge_routes(diag)

    G = build_graph_from_pd(diag)
    pos = nx.planar_layout(G)
    coords = {str(v): [float(x), float(y)] for v, (x, y) in pos.items()}
    eid2verts = {str(eid): list(pair) for eid, pair in eid2info.items()}

    z_flags = compute_z_flags(diag, eid2info, flat_vertex)

    # ---- 追加: flip_verticesの符号反転 ----
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
