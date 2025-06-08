# yamada_full_updated.py - 2025-06-07 + validate_diagram組み込み版

import sympy as sp
import networkx as nx
import copy
from collections import defaultdict, Counter

# ---------------- 妥当性チェック ----------------
def validate_format(diagram):
    errors = []
    VALID_TYPES = ('crossing', 'base_flat', 'resolved_0', 'resolved_inf', 'resolved_flat')
    for i, e in enumerate(diagram):
        if 'type' not in e or 'pd_code' not in e:
            errors.append(f"Entry {i} missing 'type' or 'pd_code'")
            continue
        if e['type'] not in VALID_TYPES:
            errors.append(f"Entry {i} has invalid type '{e['type']}'")
        pd = e['pd_code']
        if not (isinstance(pd, tuple) and len(pd) == 4 and all(isinstance(x, int) for x in pd)):
            errors.append(f"Entry {i} pd_code must be 4-int tuple, got {pd}")
    return errors

def validate_labels(diagram):
    counts = Counter()
    for e in diagram:
        counts.update(e['pd_code'])
    errors = []
    for label, cnt in counts.items():
        if cnt != 2:
            errors.append(f"Label {label} appears {cnt} times (expected 2)")
    return errors

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

def validate_graph_structure(diagram):
    errors = []
    G = build_graph_from_pd(diagram)
    # 頂点インデックス→種別(type)の辞書を作る
    vtype = {i: e['type'] for i, e in enumerate(diagram)}
    for node in G.nodes():
        tp = vtype.get(node, None)
        # crossing/resolved_0/resolved_infのみ次数4チェック
        if tp in ('crossing', 'resolved_0', 'resolved_inf'):
            deg = G.degree(node)
            if deg != 4:
                errors.append(f"Node {node} ({tp}) has degree {deg} (expected 4)")
    comps = nx.number_connected_components(G)
    if comps != 1:
        errors.append(f"Graph has {comps} components (expected 1)")
    return errors

def validate_diagram(diagram):
    errors = []
    errors += validate_format(diagram)
    errors += validate_labels(diagram)
    errors += validate_graph_structure(diagram)
    return errors

# ---------------- PD → diagram ----------------
def pd_list_to_diagram(pd_list, flat_indices):
    return [
        {'type': 'base_flat' if i in flat_indices else 'crossing',
         'pd_code': pd}
        for i, pd in enumerate(pd_list)
    ]

def compact_graph_summary(diagram):
    """
    diagram: pd_list_to_diagramで作ったリスト
    - 各頂点: 番号とtype（B=base_flat, C=crossing, R=resolved_*, F=resolved_flat）
    - 各枝: (u,v) の組をmulti edge含め全て表示
    - 各頂点のPDコードをover/under順でコンパクト表示
    """
    G = build_graph_from_pd(diagram)
    # typeの略称
    abbrev = lambda t: (
        'B' if t == 'base_flat'
        else 'C' if t == 'crossing'
        else 'R' if t in ('resolved_0', 'resolved_inf')
        else 'F' if t == 'resolved_flat'
        else t[0].upper()
    )
    v_types = [f"{i}:{abbrev(e['type'])}" for i, e in enumerate(diagram)]
    edge_list = [f"({u},{v})" for u, v in G.edges()]
    pd_codes = [
        f"{i}:[{e['pd_code'][0]},{e['pd_code'][1]},{e['pd_code'][2]},{e['pd_code'][3]}]"
        for i, e in enumerate(diagram)
    ]
    print("V:", ", ".join(v_types))
    print("E:", ", ".join(edge_list))
    print("PD(ouou):", "; ".join(pd_codes))

# ---------------- flat-only graph ----------------
def flat_graph_from_pd(diagram):
    G = nx.MultiGraph()
    
    # ---------- 0. 初期グラフ構築 ----------
    for v in range(len(diagram)):
        G.add_node(v)
    
    # ラベル → 接続頂点 の対応
    slots = defaultdict(list)
    for idx, e in enumerate(diagram):
        for lbl in e['pd_code']:
            slots[lbl].append(idx)
    
    # provenance付きエッジを追加
    for lbl, (u, v) in slots.items():
        G.add_edge(u, v, origin=None, label=lbl)
    
    # ---------- 1. 交点の解消 ----------
    survivor = {}
    already_added = set()

    for idx, e in enumerate(diagram):
        tp = e['type']
        if tp not in ('resolved_0', 'resolved_inf'):
            continue

        a, b, c, d = e['pd_code']
        nbs = {lbl: (slots[lbl][0] if slots[lbl][0] != idx else slots[lbl][1])
               for lbl in (a, b, c, d)}

        # 生存者を1つ選ぶ
        surv = next(nb for nb in nbs.values() if nb != idx)
        survivor[idx] = surv

        # (i) 4本の枝を削除
        for lbl in (a, b, c, d):
            u, v = idx, nbs[lbl]
            for key, data in list(G[u][v].items()):
                if data.get('label') == lbl:
                    G.remove_edge(u, v, key)

        # (ii) 自己ループを追加（ペアごとに一度だけ）
        keep_pairs = [(a, d), (b, c)] if tp == 'resolved_0' else [(a, b), (c, d)]
        for x, y in keep_pairs:
            pair_id = tuple(sorted((x, y)))
            tag = (surv, pair_id)
            if tag in already_added:
                continue
            already_added.add(tag)
            G.add_edge(surv, surv, label=f'{x}-{y}')

        # (iii) 不要な頂点を削除
        G.remove_node(idx)

    return G

def build_flat_graph(diagram, verbose=False):
    """
    crossing はすべて消去、flat vertex は残す方式。
    * resolved_flat も「flat vertex」とみなし残す。
    * 残る頂点数が 0 のコンポーネントは無視（独立ループ → 係数 1）。
    """

    import networkx as nx
    from collections import defaultdict

    # 1. 残す頂点集合
    KEEP = {
        idx for idx, e in enumerate(diagram)
        if e["type"] not in ("resolved_0", "resolved_inf")
    }

    # 2. Union-Find (ラベル ←→ コンポーネント)
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        parent.setdefault(x, x)
        parent.setdefault(y, y)
        parent[find(y)] = find(x)

    for e in diagram:
        a, b, c, d = e["pd_code"]
        t = e["type"]
        if t == "resolved_0":
            union(a, d); union(b, c)
        elif t == "resolved_inf":
            union(a, b); union(c, d)
        # flat vertexは union しない

    # 3. root → {残る頂点} への写像
    comp_to_keep = defaultdict(set)
    for v, e in enumerate(diagram):
        for lbl in e["pd_code"]:
            if v in KEEP:
                comp_to_keep[find(lbl)].add(v)

    # 4. グラフ構築
    G = nx.MultiGraph()
    G.add_nodes_from(KEEP)

    added = set()     # 辺の重複防止

    for root, verts in comp_to_keep.items():
        verts = sorted(verts)
        if len(verts) == 0:
            continue                    # → 独立ループ：無視
        elif len(verts) == 1:
            v = verts[0]
            tag = (v, root)
            if tag not in added:
                G.add_edge(v, v, label=f"C{root}")
                added.add(tag)
        elif len(verts) == 2:
            u, v = verts
            tag = (u, v, root)
            if tag not in added:
                G.add_edge(u, v, label=f"C{root}")
                added.add(tag)
        else:                           # 3 個以上：スター状に
            head = verts[0]
            for v in verts[1:]:
                tag = (head, v, root)
                if tag not in added:
                    G.add_edge(head, v, label=f"C{root}")
                    added.add(tag)

    if verbose:
        print("--- Nodes:", G.number_of_nodes(),
              "Edges:", G.number_of_edges())
        print("Edges detail:", list(G.edges(keys=True, data=True)))
    return G


# ---------------- Tutte recursion ----------------
class BivariatePolynomial:
    def __init__(self, data=None):
        self.data = {k: sp.Integer(v) for k, v in (data or {}).items()}
    def __add__(self, other):
        r = BivariatePolynomial(self.data.copy())
        for k, v in other.data.items():
            r.data[k] = r.data.get(k, 0) + v
            if r.data[k] == 0:
                del r.data[k]
        return r
    def mul_x(self):
        return BivariatePolynomial({(i+1, j): v for (i, j), v in self.data.items()})
    def mul_y(self):
        return BivariatePolynomial({(i, j+1): v for (i, j), v in self.data.items()})
    @staticmethod
    def one():
        return BivariatePolynomial({(0, 0): 1})

def contract_edge(G, e):
    u, v, _ = e
    w = frozenset((u, v))
    H = nx.MultiGraph()
    for n in G.nodes():
        if n not in (u, v):
            H.add_node(n)
    H.add_node(w)
    for a, b, k in G.edges(keys=True):
        if (a, b, k) == e or (b, a, k) == e:
            continue
        na = w if a in (u, v) else a
        nb = w if b in (u, v) else b
        H.add_edge(na, nb)
    return H

def tutte(G):
    if G.number_of_edges() == 0:
        return BivariatePolynomial.one()
    e = next(iter(G.edges(keys=True)))
    u, v, _ = e
    if u == v:
        H = G.copy()
        H.remove_edge(*e)
        return tutte(H).mul_y()
    H = G.copy()
    H.remove_edge(*e)
    if nx.number_connected_components(H) > nx.number_connected_components(G):
        return tutte(contract_edge(G, e)).mul_x()
    return tutte(H) + tutte(contract_edge(G, e))

# ---------------- flow polynomial ----------------
def evaluate_tutte(T, xv, yv):
    expr = 0
    for (i, j), coef in T.data.items():
        expr += coef * (xv**i) * (yv**j)
    return sp.simplify(expr)

def flow_polynomial(G, t):
    Tb = tutte(G)
    T_at = evaluate_tutte(Tb, 0, 1-t)
    E = G.number_of_edges()
    V = G.number_of_nodes()
    C = nx.number_connected_components(G)
    return sp.simplify((-1)**(E - V + C) * T_at)

# ---------------- Yamada_flat ----------------
def yamada_flat(G, q):
    Q = q + 2 + q**(-1)
    F = flow_polynomial(G, Q)
    E, N = G.number_of_edges(), G.number_of_nodes()
    sign = (-1)**(N - E )
    return sp.simplify(sign * F)

# ---------------- auxiliary ----------------
def coeffs_laurent(expr, q=sp.symbols('q')):
    expr = sp.expand(expr)
    coeffs = defaultdict(lambda: sp.Integer(0))
    for term in expr.as_ordered_terms():
        c, k = term.as_coeff_exponent(q)
        # sympyの整数や有理数以外のとき、nsimplifyで近い有理数に
        c = sp.nsimplify(c)
        # floatで整数値の場合はintに
        if isinstance(c, float) and c.is_integer():
            c = int(c)
        coeffs[int(k)] += c
    if not coeffs:
        return []
    lo, hi = min(coeffs), max(coeffs)
    # 明示的な整数変換
    result = []
    for k in range(lo, hi+1):
        val = coeffs[k]
        if isinstance(val, float) and val.is_integer():
            result.append(int(val))
        elif isinstance(val, sp.Number) and val.is_Integer:
            result.append(int(val))
        else:
            result.append(val)
    return result

def yamada_diagram(diagram, q):
    """
    Skein 再帰により diagram から Yamada 多項式を計算。
    各段階で詳細ログを print します。
    """
    # 妥当性チェック
    errors = validate_diagram(diagram)
    if errors:
        print("Diagram validation failed:")
        for err in errors:
            print("  ", err)
        raise ValueError("Invalid diagram structure")
    # 内部再帰関数: depth でインデント調整
    def recurse(dgm, depth=0):
        indent = '  ' * depth
        for i, e in enumerate(dgm):
            if e['type'] == 'crossing':
                print(f"{indent}expand crossing at index {i}: pd_code={e['pd_code']}")
                total = 0
                for coeff, typ in [(q, 'resolved_0'), (1, 'resolved_flat'), (q**-1, 'resolved_inf')]:
                    d_copy = copy.deepcopy(dgm)
                    d_copy[i]['type'] = typ
                    print(f"{indent} branch {typ} with coeff={coeff}")
                    val = recurse(d_copy, depth+1)
                    print(f"{indent}  → branch value = {sp.expand(val)}")
                    total += coeff * val
                print(f"{indent} sum@depth{depth} = {sp.expand(total)}")
                return total
        G = build_flat_graph(dgm)
        Yf = yamada_flat(G, q)
        print(f"{indent}leaf graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, Yamada_flat={sp.expand(Yf)}")
        return Yf
    # 計算開始
    return recurse(diagram, 0)

def yamada_poly_and_coeffs(expr, q=sp.symbols('q'), show_zero=True):
    """
    山田多項式を「整数係数リスト」と「多項式表示（整数係数）」両方をprint
    show_zero: 0係数も表示するか
    """
    expr = sp.expand(expr)
    coeffs = defaultdict(lambda: 0)
    for term in expr.as_ordered_terms():
        c, k = term.as_coeff_exponent(q)
        c = sp.nsimplify(c)
        if isinstance(c, float) and c.is_integer():
            c = int(c)
        if isinstance(c, sp.Number) and c.is_Integer:
            c = int(c)
        coeffs[int(k)] += c
    if not coeffs:
        print("[]")
        print("0")
        return
    lo, hi = min(coeffs), max(coeffs)
    coeff_list = [int(coeffs[k]) for k in range(lo, hi+1)]

    # --- リスト表示
    print("Laurent係数リスト:", coeff_list)

    # --- 多項式(整数係数)表示
    # 例: q^3 - 2q + 1 など（降べき or 昇べき 好みで）
    terms = []
    for i, a in enumerate(coeff_list):
        deg = i + lo
        if a == 0 and not show_zero:
            continue
        # べき表示
        if deg == 0:
            term = f"{a}"
        elif deg == 1:
            term = f"{a}*q"
        else:
            term = f"{a}*q^{deg}"
        terms.append(term)
    # きれいな和表示
    poly_str = " + ".join(terms)
    print("山田多項式:", poly_str)


# ---------------- main (with tests) ----------------
if __name__ == '__main__':
    q = sp.symbols('q')

    # double triangle test
    dt = nx.MultiGraph()
    dt.add_nodes_from([0,1,2])
    for u, v in [(0,1),(1,2),(2,0)]:
        dt.add_edge(u, v)
        dt.add_edge(u, v)
    Y_dt = sp.expand(yamada_flat(dt, q))
    print('Double Triangle Yamada')
    yamada_poly_and_coeffs(Y_dt, q)

    """
    # base cycle
    bc = nx.MultiGraph()
    bc.add_nodes_from([0,1,2])
    for u, v in [(0,1),(1,2),(2,0)]:
        bc.add_edge(u, v)
    Y_bc = sp.expand(yamada_flat(bc, q))
    print('Base 3-cycle Yamada =', Y_bc)
    print('coeffs base cycle =', coeffs_laurent(Y_bc, q))
    print()
    """

    # Theta3 test: 2 vertices with 3 parallel edges
    G_theta3 = nx.MultiGraph()
    G_theta3.add_nodes_from([0,1])
    for _ in range(3):
        G_theta3.add_edge(0,1)
    Y_theta3 = sp.expand(yamada_flat(G_theta3, q))
    print('Theta3 Yamada')
    yamada_poly_and_coeffs(Y_theta3, q)

    # Theta4 test: 2 vertices with 4 parallel edges
    G_theta4 = nx.MultiGraph()
    G_theta4.add_nodes_from([0,1])
    for _ in range(4):
        G_theta4.add_edge(0,1)
    Y_theta4 = sp.expand(yamada_flat(G_theta4, q))
    print('Theta4 Yamada')
    yamada_poly_and_coeffs(Y_theta4, q)

    PD_octagon = [
        (1, 2, 3, 4),   # flat vertex
        (1, 5, 9, 6),   # flat vertex
        (2, 6, 10, 7),  # flat vertex
        (3, 7, 11, 8),  # flat vertex
        (4, 8, 0, 5),   # flat vertex
        (9, 10, 11, 0)  # flat vertex
    ]
    diag = pd_list_to_diagram(PD_octagon, {0,1,2,3,4,5})
    errors = validate_diagram(diag)
    if errors:
        print("PD_octagon validation failed:")
        for err in errors:
            print("  ", err)
    else:
        print("build", yamada_diagram(diag, q))
        print("Yamada octahedron diagram:", diag)
        Y = sp.expand(yamada_diagram(diag, q))
        print('Yamada(octahedron) =', Y)
        print('coeffs octahedron =', coeffs_laurent(Y, q))
    print()

    # original example
    #PD_real = [(1,2,3,4), (1,3,5,6), (2,4,6,5)]
    PD_real = [(1,2,3,4),(1,2,3,4)]
    diag = pd_list_to_diagram(PD_real, {0})
    errors = validate_diagram(diag)
    if errors:
        print("PD_real validation failed:")
        for err in errors:
            print("  ", err)
    else:
        compact_graph_summary(diag)
        Y = sp.expand(yamada_diagram(diag, q))
        print('Yamada(2_1^l) =', Y)
        print('coeffs 2_1^l =', coeffs_laurent(Y, q))
    print()

    #PD_real = [(1,2,3,4), (1,3,5,6), (2,4,6,5)]
    PD_real = [(1,2,3,4), (3,2,6,5), (1,4,5,6)]
    diag = pd_list_to_diagram(PD_real, {0})
    errors = validate_diagram(diag)
    if errors:
        print("PD_real validation failed:")
        for err in errors:
            print("  ", err)
    else:
        compact_graph_summary(diag)
        Y = sp.expand(yamada_diagram(diag, q))
        print('Yamada(2_1^k) =', Y)
        print('coeffs 2_1^k =', coeffs_laurent(Y, q))
    print()

    PD_minloop = [ (1,1,2,2), (3,3,4,4) ]
    diag = [
        {'type': 'base_flat', 'pd_code': PD_minloop[0]},
        {'type': 'base_flat', 'pd_code': PD_minloop[1]},
    ]

    # デバッグ付きflat graph生成
    G = build_flat_graph(diag, verbose=True)

    PD_minloop = [(1,1,2,2), (3,3,4,4)]
    diag = pd_list_to_diagram(PD_minloop, {0,1})   # 2 つとも flat
    G = build_flat_graph(diag, verbose=True)

    # PD of a 1-crossing bouquet-like gadget
    pd = [(1, 2, 1, 2)]                 # one 4-valent vertex, labels 1,2 repeated
    flat_indices = {0}                  # it's already flat
    D = pd_list_to_diagram(pd, flat_indices)

    # Make a copy and manually mark crossing 0 as resolved_0
    D2 = copy.deepcopy(D)
    D2[0]['type'] = 'resolved_0'

    G = build_flat_graph(D2, verbose=False)
    print(G.nodes(), G.edges(keys=True))
