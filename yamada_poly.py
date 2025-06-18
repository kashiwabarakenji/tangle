# yamada_full_updated.py - 2025-06-07 + validate_diagram組み込み版

import sympy as sp
import networkx as nx
import copy
from collections import defaultdict, Counter
import json
from sympy import symbols

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
        if e['type'] in ('crossing', 'resolved_0', 'resolved_inf'):
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
        { 'type': 'base_flat' if i in flat_indices else 'crossing',
          'pd_code': tuple(pd) }         # ← ここで tuple() に
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
        for i, e in enumerate(diagram) if e['type'] in ('crossing', 'resolved_0', 'resolved_inf')
    ]
    flat_codes = [f"{i}:[{e['pd_code']}]" for i, e in enumerate(diagram) if e['type'] in ('base_flat', 'resolved_flat')]
    #print("V:", ", ".join(v_types),end=" ")
    print("E:", ", ".join(edge_list), end=" ")
    print("PD(ouou):", "; ".join(pd_codes), "flat:", "; ".join(flat_codes))

# ---------------- flat-only graph ----------------
# ここにくる時にはcrossingは全て解消されている。
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

        a, b, c, d = e['pd_code'] #ここにくる場合は、resolvedなので、次数は4。
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
    crossing は全て消去し、flat vertex だけ残して平面グラフを作る。
    さらに「どの flat 頂点集合にも属さない根 (root)」を空ループと数え、
    その個数も返す。
    """
    import networkx as nx
    from collections import defaultdict

    # ―― 1. keep する頂点（flat vertex と resolved_flat）
    KEEP = {
        idx for idx, e in enumerate(diagram)
        if e["type"] not in ("resolved_0", "resolved_inf")
    }

    # ―― 2. Union–Find on label
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
        if e["type"] == "resolved_0":
            a, b, c, d = e["pd_code"]
            union(a, d); union(b, c)
        elif e["type"] == "resolved_inf":
            a, b, c, d = e["pd_code"]
            union(a, b); union(c, d)
        # flat vertex は union しない

    # ―― 3. root → {keep 頂点} の写像（必ず root を登録）
    comp_to_keep = defaultdict(set)
    for v, e in enumerate(diagram):
        for lbl in e["pd_code"]:
            r = find(lbl)
            comp_to_keep.setdefault(r, set())   # root を必ず作成
            if v in KEEP:
                comp_to_keep[r].add(v)

    # ―― 4. グラフ構築
    G = nx.MultiGraph()
    G.add_nodes_from(KEEP)
    added = set()
    empty_loops = 0

    for root, verts in comp_to_keep.items():
        verts = sorted(verts)
        if not verts:               # ← 空集合 ⇒ 空ループ
            empty_loops += 1
            continue

        if len(verts) == 1:         # ループ
            v = verts[0]
            tag = (v, v, root)
            if tag not in added:
                G.add_edge(v, v, label=f"C{root}")
                added.add(tag)
        elif len(verts) == 2:       # 普通の辺
            u, v = verts
            tag = (u, v, root)
            if tag not in added:
                G.add_edge(u, v, label=f"C{root}")
                added.add(tag)
        else:                       # 3 頂点以上 ⇒ スター状
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
        print("空ループ数:", empty_loops)

    return G, empty_loops

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

def yamada_diagram(diagram, q, logFlag=False):
    """
    skein 展開で Yamada 多項式を計算。
    logFlag=True で展開過程のログを出力。
    空ループごとに d = q + 1 + q**(-1) を掛ける。
    """
    d = q + 1 + q**(-1)
    errors = validate_diagram(diagram)
    if errors:
        raise ValueError("Diagram validation failed: " + "; ".join(errors))

    def recurse(dgm, multiplier=1, depth=0):
        indent = '  ' * depth
        # 展開可能な crossing を探す
        for i, e in enumerate(dgm):
            if e['type'] == 'crossing':
                if logFlag:
                    print(f"{indent}expand crossing at index {i}: pd_code={e['pd_code']}")
                total = 0
                for coeff, typ in [(q, 'resolved_0'),
                                   (1, 'resolved_flat'),
                                   (q**-1, 'resolved_inf')]:
                    d_copy = copy.deepcopy(dgm)
                    d_copy[i]['type'] = typ
                    if logFlag:
                        print(f"{indent} branch {typ} with coeff={coeff}")
                    total += recurse(d_copy, multiplier*coeff, depth+1)
                return total

        # leaf なら平面グラフを構築して計算
        G, empty_loops = build_flat_graph(dgm)
        Yf = yamada_flat(G, q)
        result = multiplier * (d ** empty_loops) * Yf
        if logFlag:
            print(f"{indent}leaf graph: nodes={G.number_of_nodes()}, "
                  f"edges={G.number_of_edges()}, 空ループ数={empty_loops}, "
                  f"Yamada_flat={sp.expand(Yf)}")
            print(f"{indent}→ weighted = {sp.expand(result)}")
        return result

    return recurse(diagram, 1, 0)


def diagram_summary(diagram, logFlag=False):
    """
    ダイアグラムの概要表示と Yamada 多項式の計算。
    logFlag=True で展開ログも表示。
    """
    # グラフ概要
    compact_graph_summary(diagram)
    # 多項式計算
    q = sp.symbols('q')
    Y = sp.expand(yamada_diagram(diagram, q, logFlag=logFlag))
    coeffs = coeffs_laurent(Y, q)
    print('山田多項式 =', Y)
    print('coeffs =', coeffs)
    print(diagram)

# yamada_flat は以前の定義に加えて loop factor を yamada_diagram 側で扱う設計
def yamada_flat(G, q):
    Q = q + 2 + q**(-1)
    F = flow_polynomial(G, Q)
    E, N = G.number_of_edges(), G.number_of_nodes()
    sign = (-1)**(N - E )
    return sp.simplify(sign * F)



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
    # Colab のフォーム機能を使うなら、このセルを notebook 上で実行する
    #@title **PDコードとオプションの入力** 📝
    pd_list_str   = '[[0, 1, 2, 3], [0, 4, 5, 1], [2, 6, 7, 3], [4, 8, 9, 5], [6, 10, 11, 7], [8, 11, 10, 9]]'  #@param {type:"string"}
    flat_indices  = [0]                        #@param {type:"raw"}
    rev_crossings = []                         #@param {type:"raw"}
    logFlag       = False                      #@param {type:"boolean"}

    pd_list = json.loads(pd_list_str)
    # JSON→list なのでタプル化だけ注意
    pd_list = [tuple(p) for p in pd_list]
    diag = pd_list_to_diagram(pd_list, set(flat_indices))
    for i in rev_crossings:
        seq = diag[i]['pd_code']           # [a,b,c,d]
        rotated = seq[1:] + seq[:1]        # [b,c,d] + [a] → [b,c,d,a]
        diag[i]['pd_code'] = tuple(rotated)

    # ログ出力の on/off
    diagram_summary(diag, logFlag)

#if __name__ == '__main__':
#    q = sp.symbols('q')
    
    #examples flat vertexが複数のもの。
    #PD = [(1,2,3,4), (3,4,5),(1,2,5)]    
    #PD = [ (1,2,5),(2,3,9),(3,4,8),(1,12,4),(9,6,10,5),(6,11,7,10),(11,8,12,7)] #yamada_toolにある例。Omega 2 graph flatが0,1,2,3 [1, 1, 1, 1, 1, -1, 1, -2, 1, -1, 1, 1, 0, 1]
    #diag = pd_list_to_diagram(PD, {0,1,2,3})  # 3つのflat vertex
    #diagram_summary(diag, "triangle with 5 edges")
    #------- 5交点 plantriより。
    #PD = [[0, 1, 2, 3], [0, 4, 5, 1], [2, 6, 7, 3], [4, 8, 9, 5], [6, 10, 11, 7], [8, 11, 10, 9]] #{0} 5_1^l [1, 1, 2, 1, 1, 0, -1, -1, -1]
    #PD  = [[0, 1, 2, 3], [0, 4, 5, 1], [7, 2, 5, 6], [3, 8, 9, 4], [6, 10, 11, 7], [8, 11, 10, 9]] #{1} 5_1^k  [-1, -2, -2, -2, -2, -1, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 1]
    #PD  = [[0, 1, 2, 3], [1, 0, 4, 5], [7, 2, 5, 6], [3, 8, 9, 4], [6, 10, 11, 7], [8, 11, 10, 9]] #{3} 5_2^k  [1, 1, 1, 1, -1, -1, -2, -2, -2, -2, -1, -1, -1]
    #PD  = [[0, 1, 2, 3], [1, 0, 4, 5], [7, 2, 5, 6], [3, 8, 9, 4], [6, 10, 11, 7], [8, 11, 10, 9]] #{5} 5_2^k  [1, 1, 1, 1, -1, -1, -2, -2, -2, -2, -1, -1, -1]
    #PD  = [[0, 1, 2, 3], [1, 0, 4, 5], [7, 2, 5, 6], [3, 8, 9, 4], [6, 10, 11, 7], [8, 11, 10, 9]] #{2} 5_2^k  [1, 1, 1, 1, -1, -1, -2, -2, -2, -2, -1, -1, -1]
    #PD  = [[0, 1, 2, 3], [6, 0, 4, 5], [8, 2, 1, 7], [3, 9, 10, 11], [11, 10, 5, 4], [9, 8, 7, 6]] #{0} 5_2^l [-1, 0, 0, -1, 1, 1, 2, 1, 1, 1, 0, 0, 0, -1, -1]
    #PD  = [[0, 1, 2, 3], [6, 0, 4, 5], [8, 2, 1, 7], [3, 9, 10, 11], [11, 10, 5, 4], [9, 8, 7, 6]]  #{2} 5_2^l
    #PD = [[0, 1, 2, 3], [1, 0, 4, 5], [7, 2, 5, 6], [3, 8, 9, 4], [6, 9, 10, 11], [8, 7, 11, 10]] #{0} 5_3^l [-1, -1, 1, 1, 0, 1, 0, -1, 0, 0, 1, 0, 1, 1, -1, 0, 1]
    #PD = [[0, 1, 2, 3], [5, 1, 0, 4], [2, 6, 7, 8], [4, 3, 8, 9], [11, 5, 9, 10], [6, 11, 10, 7]] #{0} 5_3^k [1, 1, 0, 0, 0, -1, -1, 0, -1, 0, -1, -1, -2, -2, 0, -1, -1]
    #PD = [[0, 1, 2, 3], [4, 5, 1, 0], [2, 6, 7, 8], [9, 4, 3, 8], [10, 11, 5, 9], [7, 6, 11, 10]] #{2} 5_5^k [-1, 0, 2, 1, 1, 0, -1, -1, -2, -1, -2, -2, -1, -1, -1]
    #PD = [[0, 1, 2, 3], [4, 5, 1, 0], [8, 2, 6, 7], [9, 4, 3, 8], [10, 11, 5, 9], [7, 6, 11, 10]] #{3} 5_4^k [-1, -1, 1, 0, 1, 2, 1, 1, 0, 1, 0, -1, -1, -2, -3, -2, -2, -2, -1]
    #PD = [[0, 1, 2, 3], [4, 5, 1, 0], [8, 2, 6, 7], [9, 4, 3, 8], [10, 11, 5, 9], [7, 6, 11, 10]] # {5} 5_5^k [-1, 0, 2, 1, 1, 0, -1, -1, -2, -1, -2, -2, -1, -1, -1]
    #PD = [[0, 1, 2, 3], [1, 0, 4, 5], [2, 6, 7, 8], [10, 4, 3, 9], [6, 5, 10, 11], [7, 11, 9, 8]] #{4} primeでない。 [-1, -1, 0, 2, 3, 2, 1, -1, -2, -3, -3, -3, -2, -1]
    #PD = [[0, 1, 2, 3], [5, 1, 0, 4], [2, 6, 7, 8], [9, 10, 4, 3], [10, 11, 6, 5], [8, 7, 11, 9]] #{0} 5_6^k [1, 0, -2, 0, 0, -1, 0, -1, -1, -2, -1, 0, -1, 0, 1, -1, -1]
    #PD = [[0, 1, 2, 3], [4, 5, 1, 0], [8, 2, 6, 7], [3, 9, 10, 4], [10, 11, 6, 5], [9, 8, 7, 11]]  #{4} 5_7^k [-1, -1, 1, 1, 0, 2, 1, 0, 1, 0, 0, -2, -1, -1, -3, -2, -1, -2, -1]
    #PD = [[0, 1, 2, 3], [4, 5, 1, 0], [2, 6, 7, 8], [9, 10, 4, 3], [10, 11, 6, 5], [8, 7, 11, 9]] #{3} 5_8^k [1, 0, 0, 0, -1, 0, -1, -1, -2, -2, -1, -1, -1]
    #PD = [[0, 1, 2, 3], [1, 0, 4, 5], [8, 2, 6, 7], [10, 4, 3, 9], [11, 6, 5, 10], [7, 11, 9, 8]] #{5} 5_6^k [1, 0, -2, 0, 0, -1, 0, -1, -1, -2, -1, 0, -1, 0, 1, -1, -1]
    #PD = [[0, 1, 2, 3], [1, 0, 4, 5], [8, 2, 6, 7], [3, 9, 10, 4], [7, 6, 5, 11], [9, 8, 11, 10]] #{0} 5_3^l [-1, -1, 1, 1, 0, 1, 0, -1, 0, 0, 1, 0, 1, 1, -1, 0, 1]
    #diag = pd_list_to_diagram(PD, {5})
    #--------

    # examples flat vertexが1つのもの。(先頭がflat vertex)
    #PD = [(1,2,3,4),(1,2,3,4)] #[1, 1, 1]
    #PD = [(1,2,3,4), (1,3,5,6), (2,4,6,5)] # 2_1^k [-1, -2, -2, -2, -1, -1
    #PD = [(3,4,5,6), (5,3,1,2),(6,4,2,1) ]  # 2交点のリンク。頂点置換。[-1, -2, -2, -2, -1, -1]
    #PD = [(1,2,3,4), (3,2,6,5), (1,4,5,6)] #2_1^k [-1, -2, -2, -2, -1, -1]
    #PD = [(1,2,3,4), (3,2,6,5), (4,5,6,1)] #交点の上下を変えてみた。[-1, -2, -3, -2, -1]

    #PD = [(1,8,4,3),(5,1,6,2), (2,6,3,7), (8,5,7,4)] # 3_1^k [1, 1, 0, 0, -1, -1, -2, -2, -2, -2, -1]
    #PD = [(1,6,10,5),(6,1,7,2),(7,5,8,4),(10,2,9,3),(4,8,3,9)] # 3_1^kの変形 [1, 1, 0, 0, -1, -1, -2, -2, -2, -2, -1]
    #PD = [(1,5,10,6),(6,2,7,1),(2,8,3,7),(8,4,9,3),(4,10,5,9)] # 4_1^k [-1, -1, -2, -1, -1, -1, -1, -1]
    #PD = [(1,5,10,6),(9,1,8,2),(2,8,3,7),(7,3,6,4),(5,9,4,10)] # 4_2^k [1, 0, -1, 0, 0, 1, 0, 0, -1, -2, -2, -2, -2, -1]
    #PD = [(1,8,10,7),(2,6,1,7),(5,3,6,2),(3,9,4,8),(9,5,10,4)] # 4_3^k  [-1, -1, 0, 0, 0, 0, -1, -1, -2, -1, -1, -1]
    #PD = [(1,6,12,5),(11,1,10,2),(2,10,3,9),(8,4,9,3),(4,8,5,7),(6,11,7,12)] # 5_1^k [1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, -1, -2, -2, -2, -2, -1]
    #PD = [(1,9,12,10),(9,1,8,2),(2,8,3,7),(6,4,7,3),(4,11,5,12),(11,6,10,5)] # 5_2^k [-1, -1, -1, -2, -2, -2, -2, -1, -1, 1, 1, 1, 1]
    #PD = [(1,8,12,7),(7,2,6,1),(2,10,3,9),(10,4,11,3),(4,12,5,11),(8,6,9,5)] # 5_3^k [1, 1, 0, 0, 0, -1, -1, 0, -1, 0, -1, -1, -2, -2, 0, -1, -1]
    #PD = [(1,4,12,3),(2,9,1,8),(7,2,8,3),(11,7,12,6),(6,10,5,11),(10,4,9,5)] # 5_4^k  [-1, -2, -2, -2, -3, -2, -1, -1, 0, 1, 0, 1, 1, 2, 1, 0, 1, -1, -1]
    #PD = [(1,7,12,8),(4,2,5,1),(2,9,3,10),(8,3,9,4),(10,6,11,5),(7,11,6,12)] # 5_5^k  [-1, -1, -1, -2, -2, -1, -2, -1, -1, 0, 1, 1, 2, 0, -1]
    #PD = [(1,5,12,6),(8,1,9,2),(2,9,3,10),(6,4,7,3),(4,12,5,11),(11,8,10,7)] # 5_6^k [1, 0, -2, 0, 0, -1, 0, -1, -1, -2, -1, 0, -1, 0, 1, -1, -1]
    #PD = [(1,4,12,3),(1,9,2,8),(9,3,10,2),(7,4,8,5),(5,10,6,11),(12,7,11,6)] # 5_7^k [-1, -1, 1, 1, 0, 2, 1, 0, 1, 0, 0, -2, -1, -1, -3, -2, -1, -2, -1]

    #PD = [(1,10,12,9),(7,2,6,1),(2,7,3,8),(9,4,8,3),(12,5,11,4),(5,10,6,11)] # 5_8^k [1, 0, 0, 0, -1, 0, -1, -1, -2, -2, -1, -1, -1]

    #PD = [(1,4,8,5), (5,6,2,1),(3,2,6,7),(4,3,7,8)]  # 3交点のリンク。3_1^l。[1, 1, 2, 1, 0, -1, -1]
    #PD = [(1,7,6,10),(1,8,2,7),(5,3,6,2),(3,9,4,10),(9,5,8,4)] #4_1^l [1, 1, 0, 0, 0, -1, 0, 0, 1, 1, 1, 1, -1, -1]
    #PD = [(1,7,6,12),(1,8,2,7),(8,3,9,2),(3,10,4,9),(10,5,11,4),(5,12,6,11)] #5_1^l [1, 1, 2, 1, 1, 0, -1, -1, -1]
    #PD = [(1,7,6,12),(10,2,9,1),(3,11,2,10),(11,3,12,4),(4,8,5,9),(8,6,7,5)] #5_2^l  [-1, 0, 0, -1, 1, 1, 2, 1, 1, 1, 0, 0, 0, -1, -1]
    #PD = [(1,12,4,5),(5,2,6,1),(2,9,3,8),(9,4,10,3),(8,10,7,11),(11,7,12,6)] #5_3^l [1, 0, -1, 1, 1, 0, 1, 0, 0, -1, 0, 1, 0, 1, 1, -1, -1]
    #PD = [(1,5,12,6),(11,2,10,1),(3,10,2,9),(9,4,8,3),(5,8,4,7),(6,12,7,11)] #平面グラフから生成。5_1^kと同じだった。
    #PD = [(8,4,7,5),(4,8,3,9),(9,3,10,2),(2,10,1,11),(6,1,5,12),(12,7,11,6)] # 平面グラフから生成。5_2^kと同じだった。
    #PD = [(1,7,6,12),(11,7,10,8),(3,10,4,9),(9,2,8,3),(5,12,6,11),(2,4,1,5)] #平面グラフから生成。 5_3^lと多項式は同じ。 [1, 0, -1, 1, 1, 0, 1, 0, 0, -1, 0, 1, 0, 1, 1, -1, -1]

    #PD = [(1,2,3,4),(2,11,6,10),(3,12,7,11),(4,13,8,12),(13,9,14,8),(9,1,10,5),(5,6,7,14)] #D論公聴会1 [1, 1, 0, 0, 0, -1, 0, 0, 1, 1, 1, 1, -1, -1]
    #PD = [(7, 3, 14, 1), (12, 5, 9, 11),(3, 6, 5, 8), (14, 13, 9, 6), (1, 4, 2, 13), (4, 10, 11, 2), (10, 7, 8, 12)]  # 上を置換したもの。[1, 1, 0, 0, 0, -1, 0, 0, 1, 1, 1, 1, -1, -1]
    #PD = [(1,2,3,4),(1,9,5,8),(3,7,6,2),(7,4,8,10),(9,6,10,5)] #D論公聴会2 上を変形したもの [1, 1, 0, 0, 0, -1, 0, 0, 1, 1, 1, 1, -1, -1]
    #PD = [(1,2,3,4),(1,10,5,9),(10,6,11,5),(6,2,7,13),(3,8,12,7),(8,4,9,14),(13,12,14,11)] #D論公聴会 非代数的 [-1, -1, 2, 1, -2, 2, 2, -2, 1, 2, -1, 0, 0, 1, -2, -1, 3, -2, -1, 3, 0, -1]
    #PD = [(1,15,14,18),(1,7,2,6),(7,3,8,2),(3,9,4,8),(4,11,5,12),(13,6,12,5),(15,10,16,9),(10,17,11,16),(13,17,14,18)] #D論公聴会 非代数的の同値変形)] 一つ上と同じ列が得られた。

    #diag = pd_list_to_diagram(PD, {0})
    
    #diagram_summary(diag)

    #以下は古いexample 将来的には削除予定
    #PD_minloop = [(1,1,2,2), (3,3,4,4)] 連結でないとvalidationエラー
    #diag = pd_list_to_diagram(PD_minloop, {0,1})   # 2 つとも flat
    #G = build_flat_graph(diag, verbose=True)
    #diagram_summary(diag)

    # PD of a 1-crossing bouquet-like gadget
    #pd = [(1, 2, 1, 2)]                 # one 4-valent vertex, labels 1,2 repeated
    #pd = [(1, 2),(1,2,3,3)]                 # one 4-valent vertex, labels 1,2,3,4
    #flat_indices = {0}                  # it's already flat
    #D = pd_list_to_diagram(pd, flat_indices)
    #diagram_summary(D, '1-crossing bouquet-like')
    """
    # double triangle test
    dt = nx.MultiGraph()
    dt.add_nodes_from([0,1,2])
    for u, v in [(0,1),(1,2),(2,0)]:
        dt.add_edge(u, v)
        dt.add_edge(u, v)
    Y_dt = sp.expand(yamada_flat(dt, q))
    print('Double Triangle Yamada')
    yamada_poly_and_coeffs(Y_dt, q)
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
    """
    """
    PD_octahedron = [
        (1, 2, 3, 4),   # flat vertex
        (1, 5, 9, 6),   # flat vertex
        (2, 6, 10, 7),  # flat vertex
        (3, 7, 11, 8),  # flat vertex
        (4, 8, 0, 5),   # flat vertex
        (9, 10, 11, 0)  # flat vertex
    ]
    diag = pd_list_to_diagram(PD_octahedron, {0,1,2,3,4,5})
    diagram_summary(diag,"octahedron")
    """
