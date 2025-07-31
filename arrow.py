import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import math

# --- コマンドライン引数定義 ---
parser = argparse.ArgumentParser()
parser.add_argument('--flip-signs', action='store_true', help='2重辺のpreferred_signsを全部逆向きにする')
parser.add_argument('-G', '--general-graph', action='store_true', help='矢印やflat vertex強調なしの通常グラフモード')
args = parser.parse_args()

# --- JSONデータ読み込み ---
with open('data.json', encoding='utf-8') as f:
    data = json.load(f)

coords = {int(k): tuple(v) for k, v in data['coords'].items()}
eid2verts = {int(k): tuple(v) for k, v in data['eid2verts'].items()}
preferred_signs = {int(k): v for k, v in data['preferred_signs'].items()}
z_flags = {k: v for k, v in data['z_flags'].items()}
flat_vertex = data['flat_vertex']

# --- preferred_signsを反転するオプション ---
if args.flip_signs:
    preferred_signs = {k: -v if v != 0 else 0 for k, v in preferred_signs.items()}

# --- オレンジ色にする辺のセットを決める ---
routes = data.get('vertex_edge_routes', [])
orange_edges = set()
if len(routes) >= 2:
    lens = [len(route[1::2]) for route in routes]
    min_len = min(lens)
    chosen_idx = lens.index(min_len)  # 最初の最短
    orange_edges = set(routes[chosen_idx][1::2])
# 1本のみ or 無し→ orange_edgesは空集合（全部青）

# --- グラフ構築 ---
G = nx.MultiGraph()
for v in coords:
    G.add_node(v)
for eid, (u, v) in eid2verts.items():
    G.add_edge(u, v, key=eid)

fig, ax = plt.subplots(figsize=(5, 5))

# --- 頂点表示 ---
if args.general_graph:
    for v, (x, y) in coords.items():
        ax.text(x, y, str(v), fontsize=14, color='k', va='center', ha='center', zorder=6)
else:
    for v, (x, y) in coords.items():
        if v == flat_vertex:
            ax.text(x, y, str(v), fontsize=20, fontweight='bold', color='red', va='center', ha='center', zorder=6)
        else:
            ax.text(x, y, str(v), fontsize=14, color='k', va='center', ha='center', zorder=6)

# --- 辺の描画 ---
for eid, (u, v) in eid2verts.items():
    sign = preferred_signs.get(eid, 0)
    rad = 0.15 * sign if sign != 0 else 0

    # --- 始点終点をわずかに内側に補正して、頂点と重ならないように ---
    x0, y0 = coords[u]
    x1, y1 = coords[v]
    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)
    shrink_len = 0.02  # 頂点半径より少し大きめ（調整可）
    if dist > 0.0001:
        x0_ = x0 + dx * (shrink_len / dist)
        y0_ = y0 + dy * (shrink_len / dist)
        x1_ = x1 - dx * (shrink_len / dist)
        y1_ = y1 - dy * (shrink_len / dist)
    else:
        x0_, y0_, x1_, y1_ = x0, y0, x1, y1

    # --- 辺の色分け ---
    color = 'orange' if eid in orange_edges else 'royalblue'

    # --- 辺描画 ---
    if args.general_graph:
        # 通常グラフモード（矢印なし、flat_vertex強調なし）
        arrowstyle = "-"
    else:
        # 普通モード：矢印設定
        z1 = z_flags.get(f"{eid}-{u}", 1)
        z2 = z_flags.get(f"{eid}-{v}", 1)
        if z1 == -1 and z2 == -1:
            arrowstyle = "<->"
        elif z1 == -1:
            arrowstyle = "<-"
        elif z2 == -1:
            arrowstyle = "->"
        else:
            arrowstyle = "-"

    patch = mpatches.FancyArrowPatch(
        (x0_, y0_), (x1_, y1_),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=arrowstyle,
        color=color,
        linewidth=2.0,
        mutation_scale=15,
        shrinkA=5,
        shrinkB=5,
        zorder=2
    )
    ax.add_patch(patch)

    # --- 辺ラベル描画 ---
    xm = (x0 + x1) / 2
    ym = (y0 + y1) / 2
    ldx = y1 - y0
    ldy = -(x1 - x0)
    norm = math.hypot(ldx, ldy) or 1
    off = rad * 0.5 if rad != 0 else 0.02
    ax.text(
        xm + ldx / norm * off,
        ym + ldy / norm * off,
        str(eid),
        fontsize=10,
        ha="center", va="center",
        color="darkblue",
        zorder=8
    )

# --- ラベル表示（図の上部中央に余白付きで） ---
xs, ys = zip(*coords.values())
x_center = (min(xs) + max(xs)) / 2
y_top = max(ys) + 0.08
if not args.general_graph:
    ax.text(
        x_center, y_top,
        data.get('label', ''),
        fontsize=18,
        ha='center', va='bottom',
        fontweight='bold',
        color='green',
        zorder=100
    )

# --- 描画範囲/仕上げ ---
ax.set_xlim(min(xs) - 0.2, max(xs) + 0.2)
ax.set_ylim(min(ys) - 0.2, max(ys) + 0.25)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.savefig("output.png", bbox_inches="tight", pad_inches=0.1)
plt.show()