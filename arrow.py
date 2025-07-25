import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import math

parser = argparse.ArgumentParser()
parser.add_argument('--flip-signs', action='store_true', help='2重辺のpreferred_signsを全部逆向きにする')
args = parser.parse_args()


with open('data.json', encoding='utf-8') as f:
    data = json.load(f)

coords = {int(k): tuple(v) for k, v in data['coords'].items()}
eid2verts = {int(k): tuple(v) for k, v in data['eid2verts'].items()}

preferred_signs = {int(k): v for k, v in data['preferred_signs'].items()}
z_flags = {k: v for k, v in data['z_flags'].items()}
flat_vertex = data['flat_vertex']

if args.flip_signs:
    preferred_signs = {k: -v if v != 0 else 0 for k, v in preferred_signs.items()}


G = nx.MultiGraph()
for v in coords:
    G.add_node(v)
for eid, (u, v) in eid2verts.items():
    G.add_edge(u, v, key=eid)

fig, ax = plt.subplots(figsize=(5, 5))

# 頂点丸は小さくor消す
for v, (x, y) in coords.items():
    if v == flat_vertex:
        # 丸を小さく＆色も赤系
        ax.scatter(x, y, color='red', s=0, zorder=5)
        ax.text(x, y, str(v), fontsize=20, fontweight='bold', color='red', va='center', ha='center', zorder=5)
    else:
        ax.scatter(x, y, color='white', edgecolors='black', s=0, zorder=5)
        ax.text(x, y, str(v), fontsize=14, color='k', va='center', ha='center', zorder=6)

for eid, (u, v) in eid2verts.items():
    sign = preferred_signs.get(eid, 0)
    rad = 0.15 * sign if sign != 0 else 0

    # 矢印方向判定
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
        coords[u], coords[v],
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=arrowstyle,
        color='royalblue',         # ←青色に
        linewidth=2.0,
        mutation_scale=15,
        shrinkA=20,                # ←先端を短く
        shrinkB=20,
        zorder=2
    )
    ax.add_patch(patch)

    xm = (coords[u][0] + coords[v][0]) / 2
    ym = (coords[u][1] + coords[v][1]) / 2
    dx = coords[v][1] - coords[u][1]
    dy = -(coords[v][0] - coords[u][0])
    norm = math.hypot(dx, dy) or 1
    off = rad * 0.5 if rad != 0 else 0.02
    ax.text(
        xm + dx / norm * off,
        ym + dy / norm * off,
        str(eid),
        fontsize=10,
        ha="center", va="center",
        color="darkblue",
        zorder=8
    )

xs, ys = zip(*coords.values())
x_center = (min(xs) + max(xs)) / 2
y_top = max(ys) + 0.08

ax.text(
    x_center, y_top,
    data.get('label', ''),
    fontsize=18,
    ha='center', va='bottom',
    fontweight='bold',
    color='green',
    zorder=100
)

# 軸範囲調整もお忘れなく
ax.set_xlim(min(xs) - 0.2, max(xs) + 0.2)
ax.set_ylim(min(ys) - 0.2, max(ys) + 0.25)

ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.savefig("output.png", bbox_inches="tight", pad_inches=0.1)
plt.show()
