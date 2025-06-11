import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

def parse_ascii_multiedges(line):
    parts = line.strip().split()
    if len(parts) < 2:
        return []
    vertex_lists = parts[1].split(",")
    edge_counter = Counter()
    for vid, adj in enumerate(vertex_lists):
        v = chr(ord('a') + vid)
        for u in adj:
            a, b = sorted([v, u])
            edge_counter[(a, b)] += 1
    return edge_counter

def draw_planar_multigraph_ascii(line):
    edge_counter = parse_ascii_multiedges(line)
    # MultiGraphだとplanar_layout未対応。まずは普通のGraphで枝だけでレイアウト計算
    Gsimple = nx.Graph()
    for (u, v), count in edge_counter.items():
        Gsimple.add_edge(u, v)
    is_planar, _ = nx.check_planarity(Gsimple)
    if not is_planar:
        print("!! Planarity check failed !!")
        pos = nx.spring_layout(Gsimple)
    else:
        pos = nx.planar_layout(Gsimple)
    # 可視化用のMultiGraph生成
    G = nx.MultiGraph()
    for (u, v), count in edge_counter.items():
        for _ in range(count):
            G.add_edge(u, v)
    plt.figure(figsize=(5,5))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=16)
    for (u, v), count in edge_counter.items():
        width = 1.0 + (count - 1) * 3.0
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.7, edge_color='orangered')
    plt.title(line.strip())
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    lines = [
        "6 bbcc,adda,aeea,bffb,cffc,deed",
        "6 bbcd,adca,abee,affb,cffc,deed",
        "6 bccd,aeef,affa,afee,bddb,bdcc",
        "6 bbcd,adca,abef,afeb,cdff,ceed",
        "6 bbcd,adea,affd,aceb,bdff,ceec",
        "6 bbcd,adea,aeff,afeb,bdfc,cedc",
        "6 bbcd,adea,aeef,affb,bfcc,cedd",
        "6 bbcd,aefa,affd,acee,bddf,becc",
        "6 bcde,aefc,abfd,acfe,adfb,bedc",

        "6 bbcd,adca,abee,affb,cffc,deed",
        "6 bccd,aeef,affa,afee,bddb,bdcc",
        "6 bbcd,adca,abef,afeb,cdff,ceed",
        "6 bbcd,adea,affd,aceb,bdff,ceec",
        "6 bbcd,adea,aeff,afeb,bdfc,cedc",
        "6 bbcd,adea,aeef,affb,bfcc,cedd",
        "6 bbcd,aefa,affd,acee,bddf,becc",
        "6 bcde,aefc,abfd,acfe,adfb,bedc"
    ]
    for line in lines:
        draw_planar_multigraph_ascii(line)
        input("Enterキーで次へ...")
