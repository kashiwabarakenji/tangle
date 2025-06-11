import itertools
from collections import Counter, defaultdict, deque

def parse_ascii_multiedges_correct(line):
    """無向辺の多重数を正確に取得する（plantri形式用）"""
    parts = line.strip().split()
    if len(parts) < 2:
        return []
    vertex_lists = parts[1].split(",")
    n = len(vertex_lists)
    # すべての(小さい方, 大きい方)で出現回数カウント
    edge_counter = Counter()
    for vid, adj in enumerate(vertex_lists):
        v = chr(ord('a') + vid)
        for u in adj:
            a, b = sorted([v, u])
            edge_counter[(a, b)] += 1
    # 2で割って本数に
    multi_edges = []
    for (a, b), count in edge_counter.items():
        real_count = count // 2
        for i in range(real_count):
            multi_edges.append(((a, b), i))
    return multi_edges

def build_adjlist_from_edgeids(edges, remove_indices=None):
    adj = defaultdict(list)
    for idx, ((u, v), edge_num) in enumerate(edges):
        if remove_indices and idx in remove_indices:
            continue
        adj[u].append(v)
        adj[v].append(u)
    return adj

def is_connected(adj, vertices):
    if not vertices:
        return True
    visited = set()
    q = deque([vertices[0]])
    while q:
        v = q.popleft()
        if v in visited:
            continue
        visited.add(v)
        for u in adj[v]:
            if u not in visited:
                q.append(u)
    return len(visited) == len(vertices)

def find_2edge_disconnect(line, printlog=True):
    edges = parse_ascii_multiedges_correct(line)
    if len(edges) < 2:
        return False
    vertices = sorted({a for ((a, b), n) in edges} | {b for ((a, b), n) in edges})
    edge_indices = list(range(len(edges)))
    for i, j in itertools.combinations(edge_indices, 2):
        adj = build_adjlist_from_edgeids(edges, remove_indices={i, j})
        if not is_connected(adj, vertices):
            if printlog:
                e1, n1 = edges[i]
                e2, n2 = edges[j]
                print(f"除去: {line.strip()} ← 辺 {e1}(#{n1+1}), {e2}(#{n2+1}) を除くと非連結")
            return True
    return False

def main():
    lines = [
        "6 bbcb,aada,aedd,bccf,cfff,deee",
        "6 bbcb,aada,aded,bcfc,cfff,deee",
        "6 bcbb,aaad,aeee,bfff,cfcc,dedd",
        "6 bbcb,aada,aeed,bcff,cffc,deed",
        "6 bbcb,aada,aeff,bfee,cddf,cedc",
        "6 bccb,aadc,abea,beff,cffd,deed",
        "6 bbcd,adda,aeef,afbb,cffc,ceed",
        "6 bbcd,adda,adef,acbb,cfff,ceee",
        "6 bbcc,adda,aeea,bffb,cffc,deed",
        "6 bcde,aeec,abef,afff,acbb,cddd",
        "6 bbcd,adca,abee,affb,cffc,deed",
        "6 bccd,aeef,affa,afee,bddb,bdcc",
        "6 bbcd,adca,abef,afeb,cdff,ceed",
        "6 bbcd,adea,affd,aceb,bdff,ceec",
        "6 bbcd,adea,aeff,afeb,bdfc,cedc",
        "6 bbcd,adea,aeef,affb,bfcc,cedd",
        "6 bbcd,aefa,affd,acee,bddf,becc",
        "6 bcde,aefc,abfd,acfe,adfb,bedc"
    ]
    cnt = 0
    for line in lines:
        if find_2edge_disconnect(line, printlog=True):
            continue
        print(line)
        cnt += 1
    print(f"残った行数: {cnt}")

if __name__ == '__main__':
    main()
