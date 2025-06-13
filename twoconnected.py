import itertools
import sys
from collections import Counter, defaultdict, deque

def log(msg):
    print(msg, file=sys.stderr)

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
                log(f"除去: {line.strip()} ← 辺 {e1}(#{n1+1}), {e2}(#{n2+1}) を除くと非連結")
            return True
    return False

def main():
    cnt = 0
    # 標準入力から 1 行ずつ読み込む
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        # find_2edge_disconnect に渡してスキップ判定
        if find_2edge_disconnect(line, printlog=True):
            continue

        # スキップされなかったら出力・カウント
        print(line)
        cnt += 1

    log(f"残った行数: {cnt}")
if __name__ == '__main__':
    main()
