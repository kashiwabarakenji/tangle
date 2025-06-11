import sys

def has_self_loop_ascii(line):
    # フォーマット: "6 abca,abbd,acce,bddf,ceef,dffe"
    parts = line.strip().split()
    if len(parts) < 2:
        return False
    vertex_lists = parts[1].split(",")
    for vid, adj in enumerate(vertex_lists):
        v = chr(ord('a') + vid)
        if v in adj:
            return True
    return False

def main():
    for line in sys.stdin:
        if not has_self_loop_ascii(line):
            print(line, end='')  # もとの改行を保持

if __name__ == '__main__':
    main()
    
