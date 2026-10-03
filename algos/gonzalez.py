import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def _highest_degree(adj, n, exclude):
    return max((i for i in range(n) if i not in exclude), key=lambda i: len(adj[i]))

def _bfs_distances(adj, sources, n):
    dist = [math.inf] * n
    queue = []
    for s in sources:
        dist[s] = 0
        queue.append(s)
    head = 0
    while head < len(queue):
        v = queue[head]; head += 1
        for u in adj[v]:
            if dist[u] == math.inf:
                dist[u] = dist[v] + 1
                queue.append(u)
    return dist

def gonzalez(adj, n, k):
    seeds = [_highest_degree(adj, n, set())]
    seed_set = set(seeds)

    for _ in range(k - 1):
        dist = _bfs_distances(adj, seed_set, n)
        v_star = max(
            (i for i in range(n) if i not in seed_set),
            key=lambda i: dist[i]
        )
        seeds.append(v_star)
        seed_set.add(v_star)

    return seeds
