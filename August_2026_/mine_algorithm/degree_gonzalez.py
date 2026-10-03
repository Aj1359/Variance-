from mine_algorithm.helpers import _bfs_distances

def degree_gonzalez(adj, nodes, alpha, k, G_mapped, **kwargs):
    n = len(nodes)
    s0 = max(range(n), key=lambda v: len(adj[v]))
    seeds = [s0]
    
    while len(seeds) < k:
        dists = _bfs_distances(adj, seeds, n)
        best_cand = -1
        max_dist = -1
        min_deg = float('inf')
        
        for v in range(n):
            if v not in seeds:
                d = dists[v] if dists[v] != -1 else float('inf')
                if d > max_dist:
                    max_dist = d
                    min_deg = len(adj[v])
                    best_cand = v
                elif d == max_dist:
                    if len(adj[v]) < min_deg:
                        min_deg = len(adj[v])
                        best_cand = v
        seeds.append(best_cand)
        
    return seeds
