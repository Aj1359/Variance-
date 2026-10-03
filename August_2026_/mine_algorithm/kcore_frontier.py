from mine_algorithm.helpers import _bfs_distances

def kcore_frontier(adj, nodes, alpha, k, G_mapped, kcore_dict, **kwargs):
    n = len(nodes)
    c_min0 = min(kcore_dict.values())
    candidates0 = [v for v in range(n) if kcore_dict.get(v, 0) == c_min0]
    s0 = max(candidates0, key=lambda v: len(adj[v]))
    seeds = [s0]
    
    while len(seeds) < k:
        c_min = min(kcore_dict.get(v, 0) for v in range(n) if v not in seeds)
        C = [v for v in range(n) if v not in seeds and kcore_dict.get(v, 0) == c_min]
        
        dists = _bfs_distances(adj, seeds, n)
        best_cand = -1
        max_dist = -1
        best_deg = -1
        for v in C:
            d = dists[v] if dists[v] != -1 else float('inf')
            if d > max_dist:
                max_dist = d
                best_deg = len(adj[v])
                best_cand = v
            elif d == max_dist:
                if len(adj[v]) > best_deg:
                    best_deg = len(adj[v])
                    best_cand = v
        seeds.append(best_cand)
        
    return seeds
