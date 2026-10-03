import numpy as np
from mine_algorithm.helpers import _bfs_distances

def degree_median_spread(adj, nodes, alpha, k, G_mapped, **kwargs):
    n = len(nodes)
    degrees = [len(adj[v]) for v in range(n)]
    d_med = np.median(degrees)
    
    s0 = min(range(n), key=lambda v: abs(len(adj[v]) - d_med))
    seeds = [s0]
    
    while len(seeds) < k:
        dists = _bfs_distances(adj, seeds, n)
        non_seed_dists = [dists[v] if dists[v] != -1 else float('inf') for v in range(n) if v not in seeds]
        med = np.median(non_seed_dists)
        
        C = [v for v in range(n) if v not in seeds and (dists[v] if dists[v] != -1 else float('inf')) >= med]
        if not C:
            C = [v for v in range(n) if v not in seeds]
            
        best_cand = min(C, key=lambda v: abs(len(adj[v]) - d_med))
        seeds.append(best_cand)
        
    return seeds
