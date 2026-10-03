import numpy as np
from mine_algorithm.helpers import _bfs_distances

def harmonic_spread(adj, nodes, alpha, k, G_mapped, harmonic_dict, **kwargs):
    n = len(nodes)
    s0 = min(range(n), key=lambda v: harmonic_dict.get(v, 0.0))
    seeds = [s0]
    
    while len(seeds) < k:
        dists = _bfs_distances(adj, seeds, n)
        non_seed_dists = [dists[v] if dists[v] != -1 else float('inf') for v in range(n) if v not in seeds]
        med = np.median(non_seed_dists)
        
        C = [v for v in range(n) if v not in seeds and (dists[v] if dists[v] != -1 else float('inf')) >= med]
        if not C:
            C = [v for v in range(n) if v not in seeds]
            
        best_cand = min(C, key=lambda v: harmonic_dict.get(v, 0.0))
        seeds.append(best_cand)
        
    return seeds
