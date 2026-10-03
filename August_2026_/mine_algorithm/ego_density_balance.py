import networkx as nx
import numpy as np
from mine_algorithm.helpers import _bfs_distances

def ego_density_balance(adj, nodes, alpha, k, G_mapped, **kwargs):
    n = len(nodes)
    lcc = nx.clustering(G_mapped)
    c_avg = sum(lcc.values()) / n
    
    s0 = min(range(n), key=lambda v: abs(lcc.get(v, 0.0) - c_avg))
    seeds = [s0]
    
    while len(seeds) < k:
        dists = _bfs_distances(adj, seeds, n)
        non_seed_dists = [dists[v] if dists[v] != -1 else float('inf') for v in range(n) if v not in seeds]
        med = np.median(non_seed_dists)
        
        C = [v for v in range(n) if v not in seeds and (dists[v] if dists[v] != -1 else float('inf')) >= med]
        if not C:
            C = [v for v in range(n) if v not in seeds]
            
        best_cand = min(C, key=lambda v: abs(lcc.get(v, 0.0) - c_avg))
        seeds.append(best_cand)
        
    return seeds
