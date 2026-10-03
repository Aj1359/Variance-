import networkx as nx
import numpy as np
from mine_algorithm.helpers import _bfs_distances

def betweenness_gateway(adj, nodes, alpha, k, G_mapped, betweenness_dict=None, **kwargs):
    n = len(nodes)
    if betweenness_dict is not None:
        bc = betweenness_dict
    else:
        # Use sampling approximation for large graphs if not precomputed
        bc = nx.betweenness_centrality(G_mapped, k=min(n, 200))
        
    s0 = max(range(n), key=lambda v: bc.get(v, 0.0))
    seeds = [s0]
    
    while len(seeds) < k:
        dists = _bfs_distances(adj, seeds, n)
        non_seed_dists = [dists[v] if dists[v] != -1 else float('inf') for v in range(n) if v not in seeds]
        med = np.median(non_seed_dists)
        
        C = [v for v in range(n) if v not in seeds and (dists[v] if dists[v] != -1 else float('inf')) >= med]
        if not C:
            C = [v for v in range(n) if v not in seeds]
            
        best_cand = max(C, key=lambda v: bc.get(v, 0.0))
        seeds.append(best_cand)
        
    return seeds
