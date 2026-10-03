import networkx as nx
import numpy as np
from mine_algorithm.helpers import _bfs_distances

def eccentricity_spread(adj, nodes, alpha, k, G_mapped, ecc_dict=None, **kwargs):
    n = len(nodes)
    if ecc_dict is not None:
        ecc = ecc_dict
    else:
        ecc = {}
        for C_nodes in nx.connected_components(G_mapped):
            G_sub = G_mapped.subgraph(C_nodes)
            comp_ecc = nx.eccentricity(G_sub)
            ecc.update(comp_ecc)
        
    s0 = max(range(n), key=lambda v: ecc.get(v, 0))
    seeds = [s0]
    
    while len(seeds) < k:
        dists = _bfs_distances(adj, seeds, n)
        non_seed_dists = [dists[v] if dists[v] != -1 else float('inf') for v in range(n) if v not in seeds]
        med = np.median(non_seed_dists)
        
        C = [v for v in range(n) if v not in seeds and (dists[v] if dists[v] != -1 else float('inf')) >= med]
        if not C:
            C = [v for v in range(n) if v not in seeds]
            
        best_cand = max(C, key=lambda v: ecc.get(v, 0))
        seeds.append(best_cand)
        
    return seeds
