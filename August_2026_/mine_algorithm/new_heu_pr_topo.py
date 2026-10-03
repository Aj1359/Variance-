import random
import networkx as nx
from mine_algorithm.helpers import _bfs_distances, _personalized_pagerank

def new_heu_pr_topo(adj, nodes, alpha, k, G_mapped, pr_dict=None, epsilon=0.01, **kwargs):
    """
    11. PageRank Topo V1
    Same as new_heu_pr, but estimates access using Personalized PageRank restarted on seeds.
    """
    n = len(nodes)
    if pr_dict is not None:
        pr = pr_dict
    else:
        pr = nx.pagerank(G_mapped)
        
    s0 = max(range(n), key=lambda v: pr.get(v, 0.0))
    seeds = [s0]
    
    for _ in range(1, k):
        ppr = _personalized_pagerank(G_mapped, seeds, n)
        min_p = min(ppr.get(v, 0.0) for v in range(n) if v not in seeds)
        cand_indices = [v for v in range(n) if v not in seeds and ppr.get(v, 0.0) <= min_p + epsilon]
        
        dists = _bfs_distances(adj, seeds, n)
        best_cand = None
        best_dist = -2
        best_pr = -1.0
        
        for c in cand_indices:
            c_dist = dists[c] if dists[c] != -1 else float('inf')
            c_pr = pr.get(c, 0.0)
            if c_dist > best_dist:
                best_dist = c_dist
                best_pr = c_pr
                best_cand = c
            elif c_dist == best_dist:
                if c_pr > best_pr:
                    best_pr = c_pr
                    best_cand = c
        if best_cand is None:
            best_cand = cand_indices[0] if cand_indices else random.choice([v for v in range(n) if v not in seeds])
        seeds.append(best_cand)
        
    return seeds
