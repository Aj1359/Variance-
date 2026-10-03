import networkx as nx
from mine_algorithm.helpers import _bfs_distances, _personalized_pagerank

def new_heu_pr_v2_topo(adj, nodes, alpha, k, G_mapped, pr_dict=None, epsilon=0.01, hop_bucket_width=1, **kwargs):
    """
    12. PageRank Topo V2
    Same as new_heu_pr_v2, but estimates access using Personalized PageRank restarted on seeds.
    """
    n = len(nodes)
    if pr_dict is not None:
        pr = pr_dict
    else:
        pr = nx.pagerank(G_mapped)
        
    s0 = max(range(n), key=lambda v: pr.get(v, 0.0))
    seeds = [s0]
    
    avg_degree = sum(len(adj[i]) for i in range(n)) / n
    
    for _ in range(1, k):
        ppr = _personalized_pagerank(G_mapped, seeds, n)
        min_p = min(ppr.get(v, 0.0) for v in range(n) if v not in seeds)
        cand_indices = [v for v in range(n) if v not in seeds and ppr.get(v, 0.0) <= min_p + epsilon]
        
        dists = _bfs_distances(adj, seeds, n)
        def hop(i):
            return dists[i] if dists[i] != -1 else float('inf')
            
        max_hop = max((hop(c) for c in cand_indices), default=0)
        bucket = [c for c in cand_indices if max_hop - hop(c) <= hop_bucket_width]
        if not bucket:
            bucket = cand_indices
            
        min_ppr = min(ppr.get(c, 0.0) for c in bucket)
        ppr_candidates = [c for c in bucket if ppr.get(c, 0.0) <= min_ppr + epsilon]
        if not ppr_candidates:
            ppr_candidates = bucket
            
        best_cand = min(ppr_candidates, key=lambda c: abs(len(adj[c]) - avg_degree))
        seeds.append(best_cand)
        
    return seeds
