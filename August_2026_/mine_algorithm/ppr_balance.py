import random
from mine_algorithm.helpers import _personalized_pagerank

def ppr_balance(adj, nodes, alpha, k, G_mapped, **kwargs):
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    seeds = [s0]
    
    while len(seeds) < k:
        ppr = _personalized_pagerank(G_mapped, seeds, n)
        non_seed_pprs = [ppr.get(v, 0.0) for v in range(n) if v not in seeds]
        mu_ppr = sum(non_seed_pprs) / len(non_seed_pprs)
        
        underserved = [v for v in range(n) if v not in seeds and ppr.get(v, 0.0) < mu_ppr]
        if not underserved:
            underserved = [v for v in range(n) if v not in seeds]
            
        best_cand = min(underserved, key=lambda v: abs(ppr.get(v, 0.0) - mu_ppr))
        seeds.append(best_cand)
        
    return seeds
