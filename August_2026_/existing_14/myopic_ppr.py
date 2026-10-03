import random
from existing_14.helpers import _get_ppr

def myopic_ppr(adj, nodes, alpha, k, G_mapped, **kwargs):
    """
    Algorithm 3: Myopic PPR
    For each step, run PPR restarting from the current seeds, pick lowest node.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    seeds = [s0]
    
    for _ in range(1, k):
        ppr = _get_ppr(G_mapped, seeds, n)
        best_cand = -1
        min_ppr = float('inf')
        for v in range(n):
            if v not in seeds:
                if ppr.get(v, 0.0) < min_ppr:
                    min_ppr = ppr.get(v, 0.0)
                    best_cand = v
        seeds.append(best_cand)
        
    return seeds
