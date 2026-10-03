import random
from existing_14.helpers import _get_ppr

def naive_myopic_ppr(adj, nodes, alpha, k, G_mapped, **kwargs):
    """
    Algorithm 4: Naive Myopic PPR
    Run PPR once restarting from the initial seed, pick k lowest nodes.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    ppr = _get_ppr(G_mapped, [s0], n)
    
    candidates = [v for v in range(n) if v != s0]
    sorted_candidates = sorted(candidates, key=lambda v: ppr.get(v, 0.0))
    
    seeds = [s0] + sorted_candidates[:k - 1]
    return seeds
