import random
from existing_14.helpers import _get_bfs_distances

def naive_myopic_bfs(adj, nodes, alpha, k, G_mapped, **kwargs):
    """
    Algorithm 2: Naive Myopic BFS
    Estimates probabilities once from the initial seed, then picks k lowest nodes.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    dists = _get_bfs_distances(adj, s0, n)
    
    probs = []
    for v in range(n):
        d = dists[v]
        if d != -1:
            probs.append(alpha ** d)
        else:
            probs.append(0.0)
            
    candidates = [v for v in range(n) if v != s0]
    sorted_candidates = sorted(candidates, key=lambda v: probs[v])
    
    seeds = [s0] + sorted_candidates[:k - 1]
    return seeds
