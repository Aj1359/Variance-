import random
from existing_14.helpers import _get_bfs_distances

def myopic_bfs(adj, nodes, alpha, k, G_mapped, **kwargs):
    """
    Algorithm 1: Myopic BFS
    Approximates activation probabilities via BFS distances.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    seeds = [s0]
    
    # Store shortest distances from each seed
    dist_matrix = [_get_bfs_distances(adj, s0, n)]
    
    for _ in range(1, k):
        probs = []
        for v in range(n):
            if v in seeds:
                probs.append(1.0)
                continue
            fail_prob = 1.0
            for s_idx in range(len(seeds)):
                d = dist_matrix[s_idx][v]
                if d != -1:
                    fail_prob *= (1.0 - (alpha ** d))
            probs.append(1.0 - fail_prob)
            
        best_cand = -1
        min_p = float('inf')
        for v in range(n):
            if v not in seeds:
                if probs[v] < min_p:
                    min_p = probs[v]
                    best_cand = v
                    
        seeds.append(best_cand)
        dist_matrix.append(_get_bfs_distances(adj, best_cand, n))
        
    return seeds
