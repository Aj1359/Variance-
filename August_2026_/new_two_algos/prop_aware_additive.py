import random
from icm import prob_est_timed

def _get_nearby_nodes_and_dists(adj, v, L):
    """Compute BFS distance from node v to all neighbors within L hops."""
    dists = {v: 0}
    queue = [v]
    while queue:
        curr = queue.pop(0)
        curr_d = dists[curr]
        if curr_d < L:
            for nbr in adj[curr]:
                if nbr not in dists:
                    dists[nbr] = curr_d + 1
                    queue.append(nbr)
    return dists

def prop_aware_additive(adj, nodes, alpha, k, T, R_mc, pr_dict=None, kcore_dict=None, use_kcore=False, L=2, lam=0.5, **kwargs):
    """
    IIT Bhilai / LIACS: Propagation-Aware Additive Fair Seed Selection.
    """
    n = len(nodes)
    if use_kcore and kcore_dict:
        s0 = max(range(n), key=lambda v: (kcore_dict.get(v, 0), len(adj[v])))
    elif pr_dict:
        s0 = max(range(n), key=lambda v: pr_dict.get(v, 0.0))
    else:
        s0 = random.randint(0, n - 1)
        
    seeds = [s0]
    
    for _ in range(1, k):
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R_mc)
        D = [1.0 - p for p in probs]
        
        R_scores = [0.0] * n
        candidates = [v for v in range(n) if v not in seeds]
        
        for v in candidates:
            nearby = _get_nearby_nodes_and_dists(adj, v, L)
            r_val = 0.0
            for u, dist in nearby.items():
                if u != v:
                    r_val += D[u] * (alpha ** dist)
            R_scores[v] = r_val
            
        best_cand = -1
        max_score = -float('inf')
        
        for v in candidates:
            score = lam * D[v] + (1.0 - lam) * R_scores[v]
                
            if score > max_score:
                max_score = score
                best_cand = v
                
        if best_cand == -1:
            best_cand = random.choice(candidates)
            
        seeds.append(best_cand)
        
    return seeds
