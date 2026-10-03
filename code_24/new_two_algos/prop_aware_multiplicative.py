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

def get_pareto_frontier(candidates, D_scores, R_scores):
    """
    Filter candidates to the Pareto frontier.
    u dominates v if D[u] >= D[v] and R[u] >= R[v] with at least one strict inequality.
    """
    n_cand = len(candidates)
    dominated = [False] * n_cand
    for i in range(n_cand):
        v = candidates[i]
        v_d = D_scores[v]
        v_r = R_scores[v]
        for j in range(n_cand):
            if i == j:
                continue
            u = candidates[j]
            u_d = D_scores[u]
            u_r = R_scores[u]
            if u_d >= v_d and u_r >= v_r:
                if u_d > v_d or u_r > v_r:
                    dominated[i] = True
                    break
    return [candidates[i] for i in range(n_cand) if not dominated[i]]

def prop_aware_multiplicative(adj, nodes, alpha, k, T, R_mc, pr_dict=None, kcore_dict=None, use_kcore=False, L=2, lam=0.5, use_pareto=True, **kwargs):
    """
    IIT Bhilai / LIACS: Propagation-Aware Multiplicative Fair Seed Selection.
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
            
        D_raw = {v: D[v] for v in candidates}
        R_raw = {v: R_scores[v] for v in candidates}
        
        eligible_candidates = candidates
        if use_pareto:
            eligible_candidates = get_pareto_frontier(candidates, D_raw, R_raw)
            if not eligible_candidates:
                eligible_candidates = candidates
                
        best_cand = -1
        max_score = -float('inf')
        
        for v in eligible_candidates:
            # Multiplicative scoring using raw values
            score = ((D[v] + 1e-9) ** lam) * ((R_scores[v] + 1e-9) ** (1.0 - lam))
                
            if score > max_score:
                max_score = score
                best_cand = v
                
        if best_cand == -1:
            best_cand = random.choice(candidates)
            
        seeds.append(best_cand)
        
    return seeds
