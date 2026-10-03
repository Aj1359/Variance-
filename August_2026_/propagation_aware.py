import random
import networkx as nx
import numpy as np

def get_nearby_nodes_and_dists(adj, v, L):
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

def propagation_aware_fairness(adj, nodes, alpha, k, T, R_mc, pr_dict, L=2, lam=0.5, multiplicative=False, use_pareto=False, **kwargs):
    """
    Algorithm 1: Propagation-Aware Fair Seed Selection from pdf_text_2.txt.
    Supports additive/multiplicative scoring and optional Pareto candidate filtering.
    """
    from icm import prob_est_timed
    n = len(nodes)
    
    # 1. First seed node based on highest PageRank
    if pr_dict:
        s0 = max(range(n), key=lambda v: pr_dict.get(v, 0.0))
    else:
        s0 = random.randint(0, n - 1)
        
    seeds = [s0]
    
    # Run loop to select remaining k-1 seeds
    for _ in range(1, k):
        # 3. Estimate information-access probability p_v(S)
        # Using prob_est_timed as the base probability estimator
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R_mc)
        
        # 4. Compute disadvantage score D(v) = 1 - p_v(S)
        D = [1.0 - p for p in probs]
        
        # 5. Compute fair spreading power R(v)
        R_scores = [0.0] * n
        candidates = [v for v in range(n) if v not in seeds]
        
        for v in candidates:
            # Local BFS up to L hops
            nearby = get_nearby_nodes_and_dists(adj, v, L)
            r_val = 0.0
            for u, dist in nearby.items():
                if u != v:
                    r_val += D[u] * (alpha ** dist)
            R_scores[v] = r_val
            
        # 6. Normalize D(v) and R(v) among candidates (V \ S)
        cand_D = [D[v] for v in candidates]
        cand_R = [R_scores[v] for v in candidates]
        
        d_min, d_max = min(cand_D), max(cand_D)
        r_min, r_max = min(cand_R), max(cand_R)
        
        norm_D = {}
        norm_R = {}
        for v in candidates:
            if d_max - d_min < 1e-15:
                norm_D[v] = 1.0
            else:
                norm_D[v] = (D[v] - d_min) / (d_max - d_min)
                
            if r_max - r_min < 1e-15:
                norm_R[v] = 1.0
            else:
                norm_R[v] = (R_scores[v] - r_min) / (r_max - r_min)
                
        # 7. Apply Pareto-based filtering if requested
        eligible_candidates = candidates
        if use_pareto:
            eligible_candidates = get_pareto_frontier(candidates, norm_D, norm_R)
            if not eligible_candidates:  # Fallback
                eligible_candidates = candidates
                
        # 8. Compute combined score FS(v)
        best_cand = -1
        max_score = -float('inf')
        
        for v in eligible_candidates:
            nd = norm_D[v]
            nr = norm_R[v]
            if multiplicative:
                # Add tiny epsilon to avoid zero issues in power exponentiation
                score = ((nd + 1e-9) ** lam) * ((nr + 1e-9) ** (1.0 - lam))
            else:
                score = lam * nd + (1.0 - lam) * nr
                
            if score > max_score:
                max_score = score
                best_cand = v
                
        if best_cand == -1:
            best_cand = random.choice(candidates)
            
        seeds.append(best_cand)
        
    return seeds
