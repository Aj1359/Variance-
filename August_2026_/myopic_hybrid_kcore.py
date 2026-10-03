import time
import networkx as nx
from icm import prob_est_timed

def myopic_hybrid_kcore(adj, nodes, alpha, k, T, R=200, epsilon=0.01, kcore_dict=None):
    """
    Myopic-Hybrid with K-Core (Core Number) Decomposition:
    1. First seed: node with the highest core number (coreness), tie-breaking by highest degree.
    2. Step 2..k:
       - Estimate activation probabilities under current seed set.
       - Select candidates within epsilon band of the minimum probability:
         probs[i] <= min_p + epsilon
       - Among candidates, pick the one with the highest core number (coreness),
         tie-breaking by highest degree.
    """
    n = len(nodes)
    
    # Compute core numbers (coreness) if not provided
    if kcore_dict is None:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, neighbors in enumerate(adj):
            for v in neighbors:
                if u != v:
                    G.add_edge(u, v)
        kcore_dict = nx.core_number(G)
        
    # First seed selection: highest coreness, tiebreak with highest degree
    s0 = max(range(n), key=lambda i: (kcore_dict.get(i, 0), len(adj[i])))
    seeds = [s0]
    selection_log = []
    
    selection_log.append({
        "step": 1,
        "candidates_in_epsilon_band": n,
        "chosen_seed": s0,
        "seed_coreness": kcore_dict.get(s0, 0),
        "seed_degree": len(adj[s0])
    })
    
    for step in range(1, k):
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
        seed_set = set(seeds)
        
        # 1. Epsilon band candidates
        min_p = min(probs[i] for i in range(n) if i not in seed_set)
        cand_indices = [i for i in range(n) if i not in seed_set and probs[i] <= min_p + epsilon]
        
        # 2. Pick candidate with highest coreness, tie-break with highest degree
        best_cand = max(cand_indices, key=lambda i: (kcore_dict.get(i, 0), len(adj[i])))
        seeds.append(best_cand)
        
        selection_log.append({
            "step": step + 1,
            "candidates_in_epsilon_band": len(cand_indices),
            "chosen_seed": best_cand,
            "seed_coreness": kcore_dict.get(best_cand, 0),
            "seed_degree": len(adj[best_cand])
        })
        
    # Final probability estimation to get the final 'hits' distribution
    final_probs, final_hits = prob_est_timed(adj, seeds, alpha, n, T, R)
    return seeds, final_probs, final_hits, selection_log


def myopic_hybrid_kcore_batch(adj, nodes, alpha, k, T, top_kcore_pct, R=200, epsilon=0.01, kcore_dict=None):
    """
    Myopic-Hybrid with K-Core (Core Number) Batch Seeding:
    1. For each iteration, estimate activation probabilities.
    2. Filter to epsilon band candidates (probs[i] <= min_p + epsilon).
    3. Filter to Top X% of Candidates by Coreness (min 1).
    4. Sort by Degree (descending).
    5. Batch add them up to the remaining budget.
    """
    n = len(nodes)
    
    # Compute core numbers (coreness) if not provided
    if kcore_dict is None:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for u, neighbors in enumerate(adj):
            for v in neighbors:
                if u != v:
                    G.add_edge(u, v)
        kcore_dict = nx.core_number(G)
        
    seeds = []
    selection_log = []
    
    iter_count = 0
    while len(seeds) < k:
        iter_count += 1
        
        # Estimate probabilities
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
        
        # 1. Candidate Set (epsilon band)
        min_p = min(probs[i] for i in range(n) if i not in seeds)
        cand_set = [i for i in range(n) if i not in seeds and probs[i] <= min_p + epsilon]
        
        # 2. Filter by Top % Coreness
        cand_set_sorted_kcore = sorted(cand_set, key=lambda x: kcore_dict.get(x, 0), reverse=True)
        
        # Calculate how many to take
        num_to_take = int(len(cand_set) * top_kcore_pct)
        if num_to_take < 1:
            num_to_take = 1
            
        top_kcore_cands = cand_set_sorted_kcore[:num_to_take]
        
        # 3. Sort decreasing wise acc to degree (highest comes first)
        top_kcore_cands_deg = sorted(top_kcore_cands, key=lambda x: len(adj[x]), reverse=True)
        
        # 4. Batch Seeding (do not exceed remaining budget k)
        remaining = k - len(seeds)
        seeds_to_add = top_kcore_cands_deg[:remaining]
        
        seeds.extend(seeds_to_add)
        
        selection_log.append({
            "iteration": iter_count,
            "candidates_in_epsilon_band": len(cand_set),
            "candidates_after_kcore_filter": len(top_kcore_cands),
            "seeds_chosen_this_batch": len(seeds_to_add),
            "seeds_added": seeds_to_add
        })
        
    # Final probability estimation to get the final 'hits' distribution
    final_probs, final_hits = prob_est_timed(adj, seeds, alpha, n, T, R)
    return seeds, final_probs, final_hits, selection_log
