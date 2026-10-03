import time
from icm import prob_est_timed

def myopic_hybrid_cc(adj, nodes, alpha, k, T, top_cc_pct, R=200, epsilon=0.01):
    """
    Myopic-Hybrid with Closeness Centrality Batch Seeding:
    1. Epsilon band candidates
    2. Filter to Top X% of Candidates by Closeness Centrality (min 1)
    3. Sort by Degree (descending)
    4. Batch add them up to the remaining budget
    """
    n = len(nodes)
    seeds = []
    selection_log = []
    
    iter_count = 0
    while len(seeds) < k:
        iter_count += 1
        t0 = time.time()
        
        # Estimate probabilities
        probs, hits = prob_est_timed(adj, seeds, alpha, n, T, R)
        
        # 1. Candidate Set (epsilon band)
        min_p = min(probs[i] for i in range(n) if i not in seeds)
        cand_set = [i for i in range(n) if i not in seeds and probs[i] <= min_p + epsilon]
        
        # 2. Filter by Top % Closeness Centrality
        # Sort candidates descending by closeness centrality
        cand_set_sorted_cc = sorted(cand_set, key=lambda x: nodes[x].get('closeness_centrality', 0.0), reverse=True)
        
        # Calculate how many to take
        num_to_take = int(len(cand_set) * top_cc_pct)
        if num_to_take < 1:
            num_to_take = 1
            
        top_cc_cands = cand_set_sorted_cc[:num_to_take]
        
        # 3. Sort decreasing wise acc to degree (highest comes first)
        top_cc_cands_deg = sorted(top_cc_cands, key=lambda x: len(adj[x]), reverse=True)
        
        # 4. Batch Seeding (do not exceed remaining budget k)
        remaining = k - len(seeds)
        seeds_to_add = top_cc_cands_deg[:remaining]
        
        seeds.extend(seeds_to_add)
        
        selection_log.append({
            "iteration": iter_count,
            "candidates_in_epsilon_band": len(cand_set),
            "candidates_after_cc_filter": len(top_cc_cands),
            "seeds_chosen_this_batch": len(seeds_to_add),
            "seeds_added": seeds_to_add
        })
        
    # Final probability estimation to get the final 'hits' distribution
    final_probs, final_hits = prob_est_timed(adj, seeds, alpha, n, T, R)
    return seeds, final_probs, final_hits, selection_log
