import time
from icm import prob_est_timed

def myopic_hybrid_degree(adj, nodes, alpha, k, T, R=200, epsilon=0.01):
    """
    Simple Myopic-Hybrid (Average Degree Proximity):
    1. First seed: node with the highest degree (maximum initial spread capacity).
    2. Step 2..k:
       - Estimate probabilities for current seed set.
       - Find minimum probability among non-seed nodes (min_p).
       - Select all candidate nodes within epsilon band (probs[i] <= min_p + epsilon).
       - Among candidates, pick the node whose degree is closest to average graph degree:
         argmin |deg(i) - avg_degree|
    """
    n = len(nodes)
    avg_degree = sum(len(adj[i]) for i in range(n)) / n

    # First seed: highest degree node
    s0 = max(range(n), key=lambda i: len(adj[i]))
    seeds = [s0]
    selection_log = []

    selection_log.append({
        "step": 1,
        "candidates_in_epsilon_band": n,
        "chosen_seed": s0,
        "seed_degree": len(adj[s0]),
        "deg_diff_from_avg": abs(len(adj[s0]) - avg_degree)
    })

    for step in range(1, k):
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
        seed_set = set(seeds)

        # 1. Epsilon band candidates
        min_p = min(probs[i] for i in range(n) if i not in seed_set)
        cand_indices = [i for i in range(n) if i not in seed_set and probs[i] <= min_p + epsilon]

        # 2. Pick candidate with degree closest to average graph degree
        best_cand = min(cand_indices, key=lambda i: abs(len(adj[i]) - avg_degree))
        seeds.append(best_cand)

        selection_log.append({
            "step": step + 1,
            "candidates_in_epsilon_band": len(cand_indices),
            "chosen_seed": best_cand,
            "seed_degree": len(adj[best_cand]),
            "deg_diff_from_avg": abs(len(adj[best_cand]) - avg_degree)
        })

    # Final probability estimation
    final_probs, final_hits = prob_est_timed(adj, seeds, alpha, n, T, R)
    return seeds, final_probs, final_hits, selection_log
