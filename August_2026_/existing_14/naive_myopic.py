from icm import prob_est_timed

def _highest_degree(adj, n, exclude):
    return max((i for i in range(n) if i not in exclude), key=lambda i: len(adj[i]))

def naive_myopic(adj, nodes, alpha, k, T, R=200):
    """
    Naive Myopic isolated version:
    Selects highest degree node first. Runs exactly ONE Monte Carlo probability estimation.
    Then selects the next k-1 seeds by just picking the lowest probability nodes from that single estimation.
    """
    n = len(nodes)
    init_seed = _highest_degree(adj, n, set())
    
    # Run one probability estimation on the single initial seed
    probs, _ = prob_est_timed(adj, [init_seed], alpha, n, T, R)
    
    candidates = sorted(
        (i for i in range(n) if i != init_seed),
        key=lambda i: probs[i]
    )
    
    seeds = [init_seed] + candidates[: k - 1]
    
    final_probs, final_hits = prob_est_timed(adj, seeds, alpha, n, T, R)
    return seeds, final_probs, final_hits
