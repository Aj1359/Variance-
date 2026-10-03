import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ic_model import prob_est

def _highest_degree(adj, n, exclude):
    return max((i for i in range(n) if i not in exclude), key=lambda i: len(adj[i]))

def naive_myopic(adj, n, alpha, k, R=200):
    init_seed = _highest_degree(adj, n, set())
    probs, _ = prob_est(adj, [init_seed], alpha, n, R=R)

    candidates = sorted(
        (i for i in range(n) if i != init_seed),
        key=lambda i: probs[i]
    )
    seeds = [init_seed] + candidates[: k - 1]
    return seeds
