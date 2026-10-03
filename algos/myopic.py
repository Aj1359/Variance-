import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ic_model import prob_est

def _highest_degree(adj, n, exclude):
    return max((i for i in range(n) if i not in exclude), key=lambda i: len(adj[i]))

def myopic(adj, n, alpha, k, R=200):
    seeds = [_highest_degree(adj, n, set())]

    for step in range(k - 1):
        probs, _ = prob_est(adj, seeds, alpha, n, R=R)
        seed_set = set(seeds)
        v_star = min(
            (i for i in range(n) if i not in seed_set),
            key=lambda i: probs[i]
        )
        seeds.append(v_star)

    return seeds
