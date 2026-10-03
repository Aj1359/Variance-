import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ic_model import prob_est

def _highest_degree(adj, n, exclude):
    return max((i for i in range(n) if i not in exclude), key=lambda i: len(adj[i]))

def greedy(adj, n, alpha, k, R=100):
    seeds = [_highest_degree(adj, n, set())]

    for step in range(k - 1):
        seed_set = set(seeds)
        best_j, best_min_p = None, -1.0

        for j in range(n):
            if j in seed_set:
                continue
            probs, _ = prob_est(adj, seeds + [j], alpha, n, R=R)
            min_p = min(probs)
            if min_p > best_min_p:
                best_min_p = min_p
                best_j = j

        seeds.append(best_j)

    return seeds
