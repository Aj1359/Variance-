import random
from icm import prob_est_timed

def random_select(adj, nodes, alpha, k, T, R=200):
    """
    Random Seed Selection Baseline:
    Selects k seeds uniformly at random.
    """
    n = len(nodes)
    seeds = random.sample(range(n), k)
    final_probs, final_hits = prob_est_timed(adj, seeds, alpha, n, T, R)
    return seeds, final_probs, final_hits
