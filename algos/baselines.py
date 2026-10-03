import random as _random

def high_degree(adj, n, k):
    return sorted(range(n), key=lambda i: len(adj[i]), reverse=True)[:k]

def random_seeds(n, k, seed=None):
    rng = _random.Random(seed)
    return rng.sample(range(n), k)
