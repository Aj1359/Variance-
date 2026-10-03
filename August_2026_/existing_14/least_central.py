import random
import networkx as nx

def least_central(adj, nodes, alpha, k, G_mapped, closeness_dict=None, **kwargs):
    """
    Algorithm 5: LeastCentral
    Picks the nodes with the lowest closeness centrality.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    seeds = [s0]
    
    if closeness_dict is None:
        closeness_dict = nx.closeness_centrality(G_mapped)
        
    candidates = [v for v in range(n) if v != s0]
    sorted_candidates = sorted(candidates, key=lambda v: closeness_dict.get(v, 0.0))
    
    seeds.extend(sorted_candidates[:k - 1])
    return seeds
