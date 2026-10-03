import random
import networkx as nx

def least_central_n(adj, nodes, alpha, k, G_mapped, closeness_dict=None, **kwargs):
    """
    Algorithm 6: LeastCentral_n
    For each step, find the node with the lowest closeness centrality,
    and choose its highest degree neighbor.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    seeds = [s0]
    
    if closeness_dict is None:
        closeness_dict = nx.closeness_centrality(G_mapped)
        
    sorted_by_cc = sorted(range(n), key=lambda v: closeness_dict.get(v, 0.0))
    
    for _ in range(1, k):
        x = -1
        for v in sorted_by_cc:
            if v not in seeds:
                x = v
                break
        
        neighbors = adj[x]
        if neighbors:
            y = max(neighbors, key=lambda v: len(adj[v]))
        else:
            remaining = [v for v in range(n) if v not in seeds]
            y = random.choice(remaining) if remaining else x
            
        seeds.append(y)
        
    return seeds
