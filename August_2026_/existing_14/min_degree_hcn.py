import random
import networkx as nx

def min_degree_hcn(adj, nodes, alpha, k, G_mapped, harmonic_dict=None, **kwargs):
    """
    Algorithm 8: MinDegree_hcn
    MinDegree_hc node's highest degree neighbor.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    seeds = [s0]
    
    if harmonic_dict is None:
        harmonic_dict = nx.harmonic_centrality(G_mapped)
        
    for _ in range(1, k):
        min_deg = min(len(adj[v]) for v in range(n) if v not in seeds)
        V_prime = [v for v in range(n) if v not in seeds and len(adj[v]) == min_deg]
        
        x = min(V_prime, key=lambda v: harmonic_dict.get(v, 0.0))
        
        neighbors = adj[x]
        if neighbors:
            y = max(neighbors, key=lambda v: len(adj[v]))
        else:
            remaining = [v for v in range(n) if v not in seeds]
            y = random.choice(remaining) if remaining else x
            
        seeds.append(y)
        
    return seeds
