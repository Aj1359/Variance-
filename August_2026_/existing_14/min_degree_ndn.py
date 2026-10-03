import random

def min_degree_ndn(adj, nodes, alpha, k, G_mapped, **kwargs):
    """
    Algorithm 10: MinDegree_ndn
    MinDegree_nd node's highest degree neighbor.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    seeds = [s0]
    
    for _ in range(1, k):
        min_deg = min(len(adj[v]) for v in range(n) if v not in seeds)
        V_prime = [v for v in range(n) if v not in seeds and len(adj[v]) == min_deg]
        
        x = max(V_prime, key=lambda v: sum(len(adj[nbr]) for nbr in adj[v]))
        
        neighbors = adj[x]
        if neighbors:
            y = max(neighbors, key=lambda v: len(adj[v]))
        else:
            remaining = [v for v in range(n) if v not in seeds]
            y = random.choice(remaining) if remaining else x
            
        seeds.append(y)
        
    return seeds
