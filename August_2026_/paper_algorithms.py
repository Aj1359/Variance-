import random
import networkx as nx

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_bfs_distances(adj, source, n):
    """Compute BFS distance from source to all other nodes."""
    dists = [-1] * n
    dists[source] = 0
    queue = [source]
    while queue:
        curr = queue.pop(0)
        curr_d = dists[curr]
        for nbr in adj[curr]:
            if dists[nbr] == -1:
                dists[nbr] = curr_d + 1
                queue.append(nbr)
    return dists

def get_ppr(G_mapped, seeds, n):
    """Run Personalized PageRank biased towards seeds."""
    personalization = {s: 1.0 / len(seeds) for s in seeds}
    try:
        return nx.pagerank(G_mapped, personalization=personalization, dangling=personalization)
    except nx.PowerIterationFailedConvergence:
        return nx.pagerank(G_mapped, personalization=personalization, dangling=personalization, tol=1e-4, max_iter=200)

# ---------------------------------------------------------------------------
# BFS-based algorithms
# ---------------------------------------------------------------------------
def myopic_bfs(adj, nodes, alpha, k, G_mapped, **kwargs):
    """
    Algorithm 1: Myopic BFS
    Approximates activation probabilities via BFS distances.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    seeds = [s0]
    
    # Store shortest distances from each seed: dist_matrix[seed_idx][node]
    dist_matrix = [get_bfs_distances(adj, s0, n)]
    
    for _ in range(1, k):
        # Estimate probs: P(v) = 1 - prod_{s} (1 - alpha^d(v, s))
        probs = []
        for v in range(n):
            if v in seeds:
                probs.append(1.0)
                continue
            fail_prob = 1.0
            for s_idx in range(len(seeds)):
                d = dist_matrix[s_idx][v]
                if d != -1:
                    fail_prob *= (1.0 - (alpha ** d))
            probs.append(1.0 - fail_prob)
            
        # Select node with minimum activation probability
        best_cand = -1
        min_p = float('inf')
        for v in range(n):
            if v not in seeds:
                if probs[v] < min_p:
                    min_p = probs[v]
                    best_cand = v
                    
        seeds.append(best_cand)
        dist_matrix.append(get_bfs_distances(adj, best_cand, n))
        
    return seeds

def naive_myopic_bfs(adj, nodes, alpha, k, G_mapped, **kwargs):
    """
    Algorithm 2: Naive Myopic BFS
    Estimates probabilities once from the initial seed, then picks k lowest nodes.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    dists = get_bfs_distances(adj, s0, n)
    
    # Prob from s0: P(v) = alpha^d(v, s0) if reachable, else 0
    probs = []
    for v in range(n):
        d = dists[v]
        if d != -1:
            probs.append(alpha ** d)
        else:
            probs.append(0.0)
            
    # Sort non-seed nodes by probability ascending
    candidates = [v for v in range(n) if v != s0]
    sorted_candidates = sorted(candidates, key=lambda v: probs[v])
    
    seeds = [s0] + sorted_candidates[:k - 1]
    return seeds

# ---------------------------------------------------------------------------
# PPR-based algorithms
# ---------------------------------------------------------------------------
def myopic_ppr(adj, nodes, alpha, k, G_mapped, **kwargs):
    """
    Algorithm 3: Myopic PPR
    For each step, run PPR restarting from the current seeds, pick lowest node.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    seeds = [s0]
    
    for _ in range(1, k):
        ppr = get_ppr(G_mapped, seeds, n)
        # Find non-seed with minimum PPR
        best_cand = -1
        min_ppr = float('inf')
        for v in range(n):
            if v not in seeds:
                if ppr.get(v, 0.0) < min_ppr:
                    min_ppr = ppr.get(v, 0.0)
                    best_cand = v
        seeds.append(best_cand)
        
    return seeds

def naive_myopic_ppr(adj, nodes, alpha, k, G_mapped, **kwargs):
    """
    Algorithm 4: Naive Myopic PPR
    Run PPR once restarting from the initial seed, pick k lowest nodes.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    ppr = get_ppr(G_mapped, [s0], n)
    
    candidates = [v for v in range(n) if v != s0]
    sorted_candidates = sorted(candidates, key=lambda v: ppr.get(v, 0.0))
    
    seeds = [s0] + sorted_candidates[:k - 1]
    return seeds

# ---------------------------------------------------------------------------
# Topology-based algorithms
# ---------------------------------------------------------------------------
def least_central(adj, nodes, alpha, k, G_mapped, closeness_dict, **kwargs):
    """
    Algorithm 5: LeastCentral
    Picks the nodes with the lowest closeness centrality.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    seeds = [s0]
    
    # Sort non-seeds by closeness centrality ascending
    candidates = [v for v in range(n) if v != s0]
    sorted_candidates = sorted(candidates, key=lambda v: closeness_dict.get(v, 0.0))
    
    seeds.extend(sorted_candidates[:k - 1])
    return seeds

def least_central_n(adj, nodes, alpha, k, G_mapped, closeness_dict, **kwargs):
    """
    Algorithm 6: LeastCentral_n
    For each step, find the node with the lowest closeness centrality,
    and choose its highest degree neighbor.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    seeds = [s0]
    
    # Sort all nodes by closeness centrality ascending
    sorted_by_cc = sorted(range(n), key=lambda v: closeness_dict.get(v, 0.0))
    
    for _ in range(1, k):
        # Find the first node in sorted_by_cc that is not in seeds
        x = -1
        for v in sorted_by_cc:
            if v not in seeds:
                x = v
                break
        
        # Find highest degree neighbor of x
        neighbors = adj[x]
        if neighbors:
            y = max(neighbors, key=lambda v: len(adj[v]))
        else:
            # Fallback if no neighbors
            remaining = [v for v in range(n) if v not in seeds]
            y = random.choice(remaining) if remaining else x
            
        seeds.append(y)
        
    return seeds

def min_degree_hc(adj, nodes, alpha, k, G_mapped, harmonic_dict, **kwargs):
    """
    Algorithm 7: MinDegree_hc
    Find non-seeds with min degree, break ties with lowest harmonic centrality.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    seeds = [s0]
    
    for _ in range(1, k):
        # Find min degree among non-seeds
        min_deg = min(len(adj[v]) for v in range(n) if v not in seeds)
        V_prime = [v for v in range(n) if v not in seeds and len(adj[v]) == min_deg]
        
        # Choose node with lowest harmonic centrality
        x = min(V_prime, key=lambda v: harmonic_dict.get(v, 0.0))
        seeds.append(x)
        
    return seeds

def min_degree_hcn(adj, nodes, alpha, k, G_mapped, harmonic_dict, **kwargs):
    """
    Algorithm 8: MinDegree_hcn
    MinDegree_hc node's highest degree neighbor.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    seeds = [s0]
    
    for _ in range(1, k):
        # Min degree nodes among non-seeds
        min_deg = min(len(adj[v]) for v in range(n) if v not in seeds)
        V_prime = [v for v in range(n) if v not in seeds and len(adj[v]) == min_deg]
        
        # Node with lowest harmonic centrality
        x = min(V_prime, key=lambda v: harmonic_dict.get(v, 0.0))
        
        # Highest degree neighbor of x
        neighbors = adj[x]
        if neighbors:
            y = max(neighbors, key=lambda v: len(adj[v]))
        else:
            remaining = [v for v in range(n) if v not in seeds]
            y = random.choice(remaining) if remaining else x
            
        seeds.append(y)
        
    return seeds

def min_degree_nd(adj, nodes, alpha, k, G_mapped, **kwargs):
    """
    Algorithm 9: MinDegree_nd
    Find non-seeds with min degree, break ties with highest neighbor degree sum.
    """
    n = len(nodes)
    s0 = random.randint(0, n - 1)
    seeds = [s0]
    
    for _ in range(1, k):
        min_deg = min(len(adj[v]) for v in range(n) if v not in seeds)
        V_prime = [v for v in range(n) if v not in seeds and len(adj[v]) == min_deg]
        
        # Highest sum of neighbor degrees
        x = max(V_prime, key=lambda v: sum(len(adj[nbr]) for nbr in adj[v]))
        seeds.append(x)
        
    return seeds

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
