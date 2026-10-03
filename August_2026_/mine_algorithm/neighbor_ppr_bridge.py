from mine_algorithm.helpers import _personalized_pagerank

def neighbor_ppr_bridge(adj, nodes, alpha, k, G_mapped, **kwargs):
    n = len(nodes)
    s0 = max(range(n), key=lambda v: len(adj[v]))
    seeds = [s0]
    
    while len(seeds) < k:
        ppr = _personalized_pagerank(G_mapped, seeds, n)
        candidates = sorted([v for v in range(n) if v not in seeds], key=lambda v: ppr.get(v, 0.0))
        chosen = None
        for x in candidates:
            neighbors = [nbr for nbr in adj[x] if nbr not in seeds]
            if neighbors:
                y = max(neighbors, key=lambda u: len(adj[u]))
                chosen = y
                break
        if chosen is None:
            chosen = candidates[0]
        seeds.append(chosen)
        
    return seeds
