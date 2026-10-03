import networkx as nx
from mine_algorithm.helpers import _bfs_distances

def component_first(adj, nodes, alpha, k, G_mapped, **kwargs):
    n = len(nodes)
    seeds = []
    
    components = sorted(nx.connected_components(G_mapped), key=len, reverse=True)
    for C in components:
        if not any(s in C for s in seeds):
            best_node = max(C, key=lambda v: len(adj[v]))
            seeds.append(best_node)
            if len(seeds) == k:
                return seeds
                
    # Fallback: Degree Gonzalez spread within largest component or overall
    while len(seeds) < k:
        dists = _bfs_distances(adj, seeds, n)
        best_cand = -1
        max_dist = -1
        best_deg = -1
        for v in range(n):
            if v not in seeds:
                d = dists[v] if dists[v] != -1 else float('inf')
                if d > max_dist:
                    max_dist = d
                    best_deg = len(adj[v])
                    best_cand = v
                elif d == max_dist:
                    if len(adj[v]) > best_deg:
                        best_deg = len(adj[v])
                        best_cand = v
        seeds.append(best_cand)
        
    return seeds
