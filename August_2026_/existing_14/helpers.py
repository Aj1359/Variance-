import networkx as nx
from collections import deque

def _get_bfs_distances(adj, source, n):
    """Compute BFS distance from source to all other nodes."""
    dists = [-1] * n
    dists[source] = 0
    queue = deque([source])
    while queue:
        curr = queue.popleft()
        curr_d = dists[curr]
        for nbr in adj[curr]:
            if dists[nbr] == -1:
                dists[nbr] = curr_d + 1
                queue.append(nbr)
    return dists

def _pagerank_python(G, personalization, alpha=0.85, max_iter=100, tol=1e-6):
    """Pure Python fallback for PageRank to prevent ZeroDivisionError in scipy."""
    n = G.number_of_nodes()
    if n == 0:
        return {}
        
    nodes = list(G.nodes())
    p = {node: personalization.get(node, 0.0) for node in nodes}
    s_p = sum(p.values())
    if s_p == 0:
        p = {node: 1.0 / n for node in nodes}
    else:
        p = {node: val / s_p for node in nodes}
        
    x = {node: val for node, val in p.items()}
    deg = {node: G.degree(node) for node in nodes}
    
    for _ in range(max_iter):
        x_last = x
        x = {node: 0.0 for node in nodes}
        
        # Dangling sum: sum of PR of isolated nodes (degree = 0)
        dangling_sum = sum(x_last[node] for node in nodes if deg[node] == 0)
        
        for u in nodes:
            if deg[u] > 0:
                pr_u = x_last[u] / deg[u]
                for v in G.neighbors(u):
                    x[v] += alpha * pr_u
                    
        for node in nodes:
            x[node] += (1.0 - alpha) * p[node] + alpha * dangling_sum * p[node]
            
        err = sum(abs(x[node] - x_last[node]) for node in nodes)
        if err < tol:
            break
            
    return x

def pagerank_safe(G_mapped, personalization=None, **kwargs):
    """Safe PageRank wrapper with pure Python fallback."""
    if personalization is None:
        personalization = {node: 1.0 / G_mapped.number_of_nodes() for node in G_mapped.nodes()}
    try:
        return nx.pagerank(G_mapped, personalization=personalization, dangling=personalization, **kwargs)
    except Exception:
        # Fallback to pure Python pagerank
        tol = kwargs.get("tol", 1e-6)
        max_iter = kwargs.get("max_iter", 100)
        alpha = kwargs.get("alpha", 0.85)
        return _pagerank_python(G_mapped, personalization, alpha=alpha, max_iter=max_iter, tol=tol)

def _get_ppr(G_mapped, seeds, n):
    """Run Personalized PageRank biased towards seeds."""
    personalization = {s: 1.0 / len(seeds) for s in seeds}
    return pagerank_safe(G_mapped, personalization=personalization)
