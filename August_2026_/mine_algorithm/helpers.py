import networkx as nx
from collections import deque

def _bfs_distances(adj, seeds, n):
    dists = [-1] * n
    if not seeds:
        return dists
    queue = deque()
    for s in seeds:
        dists[s] = 0
        queue.append(s)
    
    while queue:
        curr = queue.popleft()
        curr_d = dists[curr]
        for nbr in adj[curr]:
            if dists[nbr] == -1:
                dists[nbr] = curr_d + 1
                queue.append(nbr)
    return dists

def _personalized_pagerank(G, seeds, n):
    if not seeds:
        return {i: 1.0 / n for i in range(n)}
    personalization = {s: 1.0 / len(seeds) for s in seeds}
    try:
        return nx.pagerank(G, personalization=personalization, dangling=personalization)
    except nx.PowerIterationFailedConvergence:
        return nx.pagerank(G, personalization=personalization, dangling=personalization, tol=1e-4, max_iter=200)
