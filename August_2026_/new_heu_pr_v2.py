import time
from collections import deque
import networkx as nx
from icm import prob_est_timed

def _bfs_distances(adj, seeds, n):
    """Unweighted shortest-hop distance from the seed set to all nodes."""
    dists = [-1] * n
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
    """
    Personalized PageRank restarted uniformly on the current seed set.
    """
    if not seeds:
        return {i: 1.0 / n for i in range(n)}
    personalization = {s: 1.0 / len(seeds) for s in seeds}
    try:
        return nx.pagerank(G, personalization=personalization, dangling=personalization)
    except nx.PowerIterationFailedConvergence:
        return nx.pagerank(G, personalization=personalization,
                           dangling=personalization, tol=1e-4, max_iter=200)

def new_heu_pr_v2(adj, nodes, alpha, k, T, R=200, epsilon=0.01, pr_dict=None, hop_bucket_width=1):
    """
    NEW_HEU_PR_V2: Cascading tiebreak
    Epsilon-Band -> Hop Bucket -> Personalized PageRank -> Degree-Normality
    """
    n = len(nodes)

    # Build directed graph once for PageRank & PPR
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i, neighbors in enumerate(adj):
        for j in neighbors:
            G.add_edge(i, j)

    # Global PageRank for first seed
    if pr_dict is not None:
        pr0 = pr_dict
    else:
        pr0 = {}
        for i, node_data in enumerate(nodes):
            if isinstance(node_data, dict):
                for key in ['pagerank', 'page_rank', 'PageRank', 'pagerank_centrality']:
                    if key in node_data:
                        pr0[i] = float(node_data[key])
                        break
        if len(pr0) != n:
            pr0 = nx.pagerank(G)

    # First seed: highest global PageRank
    s0 = max(range(n), key=lambda i: pr0.get(i, 0.0))
    seeds = [s0]
    selection_log = []

    avg_degree = sum(len(adj[i]) for i in range(n)) / n

    selection_log.append({
        "step": 1,
        "candidates_in_epsilon_band": n,
        "chosen_seed": s0,
        "seed_pr": pr0.get(s0, 0.0),
        "seed_dist": 0
    })

    for step in range(1, k):
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
        seed_set = set(seeds)

        # 1. Epsilon-band filter
        min_p = min(probs[i] for i in range(n) if i not in seed_set)
        cand_indices = [i for i in range(n) if i not in seed_set and probs[i] <= min_p + epsilon]

        # 2. Coarse hop-distance bucket
        dists = _bfs_distances(adj, seeds, n)

        def hop(i):
            return dists[i] if dists[i] != -1 else float('inf')

        max_hop = max((hop(c) for c in cand_indices), default=0)
        bucket = [c for c in cand_indices if max_hop - hop(c) <= hop_bucket_width]
        if not bucket:
            bucket = cand_indices

        # 3. Lowest personalized PageRank
        ppr = _personalized_pagerank(G, seeds, n)
        min_ppr = min(ppr.get(c, 0.0) for c in bucket)
        ppr_tol = min_ppr * 0.05 + 1e-12
        ppr_tied = [c for c in bucket if ppr.get(c, 0.0) <= min_ppr + ppr_tol]

        # 4. Degree closest to average
        best_cand = min(ppr_tied, key=lambda i: abs(avg_degree - len(adj[i])))
        seeds.append(best_cand)

        selection_log.append({
            "step": step + 1,
            "candidates_in_epsilon_band": len(cand_indices),
            "chosen_seed": best_cand,
            "seed_ppr": ppr.get(best_cand, 0.0),
            "seed_dist": dists[best_cand]
        })

    final_probs, final_hits = prob_est_timed(adj, seeds, alpha, n, T, R)
    return seeds, final_probs, final_hits, selection_log
