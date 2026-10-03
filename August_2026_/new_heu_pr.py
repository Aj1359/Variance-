import time
from collections import deque
import networkx as nx
from icm import prob_est_timed

def _bfs_distances(adj, seeds, n):
    """Compute shortest path distance from the seed set to all nodes."""
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

def new_heu_pr(adj, nodes, alpha, k, T, R=200, epsilon=0.01, pr_dict=None):
    """
    NEW_HEU_PR: PageRank Heuristic
    1. Filter: Restrict to nodes in the epsilon band (most underserved).
    2. Rank: Pick the node furthest away from any existing seed.
    3. Tiebreak: If distances are tied, pick the node with highest PageRank centrality.
    """
    n = len(nodes)
    
    # 1. Check if pr_dict is provided
    # 2. Else check if nodes contain pagerank attribute from GML
    # 3. Otherwise compute via NetworkX
    if pr_dict is not None:
        pr = pr_dict
    else:
        pr = {}
        for i, node_data in enumerate(nodes):
            if isinstance(node_data, dict):
                for key in ['pagerank', 'page_rank', 'PageRank', 'pagerank_centrality']:
                    if key in node_data:
                        pr[i] = float(node_data[key])
                        break
        if len(pr) != n:
            G = nx.Graph()
            for i, neighbors in enumerate(adj):
                for j in neighbors:
                    G.add_edge(i, j)
            pr = nx.pagerank(G)
    
    # First seed: node with highest PageRank
    s0 = max(range(n), key=lambda i: pr.get(i, 0.0))
    seeds = [s0]
    selection_log = []
    
    # Log initial seed selection
    selection_log.append({
        "step": 1,
        "candidates_in_epsilon_band": n,
        "chosen_seed": s0,
        "seed_pr": pr.get(s0, 0.0),
        "seed_dist": 0
    })
    
    for step in range(1, k):
        # Estimate probabilities
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
        seed_set = set(seeds)
        
        # 1. Epsilon-band filtering
        min_p = min(probs[i] for i in range(n) if i not in seed_set)
        cand_indices = [i for i in range(n) if i not in seed_set and probs[i] <= min_p + epsilon]
        
        # 2. Distance to seeds for all candidates
        dists = _bfs_distances(adj, seeds, n)
        
        best_cand = None
        best_dist = -2
        best_pr = -1.0
        
        for c in cand_indices:
            c_dist = dists[c] if dists[c] != -1 else float('inf') # unreachable = infinitely far
            c_pr = pr.get(c, 0.0)
            
            # 3. Maximize distance, tiebreak with max PageRank
            if c_dist > best_dist:
                best_dist = c_dist
                best_pr = c_pr
                best_cand = c
            elif c_dist == best_dist:
                if c_pr > best_pr:
                    best_pr = c_pr
                    best_cand = c
        
        # Fallback
        if best_cand is None and cand_indices:
            best_cand = cand_indices[0]
            best_dist = dists[best_cand]
            best_pr = pr.get(best_cand, 0.0)
            
        seeds.append(best_cand)
        
        selection_log.append({
            "step": step + 1,
            "candidates_in_epsilon_band": len(cand_indices),
            "chosen_seed": best_cand,
            "seed_pr": best_pr,
            "seed_dist": best_dist
        })
        
    final_probs, final_hits = prob_est_timed(adj, seeds, alpha, n, T, R)
    return seeds, final_probs, final_hits, selection_log
