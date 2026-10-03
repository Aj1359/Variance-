import time
import numpy as np
from collections import deque
import networkx as nx

from attempt2.ic_model import prob_est_timed
from attempt2.metrics import compute_all_metrics, mean_prob


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

def _log_step(step, seed, probs, nodes, lambda_, t_elapsed):
    m = compute_all_metrics(probs, nodes, lambda_)
    return {
        "step": step, "seed": seed,
        "mu": m["mu"], "var": m["var"],
        "welfare": m["welfare"], "jfi": m["jfi"],
        "min_p": m["min_p"], "gap": m["gap"],
        "disparity": m["disparity"], "time_s": t_elapsed,
    }


def new_heu_pr(adj, nodes, alpha, k, T, R=50, lambda_=1.0, verbose=False, epsilon=0.01, pr_dict=None):
    """
    NEW_HEU_PR: "Gonzales Epsilon-Band with PageRank"
    1. Filter: Restrict to nodes in the epsilon band (most underserved).
    2. Rank: Pick the node that is furthest away from any existing seed.
    3. Tiebreak: If distances are tied, pick the node with the highest PageRank centrality.
    """
    n = len(nodes)
    
    # Compute or load PageRank
    if pr_dict is not None:
        pr = pr_dict
    else:
        G = nx.DiGraph()
        for i, neighbors in enumerate(adj):
            for j in neighbors:
                G.add_edge(i, j)
                
        pr = nx.pagerank(G)
    
    # First seed: node with highest PageRank
    s0 = max(range(n), key=lambda i: pr.get(i, 0.0))
    seeds = [s0]
    log = []
    
    for step in range(k):
        t0 = time.time()
        
        # Estimate probabilities
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
        mu = mean_prob(probs)
        seed_set = set(seeds)
        
        if step < k - 1:
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
                
            seeds.append(best_cand)

        entry = _log_step(step + 1, seeds[step], probs, nodes, lambda_, time.time() - t0)
        log.append(entry)

        if verbose:
            print(f"  [NewHeuPR step {step+1:3d}] seed={seeds[step]:5d}  "
                  f" mu={mu:.4f}  var={entry['var']:.5f}  t={entry['time_s']:.1f}s")

    return seeds, log
