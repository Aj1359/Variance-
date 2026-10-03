import time
import numpy as np
from collections import deque

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


def new_heu(adj, nodes, alpha, k, T, R=50, lambda_=1.0, verbose=False, epsilon=0.01):
    """
    NEW_HEU: "Gonzales Epsilon-Band"
    1. Filter: Restrict to nodes in the epsilon band (most underserved).
    2. Rank: Pick the node that is furthest away from any existing seed.
    3. Tiebreak: If distances are tied, pick the node with the highest degree.
    """
    n = len(nodes)
    
    # First seed: node with highest degree
    s0 = max(range(n), key=lambda i: len(adj[i]))
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
            best_deg = -1
            
            for c in cand_indices:
                c_dist = dists[c] if dists[c] != -1 else float('inf') # unreachable = infinitely far
                c_deg = len(adj[c])
                
                # 3. Maximize distance, tiebreak with max degree
                if c_dist > best_dist:
                    best_dist = c_dist
                    best_deg = c_deg
                    best_cand = c
                elif c_dist == best_dist:
                    if c_deg > best_deg:
                        best_deg = c_deg
                        best_cand = c
            
            # Fallback
            if best_cand is None and cand_indices:
                best_cand = cand_indices[0]
                
            seeds.append(best_cand)

        entry = _log_step(step + 1, seeds[step], probs, nodes, lambda_, time.time() - t0)
        log.append(entry)

        if verbose:
            print(f"  [NewHeu step {step+1:3d}] seed={seeds[step]:5d}  "
                  f" mu={mu:.4f}  var={entry['var']:.5f}  t={entry['time_s']:.1f}s")

    return seeds, log
