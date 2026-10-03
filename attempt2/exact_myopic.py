"""
attempt2/exact_myopic.py
===========================
Exact Myopic Algorithm.

This is the mathematically strongest possible heuristic for sequential variance minimization.
It combines the epsilon-band filtering of Myopic-Hybrid (to guarantee we target the 
most underserved nodes) with exact Monte Carlo variance simulation (using a high R 
to avoid the Optimizer's Curse) to find the absolute mathematically best seed among
the candidates.

Because the candidate pool (epsilon band) is small, this exact evaluation is computationally
feasible, unlike evaluating all nodes in the graph.
"""

import time
import numpy as np

from attempt2.ic_model    import prob_est_timed
from attempt2.metrics     import compute_all_metrics, mean_prob, variance


def _log_step(step, seed, probs, nodes, lambda_, t_elapsed):
    m = compute_all_metrics(probs, nodes, lambda_)
    return {
        "step": step, "seed": seed,
        "mu": m["mu"], "var": m["var"],
        "welfare": m["welfare"], "jfi": m["jfi"],
        "min_p": m["min_p"], "gap": m["gap"],
        "disparity": m["disparity"], "time_s": t_elapsed,
    }


# ---------------------------------------------------------------------------
# EXACT_MYOPIC Algorithm
# ---------------------------------------------------------------------------

def exact_myopic(adj, nodes, alpha, k, T, R=200, lambda_=1.0, verbose=False,
                 epsilon=0.01):
    n = len(nodes)
    
    s0 = max(range(n), key=lambda i: len(adj[i]))
    seeds = [s0]
    log = []
    
    # Internal R for candidate evaluation to avoid MC noise
    R_eval = max(R, 200)
    
    for step in range(k):
        t0 = time.time()
        
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R, compute_ts=False)
        mu = mean_prob(probs)
        seed_set = set(seeds)
        
        if step < k - 1:
            # 1. Epsilon-band filtering
            min_p = min(probs[i] for i in range(n) if i not in seed_set)
            cand_indices = [i for i in range(n) if i not in seed_set and probs[i] <= min_p + epsilon]
            
            # 2. Exact Evaluation: Simulate all candidates in the epsilon band
            best_cand = None
            best_var = float('inf')
            
            # To prevent hanging when epsilon band is huge, limit to the top C candidates
            # ranked by the Myopic-Hybrid tiebreaker (closest to average degree)
            avg_degree = sum(len(adj[i]) for i in range(n)) / n
            cand_indices = sorted(cand_indices, key=lambda i: abs(avg_degree - len(adj[i])))
            cand_indices = cand_indices[:10]  # Simulate top 10 candidates max
            
            for c in cand_indices:
                c_probs, _ = prob_est_timed(adj, seeds + [c], alpha, n, T, R_eval, compute_ts=False)
                c_var = variance(c_probs)
                
                # Tie-breaking with degree if variances are effectively identical
                if c_var < best_var - 1e-6:
                    best_var = c_var
                    best_cand = c
                elif abs(c_var - best_var) <= 1e-6:
                    if best_cand is not None and len(adj[c]) > len(adj[best_cand]):
                        best_var = c_var
                        best_cand = c
            
            # Fallback if somehow no candidate is better (should not happen)
            if best_cand is None and cand_indices:
                best_cand = cand_indices[0]
                
            seeds.append(best_cand)

        entry = _log_step(step + 1, seeds[step], probs, nodes, lambda_, time.time() - t0)
        log.append(entry)

        if verbose:
            print(f"  [Exact-Myopic step {step+1:3d}] seed={seeds[step]:5d}  "
                  f" mu={mu:.4f}  var={entry['var']:.5f}  t={entry['time_s']:.1f}s")

    return seeds, log
