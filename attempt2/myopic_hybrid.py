"""
attempt2/myopic_hybrid.py
=========================
A clean, deterministic hybrid algorithm that combines Myopic (lowest reach)
and Degree (highest spread capacity) without suffering from Monte Carlo noise.
"""

import time
from attempt2.ic_model import prob_est_timed
from attempt2.metrics  import compute_all_metrics, mean_prob, variance

def _log_step(step, seed, probs, nodes, lambda_, t_elapsed):
    m = compute_all_metrics(probs, nodes, lambda_)
    return {
        "step":      step,   "seed":      seed,
        "mu":        m["mu"],    "var":       m["var"],
        "welfare":   m["welfare"], "jfi":       m["jfi"],
        "min_p":     m["min_p"],  "gap":       m["gap"],
        "disparity": m["disparity"], "time_s":    t_elapsed,
    }

def myopic_hybrid(adj, nodes, alpha, k, T, R=200, lambda_=1.0, verbose=False,
                  C=10, epsilon=0.01):
    """
    Myopic-Hybrid: Top-C Degree Tiebreaker.

    Parameters
    ----------
    epsilon : float
        Tolerance band around the minimum probability. Nodes with
        p_i <= min_p + epsilon are considered equally underserved and
        compete via their degree-to-average-degree proximity.
        Default=0.01.

    The Solution:
    1. Identify nodes in the epsilon band (lowest p_i).
    2. Among these, pick the one whose degree is closest to the graph average.
       This ensures reliable influence spread with minimal MC noise.
    """
    n = len(nodes)

    # First seed: strictly highest degree (maximum raw reach when all p=0)
    s0 = max(range(n), key=lambda i: len(adj[i]))
    seeds = [s0]
    log = []

    for step in range(k):
        t0 = time.time()

        # Base probability estimation for current seed set
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
        mu = mean_prob(probs)
        seed_set = set(seeds)

        if step < k - 1:
            # 1. Get minimum probability among non-seed nodes
            min_p = min(probs[i] for i in range(n) if i not in seed_set)

            # 2. Find ALL nodes within the epsilon band of the minimum probability
            cands = [i for i in range(n) if i not in seed_set and probs[i] <= min_p + epsilon]

            # 3. Among candidates, pick the node whose degree is closest to average
            avg_degree = sum(len(adj[i]) for i in range(n)) / n
            best_cand = min(cands, key=lambda i: abs(avg_degree - len(adj[i])))

            seeds.append(best_cand)

        entry = _log_step(step + 1, seeds[step], probs, nodes, lambda_, time.time() - t0)
        log.append(entry)

        if verbose:
            print(f"  [Myopic-Hybrid step {step+1:3d}] seed={seeds[step]:5d}  "
                  f" mu={mu:.4f}  var={entry['var']:.5f}  t={entry['time_s']:.1f}s")

    return seeds, log
