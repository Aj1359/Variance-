"""
attempt2/min_var_hybrid.py
===========================
True Hybrid (MIN_VAR_HYBRID) algorithm.

Combines ML-guided feature scoring (for O(N) fast candidate filtering) 
with fast Monte Carlo exact simulation (for precise greedy variance minimization).

Steps per iteration:
1. Score all non-seed nodes using the SHAP-derived ML formula.
2. Select the top C candidates.
3. Simulate each candidate exactly (using `simulate_ic_timed_fast`).
4. Pick the candidate that mathematically yields the lowest variance.
"""

import os
import json
import time
import numpy as np

from attempt2.ic_model    import prob_est_timed, simulate_ic_timed_fast
from attempt2.metrics     import compute_all_metrics, mean_prob, variance
from attempt2.ml_features import (extract_features, precompute_graph_features,
                                   FEATURE_NAMES, NUM_FEATURES)


# ---------------------------------------------------------------------------
# Default weights (mathematical fallback)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    'degree':             0.1691,
    'norm_degree':        0.0373,
    'clustering':         0.0427,
    'curr_prob':         -0.0223,
    'deficit':            0.1635,
    'min_neighbor_p':     0.0644,
    'avg_neighbor_p':    -0.0504,
    'dist_to_seeds':     -0.1266,
    'low_p_neighbors':   -0.0675,
    'degree_centrality': -0.0376,
    'local_density':      0.0293,
    'shell_reach':       -0.0259,
    'local_homophily':    0.0024,
    'group_deficit':     -0.1611,
}

def _load_weights(weights_path=None):
    if weights_path is None:
        weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'ml_data', 'derived_weights.json'
        )

    if os.path.exists(weights_path):
        with open(weights_path, 'r') as f:
            data = json.load(f)
        weights = {}
        for name, info in data['scoring_weights'].items():
            weights[name] = info['weight']
        print(f'  [MinVar-Hybrid] Loaded derived weights from {weights_path}')
        return weights
    else:
        print(f'  [MinVar-Hybrid] Using default mathematical weights')
        return DEFAULT_WEIGHTS.copy()


def _score_candidates(features, weights):
    scores = []
    for feat in features:
        s = sum(weights.get(FEATURE_NAMES[i], 0.0) * feat[i]
                for i in range(NUM_FEATURES))
        scores.append(s)
    return scores


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
# MIN_VAR_HYBRID Algorithm
# ---------------------------------------------------------------------------

def min_var_hybrid(adj, nodes, alpha, k, T, R=200, lambda_=1.0, verbose=False,
                   C=10, weights_path=None):
    """
    MinVar-Hybrid.
    
    1. Filter top C candidates using ML formula.
    2. Simulate candidates to exactly greedily minimize variance.
    """
    n = len(nodes)
    weights = _load_weights(weights_path)
    
    # 1. Precompute static graph features for ML formula
    t_pre = time.time()
    graph_features = precompute_graph_features(adj, n, nodes)
    if verbose:
        print(f"  [MinVar-Hybrid] Precomputed static features in {time.time()-t_pre:.2f}s")
    
    # First seed: strictly highest degree
    s0 = max(range(n), key=lambda i: len(adj[i]))
    seeds = [s0]
    log = []
    
    for step in range(k):
        t0 = time.time()
        
        # Base probability estimation for current seed set
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R, compute_ts=False)
        mu = mean_prob(probs)
        seed_set = set(seeds)
        
        if step < k - 1:
            # 1. ML Filtering: Score all non-seed nodes
            cand_indices, cand_features = extract_features(adj, n, seeds, probs, graph_features, T, nodes)
            scores = _score_candidates(cand_features, weights)
            
            # Sort candidates by ML score (descending)
            ranked_cands = sorted(zip(cand_indices, scores), key=lambda x: x[1], reverse=True)
            
            # Take top C
            top_c = [c[0] for c in ranked_cands[:C]]
            
            # 2. Exact Evaluation: Simulate the top C candidates to find absolute minimum variance
            best_cand = None
            best_var = float('inf')
            
            # Use a high number of iterations for candidate evaluation to avoid Optimizer's Curse (MC noise overfitting)
            R_eval = max(R, 200) 
            
            for c in top_c:
                c_probs, _ = prob_est_timed(adj, seeds + [c], alpha, n, T, R_eval, compute_ts=False)
                c_var = variance(c_probs)
                if c_var < best_var:
                    best_var = c_var
                    best_cand = c
            
            seeds.append(best_cand)

        entry = _log_step(step + 1, seeds[step], probs, nodes, lambda_, time.time() - t0)
        log.append(entry)

        if verbose:
            print(f"  [MinVar-Hybrid step {step+1:3d}] seed={seeds[step]:5d}  "
                  f" mu={mu:.4f}  var={entry['var']:.5f}  t={entry['time_s']:.1f}s")

    return seeds, log
