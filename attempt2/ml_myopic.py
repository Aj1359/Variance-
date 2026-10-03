"""
attempt2/ml_myopic.py
===========================
ML-Myopic Algorithm.

This algorithm combines the epsilon-band filtering of Myopic-Hybrid (to guarantee
we target the most underserved nodes) with the rich structural insights of the 
ML-derived scoring function (to break ties intelligently).

Steps per iteration:
1. Identify the minimum probability (min_p) among all non-seed nodes.
2. Form a candidate pool of all nodes with p_i <= min_p + epsilon.
3. Score candidates using the SHAP-derived ML formula.
4. Pick the candidate with the highest ML score.
"""

import os
import json
import time
import numpy as np

from attempt2.ic_model    import prob_est_timed
from attempt2.metrics     import compute_all_metrics, mean_prob
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
        print(f'  [ML-Myopic] Loaded derived weights from {weights_path}')
        return weights
    else:
        print(f'  [ML-Myopic] Using default mathematical weights')
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
# ML_MYOPIC Algorithm
# ---------------------------------------------------------------------------

def ml_myopic(adj, nodes, alpha, k, T, R=200, lambda_=1.0, verbose=False,
              epsilon=0.01, weights_path=None):
    """
    ML-Myopic.
    
    1. Filter candidates by `curr_prob <= min_p + epsilon`.
    2. Rank those candidates using the ML-derived formula.
    """
    n = len(nodes)
    weights = _load_weights(weights_path)
    
    t_pre = time.time()
    graph_features = precompute_graph_features(adj, n, nodes)
    if verbose:
        print(f"  [ML-Myopic] Precomputed static features in {time.time()-t_pre:.2f}s")
    
    s0 = max(range(n), key=lambda i: len(adj[i]))
    seeds = [s0]
    log = []
    
    for step in range(k):
        t0 = time.time()
        
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R, compute_ts=False)
        mu = mean_prob(probs)
        seed_set = set(seeds)
        
        if step < k - 1:
            # 1. Epsilon-band filtering (from MYOPIC_HYBRID)
            min_p = min(probs[i] for i in range(n) if i not in seed_set)
            cand_indices = [i for i in range(n) if i not in seed_set and probs[i] <= min_p + epsilon]
            
            # 2. ML Scoring
            all_cands, all_feats = extract_features(adj, n, seeds, probs, graph_features, T, nodes)
            
            # Map index to feature vector
            feat_map = {c: f for c, f in zip(all_cands, all_feats)}
            cand_features = [feat_map[c] for c in cand_indices]
            
            scores = _score_candidates(cand_features, weights)
            
            # 3. Pick the candidate with highest ML score
            best_idx = np.argmax(scores)
            best_cand = cand_indices[best_idx]
            
            seeds.append(best_cand)

        entry = _log_step(step + 1, seeds[step], probs, nodes, lambda_, time.time() - t0)
        log.append(entry)

        if verbose:
            print(f"  [ML-Myopic step {step+1:3d}] seed={seeds[step]:5d}  "
                  f" mu={mu:.4f}  var={entry['var']:.5f}  t={entry['time_s']:.1f}s")

    return seeds, log
