"""
attempt2/ml_guided_myopic.py
=============================
ML-Guided Myopic: seed selection using a scoring function derived from
SHAP-weighted feature importance analysis.

The scoring function was learned by training an XGBoost model to predict
ΔVar (variance reduction) from 12 structural node features, then distilling
the model into a linear combination.

At runtime, NO ML model is used — just the derived closed-form formula.
This makes it as fast as standard Myopic while being mathematically motivated.

Fallback weights (used if no derived_weights.json is found) are based on
the mathematical intuition from the feature analysis:
  - deficit         (positive): prioritise underserved nodes
  - norm_degree     (positive): prefer moderate-to-high degree for spread
  - low_p_neighbors (positive): prefer nodes that help underserved neighborhoods
  - dist_to_seeds   (positive): prefer nodes far from existing seeds
  - clustering      (negative): avoid highly clustered (redundant) nodes
"""

import os
import json
import time

from attempt2.ic_model    import prob_est_timed
from attempt2.metrics     import compute_all_metrics, mean_prob, variance
from attempt2.ml_features import (extract_features, precompute_graph_features,
                                   FEATURE_NAMES, NUM_FEATURES)


# ---------------------------------------------------------------------------
# Default weights (mathematical fallback)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    'degree':             0.1691,  # Raw connectivity (dominant positive factor)
    'norm_degree':        0.0373,  # Normalized spread capacity
    'clustering':         0.0427,  # Local clustering coefficient
    'curr_prob':         -0.0223,  # Avoid already-reached nodes
    'deficit':            0.1635,  # Underserved individual deficit (primary equity driver)
    'min_neighbor_p':     0.0644,  # Target neighborhoods near lowest reach
    'avg_neighbor_p':    -0.0504,  # Avoid saturated neighborhoods
    'dist_to_seeds':     -0.1266,  # Strong penalization/selection based on distance
    'low_p_neighbors':   -0.0675,  # Neighbor reach thresholding
    'degree_centrality': -0.0376,  # Global centrality penalty vs local deficit
    'local_density':      0.0293,  # Structural density
    'shell_reach':       -0.0259,  # Outer reach limit balancing
    'local_homophily':    0.0024,  # Within-group homophilic clustering weight
    'group_deficit':     -0.1611,  # Group-level deficit balancing across homophily boundaries
}


def _load_weights(weights_path=None):
    """
    Load ML-derived weights from JSON, or fall back to defaults.

    Returns dict: {feature_name: weight}
    """
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
        print(f'  [ML-Guided] Loaded derived weights from {weights_path}')
        return weights
    else:
        print(f'  [ML-Guided] Using default mathematical weights')
        return DEFAULT_WEIGHTS.copy()


def _score_candidates(features, weights):
    """
    Score each candidate using the weighted feature combination.
    Higher score = better candidate for variance reduction.
    """
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
# Main algorithm
# ---------------------------------------------------------------------------

def ml_guided_myopic(adj, nodes, alpha, k, T, R=200, lambda_=1.0,
                     verbose=False, weights_path=None):
    """
    ML-Guided Myopic: selects seeds using a feature-based scoring function
    derived from statistical analysis of variance reduction patterns.

    Parameters
    ----------
    adj          : list[set]   adjacency list
    nodes        : list[dict]  node metadata
    alpha        : float       ICM probability
    k            : int         number of seeds
    T            : int         time deadline
    R            : int         MC simulation count
    lambda_      : float       welfare trade-off parameter
    verbose      : bool        print progress
    weights_path : str|None    path to derived_weights.json

    Returns
    -------
    seeds : list[int]
    log   : list[dict]
    """
    n = len(nodes)
    weights = _load_weights(weights_path)

    # Precompute graph-level features (done once)
    graph_feats = precompute_graph_features(adj, n, nodes=nodes, T=T)

    # First seed: highest degree (standard initialisation)
    s0 = max(range(n), key=lambda i: len(adj[i]))
    seeds = [s0]
    log = []

    for step in range(k):
        t0 = time.time()

        # Estimate probabilities under current seed set
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R, compute_ts=False)

        if step < k - 1:
            # Extract features for all candidate nodes
            candidates, features = extract_features(
                adj, n, seeds, probs, graph_feats, T=T, nodes=nodes)

            # Score candidates
            scores = _score_candidates(features, weights)

            # Select the highest-scoring candidate
            best_idx = max(range(len(candidates)), key=lambda i: scores[i])
            best_cand = candidates[best_idx]
            seeds.append(best_cand)

        entry = _log_step(step + 1, seeds[step], probs, nodes, lambda_,
                          time.time() - t0)
        log.append(entry)

        if verbose:
            print(f"  [ML-Guided step {step+1:3d}] seed={seeds[step]:5d}  "
                  f"var={entry['var']:.5f}  mu={entry['mu']:.4f}")

    return seeds, log
