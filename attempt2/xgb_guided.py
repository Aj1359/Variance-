"""
attempt2/xgb_guided.py
===========================
XGBoost Guided seed selection algorithm.

Uses the actual trained XGBoost model to score candidate nodes based on their
predicted variance reduction. Since XGBoost captures non-linear interactions 
(unlike the linear ML_GUIDED formula), it should achieve significantly better 
structural filtering, avoiding the noisy optimizer's curse of explicit MC evaluations.
"""

import os
import time
import numpy as np
import xgboost as xgb

from attempt2.ic_model    import prob_est_timed
from attempt2.metrics     import compute_all_metrics, mean_prob
from attempt2.ml_features import (extract_features, precompute_graph_features,
                                   FEATURE_NAMES)


def _load_xgb_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'ml_data', 'xgboost_model.json'
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"XGBoost model not found at {model_path}")
        
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


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
# XGB_GUIDED Algorithm
# ---------------------------------------------------------------------------

def xgb_guided(adj, nodes, alpha, k, T, R=200, lambda_=1.0, verbose=False,
               model_path=None):
    n = len(nodes)
    model = _load_xgb_model(model_path)
    
    t_pre = time.time()
    graph_features = precompute_graph_features(adj, n, nodes)
    if verbose:
        print(f"  [XGB-Guided] Precomputed static features in {time.time()-t_pre:.2f}s")
    
    # First seed: strictly highest degree
    s0 = max(range(n), key=lambda i: len(adj[i]))
    seeds = [s0]
    log = []
    
    for step in range(k):
        t0 = time.time()
        
        # Base probability estimation for current seed set
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R, compute_ts=False)
        mu = mean_prob(probs)
        
        if step < k - 1:
            # 1. Extract features for all candidates
            cand_indices, cand_features = extract_features(adj, n, seeds, probs, graph_features, T, nodes)
            
            # 2. Predict variance reduction using XGBoost
            X = np.array(cand_features)
            scores = model.predict(X)
            
            # 3. Pick the absolute best candidate predicted by the model
            best_idx = np.argmax(scores)
            best_cand = cand_indices[best_idx]
            
            seeds.append(best_cand)

        entry = _log_step(step + 1, seeds[step], probs, nodes, lambda_, time.time() - t0)
        log.append(entry)

        if verbose:
            print(f"  [XGB-Guided step {step+1:3d}] seed={seeds[step]:5d}  "
                  f" mu={mu:.4f}  var={entry['var']:.5f}  t={entry['time_s']:.1f}s")

    return seeds, log
