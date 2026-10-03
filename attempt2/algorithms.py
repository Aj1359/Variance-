"""
attempt2/algorithms.py
======================
Algorithm registry – Myopic-Hybrid + ML-Guided edition.

Algorithms:
  MYOPIC_HYBRID   — Top-C degree tiebreaker on lowest-p nodes
  ML_GUIDED       — Feature-weighted scoring from SHAP analysis
  Myopic, NaiveMyopic, Gonzales — baselines
"""
import sys
import os
import time

from algos.myopic import myopic as _myopic_raw
from algos.naive_myopic import naive_myopic as _naive_myopic_raw
from algos.gonzalez import gonzalez as _gonzales_raw
from attempt2.new_heu_pr import new_heu_pr as _new_heu_pr_raw
from attempt2.heur2 import heur2 as _heur2_raw
from attempt2.myopic_hybrid import myopic_hybrid as _myopic_hybrid_raw

from attempt2.ic_model  import (prob_est_timed, prob_est_timeseries)
from attempt2.metrics   import compute_all_metrics, mean_prob, variance

ALGO_ORDER = ["NEW_HEU_PR", "Myopic", "NaiveMyopic", "Gonzales"]

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_step(step, seed, probs, nodes, lambda_, t_elapsed):
    m = compute_all_metrics(probs, nodes, lambda_)
    return {
        "step":      step,
        "seed":      seed,
        "mu":        m["mu"],
        "var":       m["var"],
        "welfare":   m["welfare"],
        "jfi":       m["jfi"],
        "min_p":     m["min_p"],
        "gap":       m["gap"],
        "disparity": m["disparity"],
        "time_s":    t_elapsed,
    }


# ---------------------------------------------------------------------------
# Myopic baseline
# ---------------------------------------------------------------------------

def myopic(adj, nodes, alpha, k, T, R=200, lambda_=1.0, verbose=False):
    n = len(nodes)
    import ic_model as _ic
    _orig = _ic.prob_est
    def _timed(adj_, s_, a_, n_, R_=200):
        return prob_est_timed(adj_, s_, a_, n_, T, R_)
    _ic.prob_est = _timed
    try:
        seeds = _myopic_raw(adj, n, alpha, k, R=R)
    finally:
        _ic.prob_est = _orig

    log, running = [], []
    for step in range(k):
        t0 = time.time()
        running.append(seeds[step])
        probs, _ = prob_est_timed(
            adj, running, alpha, n, T, R=max(R // 2, 30)
        )
        entry = _log_step(step + 1, seeds[step], probs, nodes, lambda_,
                          time.time() - t0)
        log.append(entry)
        if verbose:
            print(f"  [Myopic      step {step+1:3d}] seed={seeds[step]:5d} "
                  f"var={entry['var']:.4f}  mu={entry['mu']:.4f}")
    return seeds, log


# ---------------------------------------------------------------------------
# NaiveMyopic baseline
# ---------------------------------------------------------------------------

def naive_myopic(adj, nodes, alpha, k, T, R=200, lambda_=1.0, verbose=False):
    n = len(nodes)
    import ic_model as _ic
    _orig = _ic.prob_est
    def _timed(adj_, s_, a_, n_, R_=200):
        return prob_est_timed(adj_, s_, a_, n_, T, R_)
    _ic.prob_est = _timed
    try:
        seeds = _naive_myopic_raw(adj, n, alpha, k, R=R)
    finally:
        _ic.prob_est = _orig

    log, running = [], []
    for step in range(k):
        t0 = time.time()
        running.append(seeds[step])
        probs, _ = prob_est_timed(
            adj, running, alpha, n, T, R=max(R // 2, 30)
        )
        entry = _log_step(step + 1, seeds[step], probs, nodes, lambda_,
                          time.time() - t0)
        log.append(entry)
        if verbose:
            print(f"  [NaiveMyopic step {step+1:3d}] seed={seeds[step]:5d} "
                  f"var={entry['var']:.4f}  mu={entry['mu']:.4f}")
    return seeds, log


# ---------------------------------------------------------------------------
# Gonzales baseline
# ---------------------------------------------------------------------------

def gonzales(adj, nodes, alpha, k, T, R=200, lambda_=1.0, verbose=False,
             epsilon=0.01):
    n = len(nodes)
    seeds = _gonzales_raw(adj, n, k)
    
    log, running = [], []
    for step in range(k):
        t0 = time.time()
        running.append(seeds[step])
        probs, _ = prob_est_timed(adj, running, alpha, n, T, R=max(R // 2, 30))
        entry = _log_step(step + 1, seeds[step], probs, nodes, lambda_, time.time() - t0)
        log.append(entry)
        if verbose:
            print(f"  [Gonzales    step {step+1:3d}] seed={seeds[step]:5d} var={entry['var']:.4f}  mu={entry['mu']:.4f}")
    return seeds, log

def new_heu_pr(adj, nodes, alpha, k, T, R=200, lambda_=1.0, verbose=False,
             epsilon=0.01, pr_dict=None):
    return _new_heu_pr_raw(adj, nodes, alpha, k, T, R=R, lambda_=lambda_, verbose=verbose, epsilon=epsilon, pr_dict=pr_dict)

def heur2(adj, nodes, alpha, k, T, R=200, lambda_=1.0, verbose=False,
          epsilon=0.01):
    return _heur2_raw(adj, nodes, alpha, k, T, R=R, lambda_=lambda_, verbose=verbose, epsilon=epsilon)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def myopic_hybrid(adj, nodes, alpha, k, T, R=200, lambda_=1.0, verbose=False, epsilon=0.01):
    return _myopic_hybrid_raw(adj, nodes, alpha, k, T, R=R, lambda_=lambda_, verbose=verbose, epsilon=epsilon)

ALGORITHMS = {
    "MYOPIC_HYBRID":  myopic_hybrid,
    "HEUR2":          heur2,
    "NEW_HEU_PR":     new_heu_pr,
    "Myopic":         myopic,
    "NaiveMyopic":    naive_myopic,
    "Gonzales":       gonzales,
}


def run_algorithm(name, adj, nodes, alpha, k, T, R=200, lambda_=1.0,
                  verbose=False, epsilon=0.01, pr_dict=None):
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{name}'. "
                         f"Choose from {list(ALGORITHMS)}")
    if name == "MYOPIC_HYBRID":
        return ALGORITHMS[name](adj, nodes, alpha, k, T,
                                R=R, lambda_=lambda_, verbose=verbose,
                                epsilon=epsilon)
    if name == "NEW_HEU_PR":
        return ALGORITHMS[name](adj, nodes, alpha, k, T,
                                R=R, lambda_=lambda_, verbose=verbose,
                                epsilon=epsilon, pr_dict=pr_dict)
    if name == "ML_GUIDED":
        return ALGORITHMS[name](adj, nodes, alpha, k, T,
                                R=R, lambda_=lambda_, verbose=verbose)
    return ALGORITHMS[name](adj, nodes, alpha, k, T,
                            R=R, lambda_=lambda_, verbose=verbose)
