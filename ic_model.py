"""
ic_model.py — Compatibility shim.
Re-exports prob_est from attempt2.ic_model so that algos/*.py can import it.
The actual IC model lives in attempt2/ic_model.py.
"""
from attempt2.ic_model import prob_est_timed

def prob_est(adj, seeds, alpha, n, R=200):
    """Legacy wrapper: delegates to attempt2's timed IC model with T=8."""
    return prob_est_timed(adj, seeds, alpha, n, T=8, R=R)
