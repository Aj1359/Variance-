import time
import math
from icm import prob_est_timed

def _entropy(p, eps=1e-12):
    """Binary Shannon entropy of a single probability, in bits."""
    p = min(max(p, eps), 1 - eps)
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def _normalize(d, keys):
    """Min-max normalize a dict of scores over keys into [0, 1]."""
    vals = [d.get(k, 0.0) for k in keys]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-15:
        return {k: 0.5 for k in keys}
    return {k: (d.get(k, 0.0) - lo) / (hi - lo) for k in keys}

def _variance(probs):
    """Population variance Var(P) = (1/N) * sum((p_i - mu)^2)."""
    n = len(probs)
    if n < 2:
        return 0.0
    mu = sum(probs) / n
    return sum((p - mu) ** 2 for p in probs) / n

def entropy_hybrid(adj, nodes, alpha, k, T, R=200, R_probe=20,
                   epsilon=0.01, shortlist_size=8, own_weight=0.5):
    """
    ENTROPY_HYBRID:
    1. Epsilon-band candidate filtering.
    2. Entropy-based shortlist (Own entropy + Neighborhood entropy).
    3. Lookahead variance probe on shortlisted candidates (R_probe) to pick true argmin.
    """
    n = len(nodes)

    # First seed: highest-degree node
    s0 = max(range(n), key=lambda i: len(adj[i]))
    seeds = [s0]
    selection_log = []
    prev_var = None

    selection_log.append({
        "step": 1,
        "candidates_in_epsilon_band": n,
        "shortlist_size": 1,
        "chosen_seed": s0,
        "best_probe_var": 0.0,
        "var_trend": "first"
    })

    for step in range(1, k):
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
        cur_var = _variance(probs)
        seed_set = set(seeds)

        # 1. Epsilon-band filter
        min_p = min(probs[i] for i in range(n) if i not in seed_set)
        cand_indices = [i for i in range(n) if i not in seed_set and probs[i] <= min_p + epsilon]

        # 2. Entropy shortlist: Own entropy + Neighborhood entropy
        own_ent = {c: _entropy(probs[c]) for c in cand_indices}
        nbr_ent = {}
        for c in cand_indices:
            neighbors = adj[c]
            if neighbors:
                nbr_ent[c] = sum(_entropy(probs[j]) for j in neighbors) / len(neighbors)
            else:
                nbr_ent[c] = 0.0

        own_norm = _normalize(own_ent, cand_indices)
        nbr_norm = _normalize(nbr_ent, cand_indices)
        combined = {c: own_weight * own_norm[c] + (1 - own_weight) * nbr_norm[c] for c in cand_indices}

        shortlist = sorted(cand_indices, key=lambda c: combined[c], reverse=True)
        shortlist = shortlist[:min(shortlist_size, len(shortlist))]

        # 3. Lookahead probe on shortlisted candidates
        best_cand = None
        best_probe_var = float('inf')
        for c in shortlist:
            trial_seeds = seeds + [c]
            trial_probs, _ = prob_est_timed(adj, trial_seeds, alpha, n, T, R_probe)
            trial_var = _variance(trial_probs)
            if trial_var < best_probe_var:
                best_probe_var = trial_var
                best_cand = c

        if best_cand is None:
            best_cand = shortlist[0] if shortlist else cand_indices[0]

        seeds.append(best_cand)

        # Variance trend tracking
        if prev_var is None:
            trend = "first"
        elif cur_var < prev_var - 1e-12:
            trend = "decreasing"
        elif cur_var > prev_var + 1e-12:
            trend = "increasing"
        else:
            trend = "flat"
        prev_var = cur_var

        selection_log.append({
            "step": step + 1,
            "candidates_in_epsilon_band": len(cand_indices),
            "shortlist_size": len(shortlist),
            "chosen_seed": best_cand,
            "best_probe_var": best_probe_var,
            "var_trend": trend
        })

    final_probs, final_hits = prob_est_timed(adj, seeds, alpha, n, T, R)
    return seeds, final_probs, final_hits, selection_log
