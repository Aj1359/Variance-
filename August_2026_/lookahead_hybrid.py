import time
import networkx as nx
from icm import prob_est_timed

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

def lookahead_hybrid(adj, nodes, alpha, k, T, R=200, R_probe=20,
                     epsilon=0.01, shortlist_size=8, pr_weight=0.5, pr_dict=None):
    """
    LOOKAHEAD_HYBRID (PageRank-Hybrid):
    1. Shortlist from epsilon-band using combined Personalized PageRank (farness) + Degree normality.
    2. Run fast lookahead probe (R_probe) on shortlisted candidates to test resulting variance.
    3. Pick the candidate with the lowest probed variance.
    """
    n = len(nodes)

    # Build directed graph once for PageRank & PPR
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i, neighbors in enumerate(adj):
        for j in neighbors:
            G.add_edge(i, j)

    if pr_dict is not None:
        pr0 = pr_dict
    else:
        pr0 = {}
        for i, node_data in enumerate(nodes):
            if isinstance(node_data, dict):
                for key in ['pagerank', 'page_rank', 'PageRank', 'pagerank_centrality']:
                    if key in node_data:
                        pr0[i] = float(node_data[key])
                        break
        if len(pr0) != n:
            pr0 = nx.pagerank(G)

    avg_degree = sum(len(adj[i]) for i in range(n)) / n

    # First seed: highest global PageRank
    s0 = max(range(n), key=lambda i: pr0.get(i, 0.0))
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

        # 2. Cheap shortlist using PPR + Degree normality
        personalization = {s: 1.0 / len(seeds) for s in seeds}
        try:
            ppr = nx.pagerank(G, personalization=personalization, dangling=personalization)
        except nx.PowerIterationFailedConvergence:
            ppr = nx.pagerank(G, personalization=personalization,
                              dangling=personalization, tol=1e-4, max_iter=200)

        # Farness from seeds (low PPR = far)
        ppr_norm = _normalize(ppr, cand_indices)
        farness = {c: 1.0 - ppr_norm[c] for c in cand_indices}

        # Degree normality (close to average degree)
        deg_dist = {c: abs(avg_degree - len(adj[c])) for c in cand_indices}
        deg_dist_norm = _normalize(deg_dist, cand_indices)
        normality = {c: 1.0 - deg_dist_norm[c] for c in cand_indices}

        combined = {c: pr_weight * farness[c] + (1 - pr_weight) * normality[c] for c in cand_indices}

        shortlist = sorted(cand_indices, key=lambda c: combined[c], reverse=True)
        shortlist = shortlist[:min(shortlist_size, len(shortlist))]

        # 3. Variance-lookahead probe on shortlist
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

        # Compute variance trend
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
