import os
import sys
import time
import math
from collections import deque
import networkx as nx

from icm import prob_est_timed

# ---------------------------------------------------------------------------
# Independent Metrics Helpers (No dependencies on attempt2)
# ---------------------------------------------------------------------------

def mean_prob(probs):
    """mu = (1/n) sum p_v"""
    return sum(probs) / len(probs) if probs else 0.0

def variance(probs):
    """Population variance Var = (1/n) sum (p_v - mu)^2"""
    if len(probs) < 2:
        return 0.0
    mu = mean_prob(probs)
    return sum((p - mu) ** 2 for p in probs) / len(probs)

def std_dev(probs):
    return math.sqrt(variance(probs))

def jain_fairness_index(probs, eps=1e-12):
    """JFI = (sum p_i)^2 / (n * sum p_i^2), in (0,1]."""
    n = len(probs)
    if n == 0:
        return 0.0
    sp = sum(probs)
    sp2 = sum(p * p for p in probs)
    return sp ** 2 / (n * sp2 + eps)

def group_membership_from_nodes(nodes):
    d = {}
    for nd in nodes:
        g = nd.get("group", 0)
        nid = nd.get("id", 0)
        d.setdefault(g, [])
        d[g].append(nid)
    return d

def group_reach(probs, group_membership):
    return {
        g: (sum(probs[i] for i in ids) / len(ids) if ids else 0.0)
        for g, ids in group_membership.items()
    }

def access_gap(probs, group_membership):
    gr = group_reach(probs, group_membership)
    if len(gr) < 2:
        return 0.0
    vals = list(gr.values())
    return max(vals) - min(vals)

def tcim_disparity(probs, group_membership):
    gr = group_reach(probs, group_membership)
    vals = list(gr.values())
    if len(vals) < 2:
        return 0.0
    return max(
        abs(vals[i] - vals[j])
        for i in range(len(vals))
        for j in range(i + 1, len(vals))
    )

def compute_all_metrics(probs, nodes, lambda_=1.0):
    gm = group_membership_from_nodes(nodes)
    mu = mean_prob(probs)
    var = variance(probs)
    return {
        "mu": mu,
        "var": var,
        "welfare": mu - lambda_ * var,
        "jfi": jain_fairness_index(probs),
        "std": std_dev(probs),
        "min_p": min(probs) if probs else 0.0,
        "max_p": max(probs) if probs else 0.0,
        "gap": access_gap(probs, gm),
        "disparity": tcim_disparity(probs, gm),
    }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bfs_distances(adj, seeds, n):
    """Unweighted shortest-hop distance from the seed set (Gonzalez signal)."""
    dists = [-1] * n
    queue = deque()
    for s in seeds:
        dists[s] = 0
        queue.append(s)
    while queue:
        curr = queue.popleft()
        curr_d = dists[curr]
        for nbr in adj[curr]:
            if dists[nbr] == -1:
                dists[nbr] = curr_d + 1
                queue.append(nbr)
    return dists


def _personalized_pagerank(G, seeds, n):
    """Personalized PageRank restarted on current seeds (PageRank signal)."""
    if not seeds:
        return {i: 1.0 / n for i in range(n)}
    personalization = {s: 1.0 / len(seeds) for s in seeds}
    try:
        return nx.pagerank(G, personalization=personalization, dangling=personalization)
    except nx.PowerIterationFailedConvergence:
        return nx.pagerank(G, personalization=personalization,
                            dangling=personalization, tol=1e-4, max_iter=200)


def _normalize(d, keys):
    """Min-max normalize into [eps, 1] (never exactly 0 -- phi-means with
    phi<=0 are undefined/blow up at 0, so we floor scores at a small eps)."""
    eps = 1e-6
    vals = [d.get(k, 0.0) for k in keys]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-15:
        return {k: 0.5 for k in keys}
    return {k: eps + (1 - eps) * (d.get(k, 0.0) - lo) / (hi - lo) for k in keys}


def _phi_mean(scores, phi):
    """Generalized (Holder) mean of a list of scores in (0,1].
    phi=1 -> arithmetic mean (a plain weighted average would live here)
    phi=0 -> geometric mean (in the limit)
    phi<0 -> harmonic-mean-like; phi -> -inf -> minimum (maximin)
    All phi<=1 are concave: a candidate weak on ANY one score is punished,
    not rescued by a strong score elsewhere.
    """
    m = len(scores)
    if m == 0:
        return 0.0
    if abs(phi) < 1e-9:
        # geometric mean (phi -> 0 limit)
        prod = 1.0
        for s in scores:
            prod *= s
        return prod ** (1.0 / m)
    return (sum(s ** phi for s in scores) / m) ** (1.0 / phi)


def _log_step(step, seed, probs, nodes, lambda_, t_elapsed, var_trend):
    # Ensure nodes have id and group for compute_all_metrics
    for idx, nd in enumerate(nodes):
        if not isinstance(nd, dict):
            nodes[idx] = {}
        nodes[idx].setdefault("id", idx)
        nodes[idx].setdefault("group", 0)
        
    m = compute_all_metrics(probs, nodes, lambda_)
    return {
        "step": step, "seed": seed,
        "mu": m["mu"], "var": m["var"],
        "welfare": m["welfare"], "jfi": m["jfi"],
        "min_p": m["min_p"], "gap": m["gap"],
        "disparity": m["disparity"], "time_s": t_elapsed,
        "var_trend": var_trend,
    }


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def concave_hybrid(adj, nodes, alpha, k, T, R=200, R_probe=20, lambda_=1.0,
                    verbose=False, epsilon=0.01, shortlist_size=10,
                    phi=-1.0, pr_dict=None):
    """
    CONCAVE_HYBRID: PageRank + Gonzalez-distance + Naive-Myopic-urgency,
    combined via a concave phi-mean -> variance-lookahead final pick.
    """
    n = len(nodes)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i, neighbors in enumerate(adj):
        for j in neighbors:
            G.add_edge(i, j)

    # Initial seed selection: highest global PageRank
    pr0 = pr_dict if pr_dict is not None else nx.pagerank(G)
    s0 = max(range(n), key=lambda i: pr0.get(i, 0.0))
    seeds = [s0]
    log = []
    prev_var = None

    for step in range(k):
        t0 = time.time()

        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
        mu = mean_prob(probs)
        cur_var = variance(probs)
        seed_set = set(seeds)

        if step < k - 1:
            # 1. Epsilon-band filter.
            min_p = min(probs[i] for i in range(n) if i not in seed_set)
            cand_indices = [i for i in range(n)
                             if i not in seed_set and probs[i] <= min_p + epsilon]

            # 2a. Gonzalez signal: hop-distance from seeds (want far).
            dists = _bfs_distances(adj, seeds, n)
            gz_raw = {c: (dists[c] if dists[c] != -1 else n) for c in cand_indices}
            gz_norm = _normalize(gz_raw, cand_indices)

            # 2b. PageRank signal: personalized-PPR farness (want far, i.e.
            #     low ppr -> high farness score).
            ppr = _personalized_pagerank(G, seeds, n)
            ppr_norm = _normalize(ppr, cand_indices)
            pr_farness = {c: 1.0 - ppr_norm[c] + 1e-6 for c in cand_indices}
            pr_farness = _normalize(pr_farness, cand_indices)

            # 2c. Naive-Myopic signal: urgency = how low is p_i right now
            #     (want low p_i -> high urgency score).
            urgency_raw = {c: -probs[c] for c in cand_indices}  # lower p -> higher score
            urgency_norm = _normalize(urgency_raw, cand_indices)

            # 3. Concave combination -- a candidate must score reasonably
            #    on ALL THREE to rank highly.
            combined = {
                c: _phi_mean([gz_norm[c], pr_farness[c], urgency_norm[c]], phi)
                for c in cand_indices
            }

            shortlist = sorted(cand_indices, key=lambda c: combined[c], reverse=True)
            shortlist = shortlist[:min(shortlist_size, len(shortlist))]

            # 4. Lookahead probe: measure ACTUAL resulting variance for each
            #    shortlisted candidate, pick the true argmin.
            best_cand, best_probe_var = None, float('inf')
            for c in shortlist:
                trial_seeds = seeds + [c]
                trial_probs, _ = prob_est_timed(adj, trial_seeds, alpha, n, T, R_probe)
                trial_var = variance(trial_probs)
                if trial_var < best_probe_var:
                    best_probe_var = trial_var
                    best_cand = c

            if best_cand is None:
                best_cand = shortlist[0] if shortlist else cand_indices[0]

            seeds.append(best_cand)

        if prev_var is None:
            trend = "first"
        elif cur_var < prev_var - 1e-12:
            trend = "decreasing"
        elif cur_var > prev_var + 1e-12:
            trend = "increasing"
        else:
            trend = "flat"
        prev_var = cur_var

        entry = _log_step(step + 1, seeds[step], probs, nodes, lambda_,
                           time.time() - t0, trend)
        log.append(entry)

        if verbose:
            print(f"  [Concave-Hybrid step {step+1:3d}] seed={seeds[step]:5d}  "
                  f" mu={mu:.4f}  var={entry['var']:.5f}  trend={trend:10s}"
                  f" t={entry['time_s']:.1f}s")

    return seeds, log
