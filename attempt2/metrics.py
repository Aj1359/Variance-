"""
attempt2/metrics.py
===================
Fairness and performance metrics for variance-minimising influence maximisation
under time constraints (TCIM + Fish et al. framework).

Core objective: minimise Var(P^T) while maintaining high mean reach mu^T.

Metrics tracked
---------------
  mu        : mean p_v^(T)   — overall reach
  var       : Var(P^(T))     — fairness objective (lower = better)
  welfare   : mu - lambda * var  — joint objective
  jfi       : Jain Fairness Index  in (0,1]
  min_p     : min p_v^(T)   — worst-case access
  gap       : max(group mean) - min(group mean)  — TCIM group disparity (Eq. 2)
  disparity : TCIM max |f_T(S,Vi)/|Vi| - f_T(S,Vj)/|Vj||  (Ali et al. Eq. 2)

TCIM reachability function
--------------------------
  f_T(S, Vi, G) = E[number of nodes in Vi activated by deadline T]
                = sum_{v in Vi}  p_v^(T)

Group disparity (Ali et al. 2023, Eq. 2):
  Disparity = max_{i,j} | f_T(S,Vi)/|Vi| - f_T(S,Vj)/|Vj| |
"""

import math


# ---------------------------------------------------------------------------
# Scalar metrics
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


def welfare(probs, lambda_=1.0):
    """W = mu - lambda * Var   (jointly maximises reach and minimises variance)"""
    return mean_prob(probs) - lambda_ * variance(probs)


def jain_fairness_index(probs, eps=1e-12):
    """JFI = (sum p_i)^2 / (n * sum p_i^2),  in (0,1].  1 = perfect equality."""
    n = len(probs)
    if n == 0:
        return 0.0
    sp  = sum(probs)
    sp2 = sum(p * p for p in probs)
    return sp ** 2 / (n * sp2 + eps)


# ---------------------------------------------------------------------------
# Group-level metrics
# ---------------------------------------------------------------------------

def group_membership_from_nodes(nodes):
    """nodes: list of {id, group, ...} → {group_id: [node_ids]}"""
    d = {}
    for nd in nodes:
        g = nd["group"]
        d.setdefault(g, [])
        d[g].append(nd["id"])
    return d


def group_reach(probs, group_membership):
    """Returns {group_id: mean_p_v} for each group."""
    return {
        g: (sum(probs[i] for i in ids) / len(ids) if ids else 0.0)
        for g, ids in group_membership.items()
    }


def access_gap(probs, group_membership):
    """max(group mean p_v) - min(group mean p_v) — Fish et al. Definition 4."""
    gr = group_reach(probs, group_membership)
    if len(gr) < 2:
        return 0.0
    vals = list(gr.values())
    return max(vals) - min(vals)


def tcim_disparity(probs, group_membership):
    """
    Ali et al. (2023) Eq. (2): max_{i,j} |f_T(S,Vi)/|Vi| - f_T(S,Vj)/|Vj||

    f_T(S, Vi) = sum_{v in Vi} p_v^(T)
    Normalised by group size to be size-agnostic.

    Returns the maximum pairwise normalised-reach difference.
    """
    gr = group_reach(probs, group_membership)  # already normalised by |Vi|
    vals = list(gr.values())
    if len(vals) < 2:
        return 0.0
    return max(
        abs(vals[i] - vals[j])
        for i in range(len(vals))
        for j in range(i + 1, len(vals))
    )


# ---------------------------------------------------------------------------
# Timeseries helpers
# ---------------------------------------------------------------------------

def timeseries_metrics(probs_t, nodes, lambda_=1.0):
    """
    Given probs_t[t] = list of p_i^(t)  for t in 0..T,
    return dict of lists: mu_t, var_t, welfare_t, jfi_t, minp_t, gap_t, disp_t.
    """
    gm = group_membership_from_nodes(nodes)
    mu_t      = [mean_prob(p)           for p in probs_t]
    var_t     = [variance(p)            for p in probs_t]
    welf_t    = [mu - lambda_ * v       for mu, v in zip(mu_t, var_t)]
    jfi_t     = [jain_fairness_index(p) for p in probs_t]
    minp_t    = [min(p) if p else 0.0   for p in probs_t]
    gap_t     = [access_gap(p, gm)      for p in probs_t]
    disp_t    = [tcim_disparity(p, gm)  for p in probs_t]

    return dict(
        mu=mu_t, var=var_t, welfare=welf_t,
        jfi=jfi_t, minp=minp_t, gap=gap_t, disparity=disp_t,
    )


def compute_all_metrics(probs, nodes, lambda_=1.0):
    """Scalar metrics for a single probability vector."""
    gm = group_membership_from_nodes(nodes)
    mu  = mean_prob(probs)
    var = variance(probs)
    return {
        "mu":        mu,
        "var":       var,
        "welfare":   mu - lambda_ * var,
        "jfi":       jain_fairness_index(probs),
        "std":       std_dev(probs),
        "min_p":     min(probs) if probs else 0.0,
        "max_p":     max(probs) if probs else 0.0,
        "gap":       access_gap(probs, gm),
        "disparity": tcim_disparity(probs, gm),
    }
