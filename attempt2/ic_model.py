"""
attempt2/ic_model.py
====================
Time-Constrained Independent Cascade (IC) model.

Implements the TCIM framework from Ali et al. (2023):
  "On the Fairness of Time-Critical Influence Maximization in Social Networks"
  IEEE TKDE vol 35, no 3.

A node v activated at time t_v receives utility 1 if t_v <= T (deadline),
else 0. We estimate p_i^(T) = Pr[node i informed within T steps] via Monte
Carlo simulation.
"""

import random
from collections import deque


# ---------------------------------------------------------------------------
# Single IC run with time deadline T
# ---------------------------------------------------------------------------

def simulate_ic_timed(adj, seeds, alpha, n, T):
    """
    One Monte Carlo IC simulation with hard deadline T.

    Parameters
    ----------
    adj   : list[set]   adjacency list
    seeds : list[int]   initial seed nodes
    alpha : float       edge transmission probability (uniform)
    n     : int         number of nodes
    T     : int         deadline — only activations at t <= T count

    Returns
    -------
    informed    : set[int]    nodes activated within deadline T
    time_of_inf : list[int]  time_of_inf[v] = round activated (-1 = never)
    """
    informed = set(seeds)
    time_of_inf = [-1] * n
    for s in seeds:
        if 0 <= s < n:
            time_of_inf[s] = 0

    frontier = list(seeds)

    for t in range(1, T + 1):
        next_frontier = []
        for v in frontier:
            for u in adj[v]:
                if u not in informed and random.random() < alpha:
                    informed.add(u)
                    time_of_inf[u] = t
                    next_frontier.append(u)
        frontier = next_frontier
        if not frontier:
            break

    return informed, time_of_inf


def simulate_ic_timed_fast(adj, seeds, alpha, n, T):
    """Fast IC simulation without per-node timestamp tracking when only final reach is needed."""
    informed = set(seeds)
    frontier = list(seeds)
    for t in range(1, T + 1):
        next_frontier = []
        for v in frontier:
            for u in adj[v]:
                if u not in informed and random.random() < alpha:
                    informed.add(u)
                    next_frontier.append(u)
        frontier = next_frontier
        if not frontier:
            break
    return informed


# ---------------------------------------------------------------------------
# Monte Carlo probability estimator — returns p_i^(T) for all i
# ---------------------------------------------------------------------------

def prob_est_timed(adj, seeds, alpha, n, T, R=200, compute_ts=True):
    """
    Estimate p_i^(T) for every node i via R independent IC runs.

    Returns
    -------
    probs   : list[float]  probs[i] = fraction of R runs where i informed by T
    avg_ts  : list[float]  avg_ts[t] = average number of nodes informed by step t
    """
    hits = [0] * n

    if not compute_ts:
        for _ in range(R):
            informed = simulate_ic_timed_fast(adj, seeds, alpha, n, T)
            for v in informed:
                hits[v] += 1
        probs = [hits[v] / R for v in range(n)]
        return probs, None

    all_ts = []
    for _ in range(R):
        informed, time_of_inf = simulate_ic_timed(adj, seeds, alpha, n, T)
        for v in informed:
            hits[v] += 1

        # cumulative counts per timestep
        ts = [0] * (T + 1)
        for v in range(n):
            toi = time_of_inf[v]
            if toi >= 0:
                for step in range(toi, T + 1):
                    ts[step] += 1
        all_ts.append(ts)

    probs = [hits[v] / R for v in range(n)]
    avg_ts = [sum(run[t] for run in all_ts) / R for t in range(T + 1)]

    return probs, avg_ts


# ---------------------------------------------------------------------------
# Full timeseries: p_i^(t) for every t in {0, ..., T}
# ---------------------------------------------------------------------------

def prob_est_timeseries(adj, seeds, alpha, n, T, R=200):
    """
    Returns probs_t[t] = list of n floats = p_i^(t) for each node i.
    Runs exactly R simulations and extracts all time slices efficiently.
    """
    # hits_t[t][v] = how many runs had v informed by step t
    hits_t = [[0] * n for _ in range(T + 1)]

    for _ in range(R):
        _, time_of_inf = simulate_ic_timed(adj, seeds, alpha, n, T)
        for v in range(n):
            toi = time_of_inf[v]
            if toi >= 0:
                for t in range(toi, T + 1):
                    hits_t[t][v] += 1

    return [[hits_t[t][v] / R for v in range(n)] for t in range(T + 1)]


# ---------------------------------------------------------------------------
# BFS shortest-path distance (used by Gonzales)
# ---------------------------------------------------------------------------

def bfs_distances(adj, source, n, T=None):
    """
    BFS distances from `source` to all reachable nodes.
    If T is given, stops at depth T.
    Returns dict: node -> distance.  Unreachable nodes absent.
    """
    dist = {source: 0}
    q = deque([source])
    while q:
        v = q.popleft()
        d = dist[v]
        if T is not None and d >= T:
            continue
        for u in adj[v]:
            if u not in dist:
                dist[u] = d + 1
                q.append(u)
    return dist
