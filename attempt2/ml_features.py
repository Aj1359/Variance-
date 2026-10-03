"""
attempt2/ml_features.py
========================
Feature extraction for ML-guided variance minimisation.

Computes structural and probability-based features for every non-seed
candidate node.  These features are used both for:
  1. Training data generation (paired with MC-evaluated ΔVar)
  2. Runtime inference (the derived scoring function)

Feature vector (12 features per candidate node v):
  0. degree          — |N(v)|
  1. norm_degree     — deg(v) / avg_deg(G)
  2. clustering      — local clustering coefficient
  3. curr_prob       — p_v^(T) under current seed set
  4. deficit         — μ - p_v^(T)   (how underserved v is)
  5. min_neighbor_p  — min_{u ∈ N(v)} p_u
  6. avg_neighbor_p  — mean_{u ∈ N(v)} p_u
  7. dist_to_seeds   — min BFS distance from v to any seed
  8. low_p_neighbors — |{u ∈ N(v) : p_u < μ}|
  9. degree_centrality — deg(v) / (n - 1)
 10. local_density   — edges in N(v) / possible edges in N(v)
 11. shell_reach     — |{u : 1 ≤ dist(v,u) ≤ T}|   (T-hop neighborhood size)
"""

import math
from collections import deque


# ---------------------------------------------------------------------------
# BFS helpers
# ---------------------------------------------------------------------------

def _bfs_distances(adj, sources, n, max_depth=None):
    """Multi-source BFS returning distance array.  -1 = unreachable."""
    dist = [-1] * n
    q = deque()
    for s in sources:
        if 0 <= s < n:
            dist[s] = 0
            q.append(s)
    while q:
        v = q.popleft()
        if max_depth is not None and dist[v] >= max_depth:
            continue
        for u in adj[v]:
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist


def _bfs_shell_size(adj, v, n, T):
    """Count nodes reachable from v within T hops (excluding v itself)."""
    visited = {v}
    frontier = [v]
    count = 0
    for _ in range(T):
        next_frontier = []
        for u in frontier:
            for w in adj[u]:
                if w not in visited:
                    visited.add(w)
                    next_frontier.append(w)
                    count += 1
        frontier = next_frontier
        if not frontier:
            break
    return count


# ---------------------------------------------------------------------------
# Graph-level precomputations (done once per graph)
# ---------------------------------------------------------------------------

def precompute_graph_features(adj, n, nodes=None, T=6):
    """
    Precompute expensive graph-level features that don't change with seed set.

    Returns dict with:
        avg_degree      : float
        clustering      : list[float]  per-node clustering coefficient
        local_density   : list[float]  per-node local density
        local_homophily : list[float]  per-node homophily ratio
        groups          : list[int]    per-node group ID
        shell_reach     : list[float]  per-node T-hop reachable neighborhood size
    """
    # Average degree
    degrees = [len(adj[i]) for i in range(n)]
    avg_degree = sum(degrees) / n if n > 0 else 1.0

    # Clustering coefficient & local density & local homophily & shell_reach
    clustering = [0.0] * n
    local_density = [0.0] * n
    local_homophily = [1.0] * n
    shell_reach = [0.0] * n
    groups = [0] * n
    if nodes is not None and len(nodes) == n:
        groups = [nd.get('group', 0) if isinstance(nd, dict) else 0 for nd in nodes]

    for v in range(n):
        nbrs = list(adj[v])
        deg = len(nbrs)
        if deg > 0:
            same_g = sum(1 for u in nbrs if groups[u] == groups[v])
            local_homophily[v] = same_g / deg

            if n <= 2500:
                shell_reach[v] = float(_bfs_shell_size(adj, v, n, T))
            else:
                shell_reach[v] = float(min(deg * (avg_degree ** (T - 1)), n))

        if deg < 2:
            clustering[v] = 0.0
            local_density[v] = 0.0
            continue

        # Count edges among neighbours
        nbr_set = set(nbrs)
        triangles = 0
        for i, u in enumerate(nbrs):
            for w in nbrs[i + 1:]:
                if w in adj[u]:
                    triangles += 1

        possible = deg * (deg - 1) / 2
        clustering[v] = triangles / possible if possible > 0 else 0.0
        local_density[v] = triangles / possible if possible > 0 else 0.0

    return {
        'avg_degree':      avg_degree,
        'degrees':         degrees,
        'clustering':      clustering,
        'local_density':   local_density,
        'local_homophily': local_homophily,
        'groups':          groups,
        'shell_reach':     shell_reach,
    }


# ---------------------------------------------------------------------------
# Per-step feature extraction
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    'degree', 'norm_degree', 'clustering', 'curr_prob', 'deficit',
    'min_neighbor_p', 'avg_neighbor_p', 'dist_to_seeds',
    'low_p_neighbors', 'degree_centrality', 'local_density', 'shell_reach',
    'local_homophily', 'group_deficit',
]

NUM_FEATURES = len(FEATURE_NAMES)


def extract_features(adj, n, seeds, probs, graph_feats, T=8, nodes=None):
    """
    Extract features for all non-seed candidate nodes.

    Parameters
    ----------
    adj         : list[set]   adjacency list
    n           : int         number of nodes
    seeds       : list[int]   current seed set
    probs       : list[float] p_i^(T) for each node under current seeds
    graph_feats : dict        from precompute_graph_features()
    T           : int         time deadline

    Returns
    -------
    candidates  : list[int]          candidate node indices
    features    : list[list[float]]  features[i] = feature vector for candidates[i]
    """
    seed_set = set(seeds)
    mu = sum(probs) / n if n > 0 else 0.0

    avg_deg = graph_feats['avg_degree']
    degrees = graph_feats['degrees']
    clustering = graph_feats['clustering']
    local_dens = graph_feats['local_density']
    local_hom = graph_feats.get('local_homophily', [1.0] * n)
    groups = graph_feats.get('groups', [0] * n)
    shell_reaches = graph_feats.get('shell_reach', [0.0] * n)

    group_sums = {}
    group_counts = {}
    for i in range(n):
        g = groups[i]
        group_sums[g] = group_sums.get(g, 0.0) + probs[i]
        group_counts[g] = group_counts.get(g, 0) + 1
    group_means = {g: (group_sums[g] / group_counts[g]) if group_counts[g] > 0 else mu for g in group_sums}

    # BFS distances from all seeds
    seed_dists = _bfs_distances(adj, seeds, n, max_depth=T + 2)

    candidates = []
    features = []

    for v in range(n):
        if v in seed_set:
            continue

        deg = degrees[v]
        p_v = probs[v]

        # Neighbour statistics
        nbrs = list(adj[v])
        if nbrs:
            nbr_probs = [probs[u] for u in nbrs]
            min_nbr_p = min(nbr_probs)
            avg_nbr_p = sum(nbr_probs) / len(nbr_probs)
            low_p_count = sum(1 for p in nbr_probs if p < mu)
        else:
            min_nbr_p = 0.0
            avg_nbr_p = 0.0
            low_p_count = 0

        # Distance to nearest seed
        d2s = seed_dists[v] if seed_dists[v] >= 0 else n  # n = "unreachable"
        shell = shell_reaches[v]
        g_def = mu - group_means.get(groups[v], mu)
        feat = [
            float(deg),                          # 0: degree
            deg / avg_deg if avg_deg > 0 else 0, # 1: norm_degree
            clustering[v],                       # 2: clustering
            p_v,                                 # 3: curr_prob
            mu - p_v,                            # 4: deficit
            min_nbr_p,                           # 5: min_neighbor_p
            avg_nbr_p,                           # 6: avg_neighbor_p
            float(d2s),                          # 7: dist_to_seeds
            float(low_p_count),                  # 8: low_p_neighbors
            deg / (n - 1) if n > 1 else 0,       # 9: degree_centrality
            local_dens[v],                       # 10: local_density
            float(shell),                        # 11: shell_reach
            local_hom[v],                        # 12: local_homophily
            g_def,                               # 13: group_deficit
        ]
        candidates.append(v)
        features.append(feat)

    return candidates, features


def extract_features_for_node(v, adj, n, seeds, probs, graph_feats, T=8, nodes=None):
    """Extract features for a single candidate node (used during inference)."""
    seed_set = set(seeds)
    mu = sum(probs) / n if n > 0 else 0.0

    avg_deg = graph_feats['avg_degree']
    degrees = graph_feats['degrees']
    clustering = graph_feats['clustering']
    local_dens = graph_feats['local_density']
    local_hom = graph_feats.get('local_homophily', [1.0] * n)
    groups = graph_feats.get('groups', [0] * n)
    shell_reaches = graph_feats.get('shell_reach', [0.0] * n)

    group_sums = {}
    group_counts = {}
    for i in range(n):
        g = groups[i]
        group_sums[g] = group_sums.get(g, 0.0) + probs[i]
        group_counts[g] = group_counts.get(g, 0) + 1
    group_means = {g: (group_sums[g] / group_counts[g]) if group_counts[g] > 0 else mu for g in group_sums}

    deg = degrees[v]
    p_v = probs[v]

    nbrs = list(adj[v])
    if nbrs:
        nbr_probs = [probs[u] for u in nbrs]
        min_nbr_p = min(nbr_probs)
        avg_nbr_p = sum(nbr_probs) / len(nbr_probs)
        low_p_count = sum(1 for p in nbr_probs if p < mu)
    else:
        min_nbr_p = 0.0
        avg_nbr_p = 0.0
        low_p_count = 0

    # Quick BFS for distance to seeds
    d2s = n
    for s in seeds:
        dist = _bfs_distances(adj, [s], n, max_depth=T + 2)
        if dist[v] >= 0 and dist[v] < d2s:
            d2s = dist[v]

    shell = shell_reaches[v]
    g_def = mu - group_means.get(groups[v], mu)

    return [
        float(deg), deg / avg_deg if avg_deg > 0 else 0,
        clustering[v], p_v, mu - p_v,
        min_nbr_p, avg_nbr_p, float(d2s), float(low_p_count),
        deg / (n - 1) if n > 1 else 0, local_dens[v], float(shell),
        local_hom[v], g_def,
    ]
