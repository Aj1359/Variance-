import time
import numpy as np
from collections import deque
import networkx as nx

from attempt2.ic_model import prob_est_timed
from attempt2.metrics import compute_all_metrics, mean_prob


def _bfs_distances(adj, seeds, n):
    """Compute shortest path distance from the seed set to all nodes."""
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


def _log_step(step, seed, probs, nodes, lambda_, t_elapsed):
    m = compute_all_metrics(probs, nodes, lambda_)
    return {
        "step": step, "seed": seed,
        "mu": m["mu"], "var": m["var"],
        "welfare": m["welfare"], "jfi": m["jfi"],
        "min_p": m["min_p"], "gap": m["gap"],
        "disparity": m["disparity"], "time_s": t_elapsed,
    }


def heur2(adj, nodes, alpha, k, T, R=50, lambda_=1.0, verbose=False, epsilon=0.01):
    """
    HEUR2: Improved PageRank Heuristic
    Improvements over new_heu_pr:

    1. Composite initial seed:
       First seed = argmax(pagerank[i] * degree[i]) instead of pure PageRank.
       Avoids picking a high-PR node inside a dense cluster on sparse/citation graphs.

    2. Adaptive epsilon:
       epsilon dynamically scales as max(base_eps, 0.05 * (1 - mean_p)).
       At high alpha, probabilities saturate → epsilon adapts to keep the candidate
       pool meaningful and non-trivial.

    3. Blended distance+PageRank score:
       Instead of distance with PageRank tiebreak, candidates are scored as:
           score = 0.7 * norm_dist + 0.3 * norm_pr
       This avoids the case where a slightly-farther node with poor structural
       position beats a well-placed, nearby node.

    Changes vs NEW_HEU_PR:
       - Composite initial seed (PR * degree) instead of pure PageRank
       - Blended candidate score (0.85 dist + 0.15 pr) instead of pure distance + PR tiebreak
    """
    n = len(nodes)

    # ---------- Pre-compute PageRank ----------
    G = nx.DiGraph()
    for i, neighbors in enumerate(adj):
        for j in neighbors:
            G.add_edge(i, j)
    pr = nx.pagerank(G)

    # Degree of each node
    degrees = [len(adj[i]) for i in range(n)]
    max_degree = max(degrees) if degrees else 1

    # Improvement 1: Composite initial seed = PageRank * degree
    s0 = max(range(n), key=lambda i: pr.get(i, 0.0) * (degrees[i] / max_degree))
    seeds = [s0]
    log = []

    for step in range(k):
        t0 = time.time()

        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
        mu = mean_prob(probs)
        seed_set = set(seeds)

        if step < k - 1:
            # Fixed epsilon (adaptive was catastrophic on sparse graphs at low alpha)
            min_p = min(probs[i] for i in range(n) if i not in seed_set)
            cand_indices = [
                i for i in range(n)
                if i not in seed_set and probs[i] <= min_p + epsilon
            ]

            # BFS distances from current seed set
            dists = _bfs_distances(adj, seeds, n)

            # Improvement 3: Blended score = 0.7 * norm_dist + 0.3 * norm_pr
            cand_dists = [
                dists[c] if dists[c] != -1 else n  # unreachable = very far
                for c in cand_indices
            ]
            cand_prs   = [pr.get(c, 0.0) for c in cand_indices]

            max_dist = max(cand_dists) if cand_dists else 1
            max_pr   = max(cand_prs)   if cand_prs   else 1.0

            # Avoid divide-by-zero
            max_dist = max_dist if max_dist > 0 else 1
            max_pr   = max_pr   if max_pr   > 0 else 1.0

            scores = [
                0.85 * (d / max_dist) + 0.15 * (p / max_pr)
                for d, p in zip(cand_dists, cand_prs)
            ]

            best_idx  = int(np.argmax(scores)) if scores else 0
            best_cand = cand_indices[best_idx] if cand_indices else None

            # Fallback
            if best_cand is None:
                remaining = [i for i in range(n) if i not in seed_set]
                best_cand = remaining[0] if remaining else 0

            seeds.append(best_cand)

        entry = _log_step(step + 1, seeds[step], probs, nodes, lambda_, time.time() - t0)
        log.append(entry)

        if verbose:
            print(f"  [HEUR2 step {step+1:3d}] seed={seeds[step]:5d} "
                  f" mu={mu:.4f}  var={entry['var']:.5f}  t={entry['time_s']:.1f}s")

    return seeds, log
