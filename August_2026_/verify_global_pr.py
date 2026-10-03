import os
import sys
import networkx as nx
import time

from icm import prob_est_timed

def mean_prob(probs):
    return sum(probs) / len(probs) if probs else 0.0

def variance(probs):
    if len(probs) < 2:
        return 0.0
    mu = mean_prob(probs)
    return sum((p - mu) ** 2 for p in probs) / len(probs)

def _normalize(d, keys):
    eps = 1e-6
    vals = [d.get(k, 0.0) for k in keys]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-15:
        return {k: 0.5 for k in keys}
    return {k: eps + (1 - eps) * (d.get(k, 0.0) - lo) / (hi - lo) for k in keys}

def _phi_mean(scores, phi):
    m = len(scores)
    if m == 0:
        return 0.0
    if abs(phi) < 1e-9:
        prod = 1.0
        for s in scores:
            prod *= s
        return prod ** (1.0 / m)
    return (sum(s ** phi for s in scores) / m) ** (1.0 / phi)

def _bfs_distances(adj, seeds, n):
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

from collections import deque

def concave_hybrid_global_pr(adj, nodes, alpha, k, T, R=200, R_probe=20, epsilon=0.01, shortlist_size=10, phi=-1.0, pr_dict=None):
    n = len(nodes)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i, neighbors in enumerate(adj):
        for j in neighbors:
            G.add_edge(i, j)

    # Initial seed selection: highest global PageRank
    s0 = max(range(n), key=lambda i: pr_dict.get(i, 0.0))
    seeds = [s0]

    for step in range(k):
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
        seed_set = set(seeds)

        if step < k - 1:
            # 1. Epsilon-band filter.
            min_p = min(probs[i] for i in range(n) if i not in seed_set)
            cand_indices = [i for i in range(n) if i not in seed_set and probs[i] <= min_p + epsilon]

            # 2a. Gonzalez signal: hop-distance from seeds (want far).
            dists = _bfs_distances(adj, seeds, n)
            gz_raw = {c: (dists[c] if dists[c] != -1 else n) for c in cand_indices}
            gz_norm = _normalize(gz_raw, cand_indices)

            # 2b. PageRank signal: global PageRank centrality (want high).
            pr_norm = _normalize(pr_dict, cand_indices)

            # 2c. Naive-Myopic signal: urgency = how low is p_i right now.
            urgency_raw = {c: -probs[c] for c in cand_indices}
            urgency_norm = _normalize(urgency_raw, cand_indices)

            # 3. Concave combination
            combined = {
                c: _phi_mean([gz_norm[c], pr_norm[c], urgency_norm[c]], phi)
                for c in cand_indices
            }

            shortlist = sorted(cand_indices, key=lambda c: combined[c], reverse=True)
            shortlist = shortlist[:min(shortlist_size, len(shortlist))]

            # 4. Lookahead probe
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

    return seeds

def main():
    print("=== TESTING GLOBAL PR CENTRALITY IN CONCAVE HYBRID ===")
    
    gml_path = os.path.join("Social_Network", "Facebook.gml")
    G = nx.read_gml(gml_path, destringizer=int)
    original_nodes = sorted([int(n) for n in G.nodes()])
    id_map = {orig: new for new, orig in enumerate(original_nodes)}
    n = len(original_nodes)
    
    nodes = [{} for _ in range(n)]
    for n_id, attrs in G.nodes(data=True):
        mapped_id = id_map[int(n_id)]
        nodes[mapped_id] = attrs
        
    adj = [[] for _ in range(n)]
    G_mapped = nx.Graph()
    for u, v in G.edges():
        u, v = int(u), int(v)
        mu, mv = id_map[u], id_map[v]
        adj[mu].append(mv)
        adj[mv].append(mu)
        G_mapped.add_edge(mu, mv)
        
    pr_dict = nx.pagerank(G_mapped)
    
    alpha = 0.03
    k = 5
    T = 15
    R = 200
    R_probe = 20
    
    seeds = concave_hybrid_global_pr(adj, nodes, alpha, k, T, R=R, R_probe=R_probe, pr_dict=pr_dict)
    print("\nSelected seeds (mapped):", seeds)
    print("Selected seeds (original):", [original_nodes[s] for s in seeds])
    print("Degrees of seeds:", {s: len(adj[s]) for s in seeds})
    
    final_probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
    print(f"Final Variance: {variance(final_probs):.6f}")
    print(f"Final Mean Prob: {sum(final_probs)/n:.6f}")

if __name__ == "__main__":
    main()
