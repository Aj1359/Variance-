import os
import sys
import networkx as nx
import time

from concave_hybrid import concave_hybrid
from icm import prob_est_timed

def compute_variance(probs):
    n = len(probs)
    if n < 2:
        return 0.0
    mu = sum(probs) / n
    return sum((p - mu) ** 2 for p in probs) / n

def main():
    print("=== VERIFYING FACEBOOK ALPHA=0.03, K=5 ===")
    
    gml_path = os.path.join("Social_Network", "Facebook.gml")
    if not os.path.exists(gml_path):
        print(f"Error: {gml_path} not found.")
        return
        
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
        
    # Get PageRank from GML or precompute
    pr_dict = {}
    for i, node_data in enumerate(nodes):
        if isinstance(node_data, dict):
            for key in ['pagerank_centrality', 'pagerank', 'page_rank', 'PageRank']:
                if key in node_data:
                    pr_dict[i] = float(node_data[key])
                    break
    if len(pr_dict) != n:
        pr_dict = nx.pagerank(G_mapped)
        
    alpha = 0.03
    k = 5
    T = 15
    R = 200
    R_probe = 20
    
    print("Running concave_hybrid with verbose=True...")
    seeds, log = concave_hybrid(
        adj, nodes, alpha, k, T, R=R, R_probe=R_probe, phi=-1.0, pr_dict=pr_dict, verbose=True
    )
    
    print("\nSelected seeds (mapped):", seeds)
    print("Selected seeds (original):", [original_nodes[s] for s in seeds])
    
    print("\nRunning a final verification estimation (R=200)...")
    final_probs, final_hits = prob_est_timed(adj, seeds, alpha, n, T, R)
    var_recalculated = compute_variance(final_probs)
    print(f"Recalculated Variance: {var_recalculated:.6f}")
    print(f"Mean Activation Probability: {sum(final_probs)/n:.6f}")
    
    # Print stats of final_probs
    print(f"Min prob: {min(final_probs):.6f}")
    print(f"Max prob: {max(final_probs):.6f}")
    print(f"Number of nodes with prob > 0.05: {sum(1 for p in final_probs if p > 0.05)}")
    print(f"Number of nodes with prob > 0.5: {sum(1 for p in final_probs if p > 0.5)}")

if __name__ == "__main__":
    main()
