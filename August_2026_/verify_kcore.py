import os
import sys
import networkx as nx
import time

from myopic_hybrid_kcore import myopic_hybrid_kcore, myopic_hybrid_kcore_batch
from icm import prob_est_timed

def compute_variance(probs):
    n = len(probs)
    if n < 2:
        return 0.0
    mu = sum(probs) / n
    return sum((p - mu) ** 2 for p in probs) / n

import glob

def main():
    print("=== VERIFYING K-CORE HYBRID ALGORITHMS ON ALL NETWORKS ===")
    
    gml_files = sorted(glob.glob(os.path.join("Social_Network", "*.gml")))
    if not gml_files:
        # Try parent folder fallback
        gml_files = sorted(glob.glob(os.path.join("August_2026_", "Social_Network", "*.gml")))
        
    if not gml_files:
        print("Error: No GML files found in Social_Network/. Exiting.")
        sys.exit(1)
        
    for gml_path in gml_files:
        network_name = os.path.basename(gml_path)
        print(f"\n==================================================")
        print(f"Processing Network: {network_name}")
        print(f"==================================================")
        
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
            if mu != mv:
                adj[mu].append(mv)
                adj[mv].append(mu)
                G_mapped.add_edge(mu, mv)
            
        print(f"Graph loaded: {n} nodes, {G_mapped.number_of_edges()} edges.")
        
        # Compute K-Core decomposition (core numbers)
        print("Computing K-Core numbers...")
        t0 = time.time()
        kcore_dict = nx.core_number(G_mapped)
        print(f"K-Core numbers computed in {time.time() - t0:.3f} seconds.")
        
        # Print some statistics about coreness
        core_values = list(kcore_dict.values())
        print(f"Max Coreness: {max(core_values)}")
        print(f"Min Coreness: {min(core_values)}")
        print(f"Number of nodes in max core: {sum(1 for v in core_values if v == max(core_values))}")
        
        alpha = 0.3
        k = 5
        T = 15
        R = 100
        
        print("\n--- 1. Testing Standard K-Core Hybrid Selection ---")
        t0 = time.time()
        seeds_std, probs_std, hits_std, log_std = myopic_hybrid_kcore(
            adj, nodes, alpha, k, T, R=R, epsilon=0.01, kcore_dict=kcore_dict
        )
        t_std = time.time() - t0
        var_std = compute_variance(probs_std)
        mean_std = sum(probs_std) / n
        print(f"Seeds Chosen (mapped): {seeds_std}")
        print(f"Seeds Chosen (original): {[original_nodes[s] for s in seeds_std]}")
        print(f"Mean Activation Probability: {mean_std:.6f}")
        print(f"Variance: {var_std:.6f}")
        print(f"Time Taken: {t_std:.3f} seconds")
        print("Selection Log:")
        for entry in log_std:
            print(f"  Step {entry['step']}: Chosen node {entry['chosen_seed']} (Coreness: {entry['seed_coreness']}, Degree: {entry['seed_degree']})")
            
        print("\n--- 2. Testing Batch K-Core Hybrid Selection (top_kcore_pct = 0.10) ---")
        t0 = time.time()
        seeds_batch, probs_batch, hits_batch, log_batch = myopic_hybrid_kcore_batch(
            adj, nodes, alpha, k, T, top_kcore_pct=0.10, R=R, epsilon=0.01, kcore_dict=kcore_dict
        )
        t_batch = time.time() - t0
        var_batch = compute_variance(probs_batch)
        mean_batch = sum(probs_batch) / n
        print(f"Seeds Chosen (mapped): {seeds_batch}")
        print(f"Seeds Chosen (original): {[original_nodes[s] for s in seeds_batch]}")
        print(f"Mean Activation Probability: {mean_batch:.6f}")
        print(f"Variance: {var_batch:.6f}")
        print(f"Time Taken: {t_batch:.3f} seconds")
        print("Batch Selection Log:")
        for entry in log_batch:
            print(f"  Iteration {entry['iteration']}: candidates_in_epsilon_band={entry['candidates_in_epsilon_band']}, "
                  f"candidates_after_kcore_filter={entry['candidates_after_kcore_filter']}, "
                  f"seeds_chosen_this_batch={entry['seeds_chosen_this_batch']}, seeds={entry['seeds_added']}")
            
    print("\nAll verifications completed successfully!")

if __name__ == "__main__":
    main()
