import os
import sys
import pickle
import argparse
import networkx as nx
import numpy as np

# Add parent directory of code/ to python path so packages can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_meta_learner import compute_network_features

# Import the candidate algorithms
from existing_14.gonzalez import gonzalez
from existing_14.myopic_bfs import myopic_bfs
from existing_14.myopic_ppr import myopic_ppr
from existing_14.min_degree_hc import min_degree_hc
from existing_14.min_degree_hcn import min_degree_hcn

def load_graph(gml_path):
    """Load graph and map to sequential integers."""
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
            
    return adj, nodes, G_mapped, n, original_nodes

def run_predicted_algorithm(algo_name, adj, nodes, alpha, k, G_mapped, closeness_dict=None, harmonic_dict=None):
    """Run the predicted algorithm on the network."""
    print(f"Running predicted algorithm: {algo_name}...")
    if algo_name == "Gonzalez":
        # Gonzalez selection is topology-only (T and R are only for wrapper return)
        seeds, _, _ = gonzalez(adj, nodes, alpha, k, T=15, R=50)
        return seeds
    elif algo_name == "Myopic BFS":
        return myopic_bfs(adj, nodes, alpha, k, G_mapped)
    elif algo_name == "Myopic PPR":
        return myopic_ppr(adj, nodes, alpha, k, G_mapped)
    elif algo_name == "MinDegree_hc":
        return min_degree_hc(adj, nodes, alpha, k, G_mapped, harmonic_dict)
    elif algo_name == "MinDegree_hcn":
        return min_degree_hcn(adj, nodes, alpha, k, G_mapped, harmonic_dict)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

def main():
    parser = argparse.ArgumentParser(description="Run Random Forest Meta-Learner on a GML network.")
    parser.add_argument("--gml", type=str, required=True, help="Path to the GML network file")
    parser.add_argument("--alpha", type=float, default=0.03, help="Influence rate (alpha)")
    parser.add_argument("--k", type=int, default=10, help="Seed set size (k)")
    parser.add_argument("--model", type=str, default="../../result_comparison/meta_learner.pkl", help="Path to trained meta-learner pkl")
    args = parser.parse_args()
    
    if not os.path.exists(args.gml):
        print(f"Error: GML file not found at {args.gml}")
        return
        
    if not os.path.exists(args.model):
        # Fallback to local workspace search if run from parent folder
        if os.path.exists("result_comparison/meta_learner.pkl"):
            args.model = "result_comparison/meta_learner.pkl"
        else:
            print(f"Error: Meta-learner model not found at {args.model}")
            return
        
    print(f"Loading Meta-Learner model from {args.model}...")
    with open(args.model, "rb") as f:
        model_data = pickle.load(f)
        
    rf = model_data["model"]
    le = model_data["label_encoder"]
    
    print(f"Extracting network topological features for {args.gml}...")
    features = compute_network_features(args.gml)
    
    # Input sample: 9 features + alpha + k
    input_sample = np.array([features + [args.alpha, args.k]])
    
    # Predict best algorithm
    pred_encoded = rf.predict(input_sample)
    pred_name = le.inverse_transform(pred_encoded)[0]
    print(f"\n>>> Meta-Learner predicted best algorithm: {pred_name} <<<")
    
    # Run the predicted algorithm
    print("Loading network and precomputing centrality measures...")
    adj, nodes, G_mapped, n, original_nodes = load_graph(args.gml)
    
    print("Precomputing closeness and harmonic centralities...")
    closeness_dict = nx.closeness_centrality(G_mapped)
    harmonic_dict = nx.harmonic_centrality(G_mapped)
    
    seeds = run_predicted_algorithm(
        pred_name, adj, list(range(n)), args.alpha, args.k, G_mapped, 
        closeness_dict=closeness_dict, harmonic_dict=harmonic_dict
    )
    
    # Map seeds back to original node IDs
    original_seeds = [original_nodes[s] for s in seeds]
    print(f"\nSuccessfully selected {len(seeds)} seeds using predicted algorithm '{pred_name}':")
    print(f"Mapped Seed Indices:   {seeds}")
    print(f"Original GML Node IDs: {original_seeds}")

if __name__ == "__main__":
    main()
