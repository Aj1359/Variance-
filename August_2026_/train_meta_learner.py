import os
import glob
import json
import pickle
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

def compute_network_features(gml_path):
    """Extract 9 topological features of the GML network."""
    G = nx.read_gml(gml_path)
    G = nx.convert_node_labels_to_integers(G)
    
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)
    degree_variance = np.var(degrees)
    
    # Transitivity
    transitivity = nx.transitivity(G)
    
    # Shortest path and diameter (on largest CC if disconnected)
    if nx.is_connected(G):
        largest_cc = G
    else:
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        largest_cc = G.subgraph(components[0])
        
    if len(largest_cc) > 500:
        import random
        # Seed for reproducibility
        random.seed(42)
        sampled_nodes = random.sample(list(largest_cc.nodes()), min(100, len(largest_cc)))
        lengths = []
        for source in sampled_nodes:
            path_lengths = nx.single_source_shortest_path_length(largest_cc, source)
            lengths.extend(path_lengths.values())
        avg_shortest_path = np.mean(lengths)
        diameter = np.max(lengths)
    else:
        avg_shortest_path = nx.average_shortest_path_length(largest_cc)
        diameter = nx.diameter(largest_cc)
        
    # Assortativity
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
        if np.isnan(assortativity):
            assortativity = 0.0
    except:
        assortativity = 0.0
        
    return [
        n, m, avg_degree, max_degree, degree_variance, 
        transitivity, avg_shortest_path, diameter, assortativity
    ]

def main():
    results_path = "result_comparison/results.json"
    if not os.path.exists(results_path):
        print(f"Error: results.json not found at {results_path}")
        return
        
    with open(results_path, "r") as f:
        results = json.load(f)
        
    # GML directories to search
    gml_files = glob.glob(os.path.join("Social_Network", "*.gml"))
    if not gml_files:
        gml_files = glob.glob(os.path.join("Socials", "*.gml"))
        
    if not gml_files:
        print("Error: No GML files found in Social_Network/ or Socials/")
        return
        
    print("Extracting topological features for networks...")
    network_features = {}
    for f in gml_files:
        name = os.path.basename(f).replace(".gml", "")
        if name in results:
            print(f"  Processing {name}...")
            network_features[name] = compute_network_features(f)
            
    # The 5 candidate algorithms for the ensemble
    ensemble_algos = [
        "Gonzalez", 
        "Myopic BFS", 
        "Myopic PPR", 
        "MinDegree_hc", 
        "MinDegree_hcn"
    ]
    
    X = []
    y = []
    groups = []  # For Leave-One-Group-Out CV (group by network to avoid leakage)
    
    # Construct dataset
    for network in results:
        if network not in network_features:
            continue
        features = network_features[network]
        
        for alpha in results[network]:
            for k in results[network][alpha]:
                best_algo = None
                best_val = -1.0
                
                # Find the best performing algorithm in the ensemble
                for algo in ensemble_algos:
                    if algo in results[network][alpha][k]:
                        val = results[network][alpha][k][algo]["min_p"]
                        if val > best_val:
                            best_val = val
                            best_algo = algo
                            
                if best_algo is not None:
                    # Input feature: 9 topological features + alpha + k
                    X.append(features + [float(alpha), float(k)])
                    y.append(best_algo)
                    groups.append(network)
                    
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    
    print(f"\nPrepared {len(X)} training samples.")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    # Leave-One-Group-Out Cross Validation (train on 4 networks, test on 1)
    logo = LeaveOneGroupOut()
    scores = []
    
    for train_idx, test_idx in logo.split(X, y_encoded, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        rf.fit(X_train, y_train)
        score = rf.score(X_test, y_test)
        scores.append(score)
        
    print(f"Leave-One-Group-Out Cross Validation Accuracy: {np.mean(scores):.4f}")
    
    # Fit final model on all data
    rf.fit(X, y_encoded)
    
    # Save the model, label encoder, and feature names
    model_data = {
        "model": rf,
        "label_encoder": le,
        "features": [
            "number_nodes", "number_edges", "avg_degree", "max_degree", 
            "degree_variance", "transitivity", "avg_shortest_path", 
            "diameter", "assortativity", "alpha", "k"
        ]
    }
    
    model_dir = "result_comparison"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "meta_learner.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
        
    print(f"Successfully trained and saved Meta-Learner to {model_path}!")

if __name__ == "__main__":
    main()
