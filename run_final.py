import os
import sys
import json
import glob
import time
import networkx as nx
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attempt2.algorithms import run_algorithm
from attempt2.ic_model import prob_est_timeseries
from attempt2.metrics import timeseries_metrics, compute_all_metrics

EPSILON_FIXED = 0.01

def load_graph(name):
    txt_path = os.path.join("Social_Network", f"{name}.txt")
    if not os.path.exists(txt_path):
        return None, None, None
    edges, nodes_found = [], set()
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
                nodes_found.update([u, v])
    sorted_nodes = sorted(nodes_found)
    id_map = {o: n for n, o in enumerate(sorted_nodes)}
    n = len(sorted_nodes)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        nu, nv = id_map[u], id_map[v]
        adj[nu].add(nv); adj[nv].add(nu)
    nodes = [{'id': i, 'group': 0} for i in range(n)]
    return nodes, adj, id_map

def load_xml_pagerank(xml_path, id_map):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pr_dict = {}
    nodes_elem = root.find("nodes")
    if nodes_elem is not None:
        for node_elem in nodes_elem.findall("node"):
            orig_id = int(node_elem.get("id"))
            pr_val = float(node_elem.get("pagerank", 0.0))
            if orig_id in id_map:
                pr_dict[id_map[orig_id]] = pr_val
    return pr_dict

def run_single(algo, adj, nodes, alpha, k, T, R, lambda_, pr_dict=None):
    t0 = time.time()
    seeds, seed_log = run_algorithm(
        algo, adj, nodes, alpha=alpha, k=k, T=T, R=R,
        lambda_=lambda_, epsilon=EPSILON_FIXED, pr_dict=pr_dict
    )
    probs_t = prob_est_timeseries(adj, seeds, alpha, len(nodes), T, R=R)
    final = compute_all_metrics(probs_t[-1], nodes, lambda_)
    return {
        'seeds': seeds,
        'seed_log': seed_log,
        'final': final,
        'wall_s': time.time() - t0,
    }

def plot_variance(data_dict, title, out_path, algo_list):
    plt.figure(figsize=(8, 6))
    for algo in algo_list:
        if algo not in data_dict: continue
        x_vals = []
        y_vals = []
        # sort by k
        for k in sorted(data_dict[algo].keys()):
            x_vals.append(k)
            y_vals.append(data_dict[algo][k]["final"]["var"])
        plt.plot(x_vals, y_vals, marker='o', label=algo)
    plt.title(title)
    plt.xlabel("Seed Set Size (k)")
    plt.ylabel("Variance (Lowest is Best)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    xml_files = glob.glob(os.path.join("NodeAttributes_XML", "*.xml"))
    if not xml_files:
        print("No XML files found!")
        return

    alphas = [0.1, 0.2, 0.3]
    k_list = [20, 40, 60, 80, 100]
    T = 15
    R = 50 # Standard IC simulations
    lambda_ = 1.0
    
    hybrid_set = ["MYOPIC_HYBRID", "Myopic", "NaiveMyopic", "Gonzales"]
    pagerank_set = ["NEW_HEU_PR", "Myopic", "NaiveMyopic", "Gonzales"]

    for xml_path in xml_files:
        name = os.path.basename(xml_path).replace(".xml", "")
        print(f"\n{'='*50}\nProcessing {name}\n{'='*50}")
        nodes, adj, id_map = load_graph(name)
        if nodes is None:
            print(f"Skipping {name}, graph text file not found.")
            continue
        
        pr_dict = load_xml_pagerank(xml_path, id_map)
        
        for alpha in alphas:
            print(f"\n--- Alpha: {alpha} ---")
            alpha_dir = os.path.join("result_ultimate", name, str(alpha))
            os.makedirs(alpha_dir, exist_ok=True)
            
            res_file = os.path.join(alpha_dir, "results.json")
            if os.path.exists(res_file):
                with open(res_file, "r") as f:
                    results = json.load(f)
            else:
                results = {"hybrid": {}, "pagerank": {}}
                
            all_algos = set(hybrid_set + pagerank_set)
            
            for k in k_list:
                print(f" k = {k}")
                # We can share baselines between the two sets so we don't re-run Myopic twice!
                cache_k = {}
                for algo in all_algos:
                    # Check if already run in hybrid or pagerank structures (or skip if not requested but we just cache)
                    # To be clean, just run everything in all_algos and assign to respective sets.
                    # Wait, if we save to results directly, it's easier to just do it per set to match requirements.
                    pass
                    
                # Run hybrid set
                for algo in hybrid_set:
                    if algo not in results["hybrid"]: results["hybrid"][algo] = {}
                    if str(k) not in results["hybrid"][algo]:
                        print(f"   Running {algo} (hybrid set)...")
                        res = run_single(algo, adj, nodes, alpha, k, T, R, lambda_, pr_dict=pr_dict)
                        results["hybrid"][algo][str(k)] = res
                        
                # Run pagerank set
                for algo in pagerank_set:
                    if algo not in results["pagerank"]: results["pagerank"][algo] = {}
                    if str(k) not in results["pagerank"][algo]:
                        # If baseline was already run in hybrid set, copy it over!
                        if algo in hybrid_set and str(k) in results["hybrid"].get(algo, {}):
                            results["pagerank"][algo][str(k)] = results["hybrid"][algo][str(k)]
                        else:
                            print(f"   Running {algo} (pagerank set)...")
                            res = run_single(algo, adj, nodes, alpha, k, T, R, lambda_, pr_dict=pr_dict)
                            results["pagerank"][algo][str(k)] = res
                            
            with open(res_file, "w") as f:
                json.dump(results, f, indent=2)
                
            # Restructure dicts for plotting (keys are int k)
            plot_h = {a: {int(k): v for k, v in d.items()} for a, d in results["hybrid"].items()}
            plot_p = {a: {int(k): v for k, v in d.items()} for a, d in results["pagerank"].items()}
            
            # Generate the 2 plots
            plot_variance(plot_h, f"{name} (alpha={alpha}) - Hybrid vs Baselines", 
                          os.path.join(alpha_dir, "hybrid_vs_baselines.png"), hybrid_set)
            plot_variance(plot_p, f"{name} (alpha={alpha}) - PageRank vs Baselines", 
                          os.path.join(alpha_dir, "pagerank_vs_baselines.png"), pagerank_set)
                          
    print("\nAll simulations and plotting completed!")
    
    # Generate the markdown analysis automatically
    print("\nGenerating final markdown leaderboards...")
    import analyze_ultimate
    analyze_ultimate.main()

if __name__ == "__main__":
    main()
