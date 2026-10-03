import os
import sys
import glob
import json
import time
import random
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import baseline IC estimation
from icm import prob_est_timed

# ---------------------------------------------------------------------------
# Graph loader
# ---------------------------------------------------------------------------
def load_graph(graph_path):
    print(f"Loading {os.path.basename(graph_path)}...")
    if graph_path.endswith(".gml"):
        G = nx.read_gml(graph_path, destringizer=int)
    else:
        G = nx.Graph()
        with open(graph_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("%"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        G.add_edge(u, v)
                    except ValueError:
                        continue
                        
    original_nodes = sorted([int(n) for n in G.nodes()])
    id_map = {orig: new for new, orig in enumerate(original_nodes)}
    n = len(original_nodes)
    
    nodes = [{} for _ in range(n)]
    for n_id in G.nodes():
        mapped_id = id_map[int(n_id)]
        attrs = G.nodes[n_id]
        nodes[mapped_id] = attrs if attrs else {}
        
    adj = [[] for _ in range(n)]
    G_mapped = nx.Graph()
    G_mapped.add_nodes_from(range(n))
    for u, v in G.edges():
        u, v = int(u), int(v)
        mu, mv = id_map[u], id_map[v]
        if mu != mv:
            adj[mu].append(mv)
            adj[mv].append(mu)
            G_mapped.add_edge(mu, mv)
            
    return adj, nodes, G_mapped, n, original_nodes

# ---------------------------------------------------------------------------
# Algorithm imports from respective folders
# ---------------------------------------------------------------------------
from existing_14.myopic import myopic
from existing_14.naive_myopic import naive_myopic
from existing_14.gonzalez import gonzalez
from existing_14.random_select import random_select
from existing_14.myopic_bfs import myopic_bfs
from existing_14.naive_myopic_bfs import naive_myopic_bfs
from existing_14.myopic_ppr import myopic_ppr
from existing_14.naive_myopic_ppr import naive_myopic_ppr
from existing_14.least_central import least_central
from existing_14.least_central_n import least_central_n
from existing_14.min_degree_hc import min_degree_hc
from existing_14.min_degree_hcn import min_degree_hcn
from existing_14.min_degree_nd import min_degree_nd
from existing_14.min_degree_ndn import min_degree_ndn

from new_two_algos.prop_aware_additive import prop_aware_additive
from new_two_algos.prop_aware_multiplicative import prop_aware_multiplicative

# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------
def save_seed_report(base_dir, network, alpha, algo_name, k, seeds, original_nodes, G_mapped, pr_dict, closeness_dict, harmonic_dict, kcore_dict, L=None, lam=None):
    alpha_str = f"alpha_{alpha:.2f}"
    k_str = f"seed_set_k_{k}"
    
    if L is not None and lam is not None:
        folder_path = os.path.join(base_dir, network, alpha_str, algo_name, f"L_{L}", f"lam_{lam:.1f}", k_str)
    else:
        folder_path = os.path.join(base_dir, network, alpha_str, algo_name, k_str)
        
    os.makedirs(folder_path, exist_ok=True)
    report_file = os.path.join(folder_path, "seed_info.txt")
    
    seed_details = []
    total_deg = 0
    total_closeness = 0.0
    total_harmonic = 0.0
    total_pr = 0.0
    total_kcore = 0
    
    for s in seeds:
        deg = G_mapped.degree[s]
        cls = closeness_dict.get(s, 0.0)
        harm = harmonic_dict.get(s, 0.0)
        pr = pr_dict.get(s, 0.0)
        kc = kcore_dict.get(s, 0)
        
        orig_id = original_nodes[s]
        total_deg += deg
        total_closeness += cls
        total_harmonic += harm
        total_pr += pr
        total_kcore += kc
        
        seed_details.append(
            f"Seed Mapped ID: {s} (Original GML ID: {orig_id})\n"
            f"  Degree: {deg}\n"
            f"  Closeness Centrality: {cls:.6f}\n"
            f"  Harmonic Centrality: {harm:.6f}\n"
            f"  PageRank: {pr:.6f}\n"
            f"  K-Core Number: {kc}\n"
        )
        
    num_seeds = len(seeds)
    avg_deg = total_deg / num_seeds if num_seeds > 0 else 0
    avg_closeness = total_closeness / num_seeds if num_seeds > 0 else 0.0
    avg_harmonic = total_harmonic / num_seeds if num_seeds > 0 else 0.0
    avg_pr = total_pr / num_seeds if num_seeds > 0 else 0.0
    avg_kcore = total_kcore / num_seeds if num_seeds > 0 else 0.0
    
    with open(report_file, "w") as f:
        f.write(f"=== Seed Set Report ===\n")
        f.write(f"Network: {network}\n")
        f.write(f"Alpha: {alpha:.2f}\n")
        f.write(f"Algorithm: {algo_name}\n")
        if L is not None and lam is not None:
            f.write(f"Parameters: L={L}, Lambda={lam:.1f}\n")
        f.write(f"Seed Set Size (k): {k}\n\n")
        
        f.write(f"--- Average Metrics for Seed Set ---\n")
        f.write(f"Average Degree: {avg_deg:.2f}\n")
        f.write(f"Average Closeness Centrality: {avg_closeness:.6f}\n")
        f.write(f"Average Harmonic Centrality: {avg_harmonic:.6f}\n")
        f.write(f"Average PageRank: {avg_pr:.6f}\n")
        f.write(f"Average K-Core Number: {avg_kcore:.2f}\n\n")
        
        f.write(f"--- Individual Seed Details ---\n")
        for detail in seed_details:
            f.write(detail + "\n")

# ---------------------------------------------------------------------------
# Worker Task
# ---------------------------------------------------------------------------
def run_network_alpha_task(gml_path, alpha, output_dir):
    network = os.path.basename(gml_path).replace(".gml", "")
    print(f"Starting task: {network}, alpha={alpha:.2f}...")
    
    adj, nodes, G_mapped, n, original_nodes = load_graph(gml_path)
    
    # Precompute features
    from existing_14.helpers import pagerank_safe
    pr_dict = pagerank_safe(G_mapped)
    closeness_dict = nx.closeness_centrality(G_mapped)
    harmonic_dict = nx.harmonic_centrality(G_mapped)
    kcore_dict = nx.core_number(G_mapped)
    
    # Setup algorithms
    T = 15
    R_eval = 50
    
    # Standard algorithms
    seeds_random = random.sample(range(n), 10)
    seeds_myopic, _, _ = myopic(adj, nodes, alpha, 10, T, R=200)
    seeds_naive_myopic, _, _ = naive_myopic(adj, nodes, alpha, 10, T, R=200)
    seeds_gonzalez, _, _ = gonzalez(adj, nodes, alpha, 10, T, R=50)
    seeds_myopic_bfs = myopic_bfs(adj, nodes, alpha, 10, G_mapped)
    seeds_naive_myopic_bfs = naive_myopic_bfs(adj, nodes, alpha, 10, G_mapped)
    seeds_myopic_ppr = myopic_ppr(adj, nodes, alpha, 10, G_mapped)
    seeds_naive_myopic_ppr = naive_myopic_ppr(adj, nodes, alpha, 10, G_mapped)
    seeds_least_central = least_central(adj, nodes, alpha, 10, G_mapped, closeness_dict)
    seeds_least_central_n = least_central_n(adj, nodes, alpha, 10, G_mapped, closeness_dict)
    seeds_min_degree_hc = min_degree_hc(adj, nodes, alpha, 10, G_mapped, harmonic_dict)
    seeds_min_degree_hcn = min_degree_hcn(adj, nodes, alpha, 10, G_mapped, harmonic_dict)
    seeds_min_degree_nd = min_degree_nd(adj, nodes, alpha, 10, G_mapped)
    seeds_min_degree_ndn = min_degree_ndn(adj, nodes, alpha, 10, G_mapped)
    
    seeds_dict = {
        "Random": seeds_random,
        "Myopic": seeds_myopic,
        "Naive Myopic": seeds_naive_myopic,
        "Gonzalez": seeds_gonzalez,
        "Myopic BFS": seeds_myopic_bfs,
        "Naive Myopic BFS": seeds_naive_myopic_bfs,
        "Myopic PPR": seeds_myopic_ppr,
        "Naive Myopic PPR": seeds_naive_myopic_ppr,
        "LeastCentral": seeds_least_central,
        "LeastCentral_n": seeds_least_central_n,
        "MinDegree_hc": seeds_min_degree_hc,
        "MinDegree_hcn": seeds_min_degree_hcn,
        "MinDegree_nd": seeds_min_degree_nd,
        "MinDegree_ndn": seeds_min_degree_ndn
    }
    
    # 4 propagation-aware algorithms (dependent on L and lam)
    L_vals = [1, 2, 3]
    lam_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for L in L_vals:
        for lam in lam_vals:
            # Additive PageRank
            add_seeds = prop_aware_additive(adj, nodes, alpha, 10, T, R_mc=200, pr_dict=pr_dict, L=L, lam=lam)
            seeds_dict[f"Prop-Aware Additive (L={L}, lam={lam:.1f})"] = add_seeds
            
            # Multiplicative PageRank
            mul_seeds = prop_aware_multiplicative(adj, nodes, alpha, 10, T, R_mc=200, pr_dict=pr_dict, L=L, lam=lam)
            seeds_dict[f"Prop-Aware Multiplicative (L={L}, lam={lam:.1f})"] = mul_seeds
            
            # Additive KCore
            add_kc_seeds = prop_aware_additive(adj, nodes, alpha, 10, T, R_mc=200, kcore_dict=kcore_dict, use_kcore=True, L=L, lam=lam)
            seeds_dict[f"Prop-Aware Additive KCore (L={L}, lam={lam:.1f})"] = add_kc_seeds
            
            # Multiplicative KCore
            mul_kc_seeds = prop_aware_multiplicative(adj, nodes, alpha, 10, T, R_mc=200, kcore_dict=kcore_dict, use_kcore=True, L=L, lam=lam)
            seeds_dict[f"Prop-Aware Multiplicative KCore (L={L}, lam={lam:.1f})"] = mul_kc_seeds
            
    # Now evaluate all of them for k in 1..10
    task_res = {}
    for k in range(1, 11):
        task_res[str(k)] = {}
        for algo_name, all_seeds in seeds_dict.items():
            k_seeds = all_seeds[:k]
            probs, mean_var = prob_est_timed(adj, k_seeds, alpha, n, T, R=R_eval)
            mu = sum(probs) / n
            min_p = min(probs)
            
            task_res[str(k)][algo_name] = {
                "seeds": [original_nodes[s] for s in k_seeds],
                "variance": mean_var,
                "mean_prob": mu,
                "min_p": min_p
            }
            
            # Save report
            if " (L=" in algo_name:
                base_algo = algo_name.split(" (")[0]
                param_part = algo_name.split(" (")[1].replace(")", "")
                parts = param_part.split(", ")
                L_val = int(parts[0].split("=")[1])
                lam_val = float(parts[1].split("=")[1])
                save_seed_report(output_dir, network, alpha, base_algo, k, k_seeds, original_nodes, G_mapped, pr_dict, closeness_dict, harmonic_dict, kcore_dict, L=L_val, lam=lam_val)
            else:
                save_seed_report(output_dir, network, alpha, algo_name, k, k_seeds, original_nodes, G_mapped, pr_dict, closeness_dict, harmonic_dict, kcore_dict)
                
    print(f"Finished task: {network}, alpha={alpha:.2f}!")
    return network, alpha, task_res

# ---------------------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------------------
def plot_results(network, alpha, results_k, plot_filename, metric_name="min_p"):
    ks = sorted([int(k) for k in results_k.keys()])
    baselines = ["Random", "Myopic", "Naive Myopic", "Gonzalez"]
    heuristics = [
        "Myopic BFS", "Naive Myopic BFS", "Myopic PPR", "Naive Myopic PPR",
        "LeastCentral", "LeastCentral_n", "MinDegree_hc", "MinDegree_hcn",
        "MinDegree_nd", "MinDegree_ndn"
    ]
    
    heu_performance = {}
    for heu in heuristics:
        vals = []
        for k in ks:
            k_str = str(k)
            if heu in results_k[k_str]:
                vals.append(results_k[k_str][heu][metric_name])
        if vals:
            heu_performance[heu] = sum(vals) / len(vals)
            
    if metric_name == "variance":
        top_heuristics = sorted(heu_performance.items(), key=lambda x: x[1])[:3]
    else:
        top_heuristics = sorted(heu_performance.items(), key=lambda x: -x[1])[:3]
        
    top_heuristic_names = [x[0] for x in top_heuristics]
    
    prop_families = [
        "Prop-Aware Additive",
        "Prop-Aware Multiplicative",
        "Prop-Aware Additive KCore",
        "Prop-Aware Multiplicative KCore"
    ]
    
    best_prop_configs = {}
    for family in prop_families:
        config_performance = {}
        for algo_name in results_k["1"].keys():
            if algo_name.startswith(family + " ("):
                vals = []
                for k in ks:
                    k_str = str(k)
                    if algo_name in results_k[k_str]:
                        vals.append(results_k[k_str][algo_name][metric_name])
                if vals:
                    config_performance[algo_name] = sum(vals) / len(vals)
        if config_performance:
            if metric_name == "variance":
                best_config = min(config_performance.items(), key=lambda x: x[1])[0]
            else:
                best_config = max(config_performance.items(), key=lambda x: x[1])[0]
            best_prop_configs[family] = best_config
            
    algos_to_plot = baselines + top_heuristic_names + list(best_prop_configs.values())
    
    plt.figure(figsize=(10, 6))
    for algo in algos_to_plot:
        x_vals = []
        y_vals = []
        for k in ks:
            k_str = str(k)
            if algo in results_k[k_str]:
                x_vals.append(k)
                y_vals.append(results_k[k_str][algo][metric_name])
        if x_vals:
            plt.plot(x_vals, y_vals, label=algo, marker="o", linewidth=1.5, markersize=4)
            
    title_metric = "Variance of Access" if metric_name == "variance" else "Min Access Probability"
    plt.title(f"{title_metric} vs. Seed Set Size (k) - {network} (alpha={alpha:.2f})")
    plt.xlabel("Seed Set Size (k)")
    plt.ylabel(title_metric)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename, dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run experiment measuring accuracy ( Rawlsian fairness / min access prob ).")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel processes to use")
    parser.add_argument("--networks", type=str, default=None, help="Comma-separated list of networks to run (e.g. EU,Facebook)")
    parser.add_argument("--alphas", type=str, default=None, help="Comma-separated list of alphas to run (e.g. 0.01,0.02,0.10)")
    args = parser.parse_args()
    
    output_dir = "result_acc"
    os.makedirs(output_dir, exist_ok=True)
    master_json_path = os.path.join(output_dir, "master.json")
    
    results = {}
    if os.path.exists(master_json_path):
        try:
            with open(master_json_path, "r") as f:
                results = json.load(f)
            print(f"Loaded existing results from {master_json_path}.")
        except Exception as e:
            print(f"Error loading master.json: {e}. Starting fresh.")
            
    gml_files = []
    search_dirs = ["Small", "../Small", "code/Small", "../code/Small", "Social_Network", "../Social_Network"]
    for d in search_dirs:
        found = sorted(glob.glob(os.path.join(d, "*.gml"))) + sorted(glob.glob(os.path.join(d, "*.txt")))
        if found:
            gml_files = found
            break
            
    if not gml_files:
        print("Error: No GML or TXT files found in any search directories: " + ", ".join(search_dirs))
        return
        
    if args.networks:
        include_networks = [n.strip().lower() for n in args.networks.split(",")]
        gml_files = [f for f in gml_files if os.path.basename(f).replace(".gml", "").replace(".txt", "").lower() in include_networks]
        
    print(f"Found {len(gml_files)} network files to run.")
    
    alphas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20, 0.30]
    if args.alphas:
        alphas = [float(a.strip()) for a in args.alphas.split(",")]
    
    baselines = ["Random", "Myopic", "Naive Myopic", "Gonzalez", "Myopic BFS", "Naive Myopic BFS", 
                 "Myopic PPR", "Naive Myopic PPR", "LeastCentral", "LeastCentral_n", 
                 "MinDegree_hc", "MinDegree_hcn", "MinDegree_nd", "MinDegree_ndn"]
    prop_families = ["Prop-Aware Additive", "Prop-Aware Multiplicative", "Prop-Aware Additive KCore", "Prop-Aware Multiplicative KCore"]
    L_vals = [1, 2, 3]
    lam_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    algos_to_check = list(baselines)
    for family in prop_families:
        for L in L_vals:
            for lam in lam_vals:
                algos_to_check.append(f"{family} (L={L}, lam={lam:.1f})")
                
    tasks_to_run = []
    for gml_path in gml_files:
        network = os.path.basename(gml_path).replace(".gml", "")
        for alpha in alphas:
            alpha_str = f"{alpha:.2f}"
            complete = True
            if network not in results or alpha_str not in results[network]:
                complete = False
            else:
                for k in range(1, 11):
                    k_str = str(k)
                    if k_str not in results[network][alpha_str]:
                        complete = False
                        break
                    for algo in algos_to_check:
                        if algo not in results[network][alpha_str][k_str]:
                            complete = False
                            break
                    if not complete:
                        break
            if not complete:
                tasks_to_run.append((gml_path, alpha))
                
    print(f"Total tasks: {len(gml_files) * len(alphas)}, Tasks remaining to run: {len(tasks_to_run)}")
    
    if not tasks_to_run:
        print("All tasks already completed. Generating final plots...")
        for gml_path in gml_files:
            network = os.path.basename(gml_path).replace(".gml", "")
            for alpha in alphas:
                alpha_str = f"{alpha:.2f}"
                if network in results and alpha_str in results[network]:
                    plot_filename = os.path.join(output_dir, network, f"alpha_{alpha:.2f}", "min_acc.png")
                    plot_results(network, alpha, results[network][alpha_str], plot_filename, metric_name="min_p")
        print("Plots generated successfully!")
        return

    workers = args.num_workers if args.num_workers else os.cpu_count()
    print(f"Running execution pool with {workers} worker processes...")
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_network_alpha_task, task[0], task[1], output_dir): task for task in tasks_to_run}
        
        for future in as_completed(futures):
            gml_path, alpha = futures[future]
            network = os.path.basename(gml_path).replace(".gml", "")
            alpha_str = f"{alpha:.2f}"
            
            try:
                net_name, alpha_val, task_res = future.result()
                
                if net_name not in results:
                    results[net_name] = {}
                results[net_name][f"{alpha_val:.2f}"] = task_res
                
                with open(master_json_path, "w") as f:
                    json.dump(results, f, indent=2)
                    
                plot_filename = os.path.join(output_dir, net_name, f"alpha_{alpha_val:.2f}", "min_acc.png")
                plot_results(net_name, alpha_val, task_res, plot_filename, metric_name="min_p")
                print(f"Recorded and plotted results for network {net_name}, alpha={alpha_val:.2f}.")
                
            except Exception as exc:
                print(f"Task generated an exception: {network}, alpha={alpha:.2f}: {exc}")
                import traceback
                traceback.print_exc()

    print("\nAll remaining tasks completed successfully!")

if __name__ == "__main__":
    main()
