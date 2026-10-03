import os
import sys
import glob
import json
import networkx as nx
import matplotlib.pyplot as plt

from myopic import myopic
from naive_myopic import naive_myopic
from gonzalez import gonzalez
from concave_hybrid import concave_hybrid
from icm import prob_est_timed, simulate_timeseries

def compute_variance(probs):
    """Population variance Var(P) = (1/N) * sum((p_i - mu)^2), matching attempt2/metrics.py."""
    n = len(probs)
    if n < 2:
        return 0.0
    mu = sum(probs) / n
    return sum((p - mu) ** 2 for p in probs) / n

def compute_variance_stats(adj, seeds, alpha, n, T, R, runs=50):
    import math
    variances = []
    for _ in range(runs):
        probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
        variances.append(compute_variance(probs))
    mean_var = sum(variances) / runs
    std_var = math.sqrt(sum((v - mean_var) ** 2 for v in variances) / runs)
    return mean_var, std_var

def load_existing_baselines(network, alpha):
    """Scans other results directories to find pre-computed baseline data."""
    possible_dirs = [
        "result_hybrid_degree_multi",
        "result_pagerank_v1_multi",
        "result_pagerank_hybrid_multi",
        "result_entropy_multi"
    ]
    for d in possible_dirs:
        path = os.path.join(d, network, str(alpha), "results.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    res = json.load(f)
                    if "myopic" in res and "naive_myopic" in res and "gonzalez" in res:
                        return res["myopic"], res["naive_myopic"], res["gonzalez"]
            except Exception:
                pass
    return None, None, None

def plot_variance(myo_vars, con_vars, nm_vars, gonz_vars, ks, out_path):
    plt.figure(figsize=(8, 6))
    plt.plot(ks, myo_vars, marker='o', label='Myopic (Mean Var)')
    plt.plot(ks, nm_vars, marker='s', linestyle='--', label='Naive Myopic (Mean Var)')
    plt.plot(ks, gonz_vars, marker='^', linestyle='-.', label='Gonzalez (Mean Var)')
    plt.plot(ks, con_vars, marker='p', linewidth=2.5, label='Concave-Hybrid (Mean Var)')
    plt.title("Mean Variance vs Seed Set Size (50 runs)")
    plt.xlabel("Seed Set Size (k)")
    plt.ylabel("Mean Variance Var(P) (Lowest is Best)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_distribution(myo_hits, nm_hits, gonz_hits, con_hits, network, alpha, k, out_path):
    plt.figure(figsize=(10, 6))
    bins = range(0, 205, 5)
    plt.hist(myo_hits, bins=bins, alpha=0.4, label='Myopic', histtype='step', linewidth=1.5)
    plt.hist(nm_hits, bins=bins, alpha=0.4, label='Naive Myopic', histtype='step', linewidth=1.5)
    plt.hist(gonz_hits, bins=bins, alpha=0.4, label='Gonzalez', histtype='step', linewidth=1.5)
    plt.hist(con_hits, bins=bins, alpha=0.7, label='Concave-Hybrid', histtype='step', linewidth=2.5)
    
    plt.title(f"Influence Frequency Distribution: {network} | Alpha={alpha} | K={k}")
    plt.xlabel("Number of times influenced (out of 200 runs)")
    plt.ylabel("Frequency (Number of Nodes)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_influence_simulation(myo_ts, nm_ts, gonz_ts, con_ts, network, alpha, k, out_path):
    plt.figure(figsize=(9, 6))
    timesteps = list(range(len(myo_ts)))
    plt.plot(timesteps, myo_ts, marker='o', label='Myopic')
    plt.plot(timesteps, nm_ts, marker='s', linestyle='--', label='Naive Myopic')
    plt.plot(timesteps, gonz_ts, marker='^', linestyle='-.', label='Gonzalez')
    plt.plot(timesteps, con_ts, marker='p', linewidth=2.5, label='Concave-Hybrid')
    
    plt.title(f"Influence Cascade Simulation: {network} | Alpha={alpha} | K={k}")
    plt.xlabel("Time Step (t)")
    plt.ylabel("Cumulative Nodes Influenced")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_final_variance_bar(myo_var, nm_var, gonz_var, algo_var, algo_label, network, alpha, k, out_path):
    plt.figure(figsize=(8, 5.5))
    algos = ['Myopic', 'Naive Myopic', 'Gonzalez', algo_label]
    variances = [myo_var, nm_var, gonz_var, algo_var]
    colors = ['#1f77b4', '#aec7e8', '#ffbb78', '#2ca02c' if algo_var < myo_var else '#d62728']
    
    bars = plt.bar(algos, variances, color=colors, width=0.55, edgecolor='black', alpha=0.85)
    
    for bar, val in zip(bars, variances):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(variances)*0.01 if max(variances) > 0 else 0.001),
                 f"{val:.5f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    pct_diff = ((myo_var - algo_var) / myo_var) * 100.0 if myo_var > 0 else 0.0
    status_str = f"Variance Reduction vs Myopic: {pct_diff:+.2f}%"
    
    plt.title(f"Final Mean Variance Var(P) (50 runs): {network} | Alpha={alpha} | K={k}\n{status_str}", fontsize=11)
    plt.ylabel("Mean Population Variance Var(P)", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def run_experiment():
    gml_files = glob.glob(os.path.join("Social_Network", "*.gml"))
    if not gml_files:
        print("No .gml files found in Social_Network/")
        return
        
    alphas = [0.03, 0.06, 0.09]
    ks = [5, 10, 15, 20, 25, 30]
    T = 15
    R = 200
    R_probe = 20
    
    for gml_path in gml_files:
        network = os.path.basename(gml_path).replace(".gml", "")
        print(f"\n{'='*50}\n[CONCAVE HYBRID EXPERIMENT] Processing {network}\n{'='*50}")
        
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
            
        # Extract PageRank from GML attributes or compute if not present
        pr_dict = {}
        for i, node_data in enumerate(nodes):
            if isinstance(node_data, dict):
                for key in ['pagerank', 'page_rank', 'PageRank', 'pagerank_centrality']:
                    if key in node_data:
                        pr_dict[i] = float(node_data[key])
                        break
        
        if len(pr_dict) == n:
            print("  Loaded PageRank directly from GML node attributes.")
        else:
            print("  PageRank attribute not found in GML; precomputing PageRank...")
            pr_dict = nx.pagerank(G_mapped)
        
        for alpha in alphas:
            print(f"  Alpha: {alpha}")
            alpha_dir = os.path.join("result_concave_multi", network, str(alpha))
            os.makedirs(alpha_dir, exist_ok=True)
            
            results_json = {"myopic": {}, "naive_myopic": {}, "gonzalez": {}, "concave_hybrid": {}}
            myo_vars = []
            nm_vars = []
            gonz_vars = []
            con_vars = []
            
            # Try to load precomputed baselines to avoid redundant calculations
            myo_base, nm_base, gonz_base = load_existing_baselines(network, alpha)
            
            for k in ks:
                print(f"    K: {k}")
                k_dir = os.path.join(alpha_dir, f"k_{k}")
                os.makedirs(k_dir, exist_ok=True)
                k_str = str(k)
                
                # Check if we can reuse precomputed baselines
                has_baselines = (myo_base and k_str in myo_base and 
                                 nm_base and k_str in nm_base and 
                                 gonz_base and k_str in gonz_base)
                
                # 1. Myopic Baseline
                print("      Running Myopic...")
                if has_baselines:
                    m_seeds = myo_base[k_str]["seeds"]
                    m_probs, m_hits = prob_est_timed(adj, m_seeds, alpha, n, T, R)
                    m_ts = simulate_timeseries(adj, m_seeds, alpha, n, T, R)
                    m_var = myo_base[k_str]["var"]
                    m_std = myo_base[k_str].get("var_std", 0.0)
                else:
                    m_seeds, m_probs, m_hits = myopic(adj, nodes, alpha, k, T, R)
                    m_var, m_std = compute_variance_stats(adj, m_seeds, alpha, n, T, R, 50)
                    m_ts = simulate_timeseries(adj, m_seeds, alpha, n, T, R)
                myo_vars.append(m_var)
                results_json["myopic"][k_str] = {
                    "seeds": m_seeds, 
                    "var": m_var, 
                    "var_std": m_std,
                    "mu": sum(m_probs) / n, 
                    "final_reach": m_ts[-1]
                }
                
                # 2. Naive Myopic
                print("      Running Naive Myopic...")
                if has_baselines:
                    nm_seeds = nm_base[k_str]["seeds"]
                    nm_probs, nm_hits = prob_est_timed(adj, nm_seeds, alpha, n, T, R)
                    nm_ts = simulate_timeseries(adj, nm_seeds, alpha, n, T, R)
                    nm_var = nm_base[k_str]["var"]
                    nm_std = nm_base[k_str].get("var_std", 0.0)
                else:
                    nm_seeds, nm_probs, nm_hits = naive_myopic(adj, nodes, alpha, k, T, R)
                    nm_var, nm_std = compute_variance_stats(adj, nm_seeds, alpha, n, T, R, 50)
                    nm_ts = simulate_timeseries(adj, nm_seeds, alpha, n, T, R)
                nm_vars.append(nm_var)
                results_json["naive_myopic"][k_str] = {
                    "seeds": nm_seeds, 
                    "var": nm_var, 
                    "var_std": nm_std,
                    "mu": sum(nm_probs) / n, 
                    "final_reach": nm_ts[-1]
                }
                
                # 3. Gonzalez
                print("      Running Gonzalez...")
                if has_baselines:
                    g_seeds = gonz_base[k_str]["seeds"]
                    g_probs, g_hits = prob_est_timed(adj, g_seeds, alpha, n, T, R)
                    g_ts = simulate_timeseries(adj, g_seeds, alpha, n, T, R)
                    g_var = gonz_base[k_str]["var"]
                    g_std = gonz_base[k_str].get("var_std", 0.0)
                else:
                    g_seeds, g_probs, g_hits = gonzalez(adj, nodes, alpha, k, T, R)
                    g_var, g_std = compute_variance_stats(adj, g_seeds, alpha, n, T, R, 50)
                    g_ts = simulate_timeseries(adj, g_seeds, alpha, n, T, R)
                gonz_vars.append(g_var)
                results_json["gonzalez"][k_str] = {
                    "seeds": g_seeds, 
                    "var": g_var, 
                    "var_std": g_std,
                    "mu": sum(g_probs) / n, 
                    "final_reach": g_ts[-1]
                }
                
                # 4. Concave-Hybrid
                print("      Running Concave-Hybrid...")
                c_seeds, c_probs, c_hits, c_log = concave_hybrid(
                    adj, nodes, alpha, k, T, R=R, R_probe=R_probe, phi=-1.0, pr_dict=pr_dict
                )
                c_var, c_std = compute_variance_stats(adj, c_seeds, alpha, n, T, R, 50)
                c_ts = simulate_timeseries(adj, c_seeds, alpha, n, T, R)
                con_vars.append(c_var)
                results_json["concave_hybrid"][k_str] = {
                    "seeds": c_seeds, 
                    "var": c_var, 
                    "var_std": c_std,
                    "mu": sum(c_probs) / n, 
                    "final_reach": c_ts[-1]
                }
                
                # 5. Save report.md for this K
                report_path = os.path.join(k_dir, "report.md")
                with open(report_path, "w") as rep:
                    rep.write(f"# Concave-Hybrid Selection Report\n")
                    rep.write(f"**Network:** {network} | **Alpha:** {alpha} | **K:** {k}\n\n")
                    rep.write("| Step | Chosen Seed | Mean Prob (mu) | Var | Welfare | JFI | Min Prob | Access Gap | Disparity | Trend | Time (s) |\n")
                    rep.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
                    for entry in c_log:
                        step = entry["step"]
                        seed = entry["chosen_seed"] if "chosen_seed" in entry else entry["seed"]
                        mu = f"{entry['mu']:.4f}"
                        var_val = f"{entry['var']:.6f}"
                        welfare_val = f"{entry['welfare']:.4f}"
                        jfi_val = f"{entry['jfi']:.4f}"
                        min_p = f"{entry['min_p']:.4f}"
                        gap = f"{entry['gap']:.4f}"
                        disparity = f"{entry['disparity']:.4f}"
                        trend = entry["var_trend"]
                        t_elapsed = f"{entry['time_s']:.1f}"
                        rep.write(f"| {step} | {seed} | {mu} | {var_val} | {welfare_val} | {jfi_val} | {min_p} | {gap} | {disparity} | {trend} | {t_elapsed} |\n")
                        
                # 6. Save Plots for this K
                plot_distribution(
                    m_hits, nm_hits, g_hits, c_hits,
                    network, alpha, k, os.path.join(k_dir, "distribution.png")
                )
                plot_influence_simulation(
                    m_ts, nm_ts, g_ts, c_ts,
                    network, alpha, k, os.path.join(k_dir, "influence_simulation.png")
                )
                plot_final_variance_bar(
                    m_var, nm_var, g_var, c_var, "Concave-Hybrid",
                    network, alpha, k, os.path.join(k_dir, "final_variance_comparison.png")
                )
                
            with open(os.path.join(alpha_dir, "results.json"), "w") as f:
                json.dump(results_json, f, indent=2)
                
            plot_variance(myo_vars, con_vars, nm_vars, gonz_vars, ks, os.path.join(alpha_dir, "variance_vs_seedset.png"))
            
    print("\nConcave-Hybrid experiments completed successfully!")

if __name__ == "__main__":
    run_experiment()
