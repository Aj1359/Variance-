import os
import sys
import glob
import json
import networkx as nx
import matplotlib.pyplot as plt

from myopic import myopic
from naive_myopic import naive_myopic
from gonzalez import gonzalez
from myopic_hybrid_cc import myopic_hybrid_cc
from icm import prob_est_timed, simulate_timeseries

def compute_variance(probs):
    """Population variance Var(P) = (1/N) * sum((p_i - mu)^2), matching attempt2/metrics.py."""
    n = len(probs)
    if n < 2:
        return 0.0
    mu = sum(probs) / n
    return sum((p - mu) ** 2 for p in probs) / n

def plot_variance(myo_vars, hyb_vars, nm_vars, gonz_vars, ks, out_path, pct):
    plt.figure(figsize=(8, 6))
    plt.plot(ks, myo_vars, marker='o', label='Myopic')
    plt.plot(ks, nm_vars, marker='s', linestyle='--', label='Naive Myopic')
    plt.plot(ks, gonz_vars, marker='^', linestyle='-.', label='Gonzalez')
    plt.plot(ks, hyb_vars, marker='o', linewidth=2, label=f'Hybrid (Top {pct})')
    plt.title("Variance vs Seed Set Size")
    plt.xlabel("Seed Set Size (k)")
    plt.ylabel("Variance Var(P) (Lowest is Best)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_distribution(myo_hits, nm_hits, gonz_hits, hyb_hits, network, alpha, k, pct, out_path):
    plt.figure(figsize=(10, 6))
    bins = range(0, 205, 5)
    plt.hist(myo_hits, bins=bins, alpha=0.4, label='Myopic', histtype='step', linewidth=1.5)
    plt.hist(nm_hits, bins=bins, alpha=0.4, label='Naive Myopic', histtype='step', linewidth=1.5)
    plt.hist(gonz_hits, bins=bins, alpha=0.4, label='Gonzalez', histtype='step', linewidth=1.5)
    plt.hist(hyb_hits, bins=bins, alpha=0.7, label=f'Hybrid ({pct})', histtype='step', linewidth=2.5)
    
    plt.title(f"Influence Frequency Distribution: {network} | Alpha={alpha} | K={k}")
    plt.xlabel("Number of times influenced (out of 200 runs)")
    plt.ylabel("Frequency (Number of Nodes)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_influence_simulation(myo_ts, nm_ts, gonz_ts, hyb_ts, network, alpha, k, pct, out_path):
    plt.figure(figsize=(9, 6))
    timesteps = list(range(len(myo_ts)))
    plt.plot(timesteps, myo_ts, marker='o', label='Myopic')
    plt.plot(timesteps, nm_ts, marker='s', linestyle='--', label='Naive Myopic')
    plt.plot(timesteps, gonz_ts, marker='^', linestyle='-.', label='Gonzalez')
    plt.plot(timesteps, hyb_ts, marker='o', linewidth=2, label=f'Hybrid ({pct})')
    
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
    
    plt.title(f"Final Post-Seeding Variance Var(P): {network} | Alpha={alpha} | K={k}\n{status_str}", fontsize=11)
    plt.ylabel("Population Variance Var(P)", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def run_experiment():
    gml_files = glob.glob(os.path.join("Social_Network", "*.gml"))
    if not gml_files:
        print("No .gml files found in Social_Network/")
        return
        
    alphas = [0.2, 0.3, 0.4]
    ks = [10, 20, 30, 40, 50]
    cc_pcts = [0.05, 0.10, 0.15]
    T = 15
    R = 200
    
    for gml_path in gml_files:
        network = os.path.basename(gml_path).replace(".gml", "")
        print(f"\n{'='*50}\n[EXPERIMENT 1: HYBRID vs BASELINES] Processing {network}\n{'='*50}")
        
        G = nx.read_gml(gml_path, destringizer=int)
        original_nodes = sorted([int(n) for n in G.nodes()])
        id_map = {orig: new for new, orig in enumerate(original_nodes)}
        n = len(original_nodes)
        
        nodes = [{} for _ in range(n)]
        for n_id, attrs in G.nodes(data=True):
            mapped_id = id_map[int(n_id)]
            nodes[mapped_id] = attrs
            
        adj = [[] for _ in range(n)]
        for u, v in G.edges():
            u, v = int(u), int(v)
            mu, mv = id_map[u], id_map[v]
            adj[mu].append(mv)
            adj[mv].append(mu)
        
        for alpha in alphas:
            print(f"  Alpha: {alpha}")
            baseline_cache = {}
            
            for pct in cc_pcts:
                pct_label = f"{int(pct*100)}_percent"
                print(f"    Closeness Centrality: {pct_label}")
                
                pct_dir = os.path.join("result_hybrid", network, str(alpha), pct_label)
                os.makedirs(pct_dir, exist_ok=True)
                
                results_json = {"myopic": {}, "naive_myopic": {}, "gonzalez": {}, "hybrid": {}}
                myo_vars = []
                nm_vars = []
                gonz_vars = []
                hyb_vars = []
                
                for k in ks:
                    print(f"      K: {k}")
                    k_dir = os.path.join(pct_dir, f"k_{k}")
                    os.makedirs(k_dir, exist_ok=True)
                    
                    if k not in baseline_cache:
                        print("        Running Myopic...")
                        m_seeds, m_probs, m_hits = myopic(adj, nodes, alpha, k, T, R)
                        m_var = compute_variance(m_probs)
                        m_ts = simulate_timeseries(adj, m_seeds, alpha, n, T, R)
                        
                        print("        Running Naive Myopic...")
                        nm_seeds, nm_probs, nm_hits = naive_myopic(adj, nodes, alpha, k, T, R)
                        nm_var = compute_variance(nm_probs)
                        nm_ts = simulate_timeseries(adj, nm_seeds, alpha, n, T, R)
                        
                        print("        Running Gonzalez...")
                        g_seeds, g_probs, g_hits = gonzalez(adj, nodes, alpha, k, T, R)
                        g_var = compute_variance(g_probs)
                        g_ts = simulate_timeseries(adj, g_seeds, alpha, n, T, R)
                        
                        baseline_cache[k] = {
                            "myopic": {"seeds": m_seeds, "hits": m_hits, "var": m_var, "mu": sum(m_probs) / n, "ts": m_ts},
                            "naive_myopic": {"seeds": nm_seeds, "hits": nm_hits, "var": nm_var, "mu": sum(nm_probs) / n, "ts": nm_ts},
                            "gonzalez": {"seeds": g_seeds, "hits": g_hits, "var": g_var, "mu": sum(g_probs) / n, "ts": g_ts}
                        }
                        
                    cache_k = baseline_cache[k]
                    myo_vars.append(cache_k["myopic"]["var"])
                    nm_vars.append(cache_k["naive_myopic"]["var"])
                    gonz_vars.append(cache_k["gonzalez"]["var"])
                    
                    results_json["myopic"][str(k)] = {"seeds": cache_k["myopic"]["seeds"], "var": cache_k["myopic"]["var"], "mu": cache_k["myopic"]["mu"], "final_reach": cache_k["myopic"]["ts"][-1]}
                    results_json["naive_myopic"][str(k)] = {"seeds": cache_k["naive_myopic"]["seeds"], "var": cache_k["naive_myopic"]["var"], "mu": cache_k["naive_myopic"]["mu"], "final_reach": cache_k["naive_myopic"]["ts"][-1]}
                    results_json["gonzalez"][str(k)] = {"seeds": cache_k["gonzalez"]["seeds"], "var": cache_k["gonzalez"]["var"], "mu": cache_k["gonzalez"]["mu"], "final_reach": cache_k["gonzalez"]["ts"][-1]}
                    
                    # Run Hybrid with Closeness Centrality Batching
                    print("        Running Myopic-Hybrid-CC...")
                    h_seeds, h_probs, h_hits, h_log = myopic_hybrid_cc(adj, nodes, alpha, k, T, top_cc_pct=pct, R=R)
                    h_var = compute_variance(h_probs)
                    h_ts = simulate_timeseries(adj, h_seeds, alpha, n, T, R)
                    hyb_vars.append(h_var)
                    
                    results_json["hybrid"][str(k)] = {"seeds": h_seeds, "var": h_var, "mu": sum(h_probs) / n, "final_reach": h_ts[-1]}
                    
                    # 1. Save report.md for this K
                    report_path = os.path.join(k_dir, "report.md")
                    with open(report_path, "w") as rep:
                        rep.write(f"# Hybrid Candidate Selection Report\n")
                        rep.write(f"**Network:** {network} | **Alpha:** {alpha} | **K:** {k} | **CC:** {pct_label}\n\n")
                        rep.write("| Iteration | Candidates in Epsilon Band | Candidates Filtered by CC | Batch Size Chosen | Selected Seed IDs |\n")
                        rep.write("|---|---|---|---|---|\n")
                        for entry in h_log:
                            it = entry["iteration"]
                            c_eps = entry["candidates_in_epsilon_band"]
                            c_flt = entry["candidates_after_cc_filter"]
                            s_add = entry["seeds_chosen_this_batch"]
                            seeds_str = ", ".join(map(str, entry["seeds_added"]))
                            rep.write(f"| {it} | {c_eps} | {c_flt} | {s_add} | {seeds_str} |\n")
                            
                    # 2. Save distribution plot for this K
                    plot_distribution(
                        cache_k["myopic"]["hits"],
                        cache_k["naive_myopic"]["hits"],
                        cache_k["gonzalez"]["hits"],
                        h_hits, network, alpha, k, pct_label, os.path.join(k_dir, "distribution.png")
                    )
                    
                    # 3. Save influence simulation time-series plot for this K
                    plot_influence_simulation(
                        cache_k["myopic"]["ts"],
                        cache_k["naive_myopic"]["ts"],
                        cache_k["gonzalez"]["ts"],
                        h_ts, network, alpha, k, pct_label, os.path.join(k_dir, "influence_simulation.png")
                    )
                    
                    # 4. Save final post-seeding variance bar chart for this K
                    plot_final_variance_bar(
                        cache_k["myopic"]["var"],
                        cache_k["naive_myopic"]["var"],
                        cache_k["gonzalez"]["var"],
                        h_var,
                        f"Hybrid ({pct_label})",
                        network, alpha, k,
                        os.path.join(k_dir, "final_variance_comparison.png")
                    )
                    
                with open(os.path.join(pct_dir, "results.json"), "w") as f:
                    json.dump(results_json, f, indent=2)
                    
                plot_variance(myo_vars, hyb_vars, nm_vars, gonz_vars, ks, os.path.join(pct_dir, "variance_vs_seedset.png"), pct_label)
                
    print("\n[EXPERIMENT 1] Completed successfully! Results in result_hybrid/")

if __name__ == "__main__":
    run_experiment()
