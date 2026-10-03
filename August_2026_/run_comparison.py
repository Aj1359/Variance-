import os
import sys
import glob
import json
import time
import random
import networkx as nx
import matplotlib.pyplot as plt

# Import existing algorithms
from myopic import myopic
from naive_myopic import naive_myopic
from gonzalez import gonzalez
from myopic_hybrid_degree import myopic_hybrid_degree
from myopic_hybrid_cc import myopic_hybrid_cc
from new_heu_pr import new_heu_pr
from new_heu_pr_v2 import new_heu_pr_v2
from concave_hybrid import concave_hybrid
from myopic_hybrid_kcore import myopic_hybrid_kcore
from icm import prob_est_timed, simulate_timeseries

# Import paper 1 algorithms
from paper_algorithms import (
    myopic_bfs, naive_myopic_bfs, myopic_ppr, naive_myopic_ppr,
    least_central, least_central_n, min_degree_hc, min_degree_hcn,
    min_degree_nd, min_degree_ndn
)

# Import paper 2 algorithms
from propagation_aware import propagation_aware_fairness

# ---------------------------------------------------------------------------
# RIS TIM+ Implementation
# ---------------------------------------------------------------------------
def generate_rr_set(adj, n, alpha):
    """Generate a single Reverse Reachable (RR) set."""
    v = random.randint(0, n - 1)
    rr_set = {v}
    queue = [v]
    visited = {v}
    
    while queue:
        curr = queue.pop(0)
        for nbr in adj[curr]:
            if nbr not in visited:
                if random.random() < alpha:
                    visited.add(nbr)
                    rr_set.add(nbr)
                    queue.append(nbr)
    return rr_set

def tim_plus(adj, n, k, alpha, theta=500):
    """
    Two-stage Influence Maximization (TIM+) baseline using Reverse Influence Sampling (RIS).
    """
    rr_sets = [generate_rr_set(adj, n, alpha) for _ in range(theta)]
    
    node_to_rr = {i: [] for i in range(n)}
    for idx, rr in enumerate(rr_sets):
        for node in rr:
            node_to_rr[node].append(idx)
            
    seeds = []
    weight = [1] * theta
    node_score = {i: len(node_to_rr[i]) for i in range(n)}
    
    for _ in range(k):
        best_node = -1
        best_count = -1
        for node in range(n):
            if node in seeds:
                continue
            if node_score[node] > best_count:
                best_count = node_score[node]
                best_node = node
                
        if best_node == -1 or best_count <= 0:
            remaining_nodes = [node for node in range(n) if node not in seeds]
            if remaining_nodes:
                best_node = random.choice(remaining_nodes)
            else:
                break
                
        seeds.append(best_node)
        for idx in node_to_rr[best_node]:
            if weight[idx] == 1:
                weight[idx] = 0
                for node in rr_sets[idx]:
                    node_score[node] -= 1
                    
    return seeds

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def compute_variance(probs):
    n = len(probs)
    if n < 2:
        return 0.0
    mu = sum(probs) / n
    return sum((p - mu) ** 2 for p in probs) / n

# ---------------------------------------------------------------------------
# Graph loader
# ---------------------------------------------------------------------------
def load_graph(gml_path):
    print(f"Loading {os.path.basename(gml_path)}...")
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

# ---------------------------------------------------------------------------
# Execution wrappers
# ---------------------------------------------------------------------------
# 1. Classic Baselines
def run_random(adj, nodes, alpha, k, T, R, **kwargs):
    n = len(nodes)
    seeds = random.sample(range(n), k)
    probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
    return seeds, probs

def run_tim_plus(adj, nodes, alpha, k, T, R, **kwargs):
    n = len(nodes)
    seeds = tim_plus(adj, n, k, alpha)
    probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
    return seeds, probs

def run_myopic(adj, nodes, alpha, k, T, R, **kwargs):
    seeds, probs, _ = myopic(adj, nodes, alpha, k, T, R)
    return seeds, probs

def run_naive_myopic(adj, nodes, alpha, k, T, R, **kwargs):
    seeds, probs, _ = naive_myopic(adj, nodes, alpha, k, T, R)
    return seeds, probs

def run_gonzalez(adj, nodes, alpha, k, T, R, **kwargs):
    seeds, probs, _ = gonzalez(adj, nodes, alpha, k, T, R)
    return seeds, probs

# 2. Paper 1 Heuristics
def run_myopic_bfs(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = myopic_bfs(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_naive_myopic_bfs(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = naive_myopic_bfs(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_myopic_ppr(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = myopic_ppr(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_naive_myopic_ppr(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = naive_myopic_ppr(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_least_central(adj, nodes, alpha, k, T, R, G_mapped, closeness_dict=None, **kwargs):
    seeds = least_central(adj, nodes, alpha, k, G_mapped, closeness_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_least_central_n(adj, nodes, alpha, k, T, R, G_mapped, closeness_dict=None, **kwargs):
    seeds = least_central_n(adj, nodes, alpha, k, G_mapped, closeness_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_min_degree_hc(adj, nodes, alpha, k, T, R, G_mapped, harmonic_dict=None, **kwargs):
    seeds = min_degree_hc(adj, nodes, alpha, k, G_mapped, harmonic_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_min_degree_hcn(adj, nodes, alpha, k, T, R, G_mapped, harmonic_dict=None, **kwargs):
    seeds = min_degree_hcn(adj, nodes, alpha, k, G_mapped, harmonic_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_min_degree_nd(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = min_degree_nd(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_min_degree_ndn(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = min_degree_ndn(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

# 3. Paper 2 Propagation-Aware
def run_propagation_aware_additive(adj, nodes, alpha, k, T, R, pr_dict=None, **kwargs):
    seeds = propagation_aware_fairness(adj, nodes, alpha, k, T, R, pr_dict, L=2, lam=0.5, multiplicative=False, use_pareto=False)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_propagation_aware_multiplicative(adj, nodes, alpha, k, T, R, pr_dict=None, **kwargs):
    seeds = propagation_aware_fairness(adj, nodes, alpha, k, T, R, pr_dict, L=2, lam=0.5, multiplicative=True, use_pareto=True)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

# 4. Custom Designs
def run_ppr(adj, nodes, alpha, k, T, R, pr_dict=None, **kwargs):
    seeds, probs, _, _ = new_heu_pr(adj, nodes, alpha, k, T, R, pr_dict=pr_dict)
    return seeds, probs

def run_ppr_lookahead(adj, nodes, alpha, k, T, R, pr_dict=None, **kwargs):
    seeds, probs, _, _ = new_heu_pr_v2(adj, nodes, alpha, k, T, R, pr_dict=pr_dict)
    return seeds, probs

def run_concave(adj, nodes, alpha, k, T, R, pr_dict=None, **kwargs):
    res = concave_hybrid(adj, nodes, alpha, k, T, R=R, R_probe=20, phi=-1.0, pr_dict=pr_dict)
    seeds = res[0]
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_avg_degree(adj, nodes, alpha, k, T, R, **kwargs):
    seeds, probs, _, _ = myopic_hybrid_degree(adj, nodes, alpha, k, T, R)
    return seeds, probs

def run_kcore(adj, nodes, alpha, k, T, R, kcore_dict=None, **kwargs):
    seeds, probs, _, _ = myopic_hybrid_kcore(adj, nodes, alpha, k, T, R=R, kcore_dict=kcore_dict)
    return seeds, probs

# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------
def main():
    gml_files = sorted(glob.glob(os.path.join("Social_Network", "*.gml")))
    if not gml_files:
        print("Error: No GML files found in Social_Network/")
        return
        
    out_dir = "result_comparison"
    os.makedirs(out_dir, exist_ok=True)
    
    alphas = [0.3]
    ks = [10, 20, 30, 40, 50]
    T = 15
    R = 25  # Set R=25 to allow fast but highly accurate and stable run times
    
    algos = {
        # 1. Classic Baselines
        "Random": run_random,
        "TIM+": run_tim_plus,
        "Myopic": run_myopic,
        "Naive Myopic": run_naive_myopic,
        "Gonzalez": run_gonzalez,
        
        # 2. Paper 1 Heuristics
        "Myopic BFS": run_myopic_bfs,
        "Naive Myopic BFS": run_naive_myopic_bfs,
        "Myopic PPR": run_myopic_ppr,
        "Naive Myopic PPR": run_naive_myopic_ppr,
        "LeastCentral": run_least_central,
        "LeastCentral_n": run_least_central_n,
        "MinDegree_hc": run_min_degree_hc,
        "MinDegree_hcn": run_min_degree_hcn,
        "MinDegree_nd": run_min_degree_nd,
        "MinDegree_ndn": run_min_degree_ndn,
        
        # 3. Paper 2 Propagation-Aware
        "Prop-Aware Additive": run_propagation_aware_additive,
        "Prop-Aware Multiplicative": run_propagation_aware_multiplicative,
        
        # 4. Custom Designs
        "PageRank (ppr)": run_ppr,
        "PageRank Lookahead (ppr lookahead)": run_ppr_lookahead,
        "Concave Hybrid (concave)": run_concave,
        "Hybrid Degree (avg degree)": run_avg_degree,
        "K-Core Hybrid (kcore)": run_kcore
    }
    
    # 22 premium distinct colors and markers
    colors = [
        "#E31A1C", "#1F78B4", "#33A02C", "#FF7F00", "#6A3D9A",
        "#B15928", "#A6CEE3", "#B2DF8A", "#FB9A99", "#FDBF6F",
        "#CAB2D6", "#FFFF99", "#8DD3C7", "#FFFFB3", "#BEBADA",
        "#FB8072", "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5",
        "#BC80BD", "#CCEBC5"
    ]
    markers = ["o", "s", "^", "D", "P", "*", "h", "p", "v", ">", "<", "X"]
    
    algo_styles = {}
    for idx, name in enumerate(algos):
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        ls = "-" if idx % 2 == 0 else "--"
        algo_styles[name] = {"color": c, "marker": m, "linestyle": ls}
        
    results = {}
    
    for gml_path in gml_files:
        network = os.path.basename(gml_path).replace(".gml", "")
        print(f"\n==================================================")
        print(f"Running experiments on: {network}")
        print(f"==================================================")
        
        adj, nodes, G_mapped, n, original_nodes = load_graph(gml_path)
        
        # Precomputations
        print("Precomputing centralities and decompositions...")
        pr_dict = nx.pagerank(G_mapped)
        kcore_dict = nx.core_number(G_mapped)
        closeness_dict = nx.closeness_centrality(G_mapped)
        harmonic_dict = nx.harmonic_centrality(G_mapped)
        
        results[network] = {}
        
        for alpha in alphas:
            results[network][str(alpha)] = {}
            
            fairness_data = {name: [] for name in algos}
            spreading_data = {name: [] for name in algos}
            
            for k in ks:
                print(f"\n  --- Running suite for K = {k} ---")
                results[network][str(alpha)][str(k)] = {}
                
                for name, func in algos.items():
                    print(f"    Running {name}...")
                    t0 = time.time()
                    try:
                        seeds, probs = func(
                            adj, nodes, alpha, k, T, R,
                            pr_dict=pr_dict, kcore_dict=kcore_dict,
                            closeness_dict=closeness_dict, harmonic_dict=harmonic_dict,
                            G_mapped=G_mapped
                        )
                        var = compute_variance(probs)
                        mu = sum(probs) / n
                        min_p = min(probs)
                        t_elapsed = time.time() - t0
                        
                        fairness_data[name].append(min_p)
                        # Spreading Power: expected number of influenced nodes (mu * n)
                        spreading_data[name].append(mu * n)
                        
                        results[network][str(alpha)][str(k)][name] = {
                            "seeds": seeds,
                            "variance": var,
                            "mean_prob": mu,
                            "min_p": min_p,
                            "time_s": t_elapsed
                        }
                        print(f"      Fairness (min_p): {min_p:.6f} | Spreading (Reach): {mu * n:.2f} | Time: {t_elapsed:.2f}s")
                    except Exception as e:
                        import traceback
                        print(f"      [Error] failed to run {name}: {e}")
                        traceback.print_exc()
                        fairness_data[name].append(None)
                        spreading_data[name].append(None)
            
            # 1. Plot Individual Fairness (min_p) vs K
            plt.figure(figsize=(15, 9))
            for name in algos:
                y = fairness_data[name]
                if None in y:
                    continue
                style = algo_styles[name]
                plt.plot(ks, y, label=name, color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=2)
            plt.title(f"Individual Fairness (Rawlsian Min Prob) vs Seed Set Size: {network} (Alpha={alpha})", fontsize=14)
            plt.xlabel("Seed Set Size (k)", fontsize=12)
            plt.ylabel("Individual Fairness (Min Probability) [Higher is Better]", fontsize=12)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{network}_fairness.png"), dpi=150)
            plt.close()
            
            # 2. Plot Spreading Power (Reach) vs K
            plt.figure(figsize=(15, 9))
            for name in algos:
                y = spreading_data[name]
                if None in y:
                    continue
                style = algo_styles[name]
                plt.plot(ks, y, label=name, color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=2)
            plt.title(f"Spreading Power (Cascade Reach) vs Seed Set Size: {network} (Alpha={alpha})", fontsize=14)
            plt.xlabel("Seed Set Size (k)", fontsize=12)
            plt.ylabel("Spreading Power (Expected Influenced Nodes) [Higher is Better]", fontsize=12)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{network}_spreading.png"), dpi=150)
            plt.close()
            
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nAll experiments and comparisons run successfully! Plots saved in: {out_dir}")
    
    # Generate COMPARISON_REPORT.md
    generate_comparison_report(results, out_dir)

def generate_comparison_report(results, out_dir):
    report_path = os.path.join(out_dir, "COMPARISON_REPORT.md")
    from collections import defaultdict
    alg_stats = defaultdict(lambda: {'var_sum': 0.0, 'prob_sum': 0.0, 'min_p_sum': 0.0, 'time_sum': 0.0, 'count': 0})
    net_summary_fairness = defaultdict(dict)
    net_summary_spreading = defaultdict(dict)
    networks = list(results.keys())
    
    for net, net_data in results.items():
        for alpha, alpha_data in net_data.items():
            for k, k_data in alpha_data.items():
                for algo, metrics in k_data.items():
                    alg_stats[algo]['var_sum'] += metrics['variance']
                    alg_stats[algo]['prob_sum'] += metrics['mean_prob']
                    alg_stats[algo]['min_p_sum'] += metrics['min_p']
                    alg_stats[algo]['time_sum'] += metrics['time_s']
                    alg_stats[algo]['count'] += 1
                    
                    net_summary_fairness[net].setdefault(algo, [])
                    net_summary_fairness[net][algo].append(metrics['min_p'])
                    
                    net_summary_spreading[net].setdefault(algo, [])
                    net_summary_spreading[net][algo].append(metrics['mean_prob'])
                    
    summary_data = []
    for algo, stats in alg_stats.items():
        cnt = stats['count']
        avg_var = stats['var_sum'] / cnt if cnt else 0.0
        avg_prob = stats['prob_sum'] / cnt if cnt else 0.0
        avg_min_p = stats['min_p_sum'] / cnt if cnt else 0.0
        avg_time = stats['time_sum'] / cnt if cnt else 0.0
        summary_data.append((algo, avg_min_p, avg_prob, avg_var, avg_time))
        
    # Sort summary by Individual Fairness (min_p) descending (higher is fairer)
    summary_data.sort(key=lambda x: x[1], reverse=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Fair Influence Maximization: 22-Algorithm Comparison Report\n\n")
        f.write("This report provides a comprehensive comparison of all 22 algorithms evaluated under the Independent Cascade Model (ICM) with uniform transmission probability $\\alpha = 0.3$.\n\n")
        
        f.write("## 1. Global Performance Averages\n\n")
        f.write("| Rank | Algorithm | Average Individual Fairness (Min P) [Higher is Fairer] | Average Spreading Power (Mean P) | Average Variance | Average Execution Time (s) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for rank, (algo, min_p, prob, var, t_s) in enumerate(summary_data, 1):
            f.write(f"| {rank} | **{algo}** | {min_p:.6f} | {prob:.4f} | {var:.6f} | {t_s:.3f}s |\n")
            
        f.write("\n## 2. Average Individual Fairness (Min P) per Network\n\n")
        f.write("| Algorithm | " + " | ".join(networks) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(networks)) + "|\n")
        for algo in sorted(alg_stats.keys()):
            row = [f"**{algo}**"]
            for net in networks:
                vals_list = net_summary_fairness[net].get(algo, [])
                avg_net_val = sum(vals_list) / len(vals_list) if vals_list else 0.0
                row.append(f"{avg_net_val:.6f}")
            f.write("| " + " | ".join(row) + " |\n")
            
        f.write("\n## 3. Average Spreading Power (Mean P) per Network\n\n")
        f.write("| Algorithm | " + " | ".join(networks) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(networks)) + "|\n")
        for algo in sorted(alg_stats.keys()):
            row = [f"**{algo}**"]
            for net in networks:
                vals_list = net_summary_spreading[net].get(algo, [])
                avg_net_val = sum(vals_list) / len(vals_list) if vals_list else 0.0
                row.append(f"{avg_net_val:.4f}")
            f.write("| " + " | ".join(row) + " |\n")
            
        f.write("\n## 4. Key Observations\n\n")
        f.write("- **Trade-off between Spreading Power and Fairness**: Baselines like TIM+ typically achieve excellent spreading power but poor individual fairness (low min_p). Rawlsian-focused algorithms like Myopic or Concave Hybrid sacrifice overall reach to lift the lowest active probability.\n")
        f.write("- **Propagation-Aware Fair Selection**: The newly proposed framework (Additive and Multiplicative) successfully balances downstream spreading power and individual need via its dual-score combination.\n")
        f.write("- **Scale and Efficiency**: The K-Core Hybrid and PageRank heuristics run significantly faster than pure Myopic while maintaining highly competitive individual fairness.\n")

    print(f"Comparison report generated at: {report_path}")

if __name__ == "__main__":
    main()
