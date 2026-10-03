import os
import sys
import glob
import json
import time
import random
import networkx as nx
import matplotlib.pyplot as plt

# Import baseline IC estimation
from icm import prob_est_timed, simulate_timeseries

# ---------------------------------------------------------------------------
# RIS TIM+ Implementation
# ---------------------------------------------------------------------------
def generate_rr_set(adj, n, alpha):
    v = random.randint(0, n - 1)
    rr_set = {v}
    queue = [v]
    visited = {v}
    while queue:
        curr = queue.pop(0)
        curr_d = visited_d = 0 # simple BFS
        for nbr in adj[curr]:
            if nbr not in visited:
                if random.random() < alpha:
                    visited.add(nbr)
                    rr_set.add(nbr)
                    queue.append(nbr)
    return rr_set

def tim_plus(adj, n, k, alpha, theta=500):
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
# Imports of algorithms from the respective folders
# ---------------------------------------------------------------------------
# 1. Folder: existing_14
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

# 2. Folder: mine_algorithm
from mine_algorithm.component_first import component_first
from mine_algorithm.degree_gonzalez import degree_gonzalez
from mine_algorithm.harmonic_spread import harmonic_spread
from mine_algorithm.ppr_balance import ppr_balance
from mine_algorithm.neighbor_ppr_bridge import neighbor_ppr_bridge
from mine_algorithm.ego_density_balance import ego_density_balance
from mine_algorithm.betweenness_gateway import betweenness_gateway
from mine_algorithm.degree_median_spread import degree_median_spread
from mine_algorithm.kcore_frontier import kcore_frontier
from mine_algorithm.eccentricity_spread import eccentricity_spread
from mine_algorithm.new_heu_pr_topo import new_heu_pr_topo
from mine_algorithm.new_heu_pr_v2_topo import new_heu_pr_v2_topo

# 3. Folder: new_two_algos
from new_two_algos.prop_aware_additive import prop_aware_additive
from new_two_algos.prop_aware_multiplicative import prop_aware_multiplicative

# 4. Our own custom algorithms
from new_heu_pr import new_heu_pr
from new_heu_pr_v2 import new_heu_pr_v2
from concave_hybrid import concave_hybrid
from myopic_hybrid_degree import myopic_hybrid_degree
from myopic_hybrid_kcore import myopic_hybrid_kcore

# ---------------------------------------------------------------------------
# Execution wrappers
# ---------------------------------------------------------------------------
# Group 1: existing_14 wrappers
def run_existing_random(adj, nodes, alpha, k, T, R, **kwargs):
    return random_select(adj, nodes, alpha, k, T, R)

def run_existing_myopic(adj, nodes, alpha, k, T, R, **kwargs):
    seeds, probs, _ = myopic(adj, nodes, alpha, k, T, R)
    return seeds, probs

def run_existing_naive_myopic(adj, nodes, alpha, k, T, R, **kwargs):
    seeds, probs, _ = naive_myopic(adj, nodes, alpha, k, T, R)
    return seeds, probs

def run_existing_gonzalez(adj, nodes, alpha, k, T, R, **kwargs):
    seeds, probs, _ = gonzalez(adj, nodes, alpha, k, T, R)
    return seeds, probs

def run_existing_myopic_bfs(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = myopic_bfs(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_existing_naive_myopic_bfs(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = naive_myopic_bfs(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_existing_myopic_ppr(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = myopic_ppr(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_existing_naive_myopic_ppr(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = naive_myopic_ppr(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_existing_least_central(adj, nodes, alpha, k, T, R, G_mapped, closeness_dict=None, **kwargs):
    seeds = least_central(adj, nodes, alpha, k, G_mapped, closeness_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_existing_least_central_n(adj, nodes, alpha, k, T, R, G_mapped, closeness_dict=None, **kwargs):
    seeds = least_central_n(adj, nodes, alpha, k, G_mapped, closeness_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_existing_min_degree_hc(adj, nodes, alpha, k, T, R, G_mapped, harmonic_dict=None, **kwargs):
    seeds = min_degree_hc(adj, nodes, alpha, k, G_mapped, harmonic_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_existing_min_degree_hcn(adj, nodes, alpha, k, T, R, G_mapped, harmonic_dict=None, **kwargs):
    seeds = min_degree_hcn(adj, nodes, alpha, k, G_mapped, harmonic_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_existing_min_degree_nd(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = min_degree_nd(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_existing_min_degree_ndn(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = min_degree_ndn(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

# Group 2: mine_algorithm wrappers
def run_mine_component_first(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = component_first(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_mine_degree_gonzalez(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = degree_gonzalez(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_mine_harmonic_spread(adj, nodes, alpha, k, T, R, G_mapped, harmonic_dict=None, **kwargs):
    seeds = harmonic_spread(adj, nodes, alpha, k, G_mapped, harmonic_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_mine_ppr_balance(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = ppr_balance(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_mine_neighbor_ppr_bridge(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = neighbor_ppr_bridge(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_mine_ego_density_balance(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = ego_density_balance(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_mine_betweenness_gateway(adj, nodes, alpha, k, T, R, G_mapped, betweenness_dict=None, **kwargs):
    seeds = betweenness_gateway(adj, nodes, alpha, k, G_mapped, betweenness_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_mine_degree_median_spread(adj, nodes, alpha, k, T, R, G_mapped, **kwargs):
    seeds = degree_median_spread(adj, nodes, alpha, k, G_mapped)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_mine_kcore_frontier(adj, nodes, alpha, k, T, R, G_mapped, kcore_dict=None, **kwargs):
    seeds = kcore_frontier(adj, nodes, alpha, k, G_mapped, kcore_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_mine_eccentricity_spread(adj, nodes, alpha, k, T, R, G_mapped, ecc_dict=None, **kwargs):
    seeds = eccentricity_spread(adj, nodes, alpha, k, G_mapped, ecc_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_mine_new_heu_pr_topo(adj, nodes, alpha, k, T, R, G_mapped, pr_dict=None, **kwargs):
    seeds = new_heu_pr_topo(adj, nodes, alpha, k, G_mapped, pr_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_mine_new_heu_pr_v2_topo(adj, nodes, alpha, k, T, R, G_mapped, pr_dict=None, **kwargs):
    seeds = new_heu_pr_v2_topo(adj, nodes, alpha, k, G_mapped, pr_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

# Group 3: new_two_algos wrappers
def run_new_prop_aware_additive(adj, nodes, alpha, k, T, R, pr_dict=None, **kwargs):
    seeds = prop_aware_additive(adj, nodes, alpha, k, T, R, pr_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_new_prop_aware_multiplicative(adj, nodes, alpha, k, T, R, pr_dict=None, **kwargs):
    seeds = prop_aware_multiplicative(adj, nodes, alpha, k, T, R, pr_dict)
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

# Group 4: our own custom wrappers
def run_our_ppr(adj, nodes, alpha, k, T, R, pr_dict=None, **kwargs):
    seeds, probs, _, _ = new_heu_pr(adj, nodes, alpha, k, T, R, pr_dict=pr_dict)
    return seeds, probs

def run_our_ppr_lookahead(adj, nodes, alpha, k, T, R, pr_dict=None, **kwargs):
    seeds, probs, _, _ = new_heu_pr_v2(adj, nodes, alpha, k, T, R, pr_dict=pr_dict)
    return seeds, probs

def run_our_concave(adj, nodes, alpha, k, T, R, pr_dict=None, **kwargs):
    res = concave_hybrid(adj, nodes, alpha, k, T, R=R, R_probe=20, phi=-1.0, pr_dict=pr_dict)
    seeds = res[0]
    probs, _ = prob_est_timed(adj, seeds, alpha, len(nodes), T, R)
    return seeds, probs

def run_our_avg_degree(adj, nodes, alpha, k, T, R, **kwargs):
    seeds, probs, _, _ = myopic_hybrid_degree(adj, nodes, alpha, k, T, R)
    return seeds, probs

# Other baselines
def run_other_tim_plus(adj, nodes, alpha, k, T, R, **kwargs):
    n = len(nodes)
    seeds = tim_plus(adj, n, k, alpha)
    probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R)
    return seeds, probs

def run_other_kcore_hybrid(adj, nodes, alpha, k, T, R, kcore_dict=None, **kwargs):
    seeds, probs, _, _ = myopic_hybrid_kcore(adj, nodes, alpha, k, T, R=R, kcore_dict=kcore_dict)
    return seeds, probs

# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------
def main():
    gml_files = sorted(glob.glob(os.path.join("Social_Network", "*.gml")))
    if not gml_files:
        print("Error: No GML files found in Social_Network/")
        return
        
    out_dir = "result_comparison"
    os.makedirs(out_dir, exist_ok=True)
    
    results_path = os.path.join(out_dir, "results.json")
    results = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
            print(f"Loaded existing results.json from {results_path}.")
        except Exception as e:
            print(f"Error loading results.json: {e}. Starting fresh.")
            
    alphas = [0.3]
    ks = [10, 20, 30, 40, 50]
    T = 15
    R = 25
    
    # Organize all 34 algorithms to run/compare
    all_algos = {
        # Group 1: existing_14 baselines
        "Random": run_existing_random,
        "Myopic": run_existing_myopic,
        "Naive Myopic": run_existing_naive_myopic,
        "Gonzalez": run_existing_gonzalez,
        "Myopic BFS": run_existing_myopic_bfs,
        "Naive Myopic BFS": run_existing_naive_myopic_bfs,
        "Myopic PPR": run_existing_myopic_ppr,
        "Naive Myopic PPR": run_existing_naive_myopic_ppr,
        "LeastCentral": run_existing_least_central,
        "LeastCentral_n": run_existing_least_central_n,
        "MinDegree_hc": run_existing_min_degree_hc,
        "MinDegree_hcn": run_existing_min_degree_hcn,
        "MinDegree_nd": run_existing_min_degree_nd,
        "MinDegree_ndn": run_existing_min_degree_ndn,
        
        # Group 2: mine_algorithm (12)
        "ComponentFirst": run_mine_component_first,
        "DegreeGonzalez": run_mine_degree_gonzalez,
        "HarmonicSpread": run_mine_harmonic_spread,
        "PPR-Balance": run_mine_ppr_balance,
        "NeighborPPRBridge": run_mine_neighbor_ppr_bridge,
        "EgoDensityBalance": run_mine_ego_density_balance,
        "BetweennessGateway": run_mine_betweenness_gateway,
        "DegreeMedianSpread": run_mine_degree_median_spread,
        "KCoreFrontier": run_mine_kcore_frontier,
        "EccentricitySpread": run_mine_eccentricity_spread,
        "PageRank Topo V1": run_mine_new_heu_pr_topo,
        "PageRank Topo V2": run_mine_new_heu_pr_v2_topo,
        
        # Group 3: new_two_algos (2)
        "Prop-Aware Additive": run_new_prop_aware_additive,
        "Prop-Aware Multiplicative": run_new_prop_aware_multiplicative,
        
        # Group 4: our own (4)
        "PageRank (ppr)": run_our_ppr,
        "PageRank Lookahead (ppr lookahead)": run_our_ppr_lookahead,
        "Concave Hybrid (concave)": run_our_concave,
        "Hybrid Degree (avg degree)": run_our_avg_degree,
        
        # Group 5: other baselines (2)
        "TIM+": run_other_tim_plus,
        "K-Core Hybrid": run_other_kcore_hybrid
    }
    
    # Generate line styles for visual differentiation
    # To avoid visual clutter with 34 lines, we'll assign distinct line patterns and markers per group
    group_styles = {
        "existing_14": {"linestyle": ":", "marker": "x", "alpha": 0.5},
        "mine_algorithm": {"linestyle": "-", "marker": "o", "alpha": 0.9},
        "new_two_algos": {"linestyle": "--", "marker": "s", "alpha": 0.9},
        "our_own": {"linestyle": "-.", "marker": "^", "alpha": 0.8},
        "others": {"linestyle": "-", "marker": "D", "alpha": 0.7}
    }
    
    def get_group(name):
        if name in ["Random", "Myopic", "Naive Myopic", "Gonzalez", "Myopic BFS", "Naive Myopic BFS", 
                    "Myopic PPR", "Naive Myopic PPR", "LeastCentral", "LeastCentral_n", 
                    "MinDegree_hc", "MinDegree_hcn", "MinDegree_nd", "MinDegree_ndn"]:
            return "existing_14"
        elif name in ["ComponentFirst", "DegreeGonzalez", "HarmonicSpread", "PPR-Balance", "NeighborPPRBridge",
                      "EgoDensityBalance", "BetweennessGateway", "DegreeMedianSpread", "KCoreFrontier", 
                      "EccentricitySpread", "PageRank Topo V1", "PageRank Topo V2"]:
            return "mine_algorithm"
        elif name in ["Prop-Aware Additive", "Prop-Aware Multiplicative"]:
            return "new_two_algos"
        elif name in ["PageRank (ppr)", "PageRank Lookahead (ppr lookahead)", "Concave Hybrid (concave)", "Hybrid Degree (avg degree)"]:
            return "our_own"
        else:
            return "others"

    colors_palette = [
        "#E31A1C", "#1F78B4", "#33A02C", "#FF7F00", "#6A3D9A",
        "#B15928", "#A6CEE3", "#B2DF8A", "#FB9A99", "#FDBF6F",
        "#CAB2D6", "#FFFF99", "#8DD3C7", "#FFFFB3", "#BEBADA",
        "#FB8072", "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5",
        "#BC80BD", "#CCEBC5", "#4D4D4D", "#008080", "#808000",
        "#800080", "#000080", "#800000", "#FF00FF", "#00FFFF",
        "#FFC0CB", "#CD853F", "#4B0082", "#2E8B57"
    ]
    
    algo_styles = {}
    for idx, name in enumerate(all_algos.keys()):
        grp = get_group(name)
        styles = group_styles[grp]
        algo_styles[name] = {
            "color": colors_palette[idx % len(colors_palette)],
            "marker": styles["marker"],
            "linestyle": styles["linestyle"],
            "alpha": styles["alpha"]
        }
        
    for gml_path in gml_files:
        network = os.path.basename(gml_path).replace(".gml", "")
        print(f"\n==================================================")
        print(f"Running experiments on: {network}")
        print(f"==================================================")
        
        adj, nodes, G_mapped, n, original_nodes = load_graph(gml_path)
        
        # Precomputations for optimal execution
        print("Precomputing centralities and decompositions...")
        pr_dict = nx.pagerank(G_mapped)
        kcore_dict = nx.core_number(G_mapped)
        closeness_dict = nx.closeness_centrality(G_mapped)
        harmonic_dict = nx.harmonic_centrality(G_mapped)
        betweenness_dict = nx.betweenness_centrality(G_mapped, k=min(n, 200))
        ecc_dict = {}
        for C_nodes in nx.connected_components(G_mapped):
            G_sub = G_mapped.subgraph(C_nodes)
            comp_ecc = nx.eccentricity(G_sub)
            ecc_dict.update(comp_ecc)
            
        if network not in results:
            results[network] = {}
            
        for alpha in alphas:
            alpha_str = str(alpha)
            if alpha_str not in results[network]:
                results[network][alpha_str] = {}
                
            plot_fairness = {}
            plot_spreading = {}
            
            for k in ks:
                k_str = str(k)
                if k_str not in results[network][alpha_str]:
                    results[network][alpha_str][k_str] = {}
                    
                print(f"\n  --- Running suite for K = {k} ---")
                
                run_list = {}
                for name, func in all_algos.items():
                    if name not in results[network][alpha_str][k_str]:
                        run_list[name] = func
                        
                for name, func in run_list.items():
                    print(f"    Running {name}...")
                    t0 = time.time()
                    try:
                        seeds, probs = func(
                            adj, nodes, alpha, k, T, R,
                            pr_dict=pr_dict, kcore_dict=kcore_dict,
                            closeness_dict=closeness_dict, harmonic_dict=harmonic_dict,
                            betweenness_dict=betweenness_dict, ecc_dict=ecc_dict,
                            G_mapped=G_mapped
                        )
                        var = compute_variance(probs)
                        mu = sum(probs) / n
                        min_p = min(probs)
                        t_diff = time.time() - t0
                        
                        results[network][alpha_str][k_str][name] = {
                            "seeds": [original_nodes[s] for s in seeds],
                            "variance": var,
                            "mean_prob": mu,
                            "min_p": min_p,
                            "time_s": t_diff
                        }
                        print(f"      Fairness (min_p): {min_p:.6f} | Spreading (Reach): {mu * n:.2f} | Time: {t_diff:.2f}s")
                    except Exception as e:
                        print(f"      [Error] failed to run {name}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                # Read back all data for plotting
                for name in all_algos.keys():
                    if name in results[network][alpha_str][k_str]:
                        data = results[network][alpha_str][k_str][name]
                        if name not in plot_fairness:
                            plot_fairness[name] = []
                            plot_spreading[name] = []
                        plot_fairness[name].append((k, data["min_p"]))
                        plot_spreading[name].append((k, data["mean_prob"] * n))
                        
            # Save results progressively
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
                
            # ---------------------------------------------------------------
            # Generation of comparison plots
            # ---------------------------------------------------------------
            # Plot 1: Fairness
            plt.figure(figsize=(14, 8))
            for name, pts in plot_fairness.items():
                if not pts: continue
                pts = sorted(pts)
                x_vals = [p[0] for p in pts]
                y_vals = [p[1] for p in pts]
                style = algo_styles[name]
                plt.plot(x_vals, y_vals, label=name, color=style["color"],
                         marker=style["marker"], linestyle=style["linestyle"],
                         alpha=style["alpha"], linewidth=1.5)
            plt.title(f"Individual Fairness vs. Seed Set Size (k) on {network} (alpha={alpha})")
            plt.xlabel("Seed Set Size (k)")
            plt.ylabel("Minimum Probability of Access [Higher is Fairer]")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{network}_fairness.png"), dpi=150)
            plt.close()
            
            # Plot 2: Spreading
            plt.figure(figsize=(14, 8))
            for name, pts in plot_spreading.items():
                if not pts: continue
                pts = sorted(pts)
                x_vals = [p[0] for p in pts]
                y_vals = [p[1] for p in pts]
                style = algo_styles[name]
                plt.plot(x_vals, y_vals, label=name, color=style["color"],
                         marker=style["marker"], linestyle=style["linestyle"],
                         alpha=style["alpha"], linewidth=1.5)
            plt.title(f"Expected Cumulative Informed Nodes vs. Seed Set Size (k) on {network} (alpha={alpha})")
            plt.xlabel("Seed Set Size (k)")
            plt.ylabel("Spreading Reach (Nodes)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{network}_spreading.png"), dpi=150)
            plt.close()

    # ---------------------------------------------------------------------------
    # Generate final comparative markdown report
    # ---------------------------------------------------------------------------
    report_path = os.path.join(out_dir, "COMPARISON_REPORT.md")
    print(f"\nGenerating comprehensive comparison report at {report_path}...")
    
    # Calculate global averages
    stats = {}
    for network in results:
        for alpha_str in results[network]:
            for k_str in results[network][alpha_str]:
                for name, data in results[network][alpha_str][k_str].items():
                    if name not in stats:
                        stats[name] = {"fairness": [], "spreading": [], "variance": [], "time": []}
                    stats[name]["fairness"].append(data["min_p"])
                    # mean_prob represents fraction of nodes. Spread = mean_prob * n
                    stats[name]["spreading"].append(data["mean_prob"])
                    stats[name]["variance"].append(data["variance"])
                    stats[name]["time"].append(data["time_s"])
                    
    ranked = []
    for name, s in stats.items():
        avg_f = sum(s["fairness"]) / len(s["fairness"]) if s["fairness"] else 0.0
        avg_s = sum(s["spreading"]) / len(s["spreading"]) if s["spreading"] else 0.0
        avg_v = sum(s["variance"]) / len(s["variance"]) if s["variance"] else 0.0
        avg_t = sum(s["time"]) / len(s["time"]) if s["time"] else 0.0
        ranked.append((name, avg_f, avg_s, avg_v, avg_t))
        
    # Rank primarily by fairness (descending), then by spreading (descending)
    ranked = sorted(ranked, key=lambda x: (-x[1], -x[2]))
    
    with open(report_path, "w") as f:
        f.write("# Fair Influence Maximization: 34-Algorithm Comprehensive Benchmarking Report\n\n")
        f.write("This report provides a comparative study of 34 algorithms spanning existing baselines, topology-only variants, propagation-aware selections, and custom designs.\n\n")
        
        f.write("## 1. Global Performance Rankings\n\n")
        f.write("| Rank | Algorithm | Average Individual Fairness (Min P) [Higher is Fairer] | Average Spreading Power (Mean P) | Average Variance | Average Execution Time (s) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for idx, (name, f_val, s_val, v_val, t_val) in enumerate(ranked):
            f.write(f"| {idx + 1} | **{name}** | {f_val:.6f} | {s_val:.4f} | {v_val:.6f} | {t_val:.3f}s |\n")
            
        f.write("\n## 2. Average Individual Fairness (Min P) per Network\n\n")
        f.write("| Algorithm | " + " | ".join(results.keys()) + " |\n")
        f.write("|---| " + " | ".join("---" for _ in results.keys()) + " |\n")
        for name, _, _, _, _ in ranked:
            row = [f"**{name}**"]
            for network in results:
                # Find average min_p across all alpha and k for this network
                vals = []
                for alpha_str in results[network]:
                    for k_str in results[network][alpha_str]:
                        if name in results[network][alpha_str][k_str]:
                            vals.append(results[network][alpha_str][k_str][name]["min_p"])
                avg_net = sum(vals) / len(vals) if vals else 0.0
                row.append(f"{avg_net:.6f}")
            f.write("| " + " | ".join(row) + " |\n")
            
        f.write("\n## 3. Average Spreading Power (Mean P) per Network\n\n")
        f.write("| Algorithm | " + " | ".join(results.keys()) + " |\n")
        f.write("|---| " + " | ".join("---" for _ in results.keys()) + " |\n")
        for name, _, _, _, _ in ranked:
            row = [f"**{name}**"]
            for network in results:
                vals = []
                for alpha_str in results[network]:
                    for k_str in results[network][alpha_str]:
                        if name in results[network][alpha_str][k_str]:
                            vals.append(results[network][alpha_str][k_str][name]["mean_prob"])
                avg_net = sum(vals) / len(vals) if vals else 0.0
                row.append(f"{avg_net:.4f}")
            f.write("| " + " | ".join(row) + " |\n")
            
        f.write("\n## 4. Folder Structure Reference\n\n")
        f.write("- **`existing_14/`**: Contains the 14 baseline algorithms (Random, Myopic, Naive Myopic, Gonzalez + 10 heuristics from Windham's paper).\n")
        f.write("- **`mine_algorithm/`**: Contains our 12 custom topology-only algorithms (focusing on variance minimization and structural coverage).\n")
        f.write("- **`new_two_algos/`**: Contains the 2 propagation-aware algorithms (Additive & Multiplicative).\n")
        f.write("- **`our_own/`**: PPR, PPR Lookahead, Concave Hybrid, Average Degree, and K-Core Hybrid algorithms.\n")
        
    print("All tasks finished successfully!")

if __name__ == "__main__":
    main()
