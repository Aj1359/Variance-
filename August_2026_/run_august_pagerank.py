import os
import sys
import glob
import json
import networkx as nx
import matplotlib.pyplot as plt

from myopic import myopic
from naive_myopic import naive_myopic
from gonzalez import gonzalez
from new_heu_pr import new_heu_pr
from new_heu_pr_v2 import new_heu_pr_v2
from icm import prob_est_timed

def compute_variance(probs):
    """Population variance Var(P) = (1/N) * sum((p_i - mu)^2), matching attempt2/metrics.py."""
    n = len(probs)
    if n < 2:
        return 0.0
    mu = sum(probs) / n
    return sum((p - mu) ** 2 for p in probs) / n

def plot_variance(myo_vars, pr1_vars, pr2_vars, nm_vars, gonz_vars, ks, out_path):
    plt.figure(figsize=(9, 6))
    plt.plot(ks, myo_vars, marker='o', label='Myopic')
    plt.plot(ks, nm_vars, marker='s', linestyle='--', label='Naive Myopic')
    plt.plot(ks, gonz_vars, marker='^', linestyle='-.', label='Gonzalez')
    plt.plot(ks, pr1_vars, marker='d', linestyle=':', label='PageRank v1 (Distance+PR)')
    plt.plot(ks, pr2_vars, marker='*', linewidth=2.5, label='PageRank v2 (PPR+Degree)')
    plt.title("Variance vs Seed Set Size")
    plt.xlabel("Seed Set Size (k)")
    plt.ylabel("Variance Var(P) (Lowest is Best)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_distribution(myo_hits, nm_hits, gonz_hits, pr1_hits, pr2_hits, network, alpha, k, out_path):
    plt.figure(figsize=(10, 6))
    bins = range(0, 205, 5)
    plt.hist(myo_hits, bins=bins, alpha=0.3, label='Myopic', histtype='step', linewidth=1.5)
    plt.hist(nm_hits, bins=bins, alpha=0.3, label='Naive Myopic', histtype='step', linewidth=1.5)
    plt.hist(gonz_hits, bins=bins, alpha=0.3, label='Gonzalez', histtype='step', linewidth=1.5)
    plt.hist(pr1_hits, bins=bins, alpha=0.6, label='PageRank v1', histtype='step', linewidth=2.0)
    plt.hist(pr2_hits, bins=bins, alpha=0.8, label='PageRank v2', histtype='step', linewidth=2.5)
    
    plt.title(f"Influence Distribution: {network} | Alpha={alpha} | K={k}")
    plt.xlabel("Number of times influenced (out of 200 runs)")
    plt.ylabel("Frequency (Number of Nodes)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    gml_files = glob.glob(os.path.join("Social_Network", "*.gml"))
    if not gml_files:
        print("No .gml files found in Social_Network/")
        return
        
    alphas = [0.2, 0.3, 0.4]
    ks = [10, 20, 30, 40, 50]
    T = 15
    R = 200
    
    for gml_path in gml_files:
        network = os.path.basename(gml_path).replace(".gml", "")
        print(f"\n{'='*40}\nProcessing {network}\n{'='*40}")
        
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
            
            alpha_dir = os.path.join("result_august_pagerank", network, str(alpha))
            os.makedirs(alpha_dir, exist_ok=True)
            
            results_json = {
                "myopic": {},
                "naive_myopic": {},
                "gonzalez": {},
                "pagerank_v1": {},
                "pagerank_v2": {}
            }
            myo_vars = []
            nm_vars = []
            gonz_vars = []
            pr1_vars = []
            pr2_vars = []
            
            for k in ks:
                print(f"    K: {k}")
                k_dir = os.path.join(alpha_dir, f"k_{k}")
                os.makedirs(k_dir, exist_ok=True)
                
                # 1. Myopic Baseline
                print("      Running Myopic...")
                m_seeds, m_probs, m_hits = myopic(adj, nodes, alpha, k, T, R)
                m_var = compute_variance(m_probs)
                myo_vars.append(m_var)
                results_json["myopic"][str(k)] = {"seeds": m_seeds, "var": m_var, "mu": sum(m_probs) / n}
                
                # 2. Naive Myopic
                print("      Running Naive Myopic...")
                nm_seeds, nm_probs, nm_hits = naive_myopic(adj, nodes, alpha, k, T, R)
                nm_var = compute_variance(nm_probs)
                nm_vars.append(nm_var)
                results_json["naive_myopic"][str(k)] = {"seeds": nm_seeds, "var": nm_var, "mu": sum(nm_probs) / n}
                
                # 3. Gonzalez
                print("      Running Gonzalez...")
                g_seeds, g_probs, g_hits = gonzalez(adj, nodes, alpha, k, T, R)
                g_var = compute_variance(g_probs)
                gonz_vars.append(g_var)
                results_json["gonzalez"][str(k)] = {"seeds": g_seeds, "var": g_var, "mu": sum(g_probs) / n}
                
                # 4. PageRank v1 (New Heu PR)
                print("      Running PageRank v1 (Distance + PR)...")
                pr1_seeds, pr1_probs, pr1_hits, pr1_log = new_heu_pr(adj, nodes, alpha, k, T, R=R, pr_dict=pr_dict)
                pr1_var = compute_variance(pr1_probs)
                pr1_vars.append(pr1_var)
                results_json["pagerank_v1"][str(k)] = {"seeds": pr1_seeds, "var": pr1_var, "mu": sum(pr1_probs) / n}
                
                # 5. PageRank v2 (PPR + Degree)
                print("      Running PageRank v2 (PPR + Degree)...")
                pr2_seeds, pr2_probs, pr2_hits, pr2_log = new_heu_pr_v2(adj, nodes, alpha, k, T, R=R, pr_dict=pr_dict)
                pr2_var = compute_variance(pr2_probs)
                pr2_vars.append(pr2_var)
                results_json["pagerank_v2"][str(k)] = {"seeds": pr2_seeds, "var": pr2_var, "mu": sum(pr2_probs) / n}
                
                # 6. Save report.md for this K
                report_path = os.path.join(k_dir, "report.md")
                with open(report_path, "w") as rep:
                    rep.write(f"# PageRank Candidate Selection Report\n")
                    rep.write(f"**Network:** {network} | **Alpha:** {alpha} | **K:** {k}\n\n")
                    rep.write("## PageRank v1 (Distance + PR)\n")
                    rep.write("| Step | Candidates in Epsilon Band | Chosen Seed | PageRank Score | Distance to Seeds |\n")
                    rep.write("|---|---|---|---|---|\n")
                    for entry in pr1_log:
                        step = entry["step"]
                        c_eps = entry["candidates_in_epsilon_band"]
                        seed = entry["chosen_seed"]
                        score = f"{entry['seed_pr']:.6e}"
                        dist = entry["seed_dist"]
                        rep.write(f"| {step} | {c_eps} | {seed} | {score} | {dist} |\n")
                    
                    rep.write("\n## PageRank v2 (Hop Bucket + PPR + Degree)\n")
                    rep.write("| Step | Candidates in Epsilon Band | Chosen Seed | PPR Score | Distance to Seeds |\n")
                    rep.write("|---|---|---|---|---|\n")
                    for entry in pr2_log:
                        step = entry["step"]
                        c_eps = entry["candidates_in_epsilon_band"]
                        seed = entry["chosen_seed"]
                        score = f"{entry.get('seed_ppr', 0.0):.6e}"
                        dist = entry["seed_dist"]
                        rep.write(f"| {step} | {c_eps} | {seed} | {score} | {dist} |\n")
                        
                # 7. Save distribution plot for this K
                plot_distribution(
                    m_hits,
                    nm_hits,
                    g_hits,
                    pr1_hits,
                    pr2_hits,
                    network, alpha, k,
                    os.path.join(k_dir, "distribution.png")
                )
                
            # Save results.json and variance_vs_seedset.png under alpha level
            with open(os.path.join(alpha_dir, "results.json"), "w") as f:
                json.dump(results_json, f, indent=2)
                
            plot_variance(myo_vars, pr1_vars, pr2_vars, nm_vars, gonz_vars, ks, os.path.join(alpha_dir, "variance_vs_seedset.png"))
            
    print("\nAll PageRank experiments (v1 & v2) completed and folders generated successfully!")

if __name__ == "__main__":
    main()
