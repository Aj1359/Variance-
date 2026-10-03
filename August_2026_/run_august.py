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
from icm import prob_est_timed

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
    cc_pcts = [0.05, 0.10, 0.15]
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
        for u, v in G.edges():
            u, v = int(u), int(v)
            mu, mv = id_map[u], id_map[v]
            adj[mu].append(mv)
            adj[mv].append(mu)
        
        for alpha in alphas:
            print(f"  Alpha: {alpha}")
            # Cache baselines so we only run them once per k
            baseline_cache = {}
            
            for pct in cc_pcts:
                pct_label = f"{int(pct*100)}_percent"
                print(f"    Closeness: {pct_label}")
                
                pct_dir = os.path.join("result_august", network, str(alpha), pct_label)
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
                        # 1. Myopic Baseline
                        print("        Running Myopic...")
                        m_seeds, m_probs, m_hits = myopic(adj, nodes, alpha, k, T, R)
                        m_var = compute_variance(m_probs)
                        
                        # 2. Naive Myopic
                        print("        Running Naive Myopic...")
                        nm_seeds, nm_probs, nm_hits = naive_myopic(adj, nodes, alpha, k, T, R)
                        nm_var = compute_variance(nm_probs)
                        
                        # 3. Gonzalez
                        print("        Running Gonzalez...")
                        g_seeds, g_probs, g_hits = gonzalez(adj, nodes, alpha, k, T, R)
                        g_var = compute_variance(g_probs)
                        
                        baseline_cache[k] = {
                            "myopic": {"seeds": m_seeds, "hits": m_hits, "var": m_var, "mu": sum(m_probs) / n},
                            "naive_myopic": {"seeds": nm_seeds, "hits": nm_hits, "var": nm_var, "mu": sum(nm_probs) / n},
                            "gonzalez": {"seeds": g_seeds, "hits": g_hits, "var": g_var, "mu": sum(g_probs) / n}
                        }
                        
                    cache_k = baseline_cache[k]
                    myo_vars.append(cache_k["myopic"]["var"])
                    nm_vars.append(cache_k["naive_myopic"]["var"])
                    gonz_vars.append(cache_k["gonzalez"]["var"])
                    
                    results_json["myopic"][str(k)] = {"seeds": cache_k["myopic"]["seeds"], "var": cache_k["myopic"]["var"]}
                    results_json["naive_myopic"][str(k)] = {"seeds": cache_k["naive_myopic"]["seeds"], "var": cache_k["naive_myopic"]["var"]}
                    results_json["gonzalez"][str(k)] = {"seeds": cache_k["gonzalez"]["seeds"], "var": cache_k["gonzalez"]["var"]}
                    
                    # 4. Myopic-Hybrid-CC
                    print("        Running Hybrid...")
                    h_seeds, h_probs, h_hits, h_log = myopic_hybrid_cc(adj, nodes, alpha, k, T, top_cc_pct=pct, R=R)
                    h_var = compute_variance(h_probs)
                    hyb_vars.append(h_var)
                    
                    results_json["hybrid"][str(k)] = {"seeds": h_seeds, "var": h_var}
                    
                    # 5. Save report.md for this K
                    report_path = os.path.join(k_dir, "report.md")
                    with open(report_path, "w") as rep:
                        rep.write(f"# Candidate Selection Report\n")
                        rep.write(f"**Network:** {network} | **Alpha:** {alpha} | **K:** {k} | **CC:** {pct_label}\n\n")
                        rep.write("| Iteration | Candidates in Epsilon Band | Candidates Filtered by CC | Seeds Added |\n")
                        rep.write("|---|---|---|---|\n")
                        for entry in h_log:
                            it = entry["iteration"]
                            c_eps = entry["candidates_in_epsilon_band"]
                            c_flt = entry["candidates_after_cc_filter"]
                            s_add = entry["seeds_chosen_this_batch"]
                            rep.write(f"| {it} | {c_eps} | {c_flt} | {s_add} |\n")
                            
                    # 6. Save distribution plot for this K
                    plot_distribution(
                        cache_k["myopic"]["hits"],
                        cache_k["naive_myopic"]["hits"],
                        cache_k["gonzalez"]["hits"],
                        h_hits, network, alpha, k, pct_label, os.path.join(k_dir, "distribution.png")
                    )
                    
                with open(os.path.join(pct_dir, "results.json"), "w") as f:
                    json.dump(results_json, f, indent=2)
                    
                plot_variance(myo_vars, hyb_vars, nm_vars, gonz_vars, ks, os.path.join(pct_dir, "variance_vs_seedset.png"), pct_label)
                
    print("\nAll experiments completed and highly-nested folders generated successfully!")

if __name__ == "__main__":
    main()
