import os
import json
import matplotlib.pyplot as plt

def main():
    results_path = "result_comparison/results.json"
    if not os.path.exists(results_path):
        if os.path.exists("../result_comparison/results.json"):
            results_path = "../result_comparison/results.json"
        else:
            print(f"Error: results.json not found.")
            return
        
    with open(results_path, "r") as f:
        results = json.load(f)
        
    out_dir = "result_comparison"
    if not os.path.exists(out_dir) and os.path.exists("../result_comparison"):
        out_dir = "../result_comparison"
    
    # Categories of algorithms
    mine_algos = [
        "ComponentFirst", "DegreeGonzalez", "HarmonicSpread", "PPR-Balance", 
        "NeighborPPRBridge", "EgoDensityBalance", "BetweennessGateway", 
        "DegreeMedianSpread", "KCoreFrontier", "EccentricitySpread", 
        "PageRank Topo V1", "PageRank Topo V2"
    ]
    
    windham_algos = [
        "Random", "Myopic", "Naive Myopic", "Gonzalez", "Myopic BFS", 
        "Naive Myopic BFS", "Myopic PPR", "Naive Myopic PPR", "LeastCentral", 
        "LeastCentral_n", "MinDegree_hc", "MinDegree_hcn", "MinDegree_nd", 
        "MinDegree_ndn"
    ]
    
    prop_aware_algos = ["Prop-Aware Additive", "Prop-Aware Multiplicative"]
    baselines = ["Random", "Myopic", "Naive Myopic", "Gonzalez"]
    other_own = [
        "K-Core Hybrid", 
        "Concave Hybrid (concave)", 
        "PageRank (ppr)", 
        "PageRank Lookahead (ppr lookahead)"
    ]
    
    summary_report = []
    summary_report.append("# Selected Algorithms Performance & Comparison Report\n")
    summary_report.append("This report lists the top performing custom and baseline algorithms and compares them.\n")
    
    for network in results:
        summary_report.append(f"\n## Network: {network}\n")
        
        # Calculate average fairness (min_p) across all alphas and ks for each algorithm
        mine_fairness = {}
        windham_fairness = {}
        
        # Collect alphas and ks
        alphas = list(results[network].keys())
        ks = list(results[network][alphas[0]].keys())
        
        for name in mine_algos:
            vals = []
            for alpha in alphas:
                for k in ks:
                    if name in results[network][alpha][k]:
                        vals.append(results[network][alpha][k][name]["min_p"])
            if vals:
                mine_fairness[name] = sum(vals) / len(vals)
                
        for name in windham_algos:
            vals = []
            for alpha in alphas:
                for k in ks:
                    if name in results[network][alpha][k]:
                        vals.append(results[network][alpha][k][name]["min_p"])
            if vals:
                windham_fairness[name] = sum(vals) / len(vals)
                
        # Sort and select top 3
        top_mine = sorted(mine_fairness.items(), key=lambda x: -x[1])[:3]
        top_windham = sorted(windham_fairness.items(), key=lambda x: -x[1])[:3]
        
        top_mine_names = [x[0] for x in top_mine]
        top_windham_names = [x[0] for x in top_windham]
        
        summary_report.append("### Top 3 Custom Algorithms (mine_algorithm):\n")
        for idx, (name, val) in enumerate(top_mine):
            summary_report.append(f"{idx + 1}. **{name}** (Average Fairness: {val:.6f})\n")
            
        summary_report.append("\n### Top 3 Existing Algorithms (existing_14):\n")
        for idx, (name, val) in enumerate(top_windham):
            summary_report.append(f"{idx + 1}. **{name}** (Average Fairness: {val:.6f})\n")
            
        # Compile selected algorithms list to plot
        # Deduplicate names (e.g. Myopic might be in top_windham and baselines)
        selected_algos = list(set(
            top_mine_names + 
            top_windham_names + 
            prop_aware_algos + 
            baselines + 
            other_own
        ))
        
        print(f"\nNetwork {network}:")
        print(f"  Top 3 Mine: {top_mine_names}")
        print(f"  Top 3 Windham: {top_windham_names}")
        print(f"  Plotting {len(selected_algos)} selected algorithms.")
        
        # Plotting
        for alpha in alphas:
            plt.figure(figsize=(12, 7))
            
            for name in selected_algos:
                pts = []
                for k in ks:
                    if name in results[network][alpha][k]:
                        pts.append((int(k), results[network][alpha][k][name]["min_p"]))
                if pts:
                    pts = sorted(pts)
                    x_vals = [p[0] for p in pts]
                    y_vals = [p[1] for p in pts]
                    
                    # Highlight groups using distinct line types
                    if name in top_mine_names:
                        linestyle = "-"
                        marker = "o"
                        linewidth = 2.0
                    elif name in top_windham_names:
                        linestyle = ":"
                        marker = "x"
                        linewidth = 1.5
                    elif name in prop_aware_algos:
                        linestyle = "--"
                        marker = "s"
                        linewidth = 1.8
                    else:
                        linestyle = "-."
                        marker = "^"
                        linewidth = 1.2
                        
                    plt.plot(x_vals, y_vals, label=name, linestyle=linestyle, marker=marker, linewidth=linewidth)
                    
            # Create network/alpha directory structure
            alpha_dir = os.path.join(out_dir, network, f"alpha_{float(alpha):.2f}")
            os.makedirs(alpha_dir, exist_ok=True)
            
            plt.title(f"Fairness Comparison of Selected Top Algorithms on {network} (alpha={alpha})")
            plt.xlabel("Seed Set Size (k)")
            plt.ylabel("Individual Fairness (Min Probability of Access)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.tight_layout()
            plt.savefig(os.path.join(alpha_dir, "selected_fairness.png"), dpi=150)
            plt.close()
            
            # Spreading Reach Plot
            plt.figure(figsize=(12, 7))
            
            for name in selected_algos:
                pts = []
                for k in ks:
                    if name in results[network][alpha][k]:
                        pts.append((int(k), results[network][alpha][k][name]["mean_prob"]))
                if pts:
                    pts = sorted(pts)
                    x_vals = [p[0] for p in pts]
                    y_vals = [p[1] for p in pts]
                    
                    if name in top_mine_names:
                        linestyle = "-"
                        marker = "o"
                        linewidth = 2.0
                    elif name in top_windham_names:
                        linestyle = ":"
                        marker = "x"
                        linewidth = 1.5
                    elif name in prop_aware_algos:
                        linestyle = "--"
                        marker = "s"
                        linewidth = 1.8
                    else:
                        linestyle = "-."
                        marker = "^"
                        linewidth = 1.2
                        
                    plt.plot(x_vals, y_vals, label=name, linestyle=linestyle, marker=marker, linewidth=linewidth)
                    
            plt.title(f"Spreading Power Comparison of Selected Top Algorithms on {network} (alpha={alpha})")
            plt.xlabel("Seed Set Size (k)")
            plt.ylabel("Spreading Reach (Fraction of Nodes)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.tight_layout()
            plt.savefig(os.path.join(alpha_dir, "selected_spreading.png"), dpi=150)
            plt.close()
            
            # Variance Plot
            plt.figure(figsize=(12, 7))
            
            for name in selected_algos:
                pts = []
                for k in ks:
                    if name in results[network][alpha][k]:
                        pts.append((int(k), results[network][alpha][k][name]["variance"]))
                if pts:
                    pts = sorted(pts)
                    x_vals = [p[0] for p in pts]
                    y_vals = [p[1] for p in pts]
                    
                    if name in top_mine_names:
                        linestyle = "-"
                        marker = "o"
                        linewidth = 2.0
                    elif name in top_windham_names:
                        linestyle = ":"
                        marker = "x"
                        linewidth = 1.5
                    elif name in prop_aware_algos:
                        linestyle = "--"
                        marker = "s"
                        linewidth = 1.8
                    else:
                        linestyle = "-."
                        marker = "^"
                        linewidth = 1.2
                        
                    plt.plot(x_vals, y_vals, label=name, linestyle=linestyle, marker=marker, linewidth=linewidth)
                    
            plt.title(f"Variance Comparison of Selected Top Algorithms on {network} (alpha={alpha})")
            plt.xlabel("Seed Set Size (k)")
            plt.ylabel("Variance of Access Probabilities")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.tight_layout()
            plt.savefig(os.path.join(alpha_dir, "selected_variance.png"), dpi=150)
            plt.close()
            
    report_path = os.path.join(out_dir, "SELECTED_COMPARISON_REPORT.md")
    with open(report_path, "w") as f:
        f.writelines(summary_report)
    print(f"\nGenerated report at {report_path}")

if __name__ == "__main__":
    main()
