import os
import glob
import json
import matplotlib.pyplot as plt

def generate_outreach_plots(suffix, ks):
    """
    Scans folders matching result_*<suffix>, groups data by (network, alpha),
    and plots expected influence outreach (mu) vs seed set size (k) for all algorithms.
    """
    print(f"\nScanning results for suffix: '{suffix}'...")
    
    # 1. Map result folders to algorithm names and target JSON keys
    algo_configs = [
        {"folder": f"result_hybrid_degree{suffix}", "key": "hybrid_degree", "label": "Myopic-Hybrid (Avg Degree)"},
        {"folder": f"result_pagerank_v1{suffix}", "key": "pagerank_v1", "label": "PageRank v1"},
        {"folder": f"result_pagerank_hybrid{suffix}", "key": "pagerank_hybrid", "label": "PageRank-Hybrid"},
        {"folder": f"result_entropy{suffix}", "key": "entropy_hybrid", "label": "Entropy-Hybrid"},
        {"folder": f"result_concave{suffix}", "key": "concave_hybrid", "label": "Concave-Hybrid"}
    ]
    
    # Filter only folders that actually exist
    active_configs = []
    for cfg in algo_configs:
        if os.path.exists(cfg["folder"]):
            active_configs.append(cfg)
            
    if not active_configs:
        print(f"No results folders found matching suffix '{suffix}'.")
        return
        
    # 2. Gather all (network, alpha) combinations
    # We scan results.json from the first active folder to discover combinations
    combinations = []
    first_folder = active_configs[0]["folder"]
    json_files = glob.glob(os.path.join(first_folder, "**", "results.json"), recursive=True)
    
    for jf in json_files:
        norm_path = jf.replace("\\", "/")
        parts = norm_path.split("/")
        if len(parts) >= 4:
            network = parts[-3]
            alpha = parts[-2]
            combinations.append((network, alpha))
            
    combinations = sorted(list(set(combinations)))
    if not combinations:
        print(f"No results.json combinations found for suffix '{suffix}'.")
        return
        
    os.makedirs("reports", exist_ok=True)
    
    # 3. Plot each combination
    for network, alpha in combinations:
        plt.figure(figsize=(9, 6.5))
        
        # We will also plot the baseline curves. Since baselines (Myopic, Naive Myopic, Gonzalez)
        # are calculated in each run, we can extract them from the first folder where they are present.
        baselines_plotted = False
        
        for cfg in active_configs:
            path = os.path.join(cfg["folder"], network, str(alpha), "results.json")
            if not os.path.exists(path):
                continue
                
            with open(path, "r") as f:
                try:
                    res = json.load(f)
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    continue
            
            # Plot baselines once
            if not baselines_plotted:
                for base_key, base_label, marker, style in [
                    ("myopic", "Myopic", "o", "-"),
                    ("naive_myopic", "Naive Myopic", "s", "--"),
                    ("gonzalez", "Gonzalez", "^", "-.")
                ]:
                    if base_key in res:
                        y_vals = []
                        valid_ks = []
                        for k in ks:
                            k_str = str(k)
                            if k_str in res[base_key]:
                                # expected influence = N * mu, but plotting mu directly is standard
                                y_vals.append(res[base_key][k_str]["mu"])
                                valid_ks.append(k)
                        if y_vals:
                            plt.plot(valid_ks, y_vals, marker=marker, linestyle=style, alpha=0.6, label=base_label)
                baselines_plotted = True
            
            # Plot target hybrid algorithm
            t_key = cfg["key"]
            if t_key in res:
                y_vals = []
                valid_ks = []
                for k in ks:
                    k_str = str(k)
                    if k_str in res[t_key]:
                        y_vals.append(res[t_key][k_str]["mu"])
                        valid_ks.append(k)
                if y_vals:
                    plt.plot(valid_ks, y_vals, marker="D", linewidth=2.5, label=cfg["label"])
                    
        prefix = "legacy_" if suffix == "" else ""
        out_name = f"{prefix}influence_outreach_{network}_alpha_{alpha}.png"
        out_path = os.path.join("reports", out_name)
        
        plt.title(f"Expected Influence Outreach (mu) vs Seed Set Size\nNetwork: {network} | Alpha: {alpha}", fontsize=11, fontweight="bold")
        plt.xlabel("Seed Set Size (k)", fontsize=10)
        plt.ylabel("Mean Activation Probability (mu)", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"  Generated plot: {out_path}")

def main():
    print("=== Generating Expected Influence Outreach Comparison Plots ===")
    
    # 1. Generate plots for the active 50-run multi experiments
    generate_outreach_plots(suffix="_multi", ks=[5, 10, 15, 20, 25, 30])
    
    # 2. Generate plots for the legacy single-run experiments
    generate_outreach_plots(suffix="", ks=[10, 20, 30, 40, 50, 60])
    
    print("\nAll expected influence outreach comparison plots generated successfully in reports/ folder!")

if __name__ == "__main__":
    main()
