import os
import json
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Generate variance plots from experiment results JSON.")
    parser.add_argument("--input", type=str, default="result_comparison/results.json", help="Path to results JSON file")
    parser.add_argument("--output-dir", type=str, default="result_comparison", help="Directory to save plots")
    parser.add_argument("--alpha", type=float, default=None, help="Specific alpha to plot (default: plot all alphas)")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        # Fallback to local files if run from different context
        if os.path.exists("result_comparison/results_finalv1.json"):
            args.input = "result_comparison/results_finalv1.json"
        else:
            print(f"Error: Input JSON file not found at {args.input}")
            return
            
    print(f"Loading results from {args.input}...")
    with open(args.input, "r") as f:
        results = json.load(f)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    for network in results:
        print(f"Processing network: {network}")
        
        # Get list of alphas in the results
        alphas = sorted([float(a) for a in results[network].keys()])
        
        # Filter alpha if specified
        if args.alpha is not None:
            if args.alpha in alphas:
                alphas = [args.alpha]
            else:
                # Try string matching / rounding
                matched = [a for a in alphas if abs(a - args.alpha) < 1e-4]
                if matched:
                    alphas = matched
                else:
                    print(f"Warning: Alpha {args.alpha} not found in results for {network}. Available: {alphas}")
                    continue
                    
        for alpha in alphas:
            alpha_str = f"{alpha:.2f}"
            if alpha_str not in results[network]:
                # Fallback to float key format
                alpha_str = str(alpha)
                if alpha_str not in results[network]:
                    continue
                    
            ks = sorted([int(k) for k in results[network][alpha_str].keys()])
            
            # Find all algorithms in this subset
            algorithms = set()
            for k in results[network][alpha_str]:
                for algo in results[network][alpha_str][k]:
                    algorithms.add(algo)
                    
            plt.figure(figsize=(12, 7))
            
            for algo in sorted(algorithms):
                x_vals = []
                y_vals = []
                for k in ks:
                    k_str = str(k)
                    if k_str in results[network][alpha_str] and algo in results[network][alpha_str][k_str]:
                        # Check if variance is present in the results
                        entry = results[network][alpha_str][k_str][algo]
                        if "variance" in entry:
                            x_vals.append(k)
                            y_vals.append(entry["variance"])
                            
                if x_vals:
                    plt.plot(x_vals, y_vals, label=algo, marker="o", linewidth=1.5, markersize=4)
                    
            plt.title(f"Variance Comparison - {network} (alpha={alpha})")
            plt.xlabel("Seed Set Size (k)")
            plt.ylabel("Variance of Access Probabilities")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
            plt.tight_layout()
            
            # Create network/alpha directory structure
            alpha_dir = os.path.join(args.output_dir, network, f"alpha_{alpha:.2f}")
            os.makedirs(alpha_dir, exist_ok=True)
            out_path = os.path.join(alpha_dir, "variance.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"  Saved variance plot to {out_path}")

if __name__ == "__main__":
    main()
