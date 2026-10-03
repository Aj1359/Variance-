import os
import json
import glob
from collections import defaultdict

def generate_report(target_algo, baseline_algos, result_key, output_md, title):
    json_files = glob.glob("result_ultimate/*/*/results.json")
    
    # Store wins as: data[network][alpha] = {"win": 0, "tie": 0, "loss": 0}
    stats = defaultdict(lambda: defaultdict(lambda: {"win": 0, "tie": 0, "loss": 0}))
    
    for fpath in json_files:
        parts = fpath.replace("\\", "/").split("/")
        if len(parts) < 4: continue
        network = parts[-3]
        alpha = parts[-2]
        
        with open(fpath, "r") as f:
            data = json.load(f)
            
        if result_key not in data: continue
        dataset = data[result_key]
        
        if target_algo not in dataset: continue
        
        k_list = dataset[target_algo].keys()
        for k in k_list:
            target_var = dataset[target_algo][k]["final"]["var"]
            
            # Find the best variance among all baselines
            best_baseline_var = float('inf')
            for base in baseline_algos:
                if base in dataset and k in dataset[base]:
                    b_var = dataset[base][k]["final"]["var"]
                    if b_var < best_baseline_var:
                        best_baseline_var = b_var
                        
            if best_baseline_var == float('inf'):
                continue
                
            # Compare: lower variance is better!
            if target_var < best_baseline_var:
                stats[network][alpha]["win"] += 1
            elif target_var == best_baseline_var:
                stats[network][alpha]["tie"] += 1
            else:
                stats[network][alpha]["loss"] += 1

    # Write to Markdown
    with open(output_md, "w") as f:
        f.write(f"# {title}\n\n")
        f.write("This report tracks how many times the target algorithm achieved a strictly **lower variance** than ALL baselines.\n\n")
        f.write(f"**Target Algorithm:** {target_algo}\n")
        f.write(f"**Baselines:** {', '.join(baseline_algos)}\n\n")
        
        f.write("| Network | Alpha | Wins | Ties | Losses |\n")
        f.write("|---------|-------|------|------|--------|\n")
        
        total_wins = 0
        total_ties = 0
        total_losses = 0
        
        for network in sorted(stats.keys()):
            for alpha in sorted(stats[network].keys(), key=float):
                w = stats[network][alpha]["win"]
                t = stats[network][alpha]["tie"]
                l = stats[network][alpha]["loss"]
                total_wins += w
                total_ties += t
                total_losses += l
                f.write(f"| {network} | {alpha} | {w} | {t} | {l} |\n")
                
        f.write(f"| **TOTAL** | | **{total_wins}** | **{total_ties}** | **{total_losses}** |\n")

def main():
    baselines = ["Myopic", "NaiveMyopic", "Gonzales"]
    
    generate_report(
        target_algo="MYOPIC_HYBRID",
        baseline_algos=baselines,
        result_key="hybrid",
        output_md="result_ultimate/analysis_hybrid.md",
        title="Leaderboard: Myopic-Hybrid vs Baselines"
    )
    
    generate_report(
        target_algo="NEW_HEU_PR",
        baseline_algos=baselines,
        result_key="pagerank",
        output_md="result_ultimate/analysis_pagerank.md",
        title="Leaderboard: PageRank (NEW_HEU_PR) vs Baselines"
    )
    print("Reports generated: result_ultimate/analysis_hybrid.md and result_ultimate/analysis_pagerank.md")

if __name__ == "__main__":
    main()
