"""
generate_plots_from_json.py
===========================
Reads results_master.json and generates plots for all networks and alphas.
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attempt2.plots import generate_all_plots

def generate_win_loss_report(parsed_results, out_path):
    print("Generating comprehensive leaderboard report...")
    
    # Track overall rankings: {algo: {1: 0, 2: 0, 3: 0}}
    rankings = {}
    
    scenarios = [] # To store details for the markdown table
    total_scenarios = 0

    for alpha in sorted(parsed_results.keys()):
        for net in sorted(parsed_results[alpha].keys()):
            for k in sorted(parsed_results[alpha][net].keys()):
                run_dict = parsed_results[alpha][net][k]
                
                # Only keep NEW_HEU_PR and the 3 baselines
                INCLUDE = {"NEW_HEU_PR", "Myopic", "NaiveMyopic", "Gonzales"}
                algo_vars = []
                for algo, res in run_dict.items():
                    if algo not in INCLUDE:
                        continue
                    if res:
                        var = res.get("final", {}).get("var")
                        if var is not None:
                            algo_vars.append((algo, var))
                            
                if not algo_vars: continue
                
                # Sort by variance ascending (lowest is best)
                algo_vars.sort(key=lambda x: x[1])
                
                total_scenarios += 1
                
                top_3 = []
                for i in range(min(3, len(algo_vars))):
                    algo, var = algo_vars[i]
                    rank = i + 1
                    if algo not in rankings:
                        rankings[algo] = {1: 0, 2: 0, 3: 0}
                    rankings[algo][rank] += 1
                    top_3.append(f"{algo} ({var:.4f})")
                
                # Fill missing if less than 3
                while len(top_3) < 3: top_3.append("-")
                
                scenarios.append(f"| {net} | {alpha} | {k} | {top_3[0]} | {top_3[1]} | {top_3[2]} |")

    lines = []
    lines.append("# NEW_HEU_PR vs Baselines — Absolute Performance Leaderboard (Lowest Variance)")
    lines.append(f"Algorithms compared: **NEW_HEU_PR**, Myopic, NaiveMyopic, Gonzales  ")
    lines.append(f"Total Scenarios: {total_scenarios}\n")
    
    # 1. Overall Tally
    lines.append("## Overall Top 3 Finishes")
    lines.append("| Algorithm | 1st Place | 2nd Place | 3rd Place | Total Top 3 |")
    lines.append("|---|---|---|---|---|")
    
    # Sort algos by number of 1st places, then 2nd, then 3rd
    sorted_algos = sorted(rankings.keys(), key=lambda a: (rankings[a][1], rankings[a][2], rankings[a][3]), reverse=True)
    
    for algo in sorted_algos:
        r1 = rankings[algo][1]
        r2 = rankings[algo][2]
        r3 = rankings[algo][3]
        total = r1 + r2 + r3
        lines.append(f"| **{algo}** | {r1} | {r2} | {r3} | {total} |")
        
    lines.append("\n## Detailed Scenario Breakdown")
    lines.append("| Network | Alpha | k (Seeds) | 1st (Lowest Var) | 2nd | 3rd |")
    lines.append("|---|---|---|---|---|---|")
    lines.extend(scenarios)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Saved leaderboard report to {out_path}")

def main():
    json_path = "result_new_pagerank/results_master.json"
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        sys.exit(1)
        
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    params = data.get("params", {})
    results = data.get("results", {})
    
    class Args: pass
    args = Args()
    args.alpha = params.get("alpha", [0.1, 0.2, 0.3])
    args.k = params.get("k", [20, 40, 60, 80, 100])
    args.T = params.get("T", 8)
    
    # results_master.json format: master[alpha][network][k][algo] = result
    # We need to parse keys as appropriate types: alpha as float, k as int
    parsed_results = {}
    networks = set()
    for alpha_str, net_dict in results.items():
        try:
            a_key = float(alpha_str)
        except ValueError:
            a_key = alpha_str
            
        parsed_results[a_key] = {}
        for network_str, k_dict in net_dict.items():
            networks.add(network_str)
            parsed_results[a_key][network_str] = {}
            for k_str, algo_dict in k_dict.items():
                parsed_results[a_key][network_str][int(k_str)] = algo_dict
                
    out_dir = "result_new_pagerank/plots_master"
    os.makedirs(out_dir, exist_ok=True)
    
    report_path = os.path.join(out_dir, "win_loss_report.md")
    generate_win_loss_report(parsed_results, report_path)
    
    # Generate plots per network
    for net in networks:
        print(f"\nGenerating plots for network: {net} ...")
        net_out_dir = os.path.join(out_dir, net)
        os.makedirs(net_out_dir, exist_ok=True)
        
        # Filter parsed_results for just this network
        net_parsed = {}
        for a_key in parsed_results:
            if net in parsed_results[a_key]:
                net_parsed[a_key] = {net: parsed_results[a_key][net]}
                
        try:
            generate_all_plots(net_parsed, args, net_out_dir)
            print(f"Plots for {net} saved to {net_out_dir}")
        except Exception as e:
            print(f"Error generating plots for {net}: {e}")

if __name__ == "__main__":
    main()
