import json
import sys
import numpy as np

def analyze_master(json_path, target_algo="MYOPIC_HYBRID"):
    print(f"Loading {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    results = data.get("results", {})
    params = data.get("params", {})
    epsilon = params.get("epsilon", "N/A")
    
    # Structure: results[graph][alpha][run][k][algo]['final']['var']
    # We want to aggregate across runs.
    
    # We will store: aggregated[graph][alpha][k][algo] = list of variances
    aggregated = {}
    
    for graph, alpha_dict in results.items():
        if graph not in aggregated:
            aggregated[graph] = {}
        for alpha, run_dict in alpha_dict.items():
            if alpha not in aggregated[graph]:
                aggregated[graph][alpha] = {}
            for run, k_dict in run_dict.items():
                for k, algo_dict in k_dict.items():
                    if k not in aggregated[graph][alpha]:
                        aggregated[graph][alpha][k] = {}
                    for algo, res in algo_dict.items():
                        if res is None or 'final' not in res:
                            continue
                        if algo not in aggregated[graph][alpha][k]:
                            aggregated[graph][alpha][k][algo] = []
                        var = res['final']['var']
                        aggregated[graph][alpha][k][algo].append(var)
                        
    # Now print the top 5 for each combination
    target_wins = 0
    total_combinations = 0
    
    print(f"\n{'='*80}")
    print(f"VARIANCE RANKINGS (Epsilon: {epsilon})")
    print(f"{'='*80}")
    
    for graph in sorted(aggregated.keys()):
        for alpha in sorted(aggregated[graph].keys()):
            for k in sorted(aggregated[graph][alpha].keys()):
                total_combinations += 1
                
                algo_vars = {}
                for algo, var_list in aggregated[graph][alpha][k].items():
                    algo_vars[algo] = np.mean(var_list)
                    
                # Sort by variance ascending (lower is better)
                sorted_algos = sorted(algo_vars.items(), key=lambda x: x[1])
                
                if not sorted_algos:
                    continue
                    
                winner = sorted_algos[0][0]
                is_target_win = (winner == target_algo)
                if is_target_win:
                    target_wins += 1
                    
                win_marker = "🏆 WIN!" if is_target_win else ""
                
                print(f"\nNetwork: {graph} | Alpha: {alpha} | k: {k} {win_marker}")
                print("-" * 50)
                
                # Print top 5
                for i, (algo, mean_var) in enumerate(sorted_algos[:5]):
                    marker = "*" if algo == target_algo else " "
                    print(f"{i+1}. {marker} {algo:15s} : {mean_var:.6f}")

    print(f"\n{'='*80}")
    print(f"Summary: {target_algo} won {target_wins} out of {total_combinations} combinations.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="e:/research/results_social/results_master.json")
    parser.add_argument("--target", type=str, default="NEW_HEU")
    args = parser.parse_args()
    
    analyze_master(args.file, args.target)
