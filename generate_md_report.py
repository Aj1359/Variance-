import json
import os
import numpy as np
import argparse
import sys

def parse_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    results = data.get("results", {})
    
    aggregated = {}
    all_algos = set()
    all_metrics = set()
    
    structure_type = None
    
    for alpha, net_dict in results.items():
        for network, level3_dict in net_dict.items():
            if network not in aggregated:
                aggregated[network] = {}
            if alpha not in aggregated[network]:
                aggregated[network][alpha] = {}
                
            if structure_type is None and len(level3_dict) > 0:
                first_key = list(level3_dict.keys())[0]
                try:
                    # If it's a float < 1, it's epsilon (e.g. 0.01)
                    if "." in first_key or float(first_key) < 1.0:
                        structure_type = "epsilon_based"
                    else:
                        structure_type = "k_based"
                except:
                    structure_type = "k_based"
            
            if structure_type == "epsilon_based":
                for eps, k_dict in level3_dict.items():
                    for k_str, algo_dict in k_dict.items():
                        k = int(k_str)
                        if k not in aggregated[network][alpha]:
                            aggregated[network][alpha][k] = {}
                        if eps not in aggregated[network][alpha][k]:
                            aggregated[network][alpha][k][eps] = {}
                            
                        for algo, res in algo_dict.items():
                            if res is None or 'final' not in res or 'var' not in res['final']:
                                continue
                            all_algos.add(algo)
                            if algo not in aggregated[network][alpha][k][eps]:
                                aggregated[network][alpha][k][eps][algo] = {}
                            all_metrics.add('var')
                            aggregated[network][alpha][k][eps][algo]['var'] = res['final']['var']
            else: # k_based
                group_key = "All k"
                if group_key not in aggregated[network][alpha]:
                    aggregated[network][alpha][group_key] = {}
                    
                for k_str, algo_dict in level3_dict.items():
                    k = int(k_str)
                    if k not in aggregated[network][alpha][group_key]:
                        aggregated[network][alpha][group_key][k] = {}
                        
                    for algo, res in algo_dict.items():
                        if res is None or 'final' not in res or 'var' not in res['final']:
                            continue
                        all_algos.add(algo)
                        if algo not in aggregated[network][alpha][group_key][k]:
                            aggregated[network][alpha][group_key][k][algo] = {}
                        all_metrics.add('var')
                        aggregated[network][alpha][group_key][k][algo]['var'] = res['final']['var']
                            
    return aggregated, sorted(list(all_algos)), sorted(list(all_metrics)), structure_type

def generate_markdown(aggregated, all_algos, all_metrics, structure_type, out_file="report.md", target_algo="MYOPIC_HYBRID"):
    lines = []
    lines.append("# Evaluation Report\n")
    
    for net in sorted(aggregated.keys()):
        lines.append(f"## Network: {net}\n")
        
        for alpha in sorted(aggregated[net].keys(), key=float):
            lines.append(f"### Alpha = {alpha}\n")
            
            # Sort group_keys. If it's "All k", there's only one. Otherwise it's integer k.
            group_keys = list(aggregated[net][alpha].keys())
            if structure_type == "epsilon_based":
                group_keys = sorted(group_keys, key=int)
            
            for group_key in group_keys:
                if structure_type == "epsilon_based":
                    lines.append(f"#### k = {group_key}\n")
                    row_name = "Epsilon"
                else:
                    row_name = "k"
                
                rows_dict = aggregated[net][alpha][group_key]
                
                for metric in all_metrics:
                    if structure_type == "epsilon_based":
                        sorted_row_keys = sorted(rows_dict.keys(), key=float)
                    else:
                        sorted_row_keys = sorted(rows_dict.keys(), key=int)
                        
                    # Calculate mean variance for each algorithm to sort columns
                    algo_mean_vars = {}
                    for algo in all_algos:
                        vals = []
                        for row_key in sorted_row_keys:
                            val = rows_dict[row_key].get(algo, {}).get('var')
                            if val is not None:
                                vals.append(val)
                        # If no values, assign infinity so it goes to the end
                        algo_mean_vars[algo] = np.mean(vals) if vals else float('inf')
                        
                    # Sort algorithms by increasing variance
                    sorted_table_algos = sorted(all_algos, key=lambda x: algo_mean_vars[x])
                        
                    header = [row_name] + sorted_table_algos
                    lines.append("| " + " | ".join(header) + " |")
                    lines.append("|-" + "-|-".join(["-"*len(h) for h in header]) + "-|")
                    
                    for row_key in sorted_row_keys:
                        row_vals = [str(row_key)]
                        for algo in sorted_table_algos:
                            val = rows_dict[row_key].get(algo, {}).get(metric)
                            if val is not None:
                                if isinstance(val, float):
                                    row_vals.append(f"{val:.6f}")
                                else:
                                    row_vals.append(str(val))
                            else:
                                row_vals.append("N/A")
                        lines.append("| " + " | ".join(row_vals) + " |")
                    
                    lines.append("\n")

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Report generated at {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Markdown report from results JSON")
    parser.add_argument("--file", type=str, default="results_social/results_master.json", help="Path to results json")
    parser.add_argument("--out", type=str, default="report.md", help="Output Markdown file name")
    parser.add_argument("--target", type=str, default="MYOPIC_HYBRID", help="Algorithm to target (legacy argument)")
    
    args = parser.parse_args()
    
    print(f"Reading file: {args.file}")
    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        sys.exit(1)
        
    aggregated, all_algos, all_metrics, structure_type = parse_json(args.file)
    generate_markdown(aggregated, all_algos, all_metrics, structure_type, out_file=args.out, target_algo=args.target)

