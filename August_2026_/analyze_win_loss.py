import os
import json
import glob
from collections import defaultdict

def safe_float(val):
    try:
        return float(val)
    except:
        return float('inf')

def analyze_experiment(exp_dir, target_key, algo_name):
    """
    Scans an experiment directory for results.json and compares strictly against Myopic and all baselines.
    """
    json_files = glob.glob(os.path.join(exp_dir, "**", "results.json"), recursive=True)
    records = []
    
    for fpath in json_files:
        norm_path = fpath.replace("\\", "/")
        parts = norm_path.split("/")
        
        try:
            rel_parts = parts[parts.index(exp_dir.replace("\\", "/")):]
        except ValueError:
            rel_parts = parts
            
        network = rel_parts[1] if len(rel_parts) > 1 else "Unknown"
        alpha = rel_parts[2] if len(rel_parts) > 2 else "Unknown"
        
        # Check if closeness percentage is in path (for result_hybrid)
        closeness = "N/A"
        if len(rel_parts) > 4 and "percent" in rel_parts[3]:
            closeness = rel_parts[3]
        
        with open(fpath, "r") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                continue
                
        # Find target key
        t_key = None
        if target_key in data:
            t_key = target_key
        else:
            for k in data.keys():
                if target_key.lower() in k.lower() or k.lower() in target_key.lower():
                    if k not in ["myopic", "naive_myopic", "gonzalez"]:
                        t_key = k
                        break
                        
        if not t_key:
            continue
            
        target_dict = data[t_key]
        myopic_dict = data.get("myopic", {})
        nm_dict = data.get("naive_myopic", {})
        gonz_dict = data.get("gonzalez", {})
        
        for k_str, t_res in target_dict.items():
            k_val = int(k_str)
            t_var = t_res.get("var", float('inf'))
            t_mu = t_res.get("mu", t_res.get("final_reach", 0.0))
            
            # 1. Myopic variance
            if k_str in myopic_dict:
                myo_var = myopic_dict[k_str].get("var", float('inf'))
                myo_mu = myopic_dict[k_str].get("mu", myopic_dict[k_str].get("final_reach", 0.0))
            else:
                myo_var = float('inf')
                myo_mu = 0.0
                
            # 2. Naive Myopic variance
            if k_str in nm_dict:
                nm_var = nm_dict[k_str].get("var", float('inf'))
            else:
                nm_var = float('inf')
                
            # 3. Gonzalez variance
            if k_str in gonz_dict:
                gonz_var = gonz_dict[k_str].get("var", float('inf'))
            else:
                gonz_var = float('inf')
                
            if myo_var == float('inf'):
                continue
                
            # Find best baseline variance
            baseline_vars = [v for v in [myo_var, nm_var, gonz_var] if v != float('inf')]
            if not baseline_vars:
                continue
            best_base_var = min(baseline_vars)
                
            # Outcome vs Myopic
            if t_var < myo_var - 1e-9:
                status = "Win"
            elif abs(t_var - myo_var) <= 1e-9:
                status = "Tie"
            else:
                status = "Loss"
                
            # Outcome vs All Baselines
            if t_var < best_base_var - 1e-9:
                status_all = "Win"
            elif abs(t_var - best_base_var) <= 1e-9:
                status_all = "Tie"
            else:
                status_all = "Loss"
                
            # % variance improvement vs Myopic
            if myo_var > 0:
                pct_improvement = ((myo_var - t_var) / myo_var) * 100.0
            else:
                pct_improvement = 0.0
                
            # % variance improvement vs Best Baseline
            if best_base_var > 0:
                pct_improvement_all = ((best_base_var - t_var) / best_base_var) * 100.0
            else:
                pct_improvement_all = 0.0
                
            records.append({
                "network": network,
                "alpha": alpha,
                "closeness": closeness,
                "k": k_val,
                "target_var": t_var,
                "target_mu": t_mu,
                "myopic_var": myo_var,
                "myopic_mu": myo_mu,
                "nm_var": nm_var,
                "gonz_var": gonz_var,
                "best_base_var": best_base_var,
                "pct_improvement": pct_improvement,
                "pct_improvement_all": pct_improvement_all,
                "status": status,
                "status_all": status_all,
                "algo_name": algo_name
            })
            
    return records

def generate_markdown_report(all_records, output_file="WIN_LOSS_ANALYSIS_REPORT.md"):
    if not all_records:
        print("No simulation results found to analyze.")
        with open(output_file, "w") as f:
            f.write("# Win/Loss Analysis Report\n\nNo experimental results found. Please check your folder names and run the experiments.\n")
        return

    import math

    # Group by Algorithm Variant
    by_algo = defaultdict(list)
    for r in all_records:
        key = r["algo_name"]
        if r["closeness"] != "N/A":
            key += f" ({r['closeness']})"
        by_algo[key].append(r)

    with open(output_file, "w") as f:
        f.write("# Influence Maximization: Win/Loss Benchmark Report (vs Baselines)\n\n")
        f.write("> **Evaluation Rule:** In Fair Influence Maximization, **Lower Variance is strictly superior**.\n")
        f.write("> - **Win vs Myopic:** Target Algorithm Variance < Myopic Variance.\n")
        f.write("> - **Win vs All Baselines:** Target Algorithm Variance < Best Baseline Variance (Myopic, Naive Myopic, Gonzalez).\n\n")
        f.write("---\n\n")

        # ---------------------------------------------------------
        # 1. Master Executive Leaderboard
        # ---------------------------------------------------------
        f.write("## 1. Master Leaderboard (Target Algorithms vs Baselines)\n\n")
        f.write("| Target Algorithm | Total Runs | Wins vs Myopic | Wins vs All Baselines | Ties vs All | Losses vs All | Win Rate vs All (%) | Avg Var Reduction vs Best Baseline (%) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")

        for algo_variant, recs in sorted(by_algo.items()):
            total = len(recs)
            wins_myo = sum(1 for r in recs if r["status"] == "Win")
            wins_all = sum(1 for r in recs if r["status_all"] == "Win")
            ties_all = sum(1 for r in recs if r["status_all"] == "Tie")
            losses_all = sum(1 for r in recs if r["status_all"] == "Loss")
            win_pct_all = (wins_all / total * 100) if total > 0 else 0.0
            avg_red_all = sum(r["pct_improvement_all"] for r in recs) / total if total > 0 else 0.0
            
            f.write(f"| **{algo_variant}** | {total} | {wins_myo} | **{wins_all}** | {ties_all} | {losses_all} | **{win_pct_all:.1f}%** | **{avg_red_all:+.2f}%** |\n")

        f.write("\n---\n\n")

        # ---------------------------------------------------------
        # 1.5 Baseline Performance Matrix (Head-to-Head)
        # ---------------------------------------------------------
        f.write("## 1.5 Baseline Performance Matrix (Per-Network, Per-Alpha Average Variance)\n\n")
        f.write("This table shows the average variance (across all $k$ values) for each baseline, allowing you to see which baseline is the most competitive.\n\n")
        f.write("| Network | Alpha | Myopic (Avg Var) | Naive Myopic (Avg Var) | Gonzalez (Avg Var) | Best Performing Baseline | Hardest to Beat |\n")
        f.write("|---|---|---|---|---|---|---|\n")

        # We want to aggregate all records across all algorithms by (network, alpha) to avoid duplication
        seen_keys = set()
        for r in sorted(all_records, key=lambda x: (x["network"], float(x["alpha"]) if x["alpha"].replace('.','',1).isdigit() else 0)):
            net_alp = (r["network"], r["alpha"])
            if net_alp in seen_keys:
                continue
            seen_keys.add(net_alp)
            
            # Find all records for this (network, alpha) across any algorithm to average the baselines
            net_recs = [x for x in all_records if x["network"] == r["network"] and x["alpha"] == r["alpha"]]
            
            k_groups = defaultdict(list)
            for nr in net_recs:
                k_groups[nr["k"]].append(nr)
                
            myo_vals = []
            nm_vals = []
            gonz_vals = []
            
            for k_val, k_recs in k_groups.items():
                first = k_recs[0]
                if first["myopic_var"] != float('inf'):
                    myo_vals.append(first["myopic_var"])
                if first["nm_var"] != float('inf'):
                    nm_vals.append(first["nm_var"])
                if first["gonz_var"] != float('inf'):
                    gonz_vals.append(first["gonz_var"])
                    
            avg_myo = sum(myo_vals) / len(myo_vals) if myo_vals else float('nan')
            avg_nm = sum(nm_vals) / len(nm_vals) if nm_vals else float('nan')
            avg_gonz = sum(gonz_vals) / len(gonz_vals) if gonz_vals else float('nan')
            
            candidates = [("Myopic", avg_myo), ("Naive Myopic", avg_nm), ("Gonzalez", avg_gonz)]
            valid_candidates = [(name, val) for name, val in candidates if not math.isnan(val)]
            if valid_candidates:
                best_name, best_val = min(valid_candidates, key=lambda x: x[1])
            else:
                best_name, best_val = "N/A", float('nan')
                
            myo_str = f"{avg_myo:.4f}" if not math.isnan(avg_myo) else "N/A"
            nm_str = f"{avg_nm:.4f}" if not math.isnan(avg_nm) else "N/A"
            gonz_str = f"{avg_gonz:.4f}" if not math.isnan(avg_gonz) else "N/A"
            best_val_str = f"{best_val:.4f}" if not math.isnan(best_val) else "N/A"
            
            f.write(f"| {r['network']} | {r['alpha']} | {myo_str} | {nm_str} | {gonz_str} | **{best_name}** ({best_val_str}) | {best_name} |\n")
        
        f.write("\n---\n\n")

        # ---------------------------------------------------------
        # 2. Performance by Network vs All Baselines
        # ---------------------------------------------------------
        f.write("## 2. Performance Breakdown by Network (Wins vs All Baselines)\n\n")
        
        # Get all unique networks
        all_networks = sorted(list(set(r["network"] for r in all_records)))
        
        f.write("| Target Algorithm | " + " | ".join(all_networks) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(all_networks)) + "|\n")
        
        for algo_variant, recs in sorted(by_algo.items()):
            net_map = defaultdict(lambda: {"win": 0, "total": 0})
            for r in recs:
                net = r["network"]
                net_map[net]["total"] += 1
                if r["status_all"] == "Win":
                    net_map[net]["win"] += 1
                    
            cols = []
            for net in all_networks:
                tot = net_map[net]["total"]
                w = net_map[net]["win"]
                if tot > 0:
                    cols.append(f"**{w}/{tot}** ({w/tot*100:.0f}%)")
                else:
                    cols.append("N/A")
            f.write(f"| **{algo_variant}** | " + " | ".join(cols) + " |\n")

        f.write("\n---\n\n")

        # ---------------------------------------------------------
        # 3. Breakdown by Seed Set Size (k) vs All Baselines
        # ---------------------------------------------------------
        all_ks = sorted(list({r["k"] for r in all_records if "k" in r}))
        f.write("## 3. Win Rate Breakdown by Seed Set Size (k) (vs All Baselines)\n\n")
        header_k = " | ".join([f"k={k} (Win/Total)" for k in all_ks])
        f.write(f"| Target Algorithm | {header_k} |\n")
        f.write("|---|" + "|".join(["---" for _ in all_ks]) + "|\n")

        for algo_variant, recs in sorted(by_algo.items()):
            k_map = defaultdict(lambda: {"win": 0, "total": 0})
            for r in recs:
                k = r["k"]
                k_map[k]["total"] += 1
                if r["status_all"] == "Win":
                    k_map[k]["win"] += 1
            
            k_cols = []
            for k in all_ks:
                if k_map[k]["total"] > 0:
                    w = k_map[k]["win"]
                    tot = k_map[k]["total"]
                    k_cols.append(f"{w}/{tot} ({w/tot*100:.0f}%)")
                else:
                    k_cols.append("N/A")
            
            f.write(f"| **{algo_variant}** | " + " | ".join(k_cols) + " |\n")

        f.write("\n---\n\n")

        # ---------------------------------------------------------
        # 4. Detailed Run-by-Run Comparison Tables
        # ---------------------------------------------------------
        f.write("## 4. Comprehensive Run-by-Run Matrix (Algorithm vs Baselines)\n\n")
        
        for algo_variant, recs in sorted(by_algo.items()):
            f.write(f"### {algo_variant}\n\n")
            f.write("| Network | Alpha | k | Myopic Var | Naive Myopic Var | Gonzalez Var | Target Var | Outcome vs All | Var Reduction vs Best Baseline (%) |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            
            sorted_recs = sorted(recs, key=lambda x: (x["network"], float(x["alpha"]) if x["alpha"].replace('.','',1).isdigit() else 0, x["k"]))
            for r in sorted_recs:
                net = r["network"]
                alp = r["alpha"]
                k = r["k"]
                t_v = f"{r['target_var']:.4f}"
                m_v = f"{r['myopic_var']:.4f}"
                nm_v = f"{r['nm_var']:.4f}" if r['nm_var'] != float('inf') else "N/A"
                g_v = f"{r['gonz_var']:.4f}" if r['gonz_var'] != float('inf') else "N/A"
                res_badge = f"**{r['status_all']}**" if r['status_all'] == "Win" else r['status_all']
                pct_str = f"{r['pct_improvement_all']:+.2f}%"
                
                f.write(f"| {net} | {alp} | {k} | {m_v} | {nm_v} | {g_v} | {t_v} | {res_badge} | {pct_str} |\n")
            f.write("\n")

    print(f"Comprehensive Win/Loss Report generated at: {output_file}")

def main():
    all_records = []
    
    # 1. Collect Hybrid (Closeness Centrality Batching) records
    all_records.extend(analyze_experiment("result_hybrid", target_key="hybrid", algo_name="Myopic-Hybrid-CC"))
    
    # 2. Collect Myopic-Hybrid (Avg Degree) records
    all_records.extend(analyze_experiment("result_hybrid_degree_multi", target_key="hybrid_degree", algo_name="Myopic-Hybrid (Avg Degree)"))
    all_records.extend(analyze_experiment("result_hybrid_degree", target_key="hybrid_degree", algo_name="Myopic-Hybrid (Avg Degree)"))
    
    # 3. Collect PageRank v1 records
    all_records.extend(analyze_experiment("result_pagerank_v1_multi", target_key="pagerank_v1", algo_name="PageRank v1"))
    all_records.extend(analyze_experiment("result_pagerank_v1", target_key="pagerank_v1", algo_name="PageRank v1"))
    
    # 4. Collect PageRank v2 records
    all_records.extend(analyze_experiment("result_pagerank_v2_multi", target_key="pagerank_v2", algo_name="PageRank v2"))
    all_records.extend(analyze_experiment("result_pagerank_v2", target_key="pagerank_v2", algo_name="PageRank v2"))
    
    # 5. Collect PageRank-Hybrid (Lookahead) records
    all_records.extend(analyze_experiment("result_pagerank_hybrid_multi", target_key="pagerank_hybrid", algo_name="PageRank-Hybrid"))
    all_records.extend(analyze_experiment("result_pagerank_hybrid", target_key="pagerank_hybrid", algo_name="PageRank-Hybrid"))
    
    # 6. Collect Entropy-Hybrid records
    all_records.extend(analyze_experiment("result_entropy_multi", target_key="entropy_hybrid", algo_name="Entropy-Hybrid"))
    all_records.extend(analyze_experiment("result_entropy", target_key="entropy_hybrid", algo_name="Entropy-Hybrid"))
    
    # 7. Collect Concave-Hybrid records
    all_records.extend(analyze_experiment("result_concave_multi", target_key="concave_hybrid", algo_name="Concave-Hybrid"))
    all_records.extend(analyze_experiment("result_concave", target_key="concave_hybrid", algo_name="Concave-Hybrid"))
    
    os.makedirs("reports", exist_ok=True)
    generate_markdown_report(all_records, output_file="reports/WIN_LOSS_ANALYSIS_REPORT.md")

if __name__ == "__main__":
    main()
