import time
import run_hybrid_degree
import run_pagerank_v1
import run_pagerank_hybrid
import run_entropy
import run_concave
import analyze_win_loss
import analyze_seed_sets
import plot_influence_outreach

def main():
    start_total = time.time()
    
    print("\n" + "#"*65)
    print("# STARTING MASTER BENCHMARK PIPELINE (ACTIVE EXPERIMENTS)")
    print("# 1. Myopic-Hybrid (Avg Degree) vs Baselines")
    print("# 2. PageRank v1 vs Baselines (Distance + Global PR)")
    print("# 3. PageRank-Hybrid vs Baselines (PPR + Degree Lookahead)")
    print("# 4. Entropy-Hybrid vs Baselines (Entropy Lookahead)")
    print("# 5. Concave-Hybrid vs Baselines (Concave Holder Mean)")
    print("#"*65 + "\n")
    
    # 1. Run Myopic-Hybrid (Avg Degree)
    t0 = time.time()
    print("\n>>> STARTING EXPERIMENT 1: MYOPIC-HYBRID (AVG DEGREE) vs BASELINES")
    run_hybrid_degree.run_experiment()
    print(f">>> EXPERIMENT 1 FINISHED in {time.time() - t0:.1f}s\n")
    
    # 2. Run PageRank v1
    t0 = time.time()
    print("\n>>> STARTING EXPERIMENT 2: PAGERANK v1 vs BASELINES")
    run_pagerank_v1.run_experiment()
    print(f">>> EXPERIMENT 2 FINISHED in {time.time() - t0:.1f}s\n")
    
    # 3. Run PageRank-Hybrid
    t0 = time.time()
    print("\n>>> STARTING EXPERIMENT 3: PAGERANK-HYBRID vs BASELINES")
    run_pagerank_hybrid.run_experiment()
    print(f">>> EXPERIMENT 3 FINISHED in {time.time() - t0:.1f}s\n")
    
    # 4. Run Entropy-Hybrid
    t0 = time.time()
    print("\n>>> STARTING EXPERIMENT 4: ENTROPY-HYBRID vs BASELINES")
    run_entropy.run_experiment()
    print(f">>> EXPERIMENT 4 FINISHED in {time.time() - t0:.1f}s\n")
    
    # 5. Run Concave-Hybrid
    t0 = time.time()
    print("\n>>> STARTING EXPERIMENT 5: CONCAVE-HYBRID vs BASELINES")
    run_concave.run_experiment()
    print(f">>> EXPERIMENT 5 FINISHED in {time.time() - t0:.1f}s\n")
    
    # 6. Generate Master Win/Loss Analysis Report
    print("\n>>> GENERATING WIN/LOSS ANALYSIS REPORT (vs BASELINES)...")
    analyze_win_loss.main()
    
    # 7. Generate Seed Set Analysis Report
    print("\n>>> GENERATING SEED SET ANALYSIS REPORT...")
    analyze_seed_sets.main()
    
    # 8. Generate Influence Outreach Comparison Plots
    print("\n>>> GENERATING INFLUENCE OUTREACH COMPARISON PLOTS...")
    plot_influence_outreach.main()
    
    print("\n" + "="*65)
    print(f"ALL EXPERIMENTS COMPLETED IN {time.time() - start_total:.1f}s")
    print("Folders & Reports generated:")
    print("  1. result_hybrid_degree_multi/")
    print("  2. result_pagerank_v1_multi/")
    print("  3. result_pagerank_hybrid_multi/")
    print("  4. result_entropy_multi/")
    print("  5. result_concave_multi/")
    print("  6. reports/WIN_LOSS_ANALYSIS_REPORT.md")
    print("  7. reports/SEED_SET_ANALYSIS_REPORT.md")
    print("  8. reports/influence_outreach_*.png")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
