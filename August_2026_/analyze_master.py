"""
analyze_master.py
=================
Reads result_master/{myopic_hybrid,pagerank}/results_master.json
and result_master/results_ultimate.json to produce SUMMARY.md.

Outputs:
  result_master/SUMMARY.md  — win/loss table + variance reduction tables

Usage (from August_2026_/ on SSH):
    python analyze_master.py
    python analyze_master.py --results /path/to/result_master
"""

import os, sys, json, argparse, glob

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(_HERE, "result_master")

SUITES = {
    "myopic_hybrid": ["myopic", "naive_myopic", "gonzalez", "hybrid_degree"],
    "pagerank":      ["myopic", "naive_myopic", "gonzalez", "pagerank_heu"],
}
HEURISTICS = ["hybrid_degree", "pagerank_heu"]
BASELINES  = ["myopic", "naive_myopic", "gonzalez"]
ALGO_NAMES = {
    "myopic":        "Myopic",
    "naive_myopic":  "Naive Myopic",
    "gonzalez":      "Gonzalez",
    "hybrid_degree": "Hybrid-Degree",
    "pagerank_heu":  "PageRank-Heu",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def collect_ks(data):
    ks = set()
    for net in data.values():
        for alpha in net.values():
            for algo in alpha.values():
                ks.update(int(k) for k in algo.keys())
    return sorted(ks)


def win_loss_and_reduction(ultimate, ks):
    """
    Returns:
      win_loss[heuristic] = {"wins": int, "total": int}
      reduction[network][alpha][heuristic][k_str] = pct vs myopic
    """
    win_loss  = {h: {"wins": 0, "total": 0} for h in HEURISTICS}
    reduction = {}

    for network, net_data in sorted(ultimate.items()):
        reduction[network] = {}
        for alpha_str, algo_data in sorted(net_data.items()):
            reduction[network][alpha_str] = {h: {} for h in HEURISTICS}
            for k in ks:
                k_str = str(k)
                vars_ = {}
                for algo in list(HEURISTICS) + list(BASELINES):
                    entry = algo_data.get(algo, {}).get(k_str)
                    if entry and "var" in entry:
                        vars_[algo] = entry["var"]

                if not vars_:
                    continue

                myo_var = vars_.get("myopic")

                for h in HEURISTICS:
                    h_var = vars_.get(h)
                    if h_var is None:
                        continue
                    win_loss[h]["total"] += 1
                    baseline_vals = [vars_[b] for b in BASELINES if b in vars_]
                    if baseline_vals and h_var < min(baseline_vals):
                        win_loss[h]["wins"] += 1
                    if myo_var and myo_var > 0:
                        pct = (myo_var - h_var) / myo_var * 100.0
                        reduction[network][alpha_str][h][k_str] = round(pct, 2)

    return win_loss, reduction


def write_report(ultimate, ks, win_loss, reduction, out_path):
    L = []
    L.append("# Fair Influence Maximization — Results Summary\n\n")
    L.append("> **Metric:** Population Variance  `Var(P) = (1/N)·Σ(pᵢ−μ)²`  — **lower = fairer**\n")
    L.append(f"> **k** ∈ {ks}  |  **α** ∈ {{0.2, 0.3, 0.4}}  |  R=200  |  T=15\n\n")

    # ── Win/Loss ──────────────────────────────────────────────────────────────
    L.append("---\n## Overall Win/Loss\n\n")
    L.append("A **win** means the heuristic has lower variance than **all 3 baselines** (Myopic, Naive Myopic, Gonzalez).\n\n")
    L.append("| Algorithm | Wins | Total Configs | Win Rate |\n")
    L.append("|-----------|-----:|-------------:|---------:|\n")
    for h in HEURISTICS:
        w, t = win_loss[h]["wins"], win_loss[h]["total"]
        rate = f"{w/t*100:.1f}%" if t else "N/A"
        L.append(f"| {ALGO_NAMES[h]} | {w} | {t} | {rate} |\n")
    L.append("\n")

    # ── Variance reduction tables ─────────────────────────────────────────────
    L.append("---\n## Variance Reduction vs Myopic (%)\n\n")
    L.append("> **+X%** = heuristic is X% fairer than Myopic.  **−X%** = heuristic is worse.\n\n")

    for network in sorted(ultimate.keys()):
        L.append(f"### {network}\n\n")
        for alpha_str in sorted(ultimate[network].keys()):
            L.append(f"**α = {alpha_str}**\n\n")
            L.append("| k |" + "".join(f" {ALGO_NAMES[h]} |" for h in HEURISTICS) + "\n")
            L.append("|---|" + "---|" * len(HEURISTICS) + "\n")
            for k in ks:
                row = f"| {k} |"
                for h in HEURISTICS:
                    val = reduction.get(network, {}).get(alpha_str, {}).get(h, {}).get(str(k))
                    if val is None:
                        row += " – |"
                    elif val >= 0:
                        row += f" **+{val:.2f}%** |"
                    else:
                        row += f" {val:.2f}% |"
                L.append(row + "\n")
            L.append("\n")

    # ── Raw variance tables ───────────────────────────────────────────────────
    L.append("---\n## Raw Variance Values  `Var(P)`\n\n")
    all_algos = BASELINES + HEURISTICS

    for network in sorted(ultimate.keys()):
        for alpha_str in sorted(ultimate[network].keys()):
            L.append(f"### {network} | α={alpha_str}\n\n")
            algo_data = ultimate[network][alpha_str]
            L.append("| k |" + "".join(f" {ALGO_NAMES.get(a,a)} |" for a in all_algos) + "\n")
            L.append("|---|" + "---|" * len(all_algos) + "\n")
            for k in ks:
                k_str = str(k)
                row = f"| {k} |"
                for algo in all_algos:
                    entry = algo_data.get(algo, {}).get(k_str)
                    if entry and "var" in entry:
                        row += f" {entry['var']:.6f} |"
                    else:
                        row += " – |"
                L.append(row + "\n")
            L.append("\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(L)


def main():
    parser = argparse.ArgumentParser(description="Analyze result_master/ → SUMMARY.md")
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help="Path to result_master/ directory")
    args = parser.parse_args()

    root = os.path.abspath(args.results)
    if not os.path.isdir(root):
        print(f"[ERROR] Not found: {root}")
        sys.exit(1)

    # Try results_ultimate.json first; fall back to merging suite masters
    ultimate_path = os.path.join(root, "results_ultimate.json")
    if os.path.exists(ultimate_path):
        print(f"Loading: {ultimate_path}")
        ultimate = load_json(ultimate_path)
    else:
        print("results_ultimate.json not found – merging from suite masters ...")
        ultimate = {}
        for suite_name in SUITES:
            suite_master = os.path.join(root, suite_name, "results_master.json")
            if not os.path.exists(suite_master):
                print(f"  [SKIP] {suite_master} not found")
                continue
            data = load_json(suite_master)
            for network, net_data in data.items():
                ultimate.setdefault(network, {})
                for alpha_str, algo_data in net_data.items():
                    ultimate[network].setdefault(alpha_str, {}).update(algo_data)

    if not ultimate:
        print("[ERROR] No result data found. Run run_master.py first.")
        sys.exit(1)

    networks = sorted(ultimate.keys())
    ks = collect_ks(ultimate)
    print(f"Networks  : {networks}")
    print(f"k values  : {ks}")

    win_loss, reduction = win_loss_and_reduction(ultimate, ks)

    out_path = os.path.join(root, "SUMMARY.md")
    write_report(ultimate, ks, win_loss, reduction, out_path)
    print(f"\nReport    : {out_path}")

    print("\n── Win/Loss ──────────────────────────────────────────────")
    for h in HEURISTICS:
        w, t = win_loss[h]["wins"], win_loss[h]["total"]
        rate = f"{w/t*100:.1f}%" if t else "N/A"
        print(f"  {ALGO_NAMES[h]:<22} {w}/{t} wins  ({rate})")


if __name__ == "__main__":
    main()
