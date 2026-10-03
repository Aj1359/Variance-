"""
run_master.py
=============
Unified parallel experiment runner for Fair Influence Maximization.

OUTPUT STRUCTURE:
  result_master/
  ├── myopic_hybrid/
  │   ├── results_master.json         (all networks × alphas for hybrid run)
  │   └── <network>/<alpha>/
  │       ├── variance_vs_k.png
  │       └── k_<k>/distribution.png
  ├── pagerank/
  │   ├── results_master.json         (all networks × alphas for pagerank run)
  │   └── <network>/<alpha>/
  │       ├── variance_vs_k.png
  │       └── k_<k>/distribution.png
  └── results_ultimate.json           (merged from both)

ALGORITHMS:
  myopic_hybrid folder : myopic, naive_myopic, gonzalez, hybrid_degree
  pagerank folder      : myopic, naive_myopic, gonzalez, pagerank_heu

PARAMETERS:
  k      = [10, 20, 30, 40, 50, 60]
  alpha  = [0.2, 0.3, 0.4]
  R      = 200   Monte Carlo rounds
  T      = 15    max cascade steps

PARALLELISM:
  Each (network, alpha, suite) triple is one task → multiprocessing.Pool
  Resume: existing k entries in results_master.json are skipped

VARIANCE:
  Population Variance  Var(P) = (1/N) · Σ(pᵢ − μ)²   — lower = fairer

USAGE (from August_2026_/ on SSH):
  python run_master.py
  python run_master.py --socials ../Socials --workers 10
  nohup python run_master.py > run_master.log 2>&1 &
"""

import os, sys, glob, json, time, argparse, traceback, multiprocessing

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from myopic               import myopic
from naive_myopic         import naive_myopic
from gonzalez             import gonzalez
from myopic_hybrid_degree import myopic_hybrid_degree
from new_heu_pr           import new_heu_pr
from icm                  import prob_est_timed

# ── Experiment parameters ─────────────────────────────────────────────────────
KS     = [10, 20, 30, 40, 50, 60]
ALPHAS = [0.2, 0.3, 0.4]
R      = 200
T      = 15

DEFAULT_WORKERS = 10
DEFAULT_SOCIALS = os.path.join(_HERE, "Socials")
DEFAULT_RESULTS = os.path.join(_HERE, "result_master")

# Suites
SUITES = {
    "myopic_hybrid": ["myopic", "naive_myopic", "gonzalez", "hybrid_degree"],
    "pagerank":      ["myopic", "naive_myopic", "gonzalez", "pagerank_heu"],
}

ALGO_META = {
    "myopic":        ("Myopic",             "o",  "-",   "#4472C4"),
    "naive_myopic":  ("Naive Myopic",       "s",  "--",  "#ED7D31"),
    "gonzalez":      ("Gonzalez",           "^",  "-.",  "#70AD47"),
    "hybrid_degree": ("Hybrid-Degree",      "D",  "-",   "#7030A0"),
    "pagerank_heu":  ("PageRank Heuristic", "P",  "-",   "#C00000"),
}


# ── Metric ────────────────────────────────────────────────────────────────────
def compute_variance(probs):
    """Population variance Var(P) = (1/N)·Σ(pᵢ−μ)². Lower = fairer."""
    n = len(probs)
    if n < 2:
        return 0.0
    mu = sum(probs) / n
    return sum((p - mu) ** 2 for p in probs) / n


# ── Graph loader ──────────────────────────────────────────────────────────────
def load_graph(gml_path):
    """
    Memory-efficient GML reader that parses GML without building a NetworkX graph.
    Returns (adj, nodes, pr_dict, n).
    """
    nodes = []
    pr_dict = {}
    adj = []
    
    current_node = None
    current_edge = None
    
    with open(gml_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("node ["):
                current_node = {}
            elif line.startswith("edge ["):
                current_edge = {}
            elif line == "]":
                if current_node is not None:
                    nid = current_node['id']
                    while len(nodes) <= nid:
                        nodes.append({})
                    nodes[nid] = current_node
                    if 'pagerank' in current_node:
                        pr_dict[nid] = current_node['pagerank']
                    current_node = None
                elif current_edge is not None:
                    u = current_edge['source']
                    v = current_edge['target']
                    n_max = max(u, v)
                    while len(adj) <= n_max:
                        adj.append([])
                    adj[u].append(v)
                    adj[v].append(u)
                    current_edge = None
            else:
                # Key-value pair, e.g. "id 0"
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    key, val = parts[0], parts[1]
                    if current_node is not None:
                        if key == 'id' or key == 'degree':
                            current_node[key] = int(val)
                        elif key == 'pagerank':
                            current_node[key] = float(val)
                    elif current_edge is not None:
                        if key == 'source' or key == 'target':
                            current_edge[key] = int(val)
                        elif key == 'weight':
                            current_edge[key] = float(val)
                            
    n = len(nodes)
    while len(adj) < n:
        adj.append([])
    return adj, nodes, pr_dict, n



# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_variance_vs_k(suite_algos, data, ks, out_path, network, alpha, suite_name):
    plt.figure(figsize=(9, 6))
    for algo in suite_algos:
        label, marker, ls, color = ALGO_META[algo]
        vals = [data.get(algo, {}).get(str(k), {}).get("var") for k in ks]
        valid = [(k, v) for k, v in zip(ks, vals) if v is not None]
        if valid:
            xs, ys = zip(*valid)
            plt.plot(xs, ys, marker=marker, linestyle=ls, color=color,
                     linewidth=2, label=label)

    plt.title(f"Variance vs Seed Set Size  [{suite_name}]\n{network}  |  α={alpha}", fontsize=12)
    plt.xlabel("Seed Set Size k")
    plt.ylabel("Var(P)  [lower = fairer]")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_distribution(suite_algos, hits_by_algo, network, alpha, k, out_path):
    plt.figure(figsize=(10, 6))
    bins = range(0, R + 5, 5)
    for algo in suite_algos:
        hits = hits_by_algo.get(algo)
        if hits is None:
            continue
        label, _, ls, color = ALGO_META[algo]
        plt.hist(hits, bins=bins, histtype="step", linewidth=2,
                 linestyle=ls, color=color, label=label, alpha=0.85)

    plt.title(f"Influence Frequency Distribution\n{network}  |  α={alpha}  |  k={k}", fontsize=11)
    plt.xlabel(f"Times influenced (out of {R} simulations)")
    plt.ylabel("Number of Nodes")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ── Run one algorithm ─────────────────────────────────────────────────────────
def run_algo(algo, adj, nodes, alpha, k, T, R, pr_dict):
    """Dispatch to correct algorithm function. Returns (seeds, probs, hits)."""
    if algo == "myopic":
        seeds, probs, hits = myopic(adj, nodes, alpha, k, T, R)
    elif algo == "naive_myopic":
        seeds, probs, hits = naive_myopic(adj, nodes, alpha, k, T, R)
    elif algo == "gonzalez":
        seeds, probs, hits = gonzalez(adj, nodes, alpha, k, T, R)
    elif algo == "hybrid_degree":
        seeds, probs, hits, _ = myopic_hybrid_degree(adj, nodes, alpha, k, T, R)
    elif algo == "pagerank_heu":
        seeds, probs, hits, _ = new_heu_pr(
            adj, nodes, alpha, k, T, R,
            pr_dict=pr_dict if pr_dict else None
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    return seeds, probs, hits


# ── Worker function ───────────────────────────────────────────────────────────
def run_task(task):
    """
    Runs one (network, alpha, suite) block.
    task = (gml_path, network, alpha, suite_name, algos, ks, R, T, suite_dir)
    Writes:
      suite_dir/<network>/<alpha>/k_<k>/distribution.png
      suite_dir/<network>/<alpha>/variance_vs_k.png
    Returns (network, alpha, suite_name, status, partial_data) for merging into results_master.json
    """
    gml_path, network, alpha, suite_name, algos, ks, R, T, suite_dir = task

    net_alpha_dir = os.path.join(suite_dir, network, str(alpha))
    os.makedirs(net_alpha_dir, exist_ok=True)

    # Per-(network, alpha) checkpoint JSON inside suite folder
    checkpoint = os.path.join(net_alpha_dir, "checkpoint.json")
    if os.path.exists(checkpoint):
        with open(checkpoint) as f:
            data = json.load(f)
    else:
        data = {a: {} for a in algos}

    try:
        print(f"[START] {suite_name} | {network} | alpha={alpha}", flush=True)
        adj, nodes, pr_dict, n = load_graph(gml_path)
        print(f"  Graph loaded: {n:,} nodes", flush=True)

        for k in ks:
            k_str = str(k)
            all_done = all(k_str in data.get(a, {}) for a in algos)
            if all_done:
                print(f"  [SKIP] k={k} already done.", flush=True)
                continue

            print(f"  [k={k}]", flush=True)
            k_dir = os.path.join(net_alpha_dir, f"k_{k}")
            os.makedirs(k_dir, exist_ok=True)

            hits_for_plot = {}

            for algo in algos:
                if k_str in data.get(algo, {}):
                    hits_for_plot[algo] = None   # can't re-plot, skipped
                    continue
                print(f"    {algo} ...", flush=True)
                seeds, probs, hits = run_algo(algo, adj, nodes, alpha, k, T, R, pr_dict)
                var = compute_variance(probs)
                mu  = sum(probs) / n
                data.setdefault(algo, {})[k_str] = {
                    "seeds": seeds, "var": round(var, 8), "mu": round(mu, 8)
                }
                hits_for_plot[algo] = hits

            # Checkpoint after each k
            with open(checkpoint, "w") as f:
                json.dump(data, f, indent=2)

            # Distribution plot for this k
            dist_path = os.path.join(k_dir, "distribution.png")
            if any(h is not None for h in hits_for_plot.values()):
                plot_distribution(algos, hits_for_plot, network, alpha, k, dist_path)

        # Variance vs k plot for this (network, alpha)
        var_plot = os.path.join(net_alpha_dir, "variance_vs_k.png")
        plot_variance_vs_k(algos, data, ks, var_plot, network, alpha, suite_name)

        print(f"[DONE] {suite_name} | {network} | alpha={alpha}", flush=True)
        return (network, alpha, suite_name, "OK", data)

    except Exception as e:
        tb = traceback.format_exc()
        err_path = os.path.join(net_alpha_dir, "error.log")
        with open(err_path, "w") as ef:
            ef.write(tb)
        print(f"[ERROR] {suite_name} {network} alpha={alpha}: {e}", flush=True)
        return (network, alpha, suite_name, f"ERROR: {e}", {})


# ── Merge results_master.json for a suite ────────────────────────────────────
def build_suite_master_json(suite_dir, algos, all_results):
    """
    Aggregates per-(network,alpha) checkpoint data into one results_master.json.
    Structure: { "<network>": { "<alpha>": { "<algo>": { "<k>": {...} } } } }
    """
    master = {}
    for (network, alpha, suite_name, status, data) in all_results:
        if not data:
            continue
        master.setdefault(network, {})[str(alpha)] = data

    path = os.path.join(suite_dir, "results_master.json")
    with open(path, "w") as f:
        json.dump(master, f, indent=2)
    print(f"  Saved: {path}")
    return master


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fair-IM parallel runner → result_master/{myopic_hybrid,pagerank}/"
    )
    parser.add_argument("--socials", default=DEFAULT_SOCIALS,
                        help="Folder with enriched GML files (default: ../Socials)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help="Output root (default: August_2026_/result_master)")
    parser.add_argument("--suite", choices=["myopic_hybrid", "pagerank", "both"],
                        default="both",
                        help="Which suite to run (default: both)")
    args = parser.parse_args()

    socials_dir = os.path.abspath(args.socials)
    result_root = os.path.abspath(args.results)
    workers     = args.workers

    gml_files = sorted(glob.glob(os.path.join(socials_dir, "*.gml")))
    if not gml_files:
        print(f"[ERROR] No GML files in: {socials_dir}")
        print("  Run  build_gml.py  first.")
        sys.exit(1)

    # Determine which suites to run
    suite_keys = list(SUITES.keys()) if args.suite == "both" else [args.suite]

    print(f"Datasets  : {len(gml_files)}")
    print(f"Suites    : {suite_keys}")
    print(f"Workers   : {workers}  |  R={R}  |  T={T}")
    print(f"k values  : {KS}")
    print(f"α values  : {ALPHAS}")
    print(f"Output    : {result_root}")
    print("=" * 65)

    t0 = time.time()

    # Build task list
    tasks = []
    for suite_name in suite_keys:
        algos     = SUITES[suite_name]
        suite_dir = os.path.join(result_root, suite_name)
        os.makedirs(suite_dir, exist_ok=True)
        for gml_path in gml_files:
            network = os.path.basename(gml_path).replace(".gml", "")
            for alpha in ALPHAS:
                tasks.append((gml_path, network, alpha, suite_name, algos,
                               KS, R, T, suite_dir))

    print(f"Total tasks: {len(tasks)}")

    if workers == 1:
        all_results = [run_task(t) for t in tasks]
    else:
        with multiprocessing.Pool(processes=workers) as pool:
            all_results = pool.map(run_task, tasks)

    elapsed = time.time() - t0
    print(f"\nAll tasks done in {elapsed/3600:.2f}h  ({elapsed:.0f}s)")

    # ── Build results_master.json per suite ──────────────────────────────────
    print("\nBuilding results_master.json files ...")
    suite_masters = {}
    for suite_name in suite_keys:
        algos     = SUITES[suite_name]
        suite_dir = os.path.join(result_root, suite_name)
        suite_results = [r for r in all_results if r[2] == suite_name]
        suite_masters[suite_name] = build_suite_master_json(
            suite_dir, algos, suite_results
        )

    # ── Build results_ultimate.json ───────────────────────────────────────────
    print("\nBuilding results_ultimate.json ...")
    ultimate = {}
    for suite_name, master in suite_masters.items():
        for network, alpha_data in master.items():
            ultimate.setdefault(network, {})
            for alpha_str, algo_data in alpha_data.items():
                ultimate[network].setdefault(alpha_str, {}).update(algo_data)

    ultimate_path = os.path.join(result_root, "results_ultimate.json")
    with open(ultimate_path, "w") as f:
        json.dump(ultimate, f, indent=2)
    print(f"  Saved: {ultimate_path}")

    # ── Status summary ────────────────────────────────────────────────────────
    print("\n── Task Summary ──────────────────────────────────────────────")
    for network, alpha, suite_name, status, _ in sorted(all_results):
        tag = "✓" if status == "OK" else "✗"
        print(f"  {tag}  {suite_name:<16} {network:<25} α={alpha}  {status}")

    print(f"\nDone.  Results in: {result_root}")
    print("Run  python analyze_master.py  to generate the SUMMARY report.")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
