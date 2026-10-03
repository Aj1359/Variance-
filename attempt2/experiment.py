"""
attempt2/experiment.py
======================
Main experiment runner for the attempt2 suite.

Sweeps:
  - ICM probabilities  alpha in {0.3, 0.6, 0.9}
  - Seed set sizes     k     in {20, 40, 60, 80, 100}
  - Homophily graphs   h     in {0.2, 0.4, 0.6, 0.8}
  - Algorithms               {FGD, FGD_ADAPTIVE, FGD_DEFICIT,
                               Myopic, NaiveMyopic, Gonzales}

For each (alpha, h, k, algo) combination, the runner:
  1. Selects k seeds using the algorithm
  2. Runs the full time-step simulation (t=0..T) on the chosen seed set
  3. Records variance, mean, welfare, JFI, disparity at every time step
  4. Stores results in a nested dict and also serialises to JSON

Output directory structure:
  <out_dir>/
    results.json          -- full numerical results
    plots/                -- all generated figures

Usage:
    python attempt2/run.py
    python attempt2/run.py --alpha 0.3 0.6 --k 20 40 --T 6 --R 150
    python attempt2/run.py --fast          (small R for smoke test)
"""

import argparse
import json
import os
import sys
import time

# Force UTF-8 stdout on Windows to avoid codec errors
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attempt2.graph_loader import load_all_hichba, graph_summary
from attempt2.algorithms   import ALGO_ORDER, run_algorithm
from attempt2.ic_model     import prob_est_timeseries
from attempt2.metrics      import (timeseries_metrics, compute_all_metrics,
                                   group_membership_from_nodes,
                                   tcim_disparity)


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

SEP = "=" * 80

def hdr(msg):
    print(f"\n{SEP}\n  {msg}\n{SEP}")

def sub(msg):
    print(f"  {msg}")


# ---------------------------------------------------------------------------
# Per-run core: select seeds + full timeseries
# ---------------------------------------------------------------------------

def run_single(algo, adj, nodes, alpha, k, T, R, lambda_):
    """
    Run one (algo, alpha, k) combination on a graph.

    Returns dict with keys:
        seeds           list[int]
        seed_log        list[dict]   per-seed-step metrics
        ts_metrics      dict         {mu, var, welfare, jfi, minp, gap, disparity}
                                     each a list of T+1 values
        final           dict         scalar metrics at t=T
        group_reach_t   dict         {group: [mean_p_v for t=0..T]}
        wall_s          float        total wall time (seconds)
    """
    n  = len(nodes)
    gm = group_membership_from_nodes(nodes)

    t0 = time.time()
    seeds, seed_log = run_algorithm(
        algo, adj, nodes, alpha=alpha, k=k, T=T, R=R, lambda_=lambda_
    )

    # Full timeseries for the final seed set
    probs_t = prob_est_timeseries(adj, seeds, alpha, n, T, R=R)
    tsm     = timeseries_metrics(probs_t, nodes, lambda_)

    # Per-group timeseries
    groups = sorted(gm.keys())
    group_reach_t = {
        g: [
            sum(probs_t[t][i] for i in gm[g]) / len(gm[g]) if gm[g] else 0.0
            for t in range(T + 1)
        ]
        for g in groups
    }

    # Final scalar metrics
    final_probs = probs_t[-1]
    final       = compute_all_metrics(final_probs, nodes, lambda_)
    final["disparity"] = tcim_disparity(final_probs, gm)

    return {
        "seeds":         seeds,
        "seed_log":      seed_log,
        "ts_metrics":    tsm,
        "final":         final,
        "group_reach_t": group_reach_t,
        "wall_s":        time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    hdr(
        "attempt2 -- Fair Influence Maximisation under Time Constraints\n"
        f"  alpha={args.alpha}  k={args.k}  T={args.T}  "
        f"R={args.R}  lambda={args.lambda_}\n"
        f"  Algorithms: {ALGO_ORDER}\n"
        f"  Output: {args.out_dir}"
    )

    # Load graphs
    print(f"\nLoading HICHBA graphs from: {args.graphs_folder}")
    networks = load_all_hichba(args.graphs_folder, max_n=args.max_n)
    if not networks:
        print("ERROR: no HICHBA_*.gml files found.")
        sys.exit(1)

    h_vals = sorted(networks.keys())
    print(f"  Loaded {len(networks)} graphs: h in {h_vals}")

    # Storage: results[alpha][h][k][algo] = run_single output
    results = {}
    total_runs = len(args.alpha) * len(h_vals) * len(args.k) * len(ALGO_ORDER)
    run_idx = 0

    for alpha in args.alpha:
        results[alpha] = {}
        for h, (nodes, adj, edges) in sorted(networks.items()):
            results[alpha][h] = {}
            s = graph_summary(nodes, adj, edges)
            hdr(
                f"alpha={alpha}  h={h}  |  n={s['n']}  edges={s['edges']}  "
                f"groups={s['groups']}  avg_deg={s['avg_degree']}"
            )

            for k in args.k:
                results[alpha][h][k] = {}
                print(f"\n  -- k={k} --")

                for algo in ALGO_ORDER:
                    run_idx += 1
                    sub(f"[{run_idx}/{total_runs}] {algo} ...")

                    try:
                        res = run_single(
                            algo, adj, nodes,
                            alpha=alpha, k=k, T=args.T,
                            R=args.R, lambda_=args.lambda_
                        )
                        results[alpha][h][k][algo] = res

                        f = res["final"]
                        sub(f"  OK  var={f['var']:.4f}  mu={f['mu']:.4f}  "
                            f"disp={f['disparity']:.4f}  "
                            f"t={res['wall_s']:.1f}s")

                    except Exception as ex:
                        print(f"  FAILED {algo}: {ex}")
                        import traceback; traceback.print_exc()
                        results[alpha][h][k][algo] = None

                _print_k_table(results[alpha][h][k], k, args.T)

    # Save JSON
    json_path = os.path.join(args.out_dir, "results.json")
    _save_json(results, json_path, args)
    print(f"\n  Results saved: {json_path}")

    # Generate plots
    print(f"\nGenerating plots: {plots_dir}")
    from attempt2.plots import generate_all_plots
    generate_all_plots(results, args, plots_dir)

    hdr("DONE -- all results and plots saved.")
    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _print_k_table(k_results, k, T):
    """Print a comparison table for all algorithms at a given k."""
    print(f"\n  {'Algorithm':<14} {'mu^T':>8} {'Var^T':>8} "
          f"{'Welfare':>8} {'JFI':>7} {'Disp':>7} {'t(s)':>7}")
    print("  " + "-" * 64)
    for algo in ALGO_ORDER:
        res = k_results.get(algo)
        if res is None:
            print(f"  {algo:<14}  -- failed --")
            continue
        f = res["final"]
        print(f"  {algo:<14} {f['mu']:>8.4f} {f['var']:>8.4f} "
              f"{f['welfare']:>8.4f} {f['jfi']:>7.4f} "
              f"{f['disparity']:>7.4f} {res['wall_s']:>7.1f}")


def _save_json(results, path, args):
    """Serialise results to JSON."""
    def _clean(obj):
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(x) for x in obj]
        if isinstance(obj, set):
            return sorted(obj)
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    payload = {
        "params": {
            "alpha":  args.alpha,
            "k":      args.k,
            "T":      args.T,
            "R":      args.R,
            "lambda": args.lambda_,
            "max_n":  args.max_n,
        },
        "results": _clean(results),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="attempt2 -- Fair TCIM experiment suite")
    p.add_argument("--graphs_folder", type=str,
                   default=os.path.join(
                       os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "Synthetic Networks"),
                   help="Folder containing HICHBA_*.gml files")
    p.add_argument("--alpha",   type=float, nargs="+",
                   default=[0.3, 0.6, 0.9])
    p.add_argument("--k",       type=int,   nargs="+",
                   default=[20, 40, 60, 80, 100])
    p.add_argument("--T",       type=int,   default=8)
    p.add_argument("--R",       type=int,   default=150)
    p.add_argument("--lambda_", type=float, default=1.0)
    p.add_argument("--max_n",   type=int,   default=2000,
                   help="Subsample graphs to at most max_n nodes (0=no limit)")
    p.add_argument("--out_dir", type=str,   default="attempt2/results")
    p.add_argument("--fast",    action="store_true",
                   help="Smoke test: R=30, k=[20,40], alpha=[0.3], max_n=500")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.fast:
        args.R     = 30
        args.k     = [20, 40]
        args.alpha = [0.3]
        args.max_n = 500
        print("FAST MODE: R=30, k=[20,40], alpha=[0.3], max_n=500")
    run_experiment(args)
