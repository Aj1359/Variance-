"""
run_social.py
=============
Experiment runner: for each graph -> for each alpha -> single run.

Each run executes all algorithms independently (all are MC-based so results
vary per run). Results saved in:

  result_new/
    {graph_name}/
      {alpha}/
        results.json 
        plots/
  result_new/results_master.json

epsilon is fixed at 0.01 (best overall from sensitivity analysis).
"""
import os, sys, time, json, argparse

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attempt2.algorithms import ALGO_ORDER, run_algorithm
from attempt2.ic_model   import prob_est_timeseries
from attempt2.metrics    import timeseries_metrics, compute_all_metrics

EPSILON_FIXED = 0.01   # fixed from sensitivity analysis

# ---------------------------------------------------------------------------
# Graph loader
# ---------------------------------------------------------------------------
def load_snap_edgelist(path):
    edges, nodes_found = [], set()
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
                nodes_found.update([u, v])

    sorted_nodes = sorted(nodes_found)
    id_map = {o: n for n, o in enumerate(sorted_nodes)}
    n = len(sorted_nodes)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        nu, nv = id_map[u], id_map[v]
        adj[nu].add(nv); adj[nv].add(nu)

    return [{'id': i, 'group': 0} for i in range(n)], adj, edges


# ---------------------------------------------------------------------------
# Single algorithm run
# ---------------------------------------------------------------------------
def run_single(algo, adj, nodes, alpha, k, T, R, lambda_):
    n = len(nodes)
    t0 = time.time()

    seeds, seed_log = run_algorithm(
        algo, adj, nodes, alpha=alpha, k=k, T=T, R=R,
        lambda_=lambda_, epsilon=EPSILON_FIXED
    )

    probs_t  = prob_est_timeseries(adj, seeds, alpha, n, T, R=R)
    tsm      = timeseries_metrics(probs_t, nodes, lambda_)
    final    = compute_all_metrics(probs_t[-1], nodes, lambda_)
    final['disparity'] = 0.0

    return {
        'seeds':         seeds,
        'seed_log':      seed_log,
        'ts_metrics':    tsm,
        'final':         final,
        'group_reach_t': {0: [sum(probs_t[t])/n for t in range(T+1)]},
        'wall_s':        time.time() - t0,
    }


def _clean(obj):
    if isinstance(obj, dict):        return {str(k): _clean(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)):return [_clean(x) for x in obj]
    if isinstance(obj, set):         return sorted(obj)
    if isinstance(obj, float):       return round(obj, 6)
    return obj


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print('='*80)
    print('  Social Network Experiment')
    print('  epsilon=%.2f (fixed)  alphas=%s' % (EPSILON_FIXED, args.alpha))
    print('  k=%s  T=%d  R=%d' % (args.k, args.T, args.R))
    print('  Algorithms: %s' % ALGO_ORDER)
    print('  Output: %s' % args.out_dir)
    print('='*80)

    # Load graphs
    datasets_dir = 'Social_network'
    if not os.path.exists(datasets_dir):
        print('ERROR: Social_network/ not found.'); sys.exit(1)

    graphs = {}
    for fname in sorted(os.listdir(datasets_dir)):
        if not fname.endswith('.txt'): continue
        g_name = fname.replace('.txt', '')
        path   = os.path.join(datasets_dir, fname)
        nodes, adj, edges = load_snap_edgelist(path)
        if args.max_n > 0 and len(nodes) > args.max_n:
            print('Skipping %s (%d nodes > max_n)' % (g_name, len(nodes)))
            continue
        graphs[g_name] = (nodes, adj, edges)
        print('Loaded %-20s  n=%-6d  edges=%d' % (g_name, len(nodes), len(edges)))

    # Master accumulator: master[alpha][g][k][algo] = result
    master = {}
    for alpha in args.alpha:
        master[str(alpha)] = {}

    total = len(graphs) * len(args.alpha) * len(args.k) * len(ALGO_ORDER)
    idx   = 0

    # Outer loop: graph -> alpha
    for g_name, (nodes, adj, edges) in sorted(graphs.items()):
        n_nodes = len(nodes)

        for alpha in args.alpha:
            alpha_str = str(alpha)
            if g_name not in master[alpha_str]:
                master[alpha_str][g_name] = {}
                
            alpha_dir = os.path.join(args.out_dir, g_name, alpha_str)
            plots_dir = os.path.join(alpha_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)

            print('\n' + '='*80)
            print('  Graph=%-20s  alpha=%.2f  n=%d' % (g_name, alpha, n_nodes))
            print('='*80)

            run_results = {}   # {k: {algo: result}}

            for k in args.k:
                run_results[k] = {}
                for algo in ALGO_ORDER:
                    idx += 1
                    print('  [%d/%d] k=%d  %s ...' % (
                        idx, total, k, algo),
                        end=' ', flush=True)
                    try:
                        res = run_single(algo, adj, nodes,
                                         alpha, k, args.T, args.R,
                                         args.lambda_)
                        run_results[k][algo] = res
                        f = res['final']
                        print('OK  var=%.4f  mu=%.4f  t=%.1fs' % (
                            f['var'], f['mu'], res['wall_s']))
                    except Exception as ex:
                        print('FAILED: %s' % ex)
                        run_results[k][algo] = None

                master[alpha_str][g_name][str(k)] = run_results[k]

            # Save per-graph-alpha JSON
            run_json = os.path.join(alpha_dir, 'results.json')
            with open(run_json, 'w', encoding='utf-8') as f:
                json.dump(_clean({
                    'params':  {'alpha': alpha, 'epsilon': EPSILON_FIXED,
                                'k': args.k, 'T': args.T, 'R': args.R},
                    'network': g_name,
                    'results': run_results,
                }), f, indent=2)

            # Generate plots
            try:
                from attempt2.plots import generate_all_plots
                class _A: pass
                pa = _A()
                pa.alpha = [alpha]; pa.k = args.k; pa.T = args.T
                generate_all_plots({alpha: {g_name: run_results}}, pa, plots_dir)
            except Exception as ex:
                print('    Plot failed: %s' % ex)

    # Save master JSON
    master_path = os.path.join(args.out_dir, 'results_master.json')
    with open(master_path, 'w', encoding='utf-8') as f:
        json.dump(_clean({
            'params': {'alpha': args.alpha, 'epsilon': EPSILON_FIXED,
                       'k': args.k, 'T': args.T, 'R': args.R},
            'results': master,
        }), f, indent=2)

    print('\n' + '='*80)
    print('DONE -- master results -> %s' % master_path)
    print('Folder layout:')
    print('  %s/' % args.out_dir)
    print('    {graph}/{alpha}/results.json')
    print('    {graph}/{alpha}/plots/')
    print('='*80)

    # ---------- Auto-generate leaderboard report ----------
    try:
        from generate_plots_from_json import generate_win_loss_report
        import json as _json

        print('\nGenerating leaderboard report...')
        with open(master_path, 'r', encoding='utf-8') as f:
            data = _json.load(f)
        results_raw = data.get('results', {})

        parsed = {}
        for a_str, net_dict in results_raw.items():
            try: a_key = float(a_str)
            except: a_key = a_str
            parsed[a_key] = {}
            for net, k_dict in net_dict.items():
                parsed[a_key][net] = {}
                for k_str, algo_dict in k_dict.items():
                    parsed[a_key][net][int(k_str)] = algo_dict

        report_path = os.path.join(args.out_dir, 'leaderboard_report.md')
        generate_win_loss_report(parsed, report_path)
        print('Leaderboard report saved -> %s' % report_path)
    except Exception as ex:
        print('Report generation failed: %s' % ex)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--alpha',   type=float, nargs='+', default=[0.1, 0.2, 0.3])
    p.add_argument('--k',       type=int,   nargs='+', default=[20, 40, 60, 80, 100])
    p.add_argument('--T',       type=int,   default=8)
    p.add_argument('--R',       type=int,   default=150)
    p.add_argument('--lambda_', type=float, default=1.0)
    p.add_argument('--max_n',   type=int,   default=0,
                   help='Skip graphs larger than this (0 = no limit)')
    p.add_argument('--out_dir', type=str,   default='result_final_myopic_hybrid')
    args = p.parse_args()
    run_experiment(args)
