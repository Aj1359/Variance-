"""
build_master_and_run_remaining.py
=================================
1. Aggregates existing per-run results.json into results_master.json
2. Runs any remaining (graph, alpha, run) combinations
3. Rebuilds the master JSON with all results
"""
import os, sys, json, time, argparse

if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attempt2.algorithms import ALGO_ORDER, run_algorithm
from attempt2.ic_model   import prob_est_timeseries
from attempt2.metrics    import timeseries_metrics, compute_all_metrics

EPSILON_FIXED = 0.01


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
        'seeds': seeds, 'seed_log': seed_log,
        'ts_metrics': tsm, 'final': final,
        'group_reach_t': {0: [sum(probs_t[t])/n for t in range(T+1)]},
        'wall_s': time.time() - t0,
    }


def _clean(obj):
    if isinstance(obj, dict):        return {str(k): _clean(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)):return [_clean(x) for x in obj]
    if isinstance(obj, set):         return sorted(obj)
    if isinstance(obj, float):       return round(obj, 6)
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--alpha',   type=float, nargs='+', default=[0.1])
    p.add_argument('--k',       type=int,   nargs='+', default=[20, 40])
    p.add_argument('--T',       type=int,   default=6)
    p.add_argument('--R',       type=int,   default=50)
    p.add_argument('--lambda_', type=float, default=1.0)
    p.add_argument('--n_runs',  type=int,   default=3)
    p.add_argument('--max_n',   type=int,   default=0)
    p.add_argument('--out_dir', type=str,   default='results_social')
    args = p.parse_args()

    datasets_dir = 'Social_Network'
    graphs = {}
    for fname in sorted(os.listdir(datasets_dir)):
        if not fname.endswith('.txt'): continue
        g_name = fname.replace('.txt', '')
        path = os.path.join(datasets_dir, fname)
        nodes, adj, edges = load_snap_edgelist(path)
        if args.max_n > 0 and len(nodes) > args.max_n:
            print(f'Skipping {g_name} ({len(nodes)} nodes > max_n)')
            continue
        graphs[g_name] = (nodes, adj, edges)
        print(f'Loaded {g_name:<20s}  n={len(nodes):<6d}  edges={len(edges)}')

    master = {}

    # Step 1: Load existing results
    print('\n--- Loading existing results ---')
    for g_name in sorted(graphs.keys()):
        master[g_name] = {}
        for alpha in args.alpha:
            master[g_name][alpha] = {}
            alpha_dir = os.path.join(args.out_dir, g_name, f'icm_alpha_{alpha:.2f}')
            for run_i in range(1, args.n_runs + 1):
                run_label = f'run_{run_i:02d}'
                run_json = os.path.join(alpha_dir, run_label, 'results.json')
                if os.path.exists(run_json):
                    with open(run_json, 'r') as f:
                        data = json.load(f)
                    master[g_name][alpha][run_label] = data.get('results', {})
                    print(f'  Loaded: {g_name}/{alpha}/{run_label}')

    # Step 2: Run any missing combinations
    print('\n--- Running missing combinations ---')
    total_missing = 0
    for g_name, (nodes, adj, edges) in sorted(graphs.items()):
        n_nodes = len(nodes)
        for alpha in args.alpha:
            for run_i in range(1, args.n_runs + 1):
                run_label = f'run_{run_i:02d}'
                if run_label in master[g_name][alpha] and master[g_name][alpha][run_label]:
                    continue  # Already have this run

                total_missing += 1
                alpha_dir = os.path.join(args.out_dir, g_name, f'icm_alpha_{alpha:.2f}')
                run_dir = os.path.join(alpha_dir, run_label)
                os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)

                print(f'\n  Running: {g_name} alpha={alpha} {run_label} (n={n_nodes})')
                run_results = {}

                for k in args.k:
                    run_results[k] = {}
                    for algo in ALGO_ORDER:
                        print(f'    {algo} k={k} ...', end=' ', flush=True)
                        try:
                            res = run_single(algo, adj, nodes, alpha, k,
                                             args.T, args.R, args.lambda_)
                            run_results[k][algo] = res
                            f = res['final']
                            print(f'OK  var={f["var"]:.4f}  mu={f["mu"]:.4f}  t={res["wall_s"]:.1f}s')
                        except Exception as ex:
                            print(f'FAILED: {ex}')
                            run_results[k][algo] = None

                # Save per-run JSON
                run_json = os.path.join(run_dir, 'results.json')
                with open(run_json, 'w', encoding='utf-8') as f:
                    json.dump(_clean({
                        'params': {'alpha': alpha, 'epsilon': EPSILON_FIXED,
                                   'k': args.k, 'T': args.T, 'R': args.R, 'run': run_i},
                        'network': g_name,
                        'results': run_results,
                    }), f, indent=2)

                master[g_name][alpha][run_label] = run_results

    if total_missing == 0:
        print('  All runs already complete!')

    # Step 3: Save master JSON
    master_path = os.path.join(args.out_dir, 'results_master.json')
    with open(master_path, 'w', encoding='utf-8') as f:
        json.dump(_clean({
            'params': {'alpha': args.alpha, 'epsilon': EPSILON_FIXED,
                       'k': args.k, 'T': args.T, 'R': args.R,
                       'n_runs': args.n_runs},
            'results': master,
        }), f, indent=2)

    print(f'\n{"="*60}')
    print(f'Master results saved: {master_path}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
