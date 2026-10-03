"""
attempt2/ml_data_gen.py
========================
Training data generator for ML-guided variance minimisation.

For each graph × alpha configuration:
  1. Start with S = {highest-degree node}
  2. At each greedy step:
     a. Compute p_i^(T) for all nodes under current S
     b. Extract features for all candidates
     c. For the top-C candidates (by deficit), evaluate ΔVar via MC
     d. Record (features, ΔVar) as training samples
  3. Add the best candidate to S, repeat k times

Output: CSV file with feature columns + target column (delta_var)

Usage:
    python -m attempt2.ml_data_gen
    python -m attempt2.ml_data_gen --C 30 --k 30 --R 80
"""

import os
import sys
import csv
import time
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attempt2.ic_model    import prob_est_timed
from attempt2.metrics     import variance, mean_prob
from attempt2.ml_features import (extract_features, precompute_graph_features,
                                   FEATURE_NAMES, NUM_FEATURES)


# ---------------------------------------------------------------------------
# Graph loaders
# ---------------------------------------------------------------------------

def load_snap_edgelist(path):
    """Load a SNAP edge-list file."""
    edges, nodes_found = [], set()
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
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
        adj[nu].add(nv)
        adj[nv].add(nu)
    return [{'id': i, 'group': 0} for i in range(n)], adj, n


def load_hichba_graphs(folder, max_n=2000):
    """Load HICHBA synthetic graphs."""
    from attempt2.graph_loader import load_all_hichba
    networks = load_all_hichba(folder, max_n=max_n, verbose=False)
    result = {}
    for h, (nodes, adj, edges) in networks.items():
        result[f'HICHBA_h{h}'] = (nodes, adj, len(nodes))
    return result


# ---------------------------------------------------------------------------
# Core: evaluate ΔVar for a candidate
# ---------------------------------------------------------------------------

def evaluate_delta_var(adj, seeds, candidate, alpha, n, T, R, var_curr=None):
    """
    Compute ΔVar = Var(current seeds) - Var(seeds + candidate).
    Positive ΔVar means adding this candidate reduces variance.
    """
    if var_curr is None:
        probs_curr, _ = prob_est_timed(adj, seeds, alpha, n, T, R=R, compute_ts=False)
        var_curr = variance(probs_curr)

    # Variance after adding candidate
    probs_new, _ = prob_est_timed(adj, seeds + [candidate], alpha, n, T, R=R, compute_ts=False)
    var_new = variance(probs_new)

    return var_curr - var_new  # positive = variance reduced


# ---------------------------------------------------------------------------
# Main data generation
# ---------------------------------------------------------------------------

def generate_training_data(graphs, alphas, k, T, R, C, output_path):
    """
    Generate training dataset for ML model.

    Parameters
    ----------
    graphs : dict  {name: (nodes, adj, n)}
    alphas : list[float]
    k      : int   number of seed steps to simulate
    T      : int   time deadline
    R      : int   MC samples for probability estimation
    C      : int   number of top candidates to evaluate ΔVar for
    output_path : str   CSV output file
    """
    # Prepare CSV
    header = FEATURE_NAMES + ['delta_var', 'graph', 'alpha', 'step', 'node_id']
    rows = []

    total_configs = len(graphs) * len(alphas)
    config_idx = 0

    for g_name, (nodes, adj, n) in sorted(graphs.items()):
        # Precompute graph-level features
        print(f'\n  Precomputing graph features for {g_name} (n={n})...')
        graph_feats = precompute_graph_features(adj, n, nodes=nodes, T=T)

        for alpha in alphas:
            config_idx += 1
            print(f'\n  [{config_idx}/{total_configs}] {g_name}  alpha={alpha}  n={n}')

            # Start with highest-degree node
            s0 = max(range(n), key=lambda i: len(adj[i]))
            seeds = [s0]

            for step in range(min(k, n - 1)):
                t0 = time.time()

                # Current probabilities and variance
                probs, _ = prob_est_timed(adj, seeds, alpha, n, T, R=R, compute_ts=False)
                mu = mean_prob(probs)
                var_curr = variance(probs)

                # Extract features for all candidates
                candidates, features = extract_features(
                    adj, n, seeds, probs, graph_feats, T=T, nodes=nodes)

                # Select top-C candidates by deficit (highest deficit = most underserved)
                deficit_idx = FEATURE_NAMES.index('deficit')
                scored = [(features[i][deficit_idx], i) for i in range(len(candidates))]
                scored.sort(reverse=True)
                top_c = scored[:C]

                # Evaluate ΔVar for each top-C candidate
                best_dvar = -float('inf')
                best_cand = None

                for rank, (deficit_val, feat_idx) in enumerate(top_c):
                    cand = candidates[feat_idx]
                    feat = features[feat_idx]

                    dvar = evaluate_delta_var(adj, seeds, cand, alpha, n, T, R, var_curr=var_curr)

                    row = feat + [dvar, g_name, alpha, step, cand]
                    rows.append(row)

                    if dvar > best_dvar:
                        best_dvar = dvar
                        best_cand = cand

                # Add the best candidate to seed set
                if best_cand is not None:
                    seeds.append(best_cand)

                elapsed = time.time() - t0
                print(f'    step {step+1:3d}/{k}  '
                      f'evaluated {len(top_c)} candidates  '
                      f'best ΔVar={best_dvar:.6f}  '
                      f't={elapsed:.1f}s')

    # Write CSV
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f'\n  Training data saved: {output_path}')
    print(f'  Total samples: {len(rows)}')
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Generate ML training data')
    p.add_argument('--alpha', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    p.add_argument('--k',     type=int, default=25,
                   help='Number of greedy steps to simulate')
    p.add_argument('--T',     type=int, default=6)
    p.add_argument('--R',     type=int, default=80,
                   help='MC samples for probability estimation')
    p.add_argument('--C',     type=int, default=30,
                   help='Top-C candidates to evaluate ΔVar for per step')
    p.add_argument('--max_n', type=int, default=5000,
                   help='Skip graphs larger than this')
    p.add_argument('--output', type=str, default='attempt2/ml_data/training_data.csv')
    p.add_argument('--include_synthetic', action='store_true',
                   help='Also include HICHBA synthetic graphs')
    args = p.parse_args()

    print('=' * 70)
    print('  ML Training Data Generator')
    print(f'  alpha={args.alpha}  k={args.k}  T={args.T}  R={args.R}  C={args.C}')
    print('=' * 70)

    # Load real-world graphs
    graphs = {}
    datasets_dir = 'Social_Network'
    if os.path.exists(datasets_dir):
        for fname in sorted(os.listdir(datasets_dir)):
            if not fname.endswith('.txt'):
                continue
            g_name = fname.replace('.txt', '')
            path = os.path.join(datasets_dir, fname)
            nodes, adj, n = load_snap_edgelist(path)
            if args.max_n > 0 and n > args.max_n:
                print(f'  Skipping {g_name} ({n} nodes > max_n={args.max_n})')
                continue
            graphs[g_name] = (nodes, adj, n)
            print(f'  Loaded {g_name:<20s}  n={n}')

    # Optionally include synthetic graphs
    if args.include_synthetic:
        synth_dir = 'Synthetic Networks'
        if os.path.exists(synth_dir):
            synth = load_hichba_graphs(synth_dir, max_n=min(args.max_n, 2000))
            graphs.update(synth)
            for name in synth:
                print(f'  Loaded {name:<20s}  n={synth[name][2]}')

    if not graphs:
        print('ERROR: No graphs found.')
        sys.exit(1)

    generate_training_data(
        graphs, args.alpha, args.k, args.T, args.R, args.C, args.output
    )

    print('\nDone!')


if __name__ == '__main__':
    main()
