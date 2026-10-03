"""
attempt2/run.py
===============
Entry point for the attempt2 experiment suite.

Quick start:
    cd e:\\research
    python attempt2/run.py --fast           # smoke test (~5 min)
    python attempt2/run.py                  # full run (~hours)
    python attempt2/run.py --alpha 0.3 --k 20 40 --T 6 --R 100

Output goes to:  attempt2/results/
  results.json      — all numerical results
  plots/            — all generated figures

See experiment.py for full argument reference.
"""

import os
import sys

# Add parent directory to path so 'attempt2' package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attempt2.experiment import run_experiment, parse_args

if __name__ == "__main__":
    args = parse_args()

    if args.out_dir == "attempt2/results":
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", f"run_{timestamp}"
        )

    # Fast mode overrides
    if args.fast:
        args.R     = 30
        args.k     = [20, 40]
        args.alpha = [0.3]
        args.max_n = 500
        print("=" * 60)
        print("  FAST MODE (smoke test)")
        print("  R=30, k=[20,40], alpha=[0.3], max_n=500 nodes")
        print("=" * 60)

    run_experiment(args)
