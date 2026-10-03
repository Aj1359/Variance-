# Attempt 2: Variance Minimisation in Timed Independent Cascade Models

This directory contains the experimental setup and algorithm implementations for solving the **Minimising the Maximum Influence Gap** problem under time-constrained diffusion.

## The Algorithm: Myopic-Hybrid (Ours)

`Myopic-Hybrid` is designed to aggressively target the "True Tail" of isolated network components that drive variance. It overcomes the "False Negative" trap (where standard greedy algorithms get fooled by Monte Carlo simulation noise in dense cores) by incorporating structural tie-breaking.

1. **Primary Metric**: Evaluates the probability `p_i` of reaching each node. Candidates are all nodes tied for the lowest probability.
2. **Structural Tie-Breaker**: Among the candidates, it selects the node with the **lowest degree**. This guarantees that the seed is placed in the genuinely disconnected tail, rather than an unlucky core node.

## Baselines

- **`Myopic`**: Greedy `argmin p_i` — picks the most underserved node each round, breaking ties arbitrarily.
- **`NaiveMyopic`**: One-shot sort by initial `p_i`, no re-estimation after seeds are placed.
- **`Gonzales`**: Farthest-first traversal (k-center). Pure geometric diversity, ignores `p_i` values.

## How to Run Experiments

### 1. Running on Synthetic Data (HICHBA)
The HICHBA synthetic networks test performance across varying levels of homophily (`h=0.2` to `0.8`).
For the full evaluation across different diffusion regimes, we test highly infectious (`alpha=0.3`) and less infectious (`alpha=0.15`, `0.075`) cascades on the full 10,000 node graphs.

```bash
# Fast smoke test
python attempt2/run.py --fast

# Full experiment (runs on all 10k nodes)
python attempt2/run.py --alpha 0.3 0.15 0.075 --k 20 40 60 80 100 --T 8 --R 150 --max_n 0
```
Output results and plots are saved in `attempt2/results/run_<timestamp>/`.

### 2. Running on Real Social Networks (SNAP Datasets)
To run the full suite of experiments on empirical social network edge lists located in the `Social_network` directory:

```bash
# Full experiment (runs on all nodes, bypasses size limit)
python run_social.py --alpha 0.3 0.15 0.075 --k 20 40 60 80 100 --T 8 --R 150 --max_n 0
```
Output results and plots are saved in `results_network/`.

## File Structure
```text
attempt2/
    run.py                 -- CLI entry point for synthetic graphs
    experiment.py          -- experiment orchestration loop
    algorithms.py          -- algorithm registry
    myopic_hybrid.py       -- Implementation of the proposed algorithm
    ic_model.py            -- timed Independent Cascade simulation
    metrics.py             -- variance, JFI, disparity, welfare metrics
    graph_loader.py        -- HICHBA graph loading + subsampling
    plots.py               -- all plot generation

/ (Root)
    run_social.py          -- CLI entry point for SNAP social networks
```
