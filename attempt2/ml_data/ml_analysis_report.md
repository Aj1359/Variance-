# ML-Guided Variance Minimisation — Analysis Report

## Model Performance

- **XGBoost R²**: -1.0166 ± 1.6806
- **Linear Approximation R²**: 0.4464

## Feature Importance (SHAP)

| Rank | Feature | Importance | Direction | Weight |
|------|---------|------------|-----------|--------|
| 1 | degree | 0.1691 | ↑ (higher is better) | +0.1691 |
| 2 | deficit | 0.1635 | ↑ (higher is better) | +0.1635 |
| 3 | group_deficit | 0.1611 | ↓ (lower is better) | -0.1611 |
| 4 | dist_to_seeds | 0.1266 | ↓ (lower is better) | -0.1266 |
| 5 | low_p_neighbors | 0.0675 | ↓ (lower is better) | -0.0675 |
| 6 | min_neighbor_p | 0.0644 | ↑ (higher is better) | +0.0644 |
| 7 | avg_neighbor_p | 0.0504 | ↓ (lower is better) | -0.0504 |
| 8 | clustering | 0.0427 | ↑ (higher is better) | +0.0427 |
| 9 | degree_centrality | 0.0376 | ↓ (lower is better) | -0.0376 |
| 10 | norm_degree | 0.0373 | ↑ (higher is better) | +0.0373 |
| 11 | local_density | 0.0293 | ↑ (higher is better) | +0.0293 |
| 12 | shell_reach | 0.0259 | ↓ (lower is better) | -0.0259 |
| 13 | curr_prob | 0.0223 | ↓ (lower is better) | -0.0223 |
| 14 | local_homophily | 0.0024 | ↑ (higher is better) | +0.0024 |

## Derived Scoring Function

```
score(v) = +0.1691 × degree +0.1635 × deficit -0.1611 × group_deficit -0.1266 × dist_to_seeds -0.0675 × low_p_neighbors +0.0644 × min_neighbor_p -0.0504 × avg_neighbor_p
```

Only features with importance > 0.05 are included.
Positive weight = higher value of this feature leads to better ΔVar.
