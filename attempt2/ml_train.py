"""
attempt2/ml_train.py
=====================
Train XGBoost model on (features, ΔVar) data, extract SHAP feature importance,
and derive a closed-form scoring function.

Usage:
    python -m attempt2.ml_train
    python -m attempt2.ml_train --data attempt2/ml_data/training_data.csv
"""

import os
import sys
import csv
import json
import argparse
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attempt2.ml_features import FEATURE_NAMES, NUM_FEATURES


def load_training_data(csv_path):
    """Load CSV training data into numpy arrays."""
    X, y = [], []
    meta = []  # (graph, alpha, step, node_id)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            features = [float(row[i]) for i in range(NUM_FEATURES)]
            delta_var = float(row[NUM_FEATURES])
            graph = row[NUM_FEATURES + 1]
            alpha = float(row[NUM_FEATURES + 2])
            step = int(row[NUM_FEATURES + 3])
            node_id = int(row[NUM_FEATURES + 4])

            X.append(features)
            y.append(delta_var)
            meta.append((graph, alpha, step, node_id))

    return np.array(X), np.array(y), meta


def train_model(X, y):
    """Train XGBoost regression model."""
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    print('\n  Cross-validation (5-fold)...')
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f'  R² scores: {[f"{s:.4f}" for s in scores]}')
    print(f'  Mean R²:   {scores.mean():.4f} ± {scores.std():.4f}')

    # Fit on full data
    print('\n  Training on full dataset...')
    model.fit(X, y)

    return model, scores


def compute_shap_values(model, X):
    """Compute SHAP values for feature importance."""
    import shap

    print('\n  Computing SHAP values...')
    explainer = shap.TreeExplainer(model)

    # Use a subsample if dataset is large
    if len(X) > 2000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), 2000, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Normalise to sum to 1
    total = mean_abs_shap.sum()
    normalised = mean_abs_shap / total if total > 0 else mean_abs_shap

    return shap_values, mean_abs_shap, normalised, X_sample


def derive_scoring_function(model, X, y, shap_values, mean_abs_shap, normalised):
    """
    Derive a linear scoring function from SHAP analysis.

    Strategy:
    1. Use SHAP-weighted linear combination as scoring function
    2. Determine sign of each feature's effect from SHAP correlation
    3. Scale weights by feature importance
    """
    from sklearn.linear_model import LinearRegression

    print('\n  Deriving scoring function...')

    # Fit a linear regression to approximate the XGBoost model
    lr = LinearRegression()
    y_pred = model.predict(X)
    lr.fit(X, y_pred)  # Fit to model predictions, not raw y

    coefficients = lr.coef_
    intercept = lr.intercept_
    lr_r2 = lr.score(X, y_pred)

    print(f'  Linear approximation R² (of XGBoost): {lr_r2:.4f}')

    # Build the scoring function
    # Combine SHAP importance with linear regression coefficients
    scoring_weights = {}
    for i, name in enumerate(FEATURE_NAMES):
        # Direction from linear regression, magnitude from SHAP
        sign = 1 if coefficients[i] >= 0 else -1
        weight = normalised[i] * sign
        scoring_weights[name] = {
            'weight': float(weight),
            'shap_importance': float(normalised[i]),
            'linear_coef': float(coefficients[i]),
            'sign': sign,
        }

    return scoring_weights, intercept, lr_r2


def generate_report(scoring_weights, model_scores, lr_r2, mean_abs_shap,
                    normalised, output_dir):
    """Generate analysis report and save derived weights."""

    # Sort features by importance
    sorted_feats = sorted(scoring_weights.items(),
                          key=lambda x: abs(x[1]['shap_importance']),
                          reverse=True)

    # Print report
    print('\n' + '=' * 70)
    print('  FEATURE IMPORTANCE (SHAP)')
    print('=' * 70)
    print(f'  {"Rank":<5} {"Feature":<20} {"Importance":<12} {"Direction":<10} {"Weight":<10}')
    print(f'  {"-"*57}')
    for rank, (name, info) in enumerate(sorted_feats, 1):
        direction = '+' if info['sign'] > 0 else '−'
        print(f'  {rank:<5} {name:<20} {info["shap_importance"]:<12.4f} '
              f'{direction:<10} {info["weight"]:+.4f}')

    print(f'\n  XGBoost cross-val R²:     {model_scores.mean():.4f} ± {model_scores.std():.4f}')
    print(f'  Linear approximation R²:  {lr_r2:.4f}')

    # Derive simplified formula
    print('\n' + '=' * 70)
    print('  DERIVED SCORING FUNCTION')
    print('=' * 70)

    # Top features (importance > 0.05)
    top_feats = [(name, info) for name, info in sorted_feats
                 if info['shap_importance'] > 0.05]

    formula_parts = []
    for name, info in top_feats:
        w = info['weight']
        formula_parts.append(f'{w:+.4f} × {name}')

    formula = 'score(v) = ' + ' '.join(formula_parts)
    print(f'  {formula}')

    # Save weights to JSON
    weights_path = os.path.join(output_dir, 'derived_weights.json')
    with open(weights_path, 'w', encoding='utf-8') as f:
        json.dump({
            'feature_names': FEATURE_NAMES,
            'scoring_weights': scoring_weights,
            'model_r2': float(model_scores.mean()),
            'linear_r2': float(lr_r2),
            'formula': formula,
            'top_features': [(name, info['weight']) for name, info in top_feats],
        }, f, indent=2)

    print(f'\n  Weights saved: {weights_path}')

    # Save the report as markdown
    report_path = os.path.join(output_dir, 'ml_analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# ML-Guided Variance Minimisation — Analysis Report\n\n')
        f.write(f'## Model Performance\n\n')
        f.write(f'- **XGBoost R²**: {model_scores.mean():.4f} ± {model_scores.std():.4f}\n')
        f.write(f'- **Linear Approximation R²**: {lr_r2:.4f}\n\n')
        f.write('## Feature Importance (SHAP)\n\n')
        f.write('| Rank | Feature | Importance | Direction | Weight |\n')
        f.write('|------|---------|------------|-----------|--------|\n')
        for rank, (name, info) in enumerate(sorted_feats, 1):
            direction = '↑ (higher is better)' if info['sign'] > 0 else '↓ (lower is better)'
            f.write(f'| {rank} | {name} | {info["shap_importance"]:.4f} | '
                    f'{direction} | {info["weight"]:+.4f} |\n')
        f.write(f'\n## Derived Scoring Function\n\n')
        f.write(f'```\n{formula}\n```\n\n')
        f.write('Only features with importance > 0.05 are included.\n')
        f.write('Positive weight = higher value of this feature leads to better ΔVar.\n')

    print(f'  Report saved: {report_path}')

    return weights_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Train ML model for variance minimisation')
    p.add_argument('--data', type=str, default='attempt2/ml_data/training_data.csv')
    p.add_argument('--output_dir', type=str, default='attempt2/ml_data')
    args = p.parse_args()

    print('=' * 70)
    print('  ML Model Training for Variance Minimisation')
    print('=' * 70)

    # Load data
    print(f'\n  Loading training data: {args.data}')
    X, y, meta = load_training_data(args.data)
    print(f'  Samples: {len(X)}  Features: {X.shape[1]}')
    print(f'  Target (ΔVar) range: [{y.min():.6f}, {y.max():.6f}]')
    print(f'  Target mean: {y.mean():.6f}  std: {y.std():.6f}')

    # Train
    model, scores = train_model(X, y)
    
    # Save the model
    model_path = os.path.join(args.output_dir, 'xgboost_model.json')
    model.save_model(model_path)
    print(f'  Model saved: {model_path}')

    # SHAP analysis
    shap_values, mean_abs_shap, normalised, X_sample = compute_shap_values(model, X)

    # Derive scoring function
    scoring_weights, intercept, lr_r2 = derive_scoring_function(
        model, X, y, shap_values, mean_abs_shap, normalised)

    # Report
    os.makedirs(args.output_dir, exist_ok=True)
    generate_report(scoring_weights, scores, lr_r2, mean_abs_shap,
                    normalised, args.output_dir)

    print('\n  Done!')


if __name__ == '__main__':
    main()
