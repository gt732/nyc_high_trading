"""
Rolling Time-Series Evaluation for HRRR Error Prediction

This script performs rolling window cross-validation to assess
model performance across different seasons and time periods.

Usage:
    python -m src.eval.rolling_eval --output_dir models/nyc_err_hrrr_v2
"""

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.feature_eng import build_features


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return {
        "model": model_name,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "n_samples": len(y_true),
    }


def month_metrics(df, y_true_col, y_pred_col):
    """Calculate MAE by month."""
    if 'month' not in df.columns:
        return {}

    monthly = {}
    for month in range(1, 13):
        mask = df['month'] == month
        if mask.sum() > 0:
            mae = mean_absolute_error(df.loc[mask, y_true_col], df.loc[mask, y_pred_col])
            monthly[f'month_{month:02d}'] = round(mae, 4)

    return monthly


def create_rolling_folds(df, train_days=540, val_days=30, step_days=100, min_folds=5):
    """
    Create rolling time-series folds.

    Args:
        df: DataFrame sorted by date
        train_days: Training window size in days (18 months = 540 days)
        val_days: Validation window size in days
        step_days: Step size between folds (to sample different seasons)
        min_folds: Minimum number of folds to create

    Returns:
        List of (train_indices, val_indices) tuples
    """
    folds = []
    total_days = len(df)

    # Start from position where we have enough training data
    start = train_days

    while start + val_days <= total_days and len(folds) < min_folds:
        train_start = start - train_days
        train_end = start
        val_start = start
        val_end = start + val_days

        train_idx = list(range(train_start, train_end))
        val_idx = list(range(val_start, val_end))

        folds.append((train_idx, val_idx))

        # Move forward
        start += step_days

    # If we don't have enough folds, use a smaller step
    if len(folds) < min_folds:
        folds = []
        step_days = (total_days - train_days - val_days) // (min_folds - 1)
        if step_days < 30:
            step_days = 30

        start = train_days
        while start + val_days <= total_days:
            train_start = start - train_days
            train_end = start
            val_start = start
            val_end = start + val_days

            train_idx = list(range(train_start, train_end))
            val_idx = list(range(val_start, val_end))

            folds.append((train_idx, val_idx))

            start += step_days
            if len(folds) >= min_folds:
                break

    return folds


def main():
    parser = argparse.ArgumentParser(description='Rolling evaluation for HRRR error model')
    parser.add_argument('--csv_path', type=str, default='ml_training_data_final.csv',
                        help='Path to training data CSV')
    parser.add_argument('--hist_path', type=str, default='data/data_cleaned_ny.pkl',
                        help='Path to historical data pickle')
    parser.add_argument('--output_dir', type=str, default='models/nyc_err_hrrr_v2',
                        help='Output directory for results')
    parser.add_argument('--train_days', type=int, default=540,
                        help='Training window size in days (default: 540 = 18 months)')
    parser.add_argument('--val_days', type=int, default=30,
                        help='Validation window size in days')
    parser.add_argument('--step_days', type=int, default=100,
                        help='Step size between folds')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Rolling Time-Series Evaluation")
    print(f"{'='*80}\n")

    # Set seed
    np.random.seed(args.seed)

    # 1. Load data
    print(f"[1/4] Loading data from {args.csv_path}...")
    df = pd.read_csv(args.csv_path, parse_dates=['date']).sort_values('date').reset_index(drop=True)

    # Add month columns if not present
    if 'month' not in df.columns:
        df['month'] = df['date'].dt.month.astype('int16')
    if 'month_cat' not in df.columns:
        df['month_cat'] = df['month'].astype('category')

    print(f"  Loaded {len(df)} rows from {df['date'].min().date()} to {df['date'].max().date()}")

    # Remove rows with NaN target
    if 'err_hrrr_c' in df.columns:
        df = df.dropna(subset=['err_hrrr_c'])

    # 2. Feature engineering
    print(f"\n[2/4] Running feature engineering...")
    df_fe, label_col, metadata = build_features(
        df,
        hist_path=args.hist_path if Path(args.hist_path).exists() else None,
        use_winsorize=False,  # Don't winsorize for rolling eval
        train_mask=None
    )

    # Define features
    exclude_cols = ['date', 'target_knyc_high_c', 'err_hrrr_c', 'err_hrrr_c_wins']
    all_features = [c for c in df_fe.columns if c not in exclude_cols]

    print(f"  Using {len(all_features)} features")

    # 3. Create folds
    print(f"\n[3/4] Creating rolling folds...")
    folds = create_rolling_folds(
        df_fe,
        train_days=args.train_days,
        val_days=args.val_days,
        step_days=args.step_days,
        min_folds=args.num_folds
    )

    print(f"  Created {len(folds)} folds")
    for i, (train_idx, val_idx) in enumerate(folds):
        train_dates = df_fe.iloc[train_idx]['date']
        val_dates = df_fe.iloc[val_idx]['date']
        print(f"    Fold {i+1}: Train {train_dates.min().date()} to {train_dates.max().date()}, "
              f"Val {val_dates.min().date()} to {val_dates.max().date()}")

    # 4. Train and evaluate on each fold
    print(f"\n[4/4] Training and evaluating on each fold...")

    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n  Fold {fold_idx + 1}/{len(folds)}...")

        # Prepare data
        train_df = df_fe.iloc[train_idx]
        val_df = df_fe.iloc[val_idx]

        X_train = train_df[all_features]
        y_train = train_df['err_hrrr_c']  # Use non-winsorized for rolling eval
        X_val = val_df[all_features]
        y_val = val_df['err_hrrr_c']

        # Train Huber model (fast version)
        params = {
            'objective': 'huber',
            'alpha': 0.9,
            'huber_delta': 0.5,
            'learning_rate': 0.01,
            'num_leaves': 31,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 1,
            'reg_lambda': 1.0,
            'verbosity': -1,
            'seed': args.seed,
            'deterministic': True,
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=5000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=300), lgb.log_evaluation(period=0)]
        )

        # Predict
        y_pred = model.predict(X_val)

        # Overall metrics
        metrics = evaluate_model(y_val, y_pred, f"fold_{fold_idx+1}")

        # Monthly metrics
        val_results = val_df[['date', 'month']].copy()
        val_results['y_true'] = y_val.values
        val_results['y_pred'] = y_pred

        monthly = month_metrics(val_results, 'y_true', 'y_pred')

        # Baseline HRRR
        baseline_mae = None
        if 'knyc_hrrr_f06_c' in val_df.columns and 'target_knyc_high_c' in val_df.columns:
            baseline_mae = mean_absolute_error(val_df['target_knyc_high_c'], val_df['knyc_hrrr_f06_c'])

        # Store results
        fold_result = {
            'fold': fold_idx + 1,
            'train_start': train_df['date'].min().isoformat(),
            'train_end': train_df['date'].max().isoformat(),
            'val_start': val_df['date'].min().isoformat(),
            'val_end': val_df['date'].max().isoformat(),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'baseline_mae': round(baseline_mae, 4) if baseline_mae else None,
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'r2': metrics['r2'],
            'monthly_mae': monthly,
        }

        results.append(fold_result)

        print(f"    MAE: {metrics['mae']:.4f} C, RMSE: {metrics['rmse']:.4f} C, R2: {metrics['r2']:.4f}")
        if baseline_mae:
            print(f"    Baseline MAE: {baseline_mae:.4f} C")

    # Aggregate results
    print(f"\n{'='*80}")
    print(f"Rolling Evaluation Summary")
    print(f"{'='*80}\n")

    mae_values = [r['mae'] for r in results]
    rmse_values = [r['rmse'] for r in results]
    r2_values = [r['r2'] for r in results]

    print(f"Overall Statistics (across {len(results)} folds):")
    print(f"  MAE:  mean={np.mean(mae_values):.4f}, std={np.std(mae_values):.4f}, "
          f"min={np.min(mae_values):.4f}, max={np.max(mae_values):.4f}")
    print(f"  RMSE: mean={np.mean(rmse_values):.4f}, std={np.std(rmse_values):.4f}, "
          f"min={np.min(rmse_values):.4f}, max={np.max(rmse_values):.4f}")
    print(f"  R2:   mean={np.mean(r2_values):.4f}, std={np.std(r2_values):.4f}, "
          f"min={np.min(r2_values):.4f}, max={np.max(r2_values):.4f}")

    # Per-fold summary
    print(f"\nPer-Fold Results:")
    print(f"  {'Fold':<6} {'Val Period':<25} {'MAE':<8} {'RMSE':<8} {'R2':<8}")
    print(f"  {'-'*60}")
    for r in results:
        val_period = f"{r['val_start'][:10]} to {r['val_end'][:10]}"
        print(f"  {r['fold']:<6} {val_period:<25} {r['mae']:<8.4f} {r['rmse']:<8.4f} {r['r2']:<8.4f}")

    # Save results
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save detailed results as JSON
    summary = {
        'configuration': {
            'train_days': args.train_days,
            'val_days': args.val_days,
            'step_days': args.step_days,
            'num_folds': len(results),
            'seed': args.seed,
        },
        'aggregate_metrics': {
            'mae_mean': round(np.mean(mae_values), 4),
            'mae_std': round(np.std(mae_values), 4),
            'mae_min': round(np.min(mae_values), 4),
            'mae_max': round(np.max(mae_values), 4),
            'rmse_mean': round(np.mean(rmse_values), 4),
            'rmse_std': round(np.std(rmse_values), 4),
            'r2_mean': round(np.mean(r2_values), 4),
            'r2_std': round(np.std(r2_values), 4),
        },
        'folds': results,
    }

    with open(out_path / 'rolling_eval.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save as CSV
    records = []
    for r in results:
        record = {
            'fold': r['fold'],
            'train_start': r['train_start'],
            'train_end': r['train_end'],
            'val_start': r['val_start'],
            'val_end': r['val_end'],
            'train_samples': r['train_samples'],
            'val_samples': r['val_samples'],
            'baseline_mae': r['baseline_mae'],
            'mae': r['mae'],
            'rmse': r['rmse'],
            'r2': r['r2'],
        }

        # Add monthly metrics as separate columns
        for month, mae in r['monthly_mae'].items():
            record[f'mae_{month}'] = mae

        records.append(record)

    df_results = pd.DataFrame(records)
    df_results.to_csv(out_path / 'rolling_eval.csv', index=False)

    print(f"\nResults saved to:")
    print(f"  {out_path / 'rolling_eval.json'}")
    print(f"  {out_path / 'rolling_eval.csv'}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
