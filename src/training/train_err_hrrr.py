"""
NYC HRRR Error Modeling Training Pipeline v3 (Winter-Focused)

This script implements the v3 training pipeline with:
- Simplified feature engineering (no 8-year historical data)
- Sample weighting (1.25 for winter/cold, 1.0 otherwise)
- Three global models: L1, Huber, Quantile
- Winter specialist (Dec-Feb) for cold regime error reduction
- DOY residual debiasing (without leakage)
- Seasonal P90 metrics

Usage:
    python -m src.training.train_err_hrrr \\
        --output_dir models/nyc_err_hrrr_v3 \\
        --seed 42
"""

import argparse
import json
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.feature_eng import build_features


def evaluate_model(y_true: pd.Series, y_pred: pd.Series, model_name: str = "Model") -> dict:
    """Calculate comprehensive regression metrics."""
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


def month_metrics(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> dict:
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


def seasonal_metrics(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> dict:
    """Calculate MAE and P90 error by season."""
    results = {}

    seasons = {
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'fall': [9, 10, 11]
    }

    for season_name, months in seasons.items():
        mask = df['month'].isin(months)
        if mask.sum() > 0:
            y_true = df.loc[mask, y_true_col]
            y_pred = df.loc[mask, y_pred_col]
            abs_err = np.abs(y_true - y_pred)

            results[season_name] = {
                'mae': round(mean_absolute_error(y_true, y_pred), 4),
                'p90': round(np.percentile(abs_err, 90), 4),
                'n': int(mask.sum())
            }

    return results


def create_sample_weights(df: pd.DataFrame, winter_weight: float = 1.25) -> np.ndarray:
    """
    Create sample weights for training.

    Winter/cold samples get higher weight (1.25), others get 1.0.
    Winter is defined as: month in {12, 1, 2} OR cold==1

    Args:
        df: DataFrame with 'month' and optionally 'cold' columns
        winter_weight: Weight for winter/cold samples (default 1.25)

    Returns:
        Array of sample weights
    """
    weights = np.ones(len(df))

    # Winter months
    if 'month' in df.columns:
        winter_mask = df['month'].isin([12, 1, 2])
        weights[winter_mask] = winter_weight

    # Cold regime
    if 'cold' in df.columns:
        cold_mask = df['cold'] == 1
        weights[cold_mask] = winter_weight

    return weights


def train_lightgbm_l1(X_train, y_train, X_val, y_val, sample_weight=None, seed=42):
    """
    Train LightGBM with L1 (MAE) objective.

    Hyperparameters for v3:
    - extra_trees: True
    - learning_rate: 0.01
    - num_leaves: 31
    - min_data_in_leaf: 60
    - feature_fraction: 0.75
    - bagging_fraction: 0.7
    - reg_lambda: 1.2
    """
    print("\n  Training LightGBM with L1 objective...")

    params = {
        'objective': 'regression_l1',
        'extra_trees': True,
        'learning_rate': 0.01,
        'num_leaves': 31,
        'min_data_in_leaf': 60,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'reg_lambda': 1.2,
        'device_type': 'cpu',
        'verbosity': -1,
        'seed': seed,
        'deterministic': True,
    }

    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=20000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=400), lgb.log_evaluation(period=1000)]
    )

    return model


def train_lightgbm_huber(X_train, y_train, X_val, y_val, sample_weight=None, seed=42):
    """
    Train LightGBM with Huber objective.

    Hyperparameters for v3:
    - alpha: 0.9
    - huber_delta: 0.5
    - learning_rate: 0.005
    - num_leaves: 31
    - min_data_in_leaf: 50
    - feature_fraction: 0.8
    - bagging_fraction: 0.7
    - reg_lambda: 1.0
    """
    print("\n  Training LightGBM with Huber objective...")

    params = {
        'objective': 'huber',
        'alpha': 0.9,
        'huber_delta': 0.5,
        'learning_rate': 0.005,
        'num_leaves': 31,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'reg_lambda': 1.0,
        'device_type': 'cpu',
        'verbosity': -1,
        'seed': seed,
        'deterministic': True,
    }

    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=40000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=600), lgb.log_evaluation(period=1000)]
    )

    return model


def train_lightgbm_quantile(X_train, y_train, X_val, y_val, sample_weight=None, seed=42):
    """
    Train LightGBM with Quantile objective (alpha=0.5 for median).

    Hyperparameters for v3:
    - alpha: 0.5 (median)
    - learning_rate: 0.01
    - num_leaves: 31
    - min_data_in_leaf: 50
    - feature_fraction: 0.8
    - bagging_fraction: 0.7
    - reg_lambda: 1.0
    """
    print("\n  Training LightGBM with Quantile objective...")

    params = {
        'objective': 'quantile',
        'alpha': 0.5,  # Median
        'learning_rate': 0.01,
        'num_leaves': 31,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'reg_lambda': 1.0,
        'device_type': 'cpu',
        'verbosity': -1,
        'seed': seed,
        'deterministic': True,
    }

    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=30000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=500), lgb.log_evaluation(period=1000)]
    )

    return model


def train_winter_specialist(X_train, y_train, X_val, y_val, sample_weight=None, seed=42):
    """
    Train winter specialist for Dec-Feb months.

    Uses Huber objective with winter-focused hyperparameters.
    """
    print(f"\n  Training winter specialist (Dec-Feb)...")

    # Filter to winter months (12, 1, 2)
    train_mask = X_train['month'].isin([12, 1, 2])
    val_mask = X_val['month'].isin([12, 1, 2])

    if train_mask.sum() == 0:
        print(f"    Warning: No winter training samples")
        return None

    X_train_winter = X_train[train_mask]
    y_train_winter = y_train[train_mask]
    weight_winter = sample_weight[train_mask] if sample_weight is not None else None

    X_val_winter = X_val[val_mask] if val_mask.sum() > 0 else X_train_winter.iloc[:50]
    y_val_winter = y_val[val_mask] if val_mask.sum() > 0 else y_train_winter.iloc[:50]

    # Huber objective for winter specialist
    params = {
        'objective': 'huber',
        'alpha': 0.9,
        'huber_delta': 0.5,
        'learning_rate': 0.005,
        'num_leaves': 31,
        'min_data_in_leaf': 30,  # Lower for smaller dataset
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'reg_lambda': 1.0,
        'device_type': 'cpu',
        'verbosity': -1,
        'seed': seed,
        'deterministic': True,
    }

    feature_cols = [c for c in X_train_winter.columns if c not in ['month', 'month_cat']]
    train_data = lgb.Dataset(X_train_winter[feature_cols], label=y_train_winter, weight=weight_winter)
    val_data = lgb.Dataset(X_val_winter[feature_cols], label=y_val_winter, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=40000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=600), lgb.log_evaluation(period=500)]
    )

    print(f"    Specialist trained on {train_mask.sum()} winter samples")
    return model


def apply_doy_debiasing(df_train, df_val, global_preds_train, global_preds_val, label_col, seed=42):
    """
    Apply DOY (day-of-year) residual debiasing without leakage.

    Uses 5-fold out-of-fold predictions to compute residual bias by DOY.

    Args:
        df_train: Training DataFrame with 'day_of_year' column
        df_val: Validation DataFrame with 'day_of_year' column
        global_preds_train: Global model predictions on training set
        global_preds_val: Global model predictions on validation set
        label_col: Target label column name
        seed: Random seed

    Returns:
        Tuple of (debiased_val_preds, doy_bias_map)
    """
    print("\n  Applying DOY residual debiasing (5-fold OOF)...")

    if 'day_of_year' not in df_train.columns:
        print("    Warning: day_of_year not found, skipping debiasing")
        return global_preds_val, {}

    # We already have global predictions, so just compute residuals
    y_train = df_train[label_col].values
    residuals_oof = y_train - global_preds_train

    # Compute median residual by DOY (on training set only)
    doy_residuals = pd.DataFrame({
        'doy': df_train['day_of_year'].values,
        'residual': residuals_oof
    })

    doy_bias = doy_residuals.groupby('doy')['residual'].median().to_dict()

    # Apply debiasing to validation set
    val_doy = df_val['day_of_year'].values
    debiased_preds = global_preds_val.copy()

    for doy, bias in doy_bias.items():
        mask = val_doy == doy
        debiased_preds[mask] += bias

    # Handle DOYs not in training set (use overall median)
    overall_bias = np.median(residuals_oof)
    missing_doy_mask = ~df_val['day_of_year'].isin(doy_bias.keys())
    debiased_preds[missing_doy_mask] += overall_bias

    print(f"    Computed bias for {len(doy_bias)} unique DOYs")
    print(f"    Overall bias: {overall_bias:.4f}")

    return debiased_preds, doy_bias


def blend_winter_predictions(df, global_preds, winter_preds, weight=0.5):
    """
    Blend global and winter specialist predictions.

    For winter months (12, 1, 2) or cold==1:
        blended = weight * global + (1-weight) * winter
    Otherwise:
        blended = global

    Args:
        df: DataFrame with 'month' and optionally 'cold' columns
        global_preds: Global model predictions
        winter_preds: Winter specialist predictions
        weight: Weight for global model (default 0.5)

    Returns:
        Blended predictions
    """
    blended = global_preds.copy()

    if winter_preds is None:
        return blended

    # Identify winter samples
    winter_mask = df['month'].isin([12, 1, 2])
    if 'cold' in df.columns:
        winter_mask = winter_mask | (df['cold'] == 1)

    # Blend for winter samples
    blended[winter_mask] = (weight * global_preds[winter_mask] +
                            (1 - weight) * winter_preds[winter_mask])

    return blended


def plot_calibration_curve(y_true, y_pred, save_path):
    """Plot calibration curve showing predicted vs actual residuals."""
    plt.figure(figsize=(8, 6))

    # Sort by prediction
    sorted_idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[sorted_idx]
    y_true_sorted = y_true[sorted_idx]

    # Bin predictions and calculate mean actual in each bin
    n_bins = 20
    bin_edges = np.percentile(y_pred_sorted, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    bin_means = []

    for i in range(n_bins):
        mask = (y_pred_sorted >= bin_edges[i]) & (y_pred_sorted < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append(y_pred_sorted[mask].mean())
            bin_means.append(y_true_sorted[mask].mean())

    plt.scatter(bin_centers, bin_means, alpha=0.6, s=100, label='Binned actuals')
    plt.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()],
             'k--', lw=2, label='Perfect calibration')
    plt.xlabel('Predicted Error (C)')
    plt.ylabel('Actual Error (C)')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train v3 winter-focused HRRR error correction model')
    parser.add_argument('--csv_path', type=str, default='ml_training_data_final.csv',
                        help='Path to training data CSV')
    parser.add_argument('--output_dir', type=str, default='models/nyc_err_hrrr_v3',
                        help='Output directory for model and artifacts')
    parser.add_argument('--val_days', type=int, default=60,
                        help='Number of days to hold out for validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--skip_winter_specialist', action='store_true',
                        help='Skip training winter specialist')
    parser.add_argument('--skip_doy_debiasing', action='store_true',
                        help='Skip DOY residual debiasing')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"NYC HRRR Error Correction - v3 Winter-Focused Training")
    print(f"{'='*80}\n")

    # Set seeds
    np.random.seed(args.seed)

    # 1. Load data
    print(f"[1/8] Loading data from {args.csv_path}...")
    df = pd.read_csv(args.csv_path, parse_dates=['date']).sort_values('date').reset_index(drop=True)

    # Add month columns if not present
    if 'month' not in df.columns:
        df['month'] = df['date'].dt.month.astype('int16')
    if 'month_cat' not in df.columns:
        df['month_cat'] = df['month'].astype('category')
    if 'day_of_year' not in df.columns:
        df['day_of_year'] = df['date'].dt.dayofyear.astype('int16')

    print(f"  Loaded {len(df)} rows from {df['date'].min().date()} to {df['date'].max().date()}")

    # Remove rows with NaN target
    initial_rows = len(df)
    if 'err_hrrr_c' in df.columns:
        df = df.dropna(subset=['err_hrrr_c'])
        print(f"  Removed {initial_rows - len(df)} rows with NaN target")

    # 2. Time-ordered split
    print(f"\n[2/8] Creating time-ordered train/validation split (val_days={args.val_days})...")
    if args.val_days <= 0 or args.val_days >= len(df):
        raise ValueError(f"val_days must be in [1, {len(df)-1}]")

    train_df = df.iloc[:-args.val_days].copy()
    val_df = df.iloc[-args.val_days:].copy()

    print(f"  Training:   {len(train_df)} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  Validation: {len(val_df)} rows ({val_df['date'].min().date()} to {val_df['date'].max().date()})")

    # 3. Feature engineering
    print(f"\n[3/8] Running v3 feature engineering pipeline...")
    train_mask = pd.Series([True] * len(train_df) + [False] * len(val_df))
    df_fe, label_col, metadata = build_features(
        df,
        hist_path=None,  # No historical data in v3
        use_winsorize=True,
        train_mask=train_mask
    )

    # Split after feature engineering
    train_df = df_fe.iloc[:-args.val_days].copy()
    val_df = df_fe.iloc[-args.val_days:].copy()

    # Define features (exclude cold from features but keep for sample weights)
    exclude_cols = ['date', 'target_knyc_high_c', 'err_hrrr_c', 'err_hrrr_c_wins', 'cold']
    all_features = [c for c in train_df.columns if c not in exclude_cols]

    print(f"\n  Final feature count: {len(all_features)}")
    print(f"  Using label: {label_col}")

    # Prepare training matrices
    X_train = train_df[all_features]
    y_train = train_df[label_col]
    X_val = val_df[all_features]
    y_val = val_df[label_col]

    # Create sample weights
    print(f"\n  Creating sample weights (winter/cold=1.25, other=1.0)...")
    train_weights = create_sample_weights(train_df, winter_weight=1.25)
    winter_count = (train_weights > 1.0).sum()
    print(f"    Winter/cold samples: {winter_count}/{len(train_weights)} ({100*winter_count/len(train_weights):.1f}%)")

    # 4. Train global models
    print(f"\n[4/8] Training three global models (L1, Huber, Quantile)...")

    out_path = Path(args.output_dir)
    if out_path.exists():
        print(f"  Removing existing output directory: {out_path}")
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Train all three models
    model_l1 = train_lightgbm_l1(X_train, y_train, X_val, y_val, sample_weight=train_weights, seed=args.seed)
    model_huber = train_lightgbm_huber(X_train, y_train, X_val, y_val, sample_weight=train_weights, seed=args.seed)
    model_quantile = train_lightgbm_quantile(X_train, y_train, X_val, y_val, sample_weight=train_weights, seed=args.seed)

    # Get predictions
    val_pred_l1 = model_l1.predict(X_val)
    val_pred_huber = model_huber.predict(X_val)
    val_pred_quantile = model_quantile.predict(X_val)

    # Get training predictions for DOY debiasing
    train_pred_l1 = model_l1.predict(X_train)
    train_pred_huber = model_huber.predict(X_train)
    train_pred_quantile = model_quantile.predict(X_train)

    # Evaluate
    mae_l1 = mean_absolute_error(y_val, val_pred_l1)
    mae_huber = mean_absolute_error(y_val, val_pred_huber)
    mae_quantile = mean_absolute_error(y_val, val_pred_quantile)

    print(f"\n  Global model results:")
    print(f"    L1 MAE:       {mae_l1:.4f} C")
    print(f"    Huber MAE:    {mae_huber:.4f} C")
    print(f"    Quantile MAE: {mae_quantile:.4f} C")

    # Choose best global model
    best_mae = min(mae_l1, mae_huber, mae_quantile)
    if mae_l1 == best_mae:
        global_model = model_l1
        global_pred_val = val_pred_l1
        global_pred_train = train_pred_l1
        global_name = "l1"
    elif mae_huber == best_mae:
        global_model = model_huber
        global_pred_val = val_pred_huber
        global_pred_train = train_pred_huber
        global_name = "huber"
    else:
        global_model = model_quantile
        global_pred_val = val_pred_quantile
        global_pred_train = train_pred_quantile
        global_name = "quantile"

    print(f"  Using {global_name.upper()} model as global (best MAE: {best_mae:.4f} C)")

    # Save all models
    model_l1.save_model(str(out_path / 'model_l1.txt'))
    model_huber.save_model(str(out_path / 'model_huber.txt'))
    model_quantile.save_model(str(out_path / 'model_quantile.txt'))
    global_model.save_model(str(out_path / 'global_model.txt'))

    # 5. Train winter specialist
    winter_specialist = None
    val_pred_winter = None

    if not args.skip_winter_specialist:
        print(f"\n[5/8] Training winter specialist...")
        winter_specialist = train_winter_specialist(
            X_train, y_train, X_val, y_val,
            sample_weight=train_weights, seed=args.seed
        )

        if winter_specialist is not None:
            # Save specialist
            specialist_path = out_path / 'seasonal' / 'winter'
            specialist_path.mkdir(parents=True, exist_ok=True)
            winter_specialist.save_model(str(specialist_path / 'model.txt'))

            # Get predictions on full validation set
            feature_cols = [c for c in X_val.columns if c not in ['month', 'month_cat']]
            val_pred_winter = winter_specialist.predict(X_val[feature_cols])
    else:
        print(f"\n[5/8] Skipping winter specialist")

    # 6. Blend with winter specialist
    print(f"\n[6/8] Blending global and winter specialist predictions...")
    val_pred_blended = blend_winter_predictions(
        val_df, global_pred_val, val_pred_winter, weight=0.5
    )

    mae_blended = mean_absolute_error(y_val, val_pred_blended)
    print(f"  Blended MAE: {mae_blended:.4f} C")

    # 7. Apply DOY residual debiasing
    val_pred_debiased = global_pred_val
    doy_bias_map = {}

    if not args.skip_doy_debiasing:
        print(f"\n[7/8] Applying DOY residual debiasing...")
        val_pred_debiased, doy_bias_map = apply_doy_debiasing(
            train_df, val_df, global_pred_train, global_pred_val, label_col, seed=args.seed
        )

        mae_debiased = mean_absolute_error(y_val, val_pred_debiased)
        print(f"  Debiased MAE: {mae_debiased:.4f} C")

        # Save DOY bias map
        doy_bias_df = pd.DataFrame([
            {'day_of_year': doy, 'bias': bias}
            for doy, bias in doy_bias_map.items()
        ]).sort_values('day_of_year')
        doy_bias_df.to_csv(out_path / 'doy_bias.csv', index=False)

        # Also apply debiasing to blended
        val_pred_blended_debiased = val_pred_blended.copy()
        val_doy = val_df['day_of_year'].values
        for doy, bias in doy_bias_map.items():
            mask = val_doy == doy
            val_pred_blended_debiased[mask] += bias

        mae_blended_debiased = mean_absolute_error(y_val, val_pred_blended_debiased)
        print(f"  Blended+Debiased MAE: {mae_blended_debiased:.4f} C")
    else:
        print(f"\n[7/8] Skipping DOY debiasing")
        mae_blended_debiased = mae_blended
        val_pred_blended_debiased = val_pred_blended

    # 8. Comprehensive metrics and artifacts
    print(f"\n[8/8] Computing comprehensive metrics and saving artifacts...")

    # Baseline HRRR
    baseline_mae = None
    if 'knyc_hrrr_f06_c' in val_df.columns and 'target_knyc_high_c' in val_df.columns:
        baseline_mae = mean_absolute_error(val_df['target_knyc_high_c'], val_df['knyc_hrrr_f06_c'])
        print(f"  HRRR baseline MAE: {baseline_mae:.4f} C")

    # Monthly metrics
    val_results = val_df[['date', 'month']].copy()
    val_results['y_true'] = y_val.values
    val_results['y_pred_global'] = global_pred_val
    val_results['y_pred_blended'] = val_pred_blended_debiased

    monthly_global = month_metrics(val_results, 'y_true', 'y_pred_global')
    monthly_blended = month_metrics(val_results, 'y_true', 'y_pred_blended')

    # Seasonal metrics (with P90)
    seasonal_global = seasonal_metrics(val_results, 'y_true', 'y_pred_global')
    seasonal_blended = seasonal_metrics(val_results, 'y_true', 'y_pred_blended')

    # Add baseline seasonal metrics
    if baseline_mae is not None and 'knyc_hrrr_f06_c' in val_df.columns and 'target_knyc_high_c' in val_df.columns:
        val_results['y_true_abs'] = val_df['target_knyc_high_c'].values
        val_results['y_baseline'] = val_df['knyc_hrrr_f06_c'].values
        seasonal_baseline = seasonal_metrics(val_results, 'y_true_abs', 'y_baseline')
    else:
        seasonal_baseline = {}

    # Save seasonal P90 report
    seasonal_records = []
    for season in ['winter', 'spring', 'summer', 'fall']:
        record = {'season': season}

        if season in seasonal_blended:
            record['MAE'] = seasonal_blended[season]['mae']
            record['P90'] = seasonal_blended[season]['p90']
            record['n'] = seasonal_blended[season]['n']

        if season in seasonal_baseline:
            record['baseline_MAE'] = seasonal_baseline[season]['mae']
            record['baseline_P90'] = seasonal_baseline[season]['p90']

        seasonal_records.append(record)

    seasonal_p90_df = pd.DataFrame(seasonal_records)
    seasonal_p90_df.to_csv(out_path / 'seasonal_p90.csv', index=False)

    print(f"\n  Seasonal P90 metrics:")
    for _, row in seasonal_p90_df.iterrows():
        print(f"    {row['season']:<10} MAE={row.get('MAE', 'N/A'):>6} P90={row.get('P90', 'N/A'):>6} n={row.get('n', 0):>4}")

    # Overall metrics report
    metrics_report = {
        'training_date': datetime.now().isoformat(),
        'version': 'v3',
        'data_file': args.csv_path,
        'seed': args.seed,
        'date_range': {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat(),
            'total_days': len(df),
        },
        'split': {'train_days': len(train_df), 'val_days': len(val_df)},
        'feature_engineering': {
            'version': 'v3',
            'added_features': len(metadata['added_features']),
            'dropped_features': len(metadata['dropped_cols']),
            'final_features': len(all_features),
            'label_used': label_col,
        },
        'validation_metrics': {
            'hrrr_baseline': {'mae': round(baseline_mae, 4)} if baseline_mae else None,
            'global_l1': {'mae': round(mae_l1, 4)},
            'global_huber': {'mae': round(mae_huber, 4)},
            'global_quantile': {'mae': round(mae_quantile, 4)},
            'global_best': {'mae': round(best_mae, 4), 'model': global_name},
            'blended': {'mae': round(mae_blended, 4)},
            'debiased': {'mae': round(mae_debiased, 4)} if not args.skip_doy_debiasing else None,
            'blended_debiased': {'mae': round(mae_blended_debiased, 4)},
        },
        'monthly_mae': {
            'global': monthly_global,
            'blended': monthly_blended,
        },
        'seasonal_metrics': {
            'baseline': seasonal_baseline,
            'global': seasonal_global,
            'blended': seasonal_blended,
        },
    }

    # Save metrics
    with open(out_path / 'metrics_report.json', 'w') as f:
        json.dump(metrics_report, f, indent=2)

    # Save validation predictions
    val_output = val_df[['date']].copy()
    val_output['y_true_c'] = val_df['target_knyc_high_c'].values if 'target_knyc_high_c' in val_df.columns else y_val.values

    if 'knyc_hrrr_f06_c' in val_df.columns:
        val_output['y_pred_c_baseline_hrrr'] = val_df['knyc_hrrr_f06_c'].values
        val_output['y_pred_c_global'] = val_df['knyc_hrrr_f06_c'].values + global_pred_val
        val_output['y_pred_c_blended'] = val_df['knyc_hrrr_f06_c'].values + val_pred_blended_debiased
    else:
        val_output['y_pred_c_baseline_hrrr'] = np.nan
        val_output['y_pred_c_global'] = global_pred_val
        val_output['y_pred_c_blended'] = val_pred_blended_debiased

    val_output['abs_err_global'] = np.abs(val_output['y_true_c'] - val_output['y_pred_c_global'])
    val_output['abs_err_blended'] = np.abs(val_output['y_true_c'] - val_output['y_pred_c_blended'])

    val_output.to_csv(out_path / 'validation_predictions.csv', index=False)

    # Feature importance
    fi = pd.DataFrame({
        'feature': all_features,
        'importance': global_model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    fi.to_csv(out_path / 'feature_importance.csv', index=False)

    print(f"\n  Top 10 important features:")
    for idx, row in fi.head(10).iterrows():
        print(f"    {row['feature']:<40} {row['importance']:>10.0f}")

    # Plot calibration curve
    plot_calibration_curve(y_val.values, global_pred_val, out_path / 'calibration_curve.png')

    # Summary
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}\n")
    print(f"Final Validation Results:")
    if baseline_mae:
        print(f"  HRRR Baseline:         {baseline_mae:.4f} C")
    print(f"  Global Best ({global_name.upper()}):  {best_mae:.4f} C")
    print(f"  Blended+Debiased:      {mae_blended_debiased:.4f} C")

    # Winter performance
    if 'winter' in seasonal_blended:
        winter_mae = seasonal_blended['winter']['mae']
        winter_p90 = seasonal_blended['winter']['p90']
        print(f"\nWinter Performance (Dec-Feb):")
        print(f"  MAE: {winter_mae:.4f} C")
        print(f"  P90: {winter_p90:.4f} C")

        # Check acceptance criteria
        print(f"\nAcceptance Criteria:")
        print(f"  Holdout MAE ≤ 0.66 C:        {'✓' if mae_blended_debiased <= 0.66 else '✗'} ({mae_blended_debiased:.4f} C)")
        print(f"  Winter MAE ≤ 1.20 C:         {'✓' if winter_mae <= 1.20 else '✗'} ({winter_mae:.4f} C)")

        if 'winter' in seasonal_baseline:
            baseline_winter_p90 = seasonal_baseline['winter']['p90']
            p90_improvement = baseline_winter_p90 - winter_p90
            print(f"  Winter P90 improvement ≥ 0.15 C: {'✓' if p90_improvement >= 0.15 else '✗'} ({p90_improvement:+.4f} C)")

    print(f"\nArtifacts saved to: {out_path}")
    print(f"  - model_l1.txt, model_huber.txt, model_quantile.txt")
    print(f"  - global_model.txt")
    print(f"  - seasonal/winter/model.txt")
    print(f"  - doy_bias.csv")
    print(f"  - metrics_report.json")
    print(f"  - seasonal_p90.csv")
    print(f"  - validation_predictions.csv")
    print(f"  - feature_importance.csv")
    print(f"  - calibration_curve.png")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
