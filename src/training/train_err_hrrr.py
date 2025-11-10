"""
Upgraded HRRR Error Modeling Training Pipeline for NYC Temperature Prediction

This script implements a comprehensive training pipeline with:
- Feature engineering integration
- Multiple LightGBM variants (Huber and L1 objectives)
- Seasonal specialist models
- Ensemble predictions
- Isotonic calibration
- Comprehensive metrics and artifacts
- Ablation study

Usage:
    python -m src.training.train_err_hrrr --output_dir models/nyc_err_hrrr_v2
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
from sklearn.isotonic import IsotonicRegression
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


def train_lightgbm_huber(X_train, y_train, X_val, y_val, seed=42):
    """Train LightGBM with Huber objective."""
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

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=40000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=600), lgb.log_evaluation(period=1000)]
    )

    return model


def train_lightgbm_l1(X_train, y_train, X_val, y_val, seed=42):
    """Train LightGBM with L1 (MAE) objective and extra trees."""
    print("\n  Training LightGBM with L1 objective and extra trees...")

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

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=20000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=400), lgb.log_evaluation(period=1000)]
    )

    return model


def train_seasonal_specialist(X_train, y_train, X_val, y_val, months, model_name, seed=42):
    """Train a seasonal specialist model for specific months."""
    print(f"\n  Training seasonal specialist for months {months}...")

    # Filter training data to specific months
    train_mask = X_train['month'].isin(months)
    val_mask = X_val['month'].isin(months)

    if train_mask.sum() == 0:
        print(f"    Warning: No training samples for months {months}")
        return None

    X_train_season = X_train[train_mask]
    y_train_season = y_train[train_mask]
    X_val_season = X_val[val_mask] if val_mask.sum() > 0 else X_train_season.iloc[:100]
    y_val_season = y_val[val_mask] if val_mask.sum() > 0 else y_train_season.iloc[:100]

    # Use Huber objective for specialists
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

    feature_cols = [c for c in X_train_season.columns if c not in ['month', 'month_cat']]
    train_data = lgb.Dataset(X_train_season[feature_cols], label=y_train_season)
    val_data = lgb.Dataset(X_val_season[feature_cols], label=y_val_season, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=40000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=600), lgb.log_evaluation(period=500)]
    )

    print(f"    Specialist trained on {train_mask.sum()} samples")
    return model


def blend_predictions(months, y_global, y_specialist_apr_may, y_specialist_oct_nov):
    """Blend global and seasonal predictions based on month."""
    y_blend = y_global.copy()

    if y_specialist_apr_may is not None:
        apr_may_mask = months.isin([4, 5])
        y_blend[apr_may_mask] = 0.6 * y_global[apr_may_mask] + 0.4 * y_specialist_apr_may[apr_may_mask]

    if y_specialist_oct_nov is not None:
        oct_nov_mask = months.isin([10, 11])
        y_blend[oct_nov_mask] = 0.6 * y_global[oct_nov_mask] + 0.4 * y_specialist_oct_nov[oct_nov_mask]

    return y_blend


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


def run_ablation(df_train, df_val, label_col, baseline_mae, seed=42):
    """Run ablation study to assess feature group importance."""
    print("\n[Ablation Study] Testing feature group importance...")

    feature_groups = {
        'cross_model_spreads': [c for c in df_train.columns if c.startswith('delta_hrrr_')],
        'dewpoint_depression': [c for c in df_train.columns if c.endswith('_dp_dep')],
        'historical_context': [c for c in df_train.columns if any(x in c for x in
                                ['lag', 'trend', 'climo', 'anom', 'prec_yday'])],
        'harmonics_2_3': [c for c in df_train.columns if any(x in c for x in ['doy_sin_2', 'doy_cos_2', 'doy_sin_3', 'doy_cos_3'])],
    }

    ablation_results = []
    exclude_cols = ['date', 'target_knyc_high_c', 'err_hrrr_c', 'err_hrrr_c_wins']
    base_features = [c for c in df_train.columns if c not in exclude_cols]

    for group_name, group_cols in feature_groups.items():
        if not group_cols or not any(c in df_train.columns for c in group_cols):
            print(f"  Skipping {group_name} (no features found)")
            continue

        # Remove this group
        ablation_features = [f for f in base_features if f not in group_cols]
        print(f"\n  Testing without {group_name} ({len(group_cols)} features dropped)...")

        try:
            X_train = df_train[ablation_features]
            y_train = df_train[label_col]
            X_val = df_val[ablation_features]
            y_val = df_val[label_col]

            # Train quick Huber model
            params = {
                'objective': 'huber',
                'alpha': 0.9,
                'huber_delta': 0.5,
                'learning_rate': 0.01,
                'num_leaves': 31,
                'min_data_in_leaf': 50,
                'verbosity': -1,
                'seed': seed,
            }

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=0)]
            )

            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            delta_mae = mae - baseline_mae

            ablation_results.append({
                'feature_group': group_name,
                'features_dropped': len([c for c in group_cols if c in df_train.columns]),
                'mae_without': round(mae, 4),
                'delta_mae': round(delta_mae, 4),
                'impact': 'negative' if delta_mae > 0 else 'positive'
            })

            print(f"    MAE: {mae:.4f} (Δ = {delta_mae:+.4f})")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    return ablation_results


def main():
    parser = argparse.ArgumentParser(description='Train upgraded HRRR error correction model')
    parser.add_argument('--csv_path', type=str, default='ml_training_data_final.csv',
                        help='Path to training data CSV')
    parser.add_argument('--hist_path', type=str, default='data/data_cleaned_ny.pkl',
                        help='Path to historical data pickle')
    parser.add_argument('--output_dir', type=str, default='models/nyc_err_hrrr_v2',
                        help='Output directory for model and artifacts')
    parser.add_argument('--val_days', type=int, default=60,
                        help='Number of days to hold out for validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--skip_specialists', action='store_true',
                        help='Skip training seasonal specialists')
    parser.add_argument('--skip_ablation', action='store_true',
                        help='Skip ablation study')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"NYC HRRR Error Correction - Upgraded Training Pipeline")
    print(f"{'='*80}\n")

    # Set seeds
    np.random.seed(args.seed)

    # 1. Load data
    print(f"[1/10] Loading data from {args.csv_path}...")
    df = pd.read_csv(args.csv_path, parse_dates=['date']).sort_values('date').reset_index(drop=True)

    # Add month columns if not present
    if 'month' not in df.columns:
        df['month'] = df['date'].dt.month.astype('int16')
    if 'month_cat' not in df.columns:
        df['month_cat'] = df['month'].astype('category')

    print(f"  Loaded {len(df)} rows from {df['date'].min().date()} to {df['date'].max().date()}")

    # Remove rows with NaN target
    initial_rows = len(df)
    if 'err_hrrr_c' in df.columns:
        df = df.dropna(subset=['err_hrrr_c'])
        print(f"  Removed {initial_rows - len(df)} rows with NaN target")

    # 2. Time-ordered split
    print(f"\n[2/10] Creating time-ordered train/validation split (val_days={args.val_days})...")
    if args.val_days <= 0 or args.val_days >= len(df):
        raise ValueError(f"val_days must be in [1, {len(df)-1}]")

    train_df = df.iloc[:-args.val_days].copy()
    val_df = df.iloc[-args.val_days:].copy()

    print(f"  Training:   {len(train_df)} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  Validation: {len(val_df)} rows ({val_df['date'].min().date()} to {val_df['date'].max().date()})")

    # 3. Feature engineering
    print(f"\n[3/10] Running feature engineering pipeline...")
    train_mask = pd.Series([True] * len(train_df) + [False] * len(val_df))
    df_fe, label_col, metadata = build_features(
        df,
        hist_path=args.hist_path if Path(args.hist_path).exists() else None,
        use_winsorize=True,
        train_mask=train_mask
    )

    # Split after feature engineering
    train_df = df_fe.iloc[:-args.val_days].copy()
    val_df = df_fe.iloc[-args.val_days:].copy()

    # Define features
    exclude_cols = ['date', 'target_knyc_high_c', 'err_hrrr_c', 'err_hrrr_c_wins']
    all_features = [c for c in train_df.columns if c not in exclude_cols]

    print(f"\n  Final feature count: {len(all_features)}")

    # Prepare training matrices
    X_train = train_df[all_features]
    y_train = train_df[label_col]
    X_val = val_df[all_features]
    y_val = val_df[label_col]

    # 4. Train global models
    print(f"\n[4/10] Training global models...")

    out_path = Path(args.output_dir)
    if out_path.exists():
        print(f"  Removing existing output directory: {out_path}")
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Train both variants
    model_huber = train_lightgbm_huber(X_train, y_train, X_val, y_val, seed=args.seed)
    model_l1 = train_lightgbm_l1(X_train, y_train, X_val, y_val, seed=args.seed)

    # Evaluate both
    val_pred_huber = model_huber.predict(X_val)
    val_pred_l1 = model_l1.predict(X_val)

    mae_huber = mean_absolute_error(y_val, val_pred_huber)
    mae_l1 = mean_absolute_error(y_val, val_pred_l1)

    print(f"\n  Global model results:")
    print(f"    Huber MAE: {mae_huber:.4f} C")
    print(f"    L1 MAE:    {mae_l1:.4f} C")

    # Choose best global model
    if mae_huber <= mae_l1:
        global_model = model_huber
        global_pred_val = val_pred_huber
        global_name = "huber"
        print(f"  Using Huber model as global (best MAE)")
    else:
        global_model = model_l1
        global_pred_val = val_pred_l1
        global_name = "l1"
        print(f"  Using L1 model as global (best MAE)")

    # Save global model
    global_model.save_model(str(out_path / 'global_model.txt'))

    # 5. Train seasonal specialists
    specialist_apr_may = None
    specialist_oct_nov = None
    val_pred_apr_may = None
    val_pred_oct_nov = None

    if not args.skip_specialists:
        print(f"\n[5/10] Training seasonal specialists...")

        # Apr-May specialist
        specialist_apr_may = train_seasonal_specialist(
            X_train, y_train, X_val, y_val,
            months=[4, 5], model_name="apr_may", seed=args.seed
        )

        # Oct-Nov specialist
        specialist_oct_nov = train_seasonal_specialist(
            X_train, y_train, X_val, y_val,
            months=[10, 11], model_name="oct_nov", seed=args.seed
        )

        # Save specialists
        if specialist_apr_may is not None:
            specialist_path = out_path / 'seasonal' / 'apr_may'
            specialist_path.mkdir(parents=True, exist_ok=True)
            specialist_apr_may.save_model(str(specialist_path / 'model.txt'))

        if specialist_oct_nov is not None:
            specialist_path = out_path / 'seasonal' / 'oct_nov'
            specialist_path.mkdir(parents=True, exist_ok=True)
            specialist_oct_nov.save_model(str(specialist_path / 'model.txt'))

        # Get specialist predictions on full validation set
        feature_cols = [c for c in X_val.columns if c not in ['month', 'month_cat']]
        if specialist_apr_may is not None:
            val_pred_apr_may = specialist_apr_may.predict(X_val[feature_cols])
        if specialist_oct_nov is not None:
            val_pred_oct_nov = specialist_oct_nov.predict(X_val[feature_cols])
    else:
        print(f"\n[5/10] Skipping seasonal specialists")

    # 6. Ensemble predictions
    print(f"\n[6/10] Creating ensemble predictions...")
    val_pred_ensemble = blend_predictions(
        val_df['month'],
        global_pred_val,
        val_pred_apr_may,
        val_pred_oct_nov
    )

    mae_ensemble = mean_absolute_error(y_val, val_pred_ensemble)
    print(f"  Ensemble MAE: {mae_ensemble:.4f} C")

    # 7. Calibration
    print(f"\n[7/10] Fitting isotonic calibration...")

    # Get out-of-fold predictions on training set for calibration
    # Use 5-fold CV
    kf = KFold(n_splits=5, shuffle=False)
    train_pred_oof = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_f_train = X_train.iloc[train_idx]
        y_f_train = y_train.iloc[train_idx]
        X_f_val = X_train.iloc[val_idx]

        # Quick model for OOF predictions
        params = {
            'objective': 'huber' if global_name == 'huber' else 'regression_l1',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'verbosity': -1,
            'seed': args.seed,
        }

        train_data = lgb.Dataset(X_f_train, label=y_f_train)
        model_fold = lgb.train(params, train_data, num_boost_round=500)
        train_pred_oof[val_idx] = model_fold.predict(X_f_val)

    # Fit calibrator
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(train_pred_oof, y_train)

    # Apply calibration
    val_pred_global_cal = calibrator.predict(global_pred_val)
    val_pred_ensemble_cal = calibrator.predict(val_pred_ensemble)

    mae_global_cal = mean_absolute_error(y_val, val_pred_global_cal)
    mae_ensemble_cal = mean_absolute_error(y_val, val_pred_ensemble_cal)

    print(f"  Global calibrated MAE:   {mae_global_cal:.4f} C")
    print(f"  Ensemble calibrated MAE: {mae_ensemble_cal:.4f} C")

    # Save calibrator
    import pickle
    with open(out_path / 'calibrator.pkl', 'wb') as f:
        pickle.dump(calibrator, f)

    # Save calibration metadata
    calib_meta = {
        'oof_mae': round(mean_absolute_error(y_train, train_pred_oof), 4),
        'val_mae_before': round(mean_absolute_error(y_val, global_pred_val), 4),
        'val_mae_after': round(mae_global_cal, 4),
        'improvement': round(mean_absolute_error(y_val, global_pred_val) - mae_global_cal, 4)
    }
    with open(out_path / 'calibration.json', 'w') as f:
        json.dump(calib_meta, f, indent=2)

    # Plot calibration curve
    plot_calibration_curve(y_val.values, global_pred_val, out_path / 'calibration_curve.png')

    # 8. Comprehensive metrics
    print(f"\n[8/10] Computing comprehensive metrics...")

    # Baseline HRRR
    baseline_mae = None
    if 'knyc_hrrr_f06_c' in val_df.columns:
        baseline_mae = mean_absolute_error(val_df['target_knyc_high_c'], val_df['knyc_hrrr_f06_c'])
        print(f"  HRRR baseline MAE: {baseline_mae:.4f} C")

    # Overall metrics
    metrics_report = {
        'training_date': datetime.now().isoformat(),
        'data_file': args.csv_path,
        'hist_file': args.hist_path,
        'seed': args.seed,
        'date_range': {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat(),
            'total_days': len(df),
        },
        'split': {'train_days': len(train_df), 'val_days': len(val_df)},
        'feature_engineering': {
            'added_features': len(metadata['added_features']),
            'dropped_features': len(metadata['dropped_cols']),
            'final_features': len(all_features),
            'label_used': label_col,
        },
        'validation_metrics': {
            'hrrr_baseline': {'mae': round(baseline_mae, 4)} if baseline_mae else None,
            'global_huber': {'mae': round(mae_huber, 4)},
            'global_l1': {'mae': round(mae_l1, 4)},
            'global_best': {'mae': round(mean_absolute_error(y_val, global_pred_val), 4), 'model': global_name},
            'ensemble': {'mae': round(mae_ensemble, 4)},
            'global_calibrated': {'mae': round(mae_global_cal, 4)},
            'ensemble_calibrated': {'mae': round(mae_ensemble_cal, 4)},
        },
    }

    # Monthly metrics
    val_results = val_df[['date', 'month']].copy()
    val_results['y_true'] = y_val.values
    val_results['y_pred_global'] = global_pred_val
    val_results['y_pred_ensemble'] = val_pred_ensemble

    monthly_global = month_metrics(val_results, 'y_true', 'y_pred_global')
    monthly_ensemble = month_metrics(val_results, 'y_true', 'y_pred_ensemble')

    metrics_report['monthly_mae'] = {
        'global': monthly_global,
        'ensemble': monthly_ensemble,
    }

    # Save metrics
    with open(out_path / 'metrics_report.json', 'w') as f:
        json.dump(metrics_report, f, indent=2)

    # 9. Save validation predictions
    print(f"\n[9/10] Saving validation predictions and artifacts...")

    val_output = val_df[['date']].copy()
    val_output['y_true_c'] = val_df['target_knyc_high_c'].values
    val_output['y_pred_c_baseline_hrrr'] = val_df['knyc_hrrr_f06_c'].values if 'knyc_hrrr_f06_c' in val_df.columns else np.nan
    val_output['y_pred_c_global'] = val_df['knyc_hrrr_f06_c'].values + global_pred_val if 'knyc_hrrr_f06_c' in val_df.columns else global_pred_val
    val_output['y_pred_c_ensemble'] = val_df['knyc_hrrr_f06_c'].values + val_pred_ensemble if 'knyc_hrrr_f06_c' in val_df.columns else val_pred_ensemble
    val_output['y_pred_c_global_cal'] = val_df['knyc_hrrr_f06_c'].values + val_pred_global_cal if 'knyc_hrrr_f06_c' in val_df.columns else val_pred_global_cal
    val_output['y_pred_c_ensemble_cal'] = val_df['knyc_hrrr_f06_c'].values + val_pred_ensemble_cal if 'knyc_hrrr_f06_c' in val_df.columns else val_pred_ensemble_cal
    val_output['abs_err_global'] = (val_output['y_true_c'] - val_output['y_pred_c_global']).abs()
    val_output['abs_err_ensemble'] = (val_output['y_true_c'] - val_output['y_pred_c_ensemble']).abs()

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

    # 10. Ablation study
    if not args.skip_ablation:
        print(f"\n[10/10] Running ablation study...")
        ablation_results = run_ablation(
            train_df, val_df, label_col,
            baseline_mae=mean_absolute_error(y_val, global_pred_val),
            seed=args.seed
        )

        if ablation_results:
            ablation_df = pd.DataFrame(ablation_results)
            ablation_df.to_csv(out_path / 'ablation_report.csv', index=False)

            print(f"\n  Ablation results:")
            for result in ablation_results:
                print(f"    {result['feature_group']:<25} Δ MAE = {result['delta_mae']:+.4f} ({result['impact']})")
    else:
        print(f"\n[10/10] Skipping ablation study")

    # Summary
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}\n")
    print(f"Final Validation Results:")
    print(f"  HRRR Baseline:        {baseline_mae:.4f} C" if baseline_mae else "  HRRR Baseline:        N/A")
    print(f"  Global Model:         {mean_absolute_error(y_val, global_pred_val):.4f} C")
    print(f"  Ensemble:             {mae_ensemble:.4f} C")
    print(f"  Global Calibrated:    {mae_global_cal:.4f} C")
    print(f"  Ensemble Calibrated:  {mae_ensemble_cal:.4f} C")
    print(f"\nTarget: MAE ~0.5 C")
    if mae_ensemble_cal < 0.5:
        print(f"TARGET ACHIEVED! 🎯")
    elif baseline_mae and mean_absolute_error(y_val, global_pred_val) < baseline_mae:
        improvement = baseline_mae - mean_absolute_error(y_val, global_pred_val)
        print(f"Improved over baseline by {improvement:.4f} C")

    print(f"\nArtifacts saved to: {out_path}")
    print(f"  - global_model.txt")
    print(f"  - seasonal/apr_may/model.txt")
    print(f"  - seasonal/oct_nov/model.txt")
    print(f"  - calibrator.pkl")
    print(f"  - metrics_report.json")
    print(f"  - validation_predictions.csv")
    print(f"  - feature_importance.csv")
    print(f"  - ablation_report.csv")
    print(f"  - calibration_curve.png")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
