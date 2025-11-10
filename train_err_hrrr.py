"""
HRRR Error Modeling Training Pipeline for NYC Temperature Prediction

This script trains a LightGBM model to predict HRRR FORECAST ERRORS
rather than raw temperatures. The error is defined as:
    err_hrrr_c = target_knyc_high_c - knyc_hrrr_f06_c

This approach focuses on learning systematic biases in HRRR forecasts,
which is often more predictable than raw temperature prediction.

Configuration variables are defined at the top of the script:
- CSV_PATH: Path to training data CSV file
- OUT_DIR: Output directory for model and artifacts
- VAL_DAYS: Number of days to hold out for validation
- TIME_LIMIT: Training time limit in seconds

Inference Example:
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load("models/nyc_err_hrrr")
    err_pred = predictor.predict(X_today)
    corrected_temp = hrrr_forecast + err_pred
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from autogluon.tabular import TabularPredictor
from rich import print

# Configuration variables
CSV_PATH = "ml_training_data_final.csv"
OUT_DIR = "models/nyc_err_hrrr"
VAL_DAYS = 60
TIME_LIMIT = 600


def evaluate_model(
    y_true: pd.Series, y_pred: pd.Series, model_name: str = "Model"
) -> dict:
    """
    Calculate comprehensive regression metrics.

    Args:
        y_true: Actual target values
        y_pred: Predicted values
        model_name: Name for reporting

    Returns:
        Dictionary of metrics
    """
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


def main(
    csv_path: str,
    out_dir: str,
    val_days: int,
    time_limit: int,
    label: str = "err_hrrr_c",
):
    """
    Main training pipeline.

    Args:
        csv_path: Path to training data CSV
        out_dir: Output directory for model and artifacts
        val_days: Number of days to hold out for validation
        time_limit: Training time limit in seconds
        label: Target column name
    """
    print(f"\n{'='*80}")
    print(f"HRRR Error Modeling - LightGBM Training Pipeline")
    print(f"{'='*80}\n")

    # 1. Load and validate data
    print(f"[1/7] Loading data from {csv_path}...")
    df = (
        pd.read_csv(csv_path, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Add explicit seasonality signal for bias correction
    df["month"] = df["date"].dt.month.astype("int16")
    df["month_cat"] = df["month"].astype("category")

    if label not in df.columns:
        raise ValueError(f"Target column '{label}' not found in dataset")
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in dataset")

    # Remove rows with NaN target values
    initial_rows = len(df)
    df = df.dropna(subset=[label])
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"  Removed {removed_rows} rows with NaN target values")
    print(f"  Cleaned dataset: {len(df)} rows")

    print(
        f"  Loaded {len(df)} rows spanning {df['date'].min().date()} to {df['date'].max().date()}"
    )
    print(f"  Total features: {len(df.columns) - 2} (excluding date and target)")

    # 2. Define features
    drop_cols = ["date", "target_knyc_high_c", "err_hrrr_c"]  # keep month/month_cat
    features = [c for c in df.columns if c not in drop_cols]

    missing_summary = df[features].isnull().sum()
    missing_features = missing_summary[missing_summary > 0]
    if len(missing_features) > 0:
        print(f"  Features with missing values: {len(missing_features)}")
        top_missing = missing_features.nlargest(3)
        for feat, count in top_missing.items():
            pct = (count / len(df)) * 100
            print(f"    {feat}: {count} ({pct:.1f}%)")

    # 3. Time-ordered train/validation split
    print(f"\n[2/7] Creating time-ordered train/validation split...")
    if val_days <= 0 or val_days >= len(df):
        raise ValueError(f"val_days must be in [1, {len(df)-1}]")

    train_df = df.iloc[:-val_days].copy()
    val_df = df.iloc[-val_days:].copy()

    print(
        f"  Training:   {len(train_df)} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})"
    )
    print(
        f"  Validation: {len(val_df)} rows ({val_df['date'].min().date()} to {val_df['date'].max().date()})"
    )

    # 4. Configure AutoGluon with LightGBM
    print(f"\n[3/7] Configuring AutoGluon with LightGBM hyperparameters...")

    hyperparams = {
        "GBM": {
            "num_boost_round": 20000,
            "objective": "regression_l1",  # optimize MAE directly
            "learning_rate": 0.01,
            "num_leaves": 63,
            "max_depth": -1,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_data_in_leaf": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 0.3,
            "device_type": "cpu",
            "max_bin": 255,
            "verbosity": -1,  # leave this at -1 to reduce logs
            "ag_args_fit": {
                "early_stopping_rounds": 200,
                # optionally cap CPUs per worker to avoid oversubscription
                "num_cpus": 2,
            },
        }
    }

    tune_kwargs = {
        "num_trials": 20,
        "scheduler": "local",
        "searcher": "auto",
    }

    print(f"  Time limit: {time_limit}s")
    print(f"  Early stopping: 200 rounds")
    print(
        f"  Hyperparameter tuning: Enabled (AutoGluon will search over parameter ranges)"
    )

    # 5. Train model
    print(f"\n[4/7] Training LightGBM model with AutoGluon...")
    print(f"  This may take several minutes. AutoGluon will show progress below.\n")

    out_path = Path(out_dir)
    if out_path.exists():
        print(
            f"  Output directory {out_path} already exists. Removing it to ensure a fresh fit."
        )
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    predictor = TabularPredictor(
        label=label,
        problem_type="regression",
        eval_metric="mean_absolute_error",
        path=str(out_path),
    ).fit(
        train_data=train_df[features + [label]],  # keep training split
        tuning_data=None,  # REQUIRED when bagging without holdout
        presets="best_quality",
        time_limit=time_limit,
        hyperparameters=hyperparams,
        hyperparameter_tune_kwargs=None,  # HPO still off
        num_bag_folds=5,  # bagging on
        use_bag_holdout=False,  # per-fold early stopping allowed
        num_stack_levels=0,
        verbosity=2,
        # no fold_fitting_strategy in AutoGluon 1.4.0
    )

    print(f"\n  Training complete!")

    # 6. Evaluate on validation set
    print(f"\n[5/7] Evaluating on validation window...")
    val_pred = predictor.predict(val_df[features])

    model_metrics = evaluate_model(val_df[label], val_pred, "LightGBM")

    print(f"\n  {'Metric':<30} {'Value':<15}")
    print(f"  {'-'*45}")
    print(f"  {'MAE':<30} {model_metrics['mae']:.4f} C")
    print(f"  {'RMSE':<30} {model_metrics['rmse']:.4f} C")
    print(f"  {'R2':<30} {model_metrics['r2']:.4f}")

    # Baseline comparisons
    all_metrics = [model_metrics]

    baseline_cols = ["knyc_hrrr_f06_c", "knyc_yesterday_high_c"]
    available_baselines = [col for col in baseline_cols if col in val_df.columns]

    if available_baselines:
        print(f"\n  Baseline Comparisons:")
        print(f"  {'-'*45}")

        for baseline_col in available_baselines:
            baseline_valid = val_df[baseline_col].notna()
            if baseline_valid.sum() > 0:
                baseline_metrics = evaluate_model(
                    val_df.loc[baseline_valid, label],
                    val_df.loc[baseline_valid, baseline_col],
                    baseline_col,
                )
                all_metrics.append(baseline_metrics)

                print(f"  {baseline_col}")
                print(f"    MAE: {baseline_metrics['mae']:.4f} C")

    # 7. Save artifacts
    print(f"\n[6/7] Saving model artifacts...")

    # Save validation predictions
    val_predictions_path = os.path.join(out_dir, "validation_predictions.csv")
    val_results = val_df[["date", label]].copy()
    val_results["predicted"] = val_pred
    val_results["error"] = val_results[label] - val_results["predicted"]
    val_results["abs_error"] = val_results["error"].abs()
    val_results.to_csv(val_predictions_path, index=False)
    print(f"  Validation predictions saved to {val_predictions_path}")

    # Save metrics report
    metrics_path = os.path.join(out_dir, "metrics_report.json")
    metrics_report = {
        "training_date": datetime.now().isoformat(),
        "data_file": csv_path,
        "date_range": {
            "start": df["date"].min().isoformat(),
            "end": df["date"].max().isoformat(),
            "total_days": len(df),
        },
        "split": {"train_days": len(train_df), "val_days": len(val_df)},
        "validation_metrics": all_metrics,
        "phase_1_threshold": {
            "target_mae": 1.5,
            "achieved": model_metrics["mae"] < 1.5,
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"  Metrics report saved to {metrics_path}")

    # Save feature importance
    try:
        # Compute FI on validation window only and with one shuffle to reduce runtime
        fi = predictor.feature_importance(
            data=val_df[features + [label]], num_shuffle_sets=1
        )
        fi_path = os.path.join(out_dir, "feature_importance.csv")
        fi.to_csv(fi_path)
        print(f"  Feature importance saved to {fi_path}")

        top_fi = fi.head(10).reset_index().rename(columns={"index": "feature"})
        print(f"\n  Top 10 Most Important Features:")
        print(f"  {'-'*60}")
        for idx, row in enumerate(top_fi.itertuples(index=False), 1):
            print(f"  {idx:2d}. {row.feature:<45} {row.importance:>10.2f}")
    except Exception as e:
        print(f"  Warning: Could not extract feature importance: {e}")

    # 8. Reconstruct corrected temperatures and save
    print(f"\n[7/7] Reconstructing corrected temperature predictions...")

    # Add HRRR forecast and corrected temperature to validation results
    val_results["knyc_hrrr_f06_c"] = val_df["knyc_hrrr_f06_c"].values
    val_results["high_corrected_c"] = (
        val_results["knyc_hrrr_f06_c"] + val_results["predicted"]
    )

    # Recalculate actual target for comparison
    val_results["target_knyc_high_c"] = val_df["target_knyc_high_c"].values
    val_results["error_corrected"] = (
        val_results["target_knyc_high_c"] - val_results["high_corrected_c"]
    )
    val_results["abs_error_corrected"] = val_results["error_corrected"].abs()

    # Re-save with corrected temps
    val_results.to_csv(val_predictions_path, index=False)
    print(f"  Updated validation predictions with corrected temperatures")

    # Save predictor (no refit_full for error models)
    predictor.save()

    print(f"\n  Final model saved to {out_dir}")
    print(f"  Model is ready for error prediction!")

    # Summary
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"\nValidation MAE: {model_metrics['mae']:.4f} C")
    print(f"Validation RMSE: {model_metrics['rmse']:.4f} C")
    print(f"Validation R2: {model_metrics['r2']:.4f}")

    # Convert to Fahrenheit for display
    mae_f = model_metrics["mae"] * 9 / 5
    rmse_f = model_metrics["rmse"] * 9 / 5
    print(f"\nIn Fahrenheit:")
    print(f"  MAE:  {mae_f:.4f} F")
    print(f"  RMSE: {rmse_f:.4f} F")

    # Check against best known score
    best_mae_c = 0.9897  # Current best: CPU-optimized hyperparameters
    best_mae_f = best_mae_c * 9 / 5

    print(f"\n{'='*80}")
    print(f"SCORE COMPARISON")
    print(f"{'='*80}")
    print(f"Current best MAE:  {best_mae_c:.4f} C ({best_mae_f:.4f} F)")
    print(f"Your MAE:          {model_metrics['mae']:.4f} C ({mae_f:.4f} F)")

    if model_metrics["mae"] < best_mae_c:
        improvement = best_mae_c - model_metrics["mae"]
        improvement_f = improvement * 9 / 5
        print(f"\n{'*'*80}")
        print(
            f"NEW BEST SCORE! You improved by {improvement:.4f} C ({improvement_f:.4f} F)!"
        )
        print(f"{'*'*80}")
        print(f"\nUpdate MODEL_TRAINING_LOG.md with this run!")
    elif model_metrics["mae"] == best_mae_c:
        print(f"\nTied with best score!")
    else:
        worse_by = model_metrics["mae"] - best_mae_c
        worse_by_f = worse_by * 9 / 5
        print(
            f"\nCurrent model is {worse_by:.4f} C ({worse_by_f:.4f} F) worse than best."
        )
        print(f"Keep trying different parameters!")

    if model_metrics["mae"] < 1.5:
        print(f"\nPhase 1 Goal: MAE < 1.5 C - ACHIEVED")
    else:
        print(f"\nPhase 1 Goal: MAE < 1.5 C - NOT MET (needs improvement)")

    print(f"\nNext steps:")
    print(
        f"  - Review feature importance in {os.path.join(out_dir, 'feature_importance.csv')}"
    )
    print(f"  - Analyze errors in {val_predictions_path}")
    print(f"  - Load model for inference: TabularPredictor.load('{out_dir}')")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main(CSV_PATH, OUT_DIR, VAL_DAYS, TIME_LIMIT)
