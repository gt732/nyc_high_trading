(venv) (base) PS E:\kalshi_nyc_trading> python .\src\training\train_err_hrrr.py

================================================================================
HRRR Error Modeling - LightGBM Training Pipeline
================================================================================

[1/7] Loading data from data/training/ml_training_data_final.csv...
  Removed 1 rows with NaN target values
  Cleaned dataset: 1727 rows
  Loaded 1727 rows spanning 2021-01-01 to 2025-10-30
  Total features: 45 (excluding date and target)
  Features with missing values: 12
    klga_gfs_f08_c: 1228 (71.1%)
    knyc_obs_wind_dir_sin: 667 (38.6%)
    knyc_obs_wind_dir_cos: 667 (38.6%)

[2/7] Creating time-ordered train/validation split...
  Training:   1667 rows (2021-01-01 to 2025-08-31)
  Validation: 60 rows (2025-09-01 to 2025-10-30)

[3/7] Configuring AutoGluon with LightGBM hyperparameters...
  Time limit: 600s
  Early stopping: 200 rounds
  Hyperparameter tuning: Enabled (AutoGluon will search over parameter ranges)

[4/7] Training LightGBM model with AutoGluon...
  This may take several minutes. AutoGluon will show progress below.

  Output directory models\nyc_err_hrrr already exists. Removing it to ensure a fresh fit.
Warning: path already exists! This predictor may overwrite an existing predictor! path="models\nyc_err_hrrr"
Verbosity: 2 (Standard Logging)
=================== System Info ===================
AutoGluon Version:  1.4.0
Python Version:     3.11.5
Operating System:   Windows
Platform Machine:   AMD64
Platform Version:   10.0.19045
CPU Count:          12
Memory Avail:       62.68 GB / 79.93 GB (78.4%)
Disk Space Avail:   458.95 GB / 1907.73 GB (24.1%)
===================================================
Presets specified: ['best_quality']
Stack configuration (auto_stack=True): num_stack_levels=0, num_bag_folds=5, num_bag_sets=1
Beginning AutoGluon training ... Time limit = 600s
AutoGluon will save models to "E:\kalshi_nyc_trading\models\nyc_err_hrrr"
Train Data Rows:    1667
Train Data Columns: 44
Label Column:       err_hrrr_c
Problem Type:       regression
Preprocessing data ...
Using Feature Generators to preprocess the data ...
Fitting AutoMLPipelineFeatureGenerator...
        Available Memory:                    64188.43 MB
        Train Data (Original)  Memory Usage: 0.67 MB (0.0% of available memory)
        Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
        Stage 1 Generators:
                Fitting AsTypeFeatureGenerator...
                        Note: Converting 1 features to boolean dtype as they only contain 2 unique values.
        Stage 2 Generators:
                Fitting FillNaFeatureGenerator...
        Stage 3 Generators:
                Fitting IdentityFeatureGenerator...
                Fitting CategoryFeatureGenerator...
                        Fitting CategoryMemoryMinimizeFeatureGenerator...
        Stage 4 Generators:
                Fitting DropUniqueFeatureGenerator...
        Stage 5 Generators:
                Fitting DropDuplicatesFeatureGenerator...
        Types of features in original data (raw dtype, special dtypes):
                ('category', []) :  1 | ['month_cat']
                ('float', [])    : 37 | ['day_of_year_sin', 'day_of_year_cos', 'knyc_hrrr_f05_c', 'knyc_hrrr_f06_c', 'knyc_hrrr_f07_c', ...]
                ('int', [])      :  5 | ['day_of_year', 'data_quality_knyc_obs_hours', 'data_quality_klga_obs_hours', 'knyc_wind_speed_missing', 'month']
                ('object', [])   :  1 | ['data_quality_missing_models']
        Types of features in processed data (raw dtype, special dtypes):
                ('category', [])  :  2 | ['data_quality_missing_models', 'month_cat']
                ('float', [])     : 37 | ['day_of_year_sin', 'day_of_year_cos', 'knyc_hrrr_f05_c', 'knyc_hrrr_f06_c', 'knyc_hrrr_f07_c', ...]
                ('int', [])       :  4 | ['day_of_year', 'data_quality_knyc_obs_hours', 'data_quality_klga_obs_hours', 'month'] 
                ('int', ['bool']) :  1 | ['knyc_wind_speed_missing']
        0.1s = Fit runtime
        44 features in original data used to generate 44 features in processed data.
        Train Data (Processed) Memory Usage: 0.52 MB (0.0% of available memory)
Data preprocessing and feature engineering runtime = 0.07s ...
AutoGluon will gauge predictive performance using evaluation metric: 'mean_absolute_error'
        This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
        To change this, specify the eval_metric parameter of Predictor()
User-specified model hyperparameters to be fit:
{
        'GBM': [{'num_boost_round': 20000, 'objective': 'regression_l1', 'learning_rate': 0.01, 'num_leaves': 63, 'max_depth': -1, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'min_data_in_leaf': 20, 'reg_alpha': 0.0, 'reg_lambda': 
0.3, 'device_type': 'cpu', 'max_bin': 255, 'verbosity': 1, 'ag_args_fit': {'early_stopping_rounds': 200, 'num_cpus': 2}}],      
}
Fitting 1 L1 models, fit_strategy="sequential" ...
Fitting model: LightGBM_BAG_L1 ... Training model for up to 599.93s of the 599.93s of remaining time.
        Fitting 5 child models (S1F1 - S1F5) | Fitting with ParallelLocalFoldFittingStrategy (5 workers, per: cpus=2, gpus=0, memory=0.04%)
        -0.9482  = Validation score   (-mean_absolute_error)
        51.48s   = Training   runtime
        7.89s    = Validation runtime
Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.00s of the 529.73s of remaining time.
        Ensemble Weights: {'LightGBM_BAG_L1': 1.0}
        -0.9482  = Validation score   (-mean_absolute_error)
        0.01s    = Training   runtime
        0.0s     = Validation runtime
AutoGluon training complete, total runtime = 70.3s ... Best model: WeightedEnsemble_L2 | Estimated inference throughput: 42.3 rows/s (334 batch size)
TabularPredictor saved. To load, use: predictor = TabularPredictor.load("E:\kalshi_nyc_trading\models\nyc_err_hrrr")

  Training complete!

[5/7] Evaluating on validation window...

  Metric                         Value
  ---------------------------------------------
  MAE                            0.6606 C
  RMSE                           0.8182 C
  R2                             0.4489

  Baseline Comparisons:
  ---------------------------------------------
  knyc_hrrr_f06_c
    MAE: 21.5660 C
  knyc_yesterday_high_c
    MAE: 21.9126 C

[6/7] Saving model artifacts...
  Validation predictions saved to models/nyc_err_hrrr\validation_predictions.csv
  Metrics report saved to models/nyc_err_hrrr\metrics_report.json
Computing feature importance via permutation shuffling for 44 features using 60 rows with 1 shuffle sets...
        68.62s  = Expected runtime (68.62s per shuffle set)
        8.55s   = Actual runtime (Completed 1 of 1 shuffle sets)
  Feature importance saved to models/nyc_err_hrrr\feature_importance.csv
  Top 10 Most Important Features:
  ------------------------------------------------------------
   1. knyc_hrrr_f06_c                                     0.27
   2. klga_hrrr_f06_c                                     0.20
   3. klga_hrrr_f05_c                                     0.10
   4. knyc_hrrr_f05_c                                     0.08
   5. klga_morning_avg_dewpoint_c                         0.03
   6. knyc_obs_pressure_mslp_hpa                          0.03
   7. klga_gfs_f07_c                                      0.03
   8. knyc_morning_avg_dewpoint_c                         0.03
   9. klga_gfs_f05_c                                      0.02
  10. klga_obs_dewpoint_c                                 0.01

[7/7] Reconstructing corrected temperature predictions...
  Updated validation predictions with corrected temperatures
TabularPredictor saved. To load, use: predictor = TabularPredictor.load("E:\kalshi_nyc_trading\models\nyc_err_hrrr")

  Final model saved to models/nyc_err_hrrr
  Model is ready for error prediction!

================================================================================
Training Complete!
================================================================================

Validation MAE: 0.6606 C
Validation RMSE: 0.8182 C
Validation R2: 0.4489

In Fahrenheit:
  MAE:  1.1891 F
  RMSE: 1.4728 F

================================================================================
SCORE COMPARISON
================================================================================
Current best MAE:  0.9897 C (1.7815 F)
Your MAE:          0.6606 C (1.1891 F)

********************************************************************************
NEW BEST SCORE! You improved by 0.3291 C (0.5924 F)!
********************************************************************************

Update MODEL_TRAINING_LOG.md with this run!

Phase 1 Goal: MAE < 1.5 C - ACHIEVED

Next steps:
  - Review feature importance in models/nyc_err_hrrr\feature_importance.csv
  - Analyze errors in models/nyc_err_hrrr\validation_predictions.csv
  - Load model for inference: TabularPredictor.load('models/nyc_err_hrrr')

================================================================================