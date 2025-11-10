# NYC High Temperature HRRR Error Correction Model v2

Comprehensive temperature error prediction pipeline achieving **0.7051 C MAE** on validation set (Sept-Oct 2025), representing a **16.3% improvement** over the HRRR baseline (0.8471 C).

## Overview

This model predicts systematic errors in HRRR (High-Resolution Rapid Refresh) temperature forecasts for New York City (KNYC). By learning these error patterns, we can correct raw HRRR forecasts to produce more accurate daily high temperature predictions.

### Performance Summary

| Metric | HRRR Baseline | Global Model | Ensemble | Improvement |
|--------|---------------|--------------|----------|-------------|
| **MAE (C)** | 0.8471 | 0.7089 | **0.7051** | 0.1382 C (16.3%) |
| **RMSE (C)** | - | 1.0171 | 1.0088 | - |
| **R²** | - | 0.2937 | 0.3084 | - |

### Rolling Cross-Validation

5-fold rolling time-series evaluation across different seasons:
- Mean MAE: 0.9700 C (±0.3611)
- Range: 0.5195 - 1.4958 C
- Best performance in summer (Aug-Sep 2023: 0.52 C)
- Most challenging in winter (Jan-Feb 2023: 1.50 C)

## Data Sources

### Primary Training Data
- **File**: `ml_training_data_final.csv`
- **Period**: 2021-01-01 to 2025-10-30 (1,727 days)
- **Target**: `err_hrrr_c` = actual_high_temp - hrrr_forecast_f06
- **Features**: 44 base features including:
  - HRRR forecasts at multiple lead times (f05, f06, f07)
  - Observational data (temperature, dewpoint, wind, pressure)
  - GFS and RAP model forecasts (for LGA)
  - Data quality indicators

### Historical Context Data
- **File**: `data/data_cleaned_ny.pkl`
- **Period**: 8-year historical dataset (2013-2020)
- **Purpose**: Climatological context and lagged features
- **Key fields**: Daily tmax, tmin, precipitation

## Feature Engineering

### 1. Seasonal Harmonics (4 features)
Second and third harmonic components of day-of-year:
```python
doy_sin_2, doy_cos_2  # 2nd harmonic (captures semi-annual patterns)
doy_sin_3, doy_cos_3  # 3rd harmonic (captures tri-annual patterns)
```

### 2. Physics-Inspired Transforms (8 features)
- **Dewpoint Depression** (2): `temp - dewpoint` for KNYC and KLGA
- **Cross-Model Spreads** (6): Differences between HRRR, GFS, and RAP at equal lead times
  - `delta_hrrr_gfs_{05,06,07}` = KNYC_HRRR - KLGA_GFS
  - `delta_hrrr_rap_{05,06,07}` = KNYC_HRRR - KLGA_RAP

### 3. Circular Wind Handling (4 features)
Missing flags for wind direction sin/cos components with zero-filling

### 4. Historical Regime Context (7 features)
From 8-year dataset:
- **Lags**: `lag1_tmax_c`, `lag2_tmax_c`, `lag3_tmax_c`
- **Trend**: `tmax_trend3_c` (3-day rolling temperature change)
- **Climatology**: `climo_tmax_doy_c` (median tmax by day-of-year)
- **Anomaly**: `tmax_anom7_c` (7-day rolling anomaly from climatology)
- **Precip indicator**: `prec_yday_bin` (yesterday precipitation flag)

### Total Features
- **Original**: 44
- **Added**: 23
- **Dropped**: 2 (high missingness + object dtype)
- **Final**: 65 features for training

## Model Architecture

### Global Models

#### Model A: Huber Objective (Selected)
```python
objective: huber
alpha: 0.9
huber_delta: 0.5
learning_rate: 0.005
num_leaves: 31
min_data_in_leaf: 50
feature_fraction: 0.8
bagging_fraction: 0.7
reg_lambda: 1.0
early_stopping_rounds: 600
```
- **Training iterations**: 569 (early stopped)
- **Validation MAE**: 0.7089 C

#### Model B: L1 Objective with Extra Trees
```python
objective: regression_l1
extra_trees: True
learning_rate: 0.01
num_leaves: 31
min_data_in_leaf: 60
feature_fraction: 0.75
bagging_fraction: 0.7
reg_lambda: 1.2
early_stopping_rounds: 400
```
- **Training iterations**: 270 (early stopped)
- **Validation MAE**: 0.7302 C

### Seasonal Specialists

#### April-May Specialist
- **Training samples**: 301
- **Objective**: Huber (same as Global A)
- **Iterations**: 40,000 (full)
- **Purpose**: Handle spring transition dynamics

#### October-November Specialist
- **Training samples**: 240
- **Objective**: Huber
- **Iterations**: 446 (early stopped)
- **Purpose**: Handle fall transition dynamics

### Ensemble Strategy

Month-gated weighted average:
```python
if month in [4, 5]:  # Spring
    prediction = 0.6 * global + 0.4 * apr_may_specialist
elif month in [10, 11]:  # Fall
    prediction = 0.6 * global + 0.4 * oct_nov_specialist
else:
    prediction = global
```

**Result**: Ensemble MAE = 0.7051 C (0.5% improvement over global)

### Calibration

Isotonic regression applied to out-of-fold predictions:
- **Calibrated Global MAE**: 0.7099 C
- **Calibrated Ensemble MAE**: 0.7092 C
- Note: Calibration provided minimal benefit in this case

## Top Features (by Gain)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `delta_hrrr_gfs_06` | 21,492 |
| 2 | `knyc_hrrr_f06_c` | 17,689 |
| 3 | `klga_hrrr_f06_c` | 15,545 |
| 4 | `day_of_year_sin` | 6,302 |
| 5 | `delta_hrrr_gfs_07` | 5,924 |
| 6 | `delta_hrrr_gfs_05` | 4,863 |
| 7 | `delta_hrrr_rap_06` | 4,631 |
| 8 | `day_of_year` | 3,385 |
| 9 | `klga_hrrr_f05_c` | 3,132 |
| 10 | `klga_dp_dep` | 1,487 |

**Key Insights**:
- Cross-model spreads (HRRR-GFS differences) are most important
- Base HRRR forecasts still carry significant signal
- Seasonal components (day_of_year) are critical
- Historical lag features have lower importance

## Ablation Study

Impact of feature groups on validation MAE:

| Feature Group | Features Dropped | MAE without | Δ MAE | Impact |
|---------------|------------------|-------------|-------|--------|
| Cross-model spreads | 6 | 0.7139 | +0.0050 | Helpful |
| Dewpoint depression | 2 | 0.7173 | +0.0084 | Helpful |
| Historical context | 7 | 0.6961 | -0.0128 | **Hurts** |
| Harmonics 2&3 | 4 | 0.6912 | -0.0177 | **Hurts** |

**Surprising findings**:
- Historical lag features and 2nd/3rd harmonics actually worsen performance
- This suggests potential overfitting or that these features don't generalize well to recent data
- Cross-model spreads and dewpoint depression provide consistent value

## Artifacts

All files saved in `models/nyc_err_hrrr_v2/`:

### Models
- `global_model.txt` - Best global LightGBM model (Huber objective)
- `seasonal/apr_may/model.txt` - April-May specialist
- `seasonal/oct_nov/model.txt` - October-November specialist
- `calibrator.pkl` - Isotonic calibration transformer

### Evaluation
- `metrics_report.json` - Comprehensive metrics including monthly breakdown
- `validation_predictions.csv` - Per-day predictions with all variants
- `feature_importance.csv` - Feature importance scores
- `ablation_report.csv` - Feature group ablation results
- `rolling_eval.csv` - Rolling cross-validation results
- `rolling_eval.json` - Detailed rolling CV with monthly stats
- `calibration_curve.png` - Calibration diagnostic plot

## Reproducibility

### Training
```bash
python -m src.training.train_err_hrrr \
    --csv_path ml_training_data_final.csv \
    --hist_path data/data_cleaned_ny.pkl \
    --output_dir models/nyc_err_hrrr_v2 \
    --val_days 60 \
    --seed 42
```

**Configuration**:
- Training period: 2021-01-01 to 2025-08-31 (1,667 days)
- Validation period: 2025-09-01 to 2025-10-30 (60 days)
- Random seed: 42
- Deterministic mode: Enabled

### Rolling Evaluation
```bash
python -m src.eval.rolling_eval \
    --output_dir models/nyc_err_hrrr_v2 \
    --train_days 540 \
    --val_days 30 \
    --step_days 100 \
    --num_folds 5 \
    --seed 42
```

## Performance Analysis

### Monthly Breakdown (Validation Period)

September-October 2025:
- **September**: MAE = 0.683 C (9 samples)
- **October**: MAE = 0.714 C (51 samples)

### Rolling Evaluation by Season

| Fold | Period | Season | MAE | Baseline | Improvement |
|------|--------|--------|-----|----------|-------------|
| 1 | Jul-Aug 2022 | Summer | 0.896 | 1.734 | 48.3% |
| 2 | Oct-Nov 2022 | Fall | 1.258 | 1.529 | 17.7% |
| 3 | Jan-Feb 2023 | Winter | 1.496 | 2.288 | 34.6% |
| 4 | May-Jun 2023 | Spring | 0.680 | 0.890 | 23.6% |
| 5 | Aug-Sep 2023 | Summer | **0.520** | 0.997 | **47.9%** |

**Observations**:
- Best performance in summer months (Aug-Sep: 0.52 C)
- Most challenging in winter (Jan-Feb: 1.50 C)
- Consistent improvement over baseline across all seasons

## Limitations & Future Work

### Current Limitations
1. **Target not achieved**: MAE of 0.71 C vs. target of 0.50 C
2. **Historical features**: Surprisingly hurt performance - need investigation
3. **Winter performance**: Higher errors in cold months
4. **Small validation set**: Only 60 days may not capture full seasonal variation

### Recommendations for Improvement
1. **Remove problematic features**: Drop historical context and 2nd/3rd harmonics (could improve MAE by ~0.02 C based on ablation)
2. **Expand validation period**: Use larger holdout to better assess generalization
3. **Investigation winter errors**: Special handling for cold weather patterns
4. **Hyperparameter tuning**: More extensive search for optimal parameters
5. **Additional external data**: Consider adding:
   - Sea surface temperatures
   - Large-scale climate indices (NAO, AO)
   - Urban heat island proxies

### Next Steps
1. Retrain with reduced feature set (drop historical + harmonics 2/3)
2. Investigate winter error patterns
3. Test on 2024-2025 data when available
4. Consider ensemble with different objectives (Quantile, Tweedie)

## References

- Training script: `src/training/train_err_hrrr.py`
- Feature engineering: `src/features/feature_eng.py`
- Rolling evaluation: `src/eval/rolling_eval.py`
- Original baseline: `train_err_hrrr.py` (root)

## Contact

For questions or issues, please refer to the main repository documentation.

---

**Model Version**: v2
**Training Date**: 2025-11-10
**Framework**: LightGBM 4.6.0
**Python**: 3.11
