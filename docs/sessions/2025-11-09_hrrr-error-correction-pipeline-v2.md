# Session: HRRR Error Correction Pipeline v2 Implementation

**Date**: November 9, 2025
**Time**: 10:29 PM - 11:50 PM EST
**Duration**: ~1 hour 21 minutes
**Branch**: `claude/install-requirements-011CUyVzCRN31wvmA6UrrmUC`

## Session Objective

Implement a comprehensive temperature error correction pipeline to achieve validated MAE near 0.5°C for corrected daily high temperatures at KNYC using only data already in the repository, with no external fetches.

## What Was Accomplished

### 1. Environment Setup
- ✅ Updated `requirements.txt` to use CPU-only PyTorch (avoiding large GPU dependencies)
- ✅ Installed all required packages: AutoGluon, LightGBM, pandas, scikit-learn, matplotlib, rich

### 2. Feature Engineering Module (`src/features/feature_eng.py`)

Created comprehensive feature engineering pipeline with:

**Seasonal Harmonics** (4 features)
- 2nd and 3rd order harmonic components of day-of-year
- Captures semi-annual and tri-annual patterns

**Physics-Inspired Transforms** (8 features)
- Dewpoint depression: `temp - dewpoint` for both KNYC and KLGA
- Cross-model spreads: HRRR vs. GFS and RAP at equal lead times (f05, f06, f07)

**Circular Wind Handling** (4 features)
- Missing flags for wind direction sin/cos components
- Zero-filling for NaN values

**Historical Regime Context** (7 features)
- From 8-year dataset (`data/data_cleaned_ny.pkl`)
- Temperature lags (1, 2, 3 days)
- 3-day temperature trend
- Day-of-year climatology
- 7-day rolling anomaly from climatology
- Yesterday precipitation indicator

**Data Quality**
- High missingness pruning (>60% threshold)
- Object dtype column removal
- Target winsorization (0.5th - 99.5th percentiles)

**Final Feature Count**: 65 (44 original + 23 engineered - 2 dropped)

### 3. Upgraded Training Pipeline (`src/training/train_err_hrrr.py`)

**Global Models**
- **Model A (Huber Objective)**: Selected as best
  - Iterations: 569 (early stopped)
  - Validation MAE: 0.7089°C
  - Learning rate: 0.005, num_leaves: 31

- **Model B (L1 Objective with Extra Trees)**
  - Iterations: 270 (early stopped)
  - Validation MAE: 0.7302°C

**Seasonal Specialists**
- **April-May Specialist**
  - Training samples: 301
  - Iterations: 40,000 (full run)
  - Handles spring transition dynamics

- **October-November Specialist**
  - Training samples: 240
  - Iterations: 446 (early stopped)
  - Handles fall transition dynamics

**Ensemble Strategy**
Month-gated weighted average:
```python
if month in [4, 5]:     # Spring
    pred = 0.6 * global + 0.4 * apr_may
elif month in [10, 11]: # Fall
    pred = 0.6 * global + 0.4 * oct_nov
else:
    pred = global
```

**Calibration**
- Isotonic regression on out-of-fold predictions
- Minimal impact observed (calibration provided only 0.001°C change)

**Ablation Study**
Automated testing of feature group importance:
- Cross-model spreads: Helpful (+0.0050°C when removed)
- Dewpoint depression: Helpful (+0.0084°C when removed)
- Historical context: **Harmful** (-0.0128°C improvement when removed)
- Harmonics 2&3: **Harmful** (-0.0177°C improvement when removed)

### 4. Rolling Evaluation Module (`src/eval/rolling_eval.py`)

Implemented 5-fold rolling time-series cross-validation:
- Training window: 540 days (18 months)
- Validation window: 30 days
- Step size: 100 days (samples different seasons)

### 5. Complete Documentation

Created comprehensive `models/nyc_err_hrrr_v2/README.md` with:
- Performance summary and metrics
- Data sources and feature descriptions
- Model architecture and hyperparameters
- Top features analysis
- Ablation study insights
- Reproducibility instructions
- Limitations and future work

## Results Achieved

### Primary Validation Set (Sept-Oct 2025, 60 days)

| Model | MAE (°C) | RMSE (°C) | R² |
|-------|----------|-----------|-----|
| **HRRR Baseline** | 0.8471 | - | - |
| Global (Huber) | 0.7089 | 1.0171 | 0.2937 |
| **Ensemble** | **0.7051** | 1.0088 | 0.3084 |
| Global Calibrated | 0.7099 | 1.0125 | 0.2963 |
| Ensemble Calibrated | 0.7092 | 1.0095 | 0.3073 |

**Improvement over baseline**: 0.1382°C (16.3%)

### Rolling Cross-Validation Results (5 folds)

| Fold | Period | Season | MAE (°C) | Baseline MAE (°C) | Improvement |
|------|--------|--------|----------|-------------------|-------------|
| 1 | Jul-Aug 2022 | Summer | 0.896 | 1.734 | 48.3% |
| 2 | Oct-Nov 2022 | Fall | 1.258 | 1.529 | 17.7% |
| 3 | Jan-Feb 2023 | Winter | 1.496 | 2.288 | 34.6% |
| 4 | May-Jun 2023 | Spring | 0.680 | 0.890 | 23.6% |
| 5 | Aug-Sep 2023 | Summer | **0.520** | 0.997 | **47.9%** |

**Aggregate Statistics**:
- Mean MAE: 0.970 ± 0.361°C
- Range: 0.520 - 1.496°C
- Consistent improvement across all seasons
- Best performance in summer (Aug-Sep: 0.52°C)
- Most challenging in winter (Jan-Feb: 1.50°C)

### Top 10 Most Important Features (by Gain)

1. `delta_hrrr_gfs_06` (21,492) - Cross-model spread at 6h lead
2. `knyc_hrrr_f06_c` (17,689) - Base HRRR forecast
3. `klga_hrrr_f06_c` (15,545) - Nearby station HRRR
4. `day_of_year_sin` (6,302) - Seasonal component
5. `delta_hrrr_gfs_07` (5,924) - Cross-model spread at 7h lead
6. `delta_hrrr_gfs_05` (4,863) - Cross-model spread at 5h lead
7. `delta_hrrr_rap_06` (4,631) - HRRR-RAP spread
8. `day_of_year` (3,385) - Day of year (linear)
9. `klga_hrrr_f05_c` (3,132) - LGA HRRR at 5h lead
10. `klga_dp_dep` (1,487) - LGA dewpoint depression

**Key Insight**: Cross-model spreads (differences between HRRR and GFS/RAP) are the most predictive features, more important than the raw HRRR forecasts themselves.

## Artifacts Generated

All artifacts saved to `models/nyc_err_hrrr_v2/`:

### Models
- `global_model.txt` (37.2 MB) - Best global LightGBM model
- `seasonal/apr_may/model.txt` (138.5 MB) - Spring specialist
- `seasonal/oct_nov/model.txt` (1.3 MB) - Fall specialist
- `calibrator.pkl` - Isotonic calibration transformer

### Evaluation Files
- `metrics_report.json` - Comprehensive metrics with monthly breakdown
- `validation_predictions.csv` - Per-day predictions (all model variants)
- `feature_importance.csv` - Feature importance scores
- `ablation_report.csv` - Feature group impact analysis
- `rolling_eval.csv` - Rolling CV results (tabular)
- `rolling_eval.json` - Rolling CV results (detailed with monthly stats)
- `calibration_curve.png` - Calibration diagnostic plot

### Documentation
- `README.md` - Comprehensive model documentation

## Key Insights & Discoveries

### 1. Cross-Model Spreads Are Critical
The most important features are the differences between HRRR and other models (GFS, RAP), not the raw forecasts themselves. This suggests the model is learning about:
- Systematic biases between different NWP models
- Situations where models disagree (higher uncertainty)
- Model-specific error patterns

### 2. Historical Features Surprisingly Harmful
The ablation study revealed that removing historical context features (lags, climatology, trends) actually **improved** performance by 0.0128°C. This suggests:
- Potential overfitting on historical patterns
- Historical features may not generalize to recent data
- The 8-year dataset might be from a different regime

### 3. Higher-Order Harmonics Also Harmful
Removing 2nd and 3rd harmonic components improved performance by 0.0177°C, suggesting:
- Simple 1st-order seasonality (already in original features) is sufficient
- Higher harmonics may be capturing noise rather than signal
- Simpler is better for this problem

### 4. Seasonal Variation in Performance
- **Best**: Summer (0.52-0.90°C) - stable weather patterns
- **Worst**: Winter (1.50°C) - more variable conditions, potential snow/ice events
- **Moderate**: Spring/Fall transitions (0.68-1.26°C)

### 5. Diminishing Returns from Complexity
- Ensemble improved over global by only 0.0038°C (0.5%)
- Calibration provided minimal benefit
- Seasonal specialists had limited impact on overall performance
- Suggests simpler models might be nearly as effective

## Target Achievement

**Target**: MAE ≈ 0.5°C
**Achieved**: 0.7051°C (ensemble)
**Gap**: 0.2051°C (29% above target)

**Status**: ⚠️ Target not met, but significant progress made

## Recommendations for Reaching 0.5°C Target

Based on this session's findings:

### 1. Simplify Feature Set (Highest Priority)
- **Remove historical context features** → Could improve by ~0.013°C
- **Remove 2nd/3rd harmonics** → Could improve by ~0.018°C
- **Combined potential improvement**: ~0.031°C → MAE ≈ 0.67°C

### 2. Investigate Winter Errors
- Winter MAE (1.50°C) is 3x higher than summer (0.52°C)
- Create winter-specific specialist or features
- Consider snow/ice indicators if available

### 3. Expand Validation Set
- Current validation: Only 60 days (Sept-Oct 2025)
- Limited seasonal coverage
- Recommend: 6-12 months validation for robust assessment

### 4. Hyperparameter Optimization
- Current: Quick manual tuning
- Recommend: Bayesian optimization or grid search
- Focus on learning rate, num_leaves, regularization

### 5. Additional Data Sources (if available)
- Sea surface temperatures (SST)
- Large-scale climate indices (NAO, AO, MJO)
- Urban heat island proxies
- Ensemble spread from HRRR members

## Technical Challenges Resolved

### Challenge 1: Index Alignment in Winsorization
**Issue**: `AssertionError` when using `train_mask` with `.loc` indexing
**Solution**: Reset index on Series and use array-based indexing instead

### Challenge 2: Object Dtype Columns
**Issue**: LightGBM can't handle object dtype columns
**Solution**: Automatic detection and removal in feature engineering pipeline

### Challenge 3: Large CUDA Dependencies
**Issue**: torch with CUDA was downloading 900+ MB of GPU libraries
**Solution**: Switched to CPU-only torch (184 MB), saving bandwidth and install time

## Code Quality & Reproducibility

✅ **Deterministic Training**: Seed=42, deterministic mode enabled
✅ **Comprehensive Logging**: All steps logged with rich formatting
✅ **Modular Architecture**: Separate modules for features, training, evaluation
✅ **Full Documentation**: README with complete reproducibility instructions
✅ **Automated Ablation**: Feature group importance testing
✅ **Version Control**: All changes committed with detailed message

## Commands to Reproduce

```bash
# Install dependencies (CPU-only)
pip install autogluon.tabular[lightgbm]==1.4.0 pandas scikit-learn matplotlib rich

# Run training pipeline
python -m src.training.train_err_hrrr \
    --output_dir models/nyc_err_hrrr_v2 \
    --val_days 60 \
    --seed 42

# Run rolling evaluation
python -m src.eval.rolling_eval \
    --output_dir models/nyc_err_hrrr_v2 \
    --train_days 540 \
    --val_days 30 \
    --step_days 100 \
    --num_folds 5 \
    --seed 42
```

## Git History

**Branch**: `claude/install-requirements-011CUyVzCRN31wvmA6UrrmUC`
**Commit**: `4a34822`

**Commit Message**:
```
feat: Implement comprehensive HRRR error correction pipeline v2

- Add feature engineering module (src/features/feature_eng.py)
  * Seasonal harmonics (2nd/3rd order)
  * Physics transforms (dewpoint depression, cross-model spreads)
  * Historical context from 8-year dataset
  * High missingness pruning and winsorization

- Upgrade training pipeline (src/training/train_err_hrrr.py)
  * Dual LightGBM models (Huber + L1 objectives)
  * Seasonal specialists for Apr-May and Oct-Nov
  * Month-gated ensemble strategy
  * Isotonic calibration
  * Comprehensive ablation study

- Add rolling evaluation (src/eval/rolling_eval.py)
  * 5-fold time-series cross-validation
  * Monthly metrics breakdowns

Results:
- Validation MAE: 0.7051 C (ensemble)
- 16.3% improvement over HRRR baseline (0.8471 C)
- Rolling CV: 0.97 ± 0.36 C across seasons
```

**Files Changed**: 21 files, 782,148 insertions

## Next Session Recommendations

1. **Quick win**: Retrain with simplified features (drop historical + harmonics 2/3)
2. **Deep dive**: Analyze winter error patterns in detail
3. **Expansion**: Test on additional years if data becomes available
4. **Alternative approaches**: Try gradient boosting variants (CatBoost, XGBoost)
5. **Ensemble methods**: Test stacking with diverse base models

## Session Statistics

- **Lines of Code Written**: ~1,500
- **Models Trained**: 7 (2 global + 2 specialists + 3 calibrated variants)
- **Features Engineered**: 23 new features
- **Evaluation Folds**: 5 rolling windows
- **Artifacts Generated**: 13 files
- **Documentation Pages**: 1 comprehensive README (~400 lines)
- **Improvement Over Baseline**: 16.3%
- **Distance to Target**: 0.205°C (29%)

---

**Session Status**: ✅ Complete
**Branch Status**: ✅ Committed and pushed
**Next Actions**: Consider retraining with simplified feature set based on ablation insights
