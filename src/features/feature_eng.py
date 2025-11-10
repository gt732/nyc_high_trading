"""
Feature Engineering Module for NYC Temperature Error Prediction

This module implements comprehensive feature engineering including:
- Seasonal harmonics (2nd and 3rd order)
- Physics-inspired transforms (dewpoint depression, model spreads)
- Circular wind handling
- Historical regime context from 8-year dataset
- High missingness pruning
- Target winsorization

All features are designed to avoid lookahead bias.
"""

import pickle
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def add_harmonics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 1st harmonic component of day-of-year seasonality.

    Args:
        df: DataFrame with day_of_year column

    Returns:
        DataFrame with added harmonic features
    """
    if 'day_of_year' not in df.columns:
        warnings.warn("day_of_year column not found, skipping harmonics")
        return df

    df = df.copy()
    doy = df['day_of_year'].astype(float)

    # Only 1st harmonic for v3
    df['doy_sin_1'] = np.sin(2 * np.pi * 1 * doy / 365.0)
    df['doy_cos_1'] = np.cos(2 * np.pi * 1 * doy / 365.0)

    return df


def add_physics_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physics-inspired features:
    - Dewpoint depression (temp - dewpoint)
    - Cross-model spreads at matching forecast leads

    Args:
        df: DataFrame with temperature and model forecast columns

    Returns:
        DataFrame with added physics features
    """
    df = df.copy()

    # Dewpoint depression
    if 'knyc_morning_avg_temp_c' in df.columns and 'knyc_morning_avg_dewpoint_c' in df.columns:
        df['knyc_dp_dep'] = df['knyc_morning_avg_temp_c'] - df['knyc_morning_avg_dewpoint_c']

    if 'klga_morning_avg_temp_c' in df.columns and 'klga_morning_avg_dewpoint_c' in df.columns:
        df['klga_dp_dep'] = df['klga_morning_avg_temp_c'] - df['klga_morning_avg_dewpoint_c']

    # Cross-model spreads at equal leads
    for lead in ['05', '06', '07']:
        hrrr_col = f'knyc_hrrr_f{lead}_c'
        gfs_col = f'klga_gfs_f{lead}_c'
        rap_col = f'klga_rap_f{lead}_c'

        if hrrr_col in df.columns and gfs_col in df.columns:
            df[f'delta_hrrr_gfs_{lead}'] = df[hrrr_col] - df[gfs_col]

        if hrrr_col in df.columns and rap_col in df.columns:
            df[f'delta_hrrr_rap_{lead}'] = df[hrrr_col] - df[rap_col]

    return df


def add_lead_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lead slopes for model forecasts.

    Slopes capture temperature change trends:
    - hrrr_slope_57 = (f07 - f05) / 2
    - gfs_slope_57 = (f07 - f05) / 2
    - rap_slope_57 = (f07 - f05) / 2

    Args:
        df: DataFrame with forecast columns

    Returns:
        DataFrame with added slope features
    """
    df = df.copy()

    # HRRR slope
    if 'knyc_hrrr_f05_c' in df.columns and 'knyc_hrrr_f07_c' in df.columns:
        df['hrrr_slope_57'] = (df['knyc_hrrr_f07_c'] - df['knyc_hrrr_f05_c']) / 2.0

    # GFS slope
    if 'klga_gfs_f05_c' in df.columns and 'klga_gfs_f07_c' in df.columns:
        df['gfs_slope_57'] = (df['klga_gfs_f07_c'] - df['klga_gfs_f05_c']) / 2.0

    # RAP slope
    if 'klga_rap_f05_c' in df.columns and 'klga_rap_f07_c' in df.columns:
        df['rap_slope_57'] = (df['klga_rap_f07_c'] - df['klga_rap_f05_c']) / 2.0

    return df


def add_cold_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cold regime indicators and interactions.

    Cold regime: knyc_hrrr_f06_c < 5°C
    Interactions with:
    - Dewpoint depression at both stations
    - Mean sea level pressure

    Args:
        df: DataFrame with forecast and observation columns

    Returns:
        DataFrame with added cold regime features
    """
    df = df.copy()

    # Cold regime flag
    if 'knyc_hrrr_f06_c' in df.columns:
        df['cold'] = (df['knyc_hrrr_f06_c'] < 5.0).astype('int8')

        # Interactions with dewpoint depression
        if 'knyc_dp_dep' in df.columns:
            df['cold_x_dp_knyc'] = df['cold'] * df['knyc_dp_dep']

        if 'klga_dp_dep' in df.columns:
            df['cold_x_dp_klga'] = df['cold'] * df['klga_dp_dep']

        # Interaction with MSLP (mean sea level pressure)
        # Look for MSLP columns
        mslp_cols = [c for c in df.columns if 'mslp' in c.lower() or 'pressure' in c.lower()]
        if mslp_cols:
            # Use the first available pressure column
            df['cold_x_mslp'] = df['cold'] * df[mslp_cols[0]]

    return df


def handle_circular_wind(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle circular wind direction features by:
    - Creating missing flags for wind direction sin/cos
    - Filling NaN with 0.0

    Args:
        df: DataFrame with wind direction sin/cos columns

    Returns:
        DataFrame with wind features handled
    """
    df = df.copy()

    wind_cols = [
        'knyc_obs_wind_dir_sin', 'knyc_obs_wind_dir_cos',
        'klga_obs_wind_dir_sin', 'klga_obs_wind_dir_cos'
    ]

    for col in wind_cols:
        if col in df.columns:
            df[f'{col}_isna'] = df[col].isna().astype('int8')
            df[col] = df[col].fillna(0.0)

    return df


def add_hist_context(main_df: pd.DataFrame, hist_path: Optional[str]) -> pd.DataFrame:
    """
    Add historical regime context from 8-year dataset.

    Features include:
    - Lagged tmax (1, 2, 3 days)
    - 3-day temperature trend
    - Day-of-year climatology
    - 7-day anomaly from climatology
    - Yesterday precipitation indicator

    All features use only past information (no lookahead).

    Args:
        main_df: Main training DataFrame
        hist_path: Path to historical pickle file

    Returns:
        DataFrame with historical features merged
    """
    if hist_path is None or not Path(hist_path).exists():
        warnings.warn(f"Historical data not found at {hist_path}, skipping historical features")
        return main_df

    # Load historical dataset
    try:
        with open(hist_path, 'rb') as f:
            hd = pickle.load(f)
    except Exception as e:
        warnings.warn(f"Could not load historical data: {e}")
        return main_df

    hd = hd.copy()

    # Ensure date column
    if 'date' not in hd.columns:
        warnings.warn("No date column in historical data")
        return main_df

    hd['date'] = pd.to_datetime(hd['date'])
    hd = hd.sort_values('date').reset_index(drop=True)

    # Convert Fahrenheit to Celsius if needed
    # Look for tmax column (could be tmax_avg, tmax, etc.)
    tmax_col = None
    for col in ['tmax_avg', 'tmax', 'temp_max', 'temperature_max']:
        if col in hd.columns:
            tmax_col = col
            break

    if tmax_col is None:
        warnings.warn("No tmax column found in historical data")
        return main_df

    # Check if values are in Fahrenheit (reasonable temp range check)
    sample_val = hd[tmax_col].dropna().iloc[0] if len(hd[tmax_col].dropna()) > 0 else 20
    if sample_val > 50:  # Likely Fahrenheit
        hd['tmax_hist_c'] = (hd[tmax_col] - 32.0) * 5.0 / 9.0
    else:
        hd['tmax_hist_c'] = hd[tmax_col]

    # Similar for tmin
    tmin_col = None
    for col in ['tmin_avg', 'tmin', 'temp_min', 'temperature_min']:
        if col in hd.columns:
            tmin_col = col
            break

    if tmin_col is not None:
        sample_val = hd[tmin_col].dropna().iloc[0] if len(hd[tmin_col].dropna()) > 0 else 10
        if sample_val > 50:
            hd['tmin_hist_c'] = (hd[tmin_col] - 32.0) * 5.0 / 9.0
        else:
            hd['tmin_hist_c'] = hd[tmin_col]

    # Create lagged features (shift pushes data down, so lag1 uses yesterday's value)
    hd['lag1_tmax_c'] = hd['tmax_hist_c'].shift(1)
    hd['lag2_tmax_c'] = hd['tmax_hist_c'].shift(2)
    hd['lag3_tmax_c'] = hd['tmax_hist_c'].shift(3)

    # Temperature trend: mean of first differences over last 3 days
    hd['tmax_trend3_c'] = hd['tmax_hist_c'].diff().rolling(3, min_periods=1).mean()

    # Day-of-year climatology
    hd['doy'] = hd['date'].dt.dayofyear
    climo = hd.groupby('doy')['tmax_hist_c'].median()
    hd['climo_tmax_doy_c'] = hd['doy'].map(climo)

    # 7-day rolling mean anomaly from climatology
    hd['tmax_anom7_c'] = hd['tmax_hist_c'].rolling(7, min_periods=3).mean() - hd['climo_tmax_doy_c']

    # Precipitation yesterday indicator
    prec_cols = ['prec_om', 'prec', 'precipitation', 'precip']
    prec_col = None
    for col in prec_cols:
        if col in hd.columns:
            prec_col = col
            break

    if prec_col is not None:
        hd['prec_yday_bin'] = (hd[prec_col].shift(1) > 0).astype('int8')

    # Select features to merge
    merge_cols = ['date', 'lag1_tmax_c', 'lag2_tmax_c', 'lag3_tmax_c',
                  'tmax_trend3_c', 'climo_tmax_doy_c', 'tmax_anom7_c']
    if prec_col is not None:
        merge_cols.append('prec_yday_bin')

    hd_features = hd[[c for c in merge_cols if c in hd.columns]]

    # Merge on date (left join, no forward fill)
    result = main_df.merge(hd_features, on='date', how='left')

    return result


def prune_high_missingness(df: pd.DataFrame, threshold: float = 0.5) -> Tuple[pd.DataFrame, list]:
    """
    Remove features with missing ratio > threshold.

    Binary missing flags created by this module are preserved.

    Args:
        df: DataFrame with features
        threshold: Maximum allowed missing ratio (default 0.5)

    Returns:
        Tuple of (pruned DataFrame, list of dropped column names)
    """
    df = df.copy()

    # Identify features to check (exclude date and targets)
    exclude_cols = ['date', 'target_knyc_high_c', 'err_hrrr_c']
    check_cols = [c for c in df.columns if c not in exclude_cols]

    # Calculate missing ratios
    missing_ratios = df[check_cols].isnull().sum() / len(df)

    # Identify columns to drop (but preserve _isna flags)
    drop_cols = []
    for col in check_cols:
        if missing_ratios[col] > threshold and not col.endswith('_isna'):
            drop_cols.append(col)

    # Drop columns
    df = df.drop(columns=drop_cols)

    return df, drop_cols


def winsorize_label(df: pd.DataFrame, label_col: str = 'err_hrrr_c',
                    train_mask: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, str]:
    """
    Winsorize the target label at 0.5th and 99.5th percentiles.

    Percentiles are computed only on the training portion if train_mask is provided.

    Args:
        df: DataFrame with target label
        label_col: Name of the target column
        train_mask: Boolean mask indicating training rows (optional)

    Returns:
        Tuple of (DataFrame with winsorized label, winsorized label column name)
    """
    df = df.copy()

    if label_col not in df.columns:
        warnings.warn(f"Label column {label_col} not found")
        return df, label_col

    # Determine which rows to use for percentile calculation
    if train_mask is not None:
        # Reset index to ensure alignment
        if isinstance(train_mask, pd.Series):
            train_mask = train_mask.reset_index(drop=True)
        y = df[label_col].values[train_mask.values]
    else:
        y = df[label_col].values

    # Calculate percentiles
    lo, hi = np.nanpercentile(y, [0.5, 99.5])

    # Create winsorized version
    wins_col = f'{label_col}_wins'
    df[wins_col] = np.clip(df[label_col], lo, hi)

    return df, wins_col


def build_features(df: pd.DataFrame, hist_path: Optional[str] = None,
                  use_winsorize: bool = True,
                  train_mask: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, str, dict]:
    """
    Main feature engineering pipeline (v3 simplified).

    Applies all feature engineering steps in sequence:
    1. Seasonal harmonics (1st order only)
    2. Physics transforms (dewpoint depression, cross-model spreads)
    3. Lead slopes (HRRR, GFS, RAP)
    4. Cold regime features and interactions
    5. Circular wind handling
    6. High missingness pruning (>50%)
    7. Target winsorization (optional)

    Note: Historical context features removed in v3.

    Args:
        df: Input DataFrame
        hist_path: Path to historical pickle file (deprecated, ignored in v3)
        use_winsorize: Whether to winsorize the target (default True)
        train_mask: Boolean mask for training rows (for winsorization)

    Returns:
        Tuple of:
        - Transformed DataFrame
        - Label column name to use for training
        - Metadata dictionary with dropped columns and other info
    """
    metadata = {
        'dropped_cols': [],
        'added_features': [],
        'hist_path': None,  # Not used in v3
        'winsorized': use_winsorize,
        'version': 'v3'
    }

    original_cols = set(df.columns)

    print("Feature Engineering Pipeline (v3 Simplified)")
    print("=" * 80)

    # 1. Harmonics (1st order only)
    print("\n[1/7] Adding seasonal harmonics (1st order)...")
    df = add_harmonics(df)
    new_cols = set(df.columns) - original_cols
    print(f"  Added {len(new_cols)} harmonic features: {sorted(new_cols)}")
    metadata['added_features'].extend(new_cols)
    original_cols = set(df.columns)

    # 2. Physics transforms
    print("\n[2/7] Adding physics-inspired transforms...")
    df = add_physics_transforms(df)
    new_cols = set(df.columns) - original_cols
    print(f"  Added {len(new_cols)} physics features")
    if new_cols:
        print(f"    {sorted(new_cols)}")
    metadata['added_features'].extend(new_cols)
    original_cols = set(df.columns)

    # 3. Lead slopes
    print("\n[3/7] Adding lead slopes...")
    df = add_lead_slopes(df)
    new_cols = set(df.columns) - original_cols
    print(f"  Added {len(new_cols)} slope features")
    if new_cols:
        print(f"    {sorted(new_cols)}")
    metadata['added_features'].extend(new_cols)
    original_cols = set(df.columns)

    # 4. Cold regime features
    print("\n[4/7] Adding cold regime features and interactions...")
    df = add_cold_regime_features(df)
    new_cols = set(df.columns) - original_cols
    print(f"  Added {len(new_cols)} cold regime features")
    if new_cols:
        print(f"    {sorted(new_cols)}")
    metadata['added_features'].extend(new_cols)
    original_cols = set(df.columns)

    # 5. Wind handling
    print("\n[5/7] Handling circular wind features...")
    df = handle_circular_wind(df)
    new_cols = set(df.columns) - original_cols
    print(f"  Added {len(new_cols)} wind missing flags")
    metadata['added_features'].extend(new_cols)
    original_cols = set(df.columns)

    # 6. Prune high missingness
    print("\n[6/7] Pruning features with high missingness...")
    df, dropped = prune_high_missingness(df, threshold=0.5)
    print(f"  Dropped {len(dropped)} features with >50% missing")
    if dropped:
        print(f"    Top dropped: {dropped[:5]}")
    metadata['dropped_cols'] = dropped

    # Drop object dtype columns (can't be used in ML models)
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        object_cols = [c for c in object_cols if c not in ['date']]  # Keep date
        if object_cols:
            print(f"  Dropping {len(object_cols)} object dtype columns: {object_cols}")
            df = df.drop(columns=object_cols)
            metadata['dropped_cols'].extend(object_cols)

    # 7. Winsorize target
    label_col = 'err_hrrr_c'
    if use_winsorize and label_col in df.columns:
        print("\n[7/7] Winsorizing target label...")
        df, label_col = winsorize_label(df, label_col, train_mask)
        print(f"  Using label: {label_col}")
    else:
        print("\n[7/7] Skipping winsorization")

    metadata['label_col'] = label_col

    # Final feature count
    exclude_cols = ['date', 'target_knyc_high_c', 'err_hrrr_c', 'err_hrrr_c_wins',
                    'month', 'month_cat', 'day_of_year', 'cold']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    metadata['final_feature_count'] = len(feature_cols)
    metadata['final_features'] = feature_cols

    print(f"\n{'=' * 80}")
    print(f"Feature engineering complete! (v3)")
    print(f"  Final feature count: {len(feature_cols)}")
    print(f"  Label column: {label_col}")
    print(f"{'=' * 80}\n")

    return df, label_col, metadata


if __name__ == "__main__":
    # Test the feature engineering pipeline
    print("Feature Engineering Module - Test Run")
    print("=" * 80)

    # Load training data
    df = pd.read_csv("ml_training_data_final.csv", parse_dates=['date'])
    print(f"Loaded {len(df)} rows")

    # Run feature engineering
    hist_path = "data/data_cleaned_ny.pkl"
    df_fe, label, meta = build_features(df, hist_path=hist_path)

    print("\nMetadata:")
    print(f"  Added {len(meta['added_features'])} new features")
    print(f"  Dropped {len(meta['dropped_cols'])} high-missing features")
    print(f"  Final feature count: {meta['final_feature_count']}")
    print(f"  Label: {meta['label_col']}")
