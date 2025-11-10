# NY Cleaned Dataset Documentation

## Overview
The `data_cleaned_ny.pkl` file contains the preprocessed weather dataset for New York (Central Park) used in the LSTM automated trading system. This dataset combines data from multiple weather APIs and has been cleaned and prepared for machine learning model training.

## File Information
- **File Location**: `Data/data_cleaned_ny.pkl`
- **File Format**: Pickle (.pkl) - Python serialized data
- **File Size**: ~278 KB
- **Data Structure**: pandas DataFrame

## Dataset Details

### Temporal Coverage
- **Start Date**: January 1, 2016
- **End Date**: March 24, 2024
- **Total Days**: 3,006 days (approximately 8.2 years)

### Dataset Shape
- **Rows**: 3,006 (one row per day)
- **Columns**: 12 (including date and derived features)

## Column Description

| Column Name | Data Type | Description | Non-Null Count | Range/Values |
|-------------|-----------|-------------|----------------|--------------|
| `date` | datetime64[ns] | Date of observation | 3,006/3,006 | 2016-01-01 to 2024-03-24 |
| `tmax_vc` | float64 | Max temperature from Visual Crossing (°F) | 3,006/3,006 | 12.2 to 97.0 |
| `tmax_om` | float64 | Max temperature from Open Meteo (°F) | 3,005/3,006 | 11.29 to 98.59 |
| `tmax_ms` | float64 | Max temperature from MeteoStat (°F) | 1,461/3,006 | 18.5 to 98.24 |
| `tmax_ncei` | float64 | Max temperature from NCEI (°F) | 3,003/3,006 | 13.0 to 98.0 |
| `humi_vc` | float64 | Humidity from Visual Crossing (%) | 3,006/3,006 | 19.5 to 96.7 |
| `prec_om` | float64 | Precipitation from Open Meteo (inches) | 3,006/3,006 | 0.0 to 24.0 |
| `tmin_ms` | float64 | Min temperature from MeteoStat (°F) | 1,461/3,006 | 2.3 to 79.0 |
| `tmin_ncei` | float64 | Min temperature from NCEI (°F) | 3,003/3,006 | -1.0 to 82.0 |
| `day` | int32 | Day of year (1-366) | 3,006/3,006 | 1 to 366 |
| `tmax_avg` | float64 | **Average max temperature** (°F) | 3,006/3,006 | 11.74 to 97.41 |
| `tmin_avg` | float64 | **Average min temperature** (°F) | 1,461/3,006 | 2.3 to 79.0 |

## Key Features for LSTM Training

The final cleaned dataset used for LSTM training focuses on these 5 main features:

1. **Day of Year** (`day`) - Seasonal indicator
2. **Average Maximum Temperature** (`tmax_avg`) - Weighted average from multiple sources
3. **Average Minimum Temperature** (`tmin_avg`) - Average of MeteoStat and NCEI data
4. **Precipitation** (`prec_om`) - From Open Meteo
5. **Humidity** (`humi_vc`) - From Visual Crossing

### Temperature Calculations
- `tmax_avg` = average of available: Visual Crossing + Open Meteo + MeteoStat
- `tmin_avg` = average of available: MeteoStat + NCEI

## How to Load and Use the Dataset

### Basic Loading
```python
import pandas as pd

# Load the dataset
df = pd.read_pickle('Data/data_cleaned_ny.pkl')

# Display basic information
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
```

### Full Data Inspection
```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_pickle('Data/data_cleaned_ny.pkl')

# Basic information
print("=== Dataset Overview ===")
print(f"Shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Total days: {len(df)}")

# Data types
print("\n=== Data Types ===")
print(df.dtypes)

# Missing data analysis
print("\n=== Missing Data Analysis ===")
for col in df.columns:
    missing = df[col].isna().sum()
    percentage = (missing / len(df)) * 100
    print(f"{col}: {missing}/{len(df)} ({percentage:.1f}% missing)")

# Statistical summary
print("\n=== Statistical Summary ===")
print(df.describe())

# Sample values for each column
print("\n=== Sample Values ===")
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        sample_values = df[col].dropna().head(3).tolist()
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"\n{col}:")
        print(f"  Sample: {sample_values}")
        print(f"  Range: {min_val} to {max_val}")
```

### Preparing Data for LSTM Training
```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_pickle('Data/data_cleaned_ny.pkl')

# Select key features for LSTM
features = ['day', 'tmax_avg', 'tmin_avg', 'prec_om', 'humi_vc']
lstm_data = df[features].copy()

# Handle missing tmin_avg (if needed)
# Option 1: Forward fill
lstm_data['tmin_avg'] = lstm_data['tmin_avg'].fillna(method='ffill')

# Option 2: Use only complete rows
complete_data = lstm_data.dropna()

print(f"Complete data points: {len(complete_data)}/{len(lstm_data)}")
print("Features used:", features)
```

### Time Series Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_pickle('Data/data_cleaned_ny.pkl')
df.set_index('date', inplace=True)

# Plot temperature trends
plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(df.index, df['tmax_avg'], label='Max Temp', alpha=0.7)
plt.plot(df.index, df['tmin_avg'], label='Min Temp', alpha=0.7)
plt.title('NY Temperature Trends (2016-2024)')
plt.ylabel('Temperature (°F)')
plt.legend()

# Plot precipitation and humidity
plt.subplot(2, 1, 2)
plt.scatter(df.index, df['prec_om'], alpha=0.5, label='Precipitation')
plt.ylabel('Precipitation (inches)')
plt.twinx()
plt.plot(df.index, df['humi_vc'], 'orange', alpha=0.5, label='Humidity')
plt.ylabel('Humidity (%)')
plt.title('Precipitation and Humidity')
plt.legend()

plt.tight_layout()
plt.show()
```

## Data Quality Notes

### Completeness
- **tmax data**: Nearly complete (99.97% across all sources)
- **tmin data**: Only 48.6% complete due to limited MeteoStat coverage in early years
- **Humidity & Precipitation**: 100% complete

### Temperature Sources
- **Visual Crossing**: Most consistent, full coverage
- **Open Meteo**: Nearly complete (missing 1 day)
- **MeteoStat**: Partial coverage (only 1,461 days available)
- **NCEI**: Nearly complete (missing 3 days)

### Missing Data Strategy
The dataset uses multi-source averaging to maximize data availability and accuracy. Missing values in individual sources don't affect the final averaged features as long as at least one source provides data.

## Usage in Trading System

This dataset is specifically designed for training LSTM models to predict temperature patterns that may be relevant to weather-based trading strategies. The multi-source approach provides robust temperature measurements while the inclusion of humidity and precipitation data offers additional predictive features.

## Related Files
- `merged_df_ny.pkl` - Raw data from all sources (48 columns)
- `data_cleaned_ny_v2.csv` - CSV version of this dataset
- `model_ny.keras` - Trained LSTM model using this data