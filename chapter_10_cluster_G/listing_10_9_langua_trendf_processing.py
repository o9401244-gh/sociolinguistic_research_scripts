import pandas as pd
import numpy as np

# Load raw time-series frequency dataset directly.
# This method uses a time-series file, not a speaker
# corpus; it does not pass through the Cluster G
# preprocessing block. Prepare trend_raw.csv
# separately.
df = pd.read_csv("trend_raw.csv", encoding='utf-8')
df = df.drop_duplicates()
print(f"Records loaded: {len(df)}")

##### 1. Validate required columns
required = ['Year', 'Variant_Freq']
missing = [c for c in required
           if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}")
df = df.dropna(subset=['Year'])

##### 2. Coerce and validate numeric fields
df['Year'] = pd.to_numeric(
    df['Year'], errors='coerce')
df['Variant_Freq'] = pd.to_numeric(
    df['Variant_Freq'], errors='coerce')
invalid_freq = (
    df['Variant_Freq'].notna()
    & ((df['Variant_Freq'] < 0)
       | (df['Variant_Freq'] > 1)))
if invalid_freq.any():
    print(f"Warning: {invalid_freq.sum()} "
          f"Variant_Freq values outside range 0--1.")

if 'N_Tokens' in df.columns:
    df['N_Tokens'] = pd.to_numeric(
        df['N_Tokens'], errors='coerce')

##### 3. Sort and check temporal regularity
df = df.sort_values('Year').reset_index(drop=True)
years = df['Year'].dropna().astype(int)
year_gaps = years.diff().dropna()
if year_gaps.nunique() > 1:
    print(f"Warning: irregular year spacing detected. "
          f"Gap sizes: "
          f"{sorted(year_gaps.unique().tolist())}")
print(f"Year range: {years.min()} to {years.max()} "
      f"({len(years)} observations)")

##### 4. Create complete year index and interpolate
# Fill gaps in the time series with a complete annual
# index, then interpolate Variant_Freq linearly.
year_index = pd.DataFrame({
    'Year': range(
        int(years.min()), int(years.max()) + 1)})
df = year_index.merge(df, on='Year', how='left')
n_missing = df['Variant_Freq'].isna().sum()
if n_missing > 0:
    df['Variant_Freq'] = (df['Variant_Freq']
                          .interpolate(method='linear'))
    print(f"{n_missing} missing Variant_Freq values "
          f"interpolated.")

##### 5. Save processed time series
df.to_csv(
    "out_G_langua_trendf_processed.csv", index=False)
print("Language trend forecasting data processing "
      "completed.")
