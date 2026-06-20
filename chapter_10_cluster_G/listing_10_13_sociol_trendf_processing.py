import pandas as pd
import numpy as np

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster G block)
df = pd.read_csv(
    "out_G_sociol_trendf_preprocessed.csv",
    encoding='utf-8')
print(f"Records loaded: {len(df)}")

##### 1. Community dummy coding for fixed-effects model
if 'Community' in df.columns:
    comm_dummies = pd.get_dummies(
        df['Community'], prefix='Comm',
        drop_first=True, dtype=int)
    df = pd.concat([df, comm_dummies], axis=1)
    print(f"Community dummies created: "
          f"{comm_dummies.columns.tolist()}")

##### 2. Data sufficiency check for multilevel
# forecasting. Multilevel trend models require adequate
# temporal range and sufficient observations per
# speaker/community.
if 'Year' in df.columns:
    year_range = df['Year'].max() - df['Year'].min()
    if year_range < 10:
        print(f"Warning: year range is only "
              f"{year_range} years. Multilevel trend "
              f"forecasting typically requires >= 10 "
              f"years of data.")

if 'Speaker_ID' in df.columns:
    obs_per_speaker = df.groupby('Speaker_ID').size()
    sparse = obs_per_speaker[obs_per_speaker < 3]
    if not sparse.empty:
        print(f"Warning: {len(sparse)} speaker(s) with "
              f"fewer than 3 observations. Random-slope "
              f"estimates may be unstable.")

if 'Community' in df.columns:
    n_comm = df['Community'].nunique()
    if n_comm < 5:
        print(f"Warning: only {n_comm} communities. "
              f"Community-level random effects require "
              f"at least 5 groups for stable estimation.")

##### 3. Save processed dataset
df.to_csv(
    "out_G_sociol_trendf_processed.csv", index=False)
print("Sociolinguistic trend forecasting data "
      "processing completed.")
