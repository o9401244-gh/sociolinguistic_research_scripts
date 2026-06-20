import pandas as pd
import numpy as np

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster E block)
df = pd.read_csv("out_E_causal_analys_preprocessed.csv",
                 encoding='utf-8')
print(f"Records loaded: {len(df)}")

##### 1. Encode binary outcome variable
# Outcome_Variant must contain exactly two distinct values.
# The alphabetically second variant is coded as 1
# (the outcome of theoretical interest).
# Adjust this assignment to match the research question.
variants = df['Outcome_Variant'].str.strip().unique()
print(f"Outcome variants detected: {variants.tolist()}")
if len(variants) != 2:
    print(f"Warning: expected exactly 2 variants, "
          f"found {len(variants)}. "
          f"Manual review required.")
variant_labels = sorted(variants)
df['Outcome_Variant'] = (df['Outcome_Variant']
                         .str.strip())
df['Outcome_Binary'] = (
    df['Outcome_Variant'] == variant_labels[1]
).astype(int)
print(f"Outcome encoding: "
      f"'{variant_labels[1]}' = 1, "
      f"'{variant_labels[0]}' = 0")
print(f"Outcome distribution:\n"
      f"{df['Outcome_Binary'].value_counts()}")

##### 2. Normalize and encode predictor variables
# Age: center on mean for interpretability
if 'Age' in df.columns:
    age_mean = df['Age'].mean()
    df['Age_Centered'] = df['Age'] - age_mean
    print(f"Age centered at mean: {age_mean:.2f} years")

# Gender: dummy-code with first category as reference
if 'Gender' in df.columns:
    df['Gender'] = (df['Gender'].str.strip()
                    .str.title())
    gender_dummies = pd.get_dummies(
        df['Gender'], prefix='Gender',
        drop_first=True, dtype=int)
    df = pd.concat([df, gender_dummies], axis=1)
    print(f"Gender dummy columns created: "
          f"{gender_dummies.columns.tolist()}")

# Education: ordinal encoding
if 'Education' in df.columns:
    df['Education'] = (df['Education'].str.strip()
                       .str.lower())
    edu_map = {
        'primary': 1, 'secondary': 2, 'tertiary': 3}
    df['Education_Numeric'] = (
        df['Education'].map(edu_map))
    unrecognized = df['Education_Numeric'].isna()
    if unrecognized.any():
        vals = (df.loc[unrecognized, 'Education']
                .unique().tolist())
        print(f"Warning: {unrecognized.sum()} records "
              f"with unrecognized Education values: "
              f"{vals}")

# Register: dummy-code
if 'Register' in df.columns:
    df['Register'] = (df['Register'].str.strip()
                      .str.lower())
    register_dummies = pd.get_dummies(
        df['Register'], prefix='Register',
        drop_first=True, dtype=int)
    df = pd.concat([df, register_dummies], axis=1)
    print(f"Register dummy columns created: "
          f"{register_dummies.columns.tolist()}")

# Linguistic_Context: dummy-code
if 'Linguistic_Context' in df.columns:
    df['Linguistic_Context'] = (
        df['Linguistic_Context'].str.strip()
        .str.lower())
    lc_dummies = pd.get_dummies(
        df['Linguistic_Context'], prefix='LingCtx',
        drop_first=True, dtype=int)
    df = pd.concat([df, lc_dummies], axis=1)
    print(f"Linguistic_Context dummy columns created: "
          f"{lc_dummies.columns.tolist()}")

# Community: grouping variable; normalize only
if 'Community' in df.columns:
    df['Community'] = (df['Community'].str.strip()
                       .str.title())
    print(f"Communities detected: "
          f"{df['Community'].nunique()}")

##### 3. Check distributional balance by speaker
# Flag speakers with fewer than 5 observations;
# sparse speakers produce unstable random-effects
# estimates.
obs_per_speaker = df.groupby('Speaker_ID').size()
sparse = obs_per_speaker[obs_per_speaker < 5]
if not sparse.empty:
    print(f"Warning: {len(sparse)} speaker(s) with "
          f"fewer than 5 observations. Random-effects "
          f"estimates may be unstable.")
print(f"Median observations per speaker: "
      f"{obs_per_speaker.median():.1f}")
print(f"Speakers with >= 5 observations: "
      f"{(obs_per_speaker >= 5).sum()} of "
      f"{len(obs_per_speaker)}")

##### 4. Check pairwise correlations among
# numeric predictors.
# Flag predictor pairs with |r| > 0.7 as
# collinearity risks.
numeric_preds = [
    p for p in ['Age_Centered', 'Education_Numeric']
    if p in df.columns]
dummy_preds = [
    c for c in df.columns
    if c.startswith(('Gender_', 'Register_',
                     'LingCtx_'))]
all_preds = numeric_preds + dummy_preds
if len(all_preds) >= 2:
    corr = df[all_preds].corr().abs()
    high_corr = [
        (all_preds[i], all_preds[j],
         round(corr.iloc[i, j], 3))
        for i in range(len(all_preds))
        for j in range(i + 1, len(all_preds))
        if corr.iloc[i, j] > 0.7]
    if high_corr:
        print("Warning: high collinearity detected "
              "(|r| > 0.7):")
        for pair in high_corr:
            print(f"  {pair[0]} x {pair[1]}: "
                  f"r = {pair[2]}")
    else:
        print("Collinearity check passed: no predictor "
              "pairs with |r| > 0.7.")

##### 5. Export model-ready dataset
# Drop raw categorical columns now represented
# by dummies.
drop_cols = [
    c for c in [
        'Outcome_Variant', 'Gender', 'Register',
        'Linguistic_Context', 'Age', 'Education']
    if c in df.columns]
df_model = df.drop(columns=drop_cols)
df_model.to_csv(
    "out_E_causal_analys_processed.csv", index=False)
print("Causal analysis of language variation data "
      "processing completed.")
print("Output files:")
print("  out_E_causal_analys_processed.csv  "
      "-- model-ready dataset with binary outcome, "
      "centered/dummy-coded predictors, and grouping "
      "variables")
