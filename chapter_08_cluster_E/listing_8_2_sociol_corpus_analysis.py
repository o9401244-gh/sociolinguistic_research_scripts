import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Load processed and coded corpus (output of Listing 8.1)
df      = pd.read_csv(
    "out_E_sociol_corpus_processed.csv",
    encoding='utf-8')
summary = pd.read_csv(
    "out_E_sociol_corpus_summary.csv",
    encoding='utf-8')

# Identify Code_ columns present in the dataset
code_cols = [c for c in df.columns
             if c.startswith('Code_')]
print(f"Coded feature columns detected: {code_cols}")

# Drop rows not yet coded
# (all Code_ columns still NaN)
coded_mask = df[code_cols].notna().any(axis=1)
df_coded = df[coded_mask].copy()
print(f"Coded texts available for analysis: "
      f"{len(df_coded)} of {len(df)} total")

# Social variable columns to stratify by
social_vars = [
    v for v in ['Genre', 'Gender', 'Community', 'Year']
    if v in df_coded.columns]
print(f"Social stratification variables: {social_vars}")

##### 1. Overall feature frequency distributions
results = []
total_tokens = df_coded['Token_Count'].sum()
for col in code_cols:
    valid = df_coded[col].dropna()
    n_present = int((valid == 1).sum())
    n_absent  = int((valid == 0).sum())
    n_total   = n_present + n_absent
    prop = (round(n_present / n_total, 4)
            if n_total > 0 else np.nan)
    norm_freq = (round(
        (n_present / total_tokens) * 1000, 4)
        if total_tokens > 0 else np.nan)
    results.append({
        'Feature': col,
        'N_Present': n_present,
        'N_Absent': n_absent,
        'N_Total': n_total,
        'Proportion': prop,
        'Freq_Per_1000_Tokens': norm_freq,
    })
freq_df = pd.DataFrame(results)
print("\n--- Overall Feature Frequencies ---")
print(freq_df.to_string(index=False))

##### 2. Feature frequency by social variable
breakdown_records = []
for var in social_vars:
    if (df_coded[var].dtype == object
            or df_coded[var].nunique() <= 10):
        for col in code_cols:
            sub = df_coded[[var, col]].dropna()
            if sub.empty:
                continue
            grp = (sub.groupby(var)[col]
                   .agg(
                       N_Present=lambda x:
                           (x == 1).sum(),
                       N_Total='count')
                   .reset_index())
            grp['Proportion'] = (
                grp['N_Present']
                / grp['N_Total']).round(4)
            grp['Feature'] = col
            grp['Variable'] = var
            breakdown_records.append(grp)
if breakdown_records:
    breakdown_df = pd.concat(
        breakdown_records, ignore_index=True)
    breakdown_df = breakdown_df[[
        'Variable', 'Feature',
        'N_Present', 'N_Total', 'Proportion']]
else:
    breakdown_df = pd.DataFrame()

##### 3. Chi-square tests of independence
chisq_records = []
for var in social_vars:
    if (df_coded[var].dtype == object
            or df_coded[var].nunique() <= 10):
        for col in code_cols:
            sub = df_coded[[var, col]].dropna()
            sub = sub[sub[col].isin([0, 1])]
            if sub.empty or sub[var].nunique() < 2:
                continue
            contingency = pd.crosstab(
                sub[var], sub[col])
            if (contingency.shape[0] < 2
                    or contingency.shape[1] < 2):
                continue
            try:
                chi2, p, dof, expected = (
                    chi2_contingency(contingency))
                low_expected = (expected < 5).any()
                chisq_records.append({
                    'Variable': var,
                    'Feature': col,
                    'Chi2': round(chi2, 4),
                    'df': dof,
                    'p_value': round(p, 6),
                    'Significant_p05': p < 0.05,
                    'Low_Expected_Warning':
                        low_expected,
                })
            except Exception as e:
                print(f"Chi-square failed for "
                      f"{var} x {col}: {e}")
chisq_df = pd.DataFrame(chisq_records)
if not chisq_df.empty:
    print("\n--- Chi-Square Tests of Independence ---")
    print(chisq_df.to_string(index=False))

##### 4. Co-occurrence counts between Code_ features
cooccurrence_records = []
for i, col_a in enumerate(code_cols):
    for col_b in code_cols[i+1:]:
        sub = df_coded[[col_a, col_b]].dropna()
        sub = sub[
            sub[col_a].isin([0, 1])
            & sub[col_b].isin([0, 1])]
        if sub.empty:
            continue
        n_both = int(
            ((sub[col_a] == 1)
             & (sub[col_b] == 1)).sum())
        n_total = len(sub)
        prop = (round(n_both / n_total, 4)
                if n_total > 0 else np.nan)
        cooccurrence_records.append({
            'Feature_A': col_a,
            'Feature_B': col_b,
            'N_Cooccurrence': n_both,
            'N_Total': n_total,
            'Cooccurrence_Proportion': prop,
        })
cooccurrence_df = pd.DataFrame(cooccurrence_records)
if not cooccurrence_df.empty:
    print("\n--- Feature Co-occurrence Counts ---")
    print(cooccurrence_df.to_string(index=False))

##### 5. Diachronic trend: feature frequency by year
# Only computed if Year column is available and
# has sufficient range.
if ('Year' in df_coded.columns
        and df_coded['Year'].notna().sum() > 10):
    year_trend_records = []
    for col in code_cols:
        sub = df_coded[
            ['Year', col, 'Token_Count']].dropna()
        sub = sub[sub[col].isin([0, 1])]
        if sub.empty:
            continue
        annual = sub.groupby('Year').agg(
            N_Present=(col,
                       lambda x: (x == 1).sum()),
            N_Total=(col, 'count'),
            Total_Tokens=(
                'Token_Count', 'sum')
        ).reset_index()
        annual['Proportion'] = (
            annual['N_Present']
            / annual['N_Total']).round(4)
        annual['Freq_Per_1000'] = (
            (annual['N_Present']
             / annual['Total_Tokens'])
            * 1000).round(4)
        annual['Feature'] = col
        year_trend_records.append(annual)
    trend_df = (pd.concat(
        year_trend_records, ignore_index=True)
        if year_trend_records
        else pd.DataFrame())
else:
    trend_df = pd.DataFrame()

##### 6. Export all results
freq_df.to_csv(
    "out_E_sociol_corpus_feature_frequencies.csv",
    index=False)
chisq_df.to_csv(
    "out_E_sociol_corpus_chisquare_tests.csv",
    index=False)
cooccurrence_df.to_csv(
    "out_E_sociol_corpus_cooccurrence.csv",
    index=False)
if not breakdown_df.empty:
    breakdown_df.to_csv(
        "out_E_sociol_corpus_breakdown.csv",
        index=False)
if not trend_df.empty:
    trend_df.to_csv(
        "out_E_sociol_corpus_diachronic_trends.csv",
        index=False)
print("\nSociolinguistic corpus analysis data analysis "
      "completed.")
print("Output files:")
print("  out_E_sociol_corpus_feature_frequencies.csv")
print("  out_E_sociol_corpus_chisquare_tests.csv")
print("  out_E_sociol_corpus_cooccurrence.csv")
print("  out_E_sociol_corpus_breakdown.csv")
print("  out_E_sociol_corpus_diachronic_trends.csv")
