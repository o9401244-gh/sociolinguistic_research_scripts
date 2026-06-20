import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Load dataset (output of Listing 10.13)
df = pd.read_csv(
    "out_G_sociol_trendf_processed.csv",
    encoding='utf-8')
print(f"Records: {len(df)}, "
      f"Speakers: {df['Speaker_ID'].nunique()}, "
      f"Communities: {df['Community'].nunique()}, "
      f"Years: {df['Year'].min():.0f}--"
      f"{df['Year'].max():.0f}")

##### 1. Multilevel model: random intercepts for
# Speaker and Community (two-level nesting)
pred_parts = ['Year_Centered']
if 'Age_Centered' in df.columns:
    pred_parts.append('Age_Centered')
if 'Education_Num' in df.columns:
    pred_parts.append('Education_Num')
formula = ("Outcome_Binary ~ "
           + " + ".join(pred_parts))
model_data = df.dropna(
    subset=pred_parts + ['Outcome_Binary'])
lmm = None
try:
    lmm = smf.mixedlm(
        formula, data=model_data,
        groups=model_data['Speaker_ID']
    ).fit(reml=False)
    print(f"\n--- Multilevel Model "
          f"(speaker-level RE) ---")
    print(lmm.summary())
    year_coef = lmm.params.get(
        'Year_Centered', np.nan)
    year_pval = lmm.pvalues.get(
        'Year_Centered', np.nan)
    print(f"\nYear coefficient: {year_coef:.6f} "
          f"(p = {year_pval:.4f})")
    change_per_decade = year_coef * 10
    print(f"Estimated change in P(variant) per "
          f"decade: {change_per_decade:.4f}")
except Exception as e:
    print(f"Multilevel model failed: {e}")

##### 2. Community-level trend estimates
comm_trends = []
for comm, grp in model_data.groupby('Community'):
    if (grp['Year_Centered'].std() == 0
            or len(grp) < 5):
        continue
    try:
        sub_lmm = smf.mixedlm(
            "Outcome_Binary ~ Year_Centered",
            data=grp,
            groups=grp['Speaker_ID']
        ).fit(reml=False, disp=False)
        comm_trends.append({
            'Community': comm,
            'Year_Coef': round(
                sub_lmm.params.get(
                    'Year_Centered', np.nan), 6),
            'Year_p': round(
                sub_lmm.pvalues.get(
                    'Year_Centered', np.nan), 4),
            'N_Obs': len(grp),
            'N_Speakers': grp[
                'Speaker_ID'].nunique(),
        })
    except Exception:
        pass
comm_df = pd.DataFrame(comm_trends)
if not comm_df.empty:
    print(f"\n--- Community-Level Trends ---")
    print(comm_df.to_string(index=False))

##### 3. Conditional forecasts for a future year
year_mean = df['Year'].mean()
age_mean  = (df['Age'].mean()
             if 'Age' in df.columns else 0)
FUTURE_YEAR   = int(
    df['Year_Centered'].max()) + 10
future_year_c = FUTURE_YEAR - year_mean
scenario_df = pd.DataFrame()
if lmm is not None:
    scenarios  = []
    edu_levels = (
        [1, 2, 3]
        if 'Education_Num' in df.columns
        else [None])
    age_groups = (
        [25, 45, 65]
        if 'Age_Centered' in df.columns
        else [None])
    for edu in edu_levels:
        for age in age_groups:
            row = {'Year_Centered': future_year_c}
            if edu is not None:
                row['Education_Num'] = edu
            if age is not None:
                row['Age_Centered'] = (
                    age - age_mean)
            pred_val = lmm.params['Intercept']
            for col, val in row.items():
                if col in lmm.params:
                    pred_val += (
                        lmm.params[col] * val)
            pred_val = float(
                np.clip(pred_val, 0.0, 1.0))
            edu_label = {
                1: 'primary',
                2: 'secondary',
                3: 'tertiary'
            }.get(edu, 'NA')
            scenarios.append({
                'Future_Year': FUTURE_YEAR,
                'Education_Num': edu_label,
                'Age_Centered': (
                    age if age else 'NA'),
                'Predicted_Prob': round(
                    pred_val, 4),
            })
    scenario_df = pd.DataFrame(scenarios)
    print(f"\n--- Conditional Forecasts for "
          f"{FUTURE_YEAR} ---")
    print(scenario_df.to_string(index=False))

##### 4. Bootstrap uncertainty estimation
N_BOOTSTRAP = 200
boot_coefs  = []
rng = np.random.default_rng(seed=42)
speakers = model_data['Speaker_ID'].unique()
for _ in range(N_BOOTSTRAP):
    boot_speakers = rng.choice(
        speakers,
        size=len(speakers), replace=True)
    boot_df = pd.concat(
        [model_data[
             model_data['Speaker_ID'] == s]
         for s in boot_speakers],
        ignore_index=True)
    try:
        boot_model = smf.mixedlm(
            formula, data=boot_df,
            groups=boot_df['Speaker_ID']
        ).fit(reml=False, disp=False)
        boot_coefs.append(
            boot_model.params.get(
                'Year_Centered', np.nan))
    except Exception:
        pass
boot_coefs = np.array(
    [c for c in boot_coefs
     if not np.isnan(c)])
if len(boot_coefs) > 10:
    boot_ci = np.percentile(
        boot_coefs, [2.5, 97.5])
    print(f"\nBootstrap 95% CI for Year "
          f"coefficient: "
          f"[{boot_ci[0]:.6f}, "
          f"{boot_ci[1]:.6f}] "
          f"(n = {len(boot_coefs)} "
          f"replications)")

##### 5. Export results
if not comm_df.empty:
    comm_df.to_csv(
        "out_G_sociol_trendf_community.csv",
        index=False)
if not scenario_df.empty:
    scenario_df.to_csv(
        "out_G_sociol_trendf_forecasts.csv",
        index=False)
if len(boot_coefs) > 10:
    pd.DataFrame({
        'Bootstrap_Year_Coef': (
            boot_coefs.round(6))
    }).to_csv(
        "out_G_sociol_trendf_bootstrap.csv",
        index=False)
print("\nSociolinguistic trend forecasting "
      "analysis completed.")
print("Output files:")
print("  out_G_sociol_trendf_community.csv")
print("  out_G_sociol_trendf_forecasts.csv")
print("  out_G_sociol_trendf_bootstrap.csv")
