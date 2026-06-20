import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load prestige data (output of Listing 10.7)
df = pd.read_csv(
    "out_G_langua_presti_processed.csv",
    encoding='utf-8')

##### 1. Aggregate annual prestige index
annual_prestige = df.groupby('Year').agg(
    Prestige_Index=('Prestige_Index', 'mean'),
    N_Domains=('Domain', 'nunique'),
    N_Records=('Prestige_Index', 'count')
).reset_index()
annual_prestige['Prestige_Index'] = (
    annual_prestige['Prestige_Index'].round(4))
print("\nAnnual prestige index:")
print(annual_prestige.to_string(index=False))

##### 2. Fit linear trend model with
# Hodrick-Prescott smoothing
# Smooth the index to separate trend from
# cyclical noise
from statsmodels.tsa.filters.hp_filter import hpfilter
cycle, trend = hpfilter(
    annual_prestige['Prestige_Index'].values,
    lamb=1600)
annual_prestige['Trend'] = trend.round(4)
annual_prestige['Cycle'] = cycle.round(4)

# OLS on smoothed trend
years  = annual_prestige['Year'].values
year_c = years - years.mean()
X = sm.add_constant(year_c)
y = annual_prestige['Trend'].values
ols = sm.OLS(y, X).fit()
print(f"\n--- Linear Trend Model ---")
print(f"Intercept: {ols.params[0]:.4f}, "
      f"Slope: {ols.params[1]:.6f} per year")
print(f"R2: {ols.rsquared:.4f}, "
      f"p (slope): {ols.pvalues[1]:.4f}")
slope_per_decade = ols.params[1] * 10
print(f"Estimated change per decade: "
      f"{slope_per_decade:.4f}")

##### 3. Domain-level breakdown
domain_trend = (
    df.groupby(['Domain', 'Year'])
    .agg(Prestige_Index=('Prestige_Index', 'mean'))
    .reset_index())
domain_summary = (
    domain_trend.groupby('Domain')
    .agg(
        Mean_Prestige=('Prestige_Index', 'mean'),
        Trend_Direction=(
            'Prestige_Index',
            lambda x: (
                'rising' if x.iloc[-1] > x.iloc[0]
                else 'declining' if len(x) > 1
                else 'stable')))
    .reset_index())
domain_summary['Mean_Prestige'] = (
    domain_summary['Mean_Prestige'].round(4))
print(f"\nPrestige by domain:")
print(domain_summary.to_string(index=False))

##### 4. Forward projection (10-year horizon)
FORECAST_YEARS = list(range(
    int(years.max()) + 1,
    int(years.max()) + 11))
year_c_future = np.array(FORECAST_YEARS) - years.mean()
X_future = sm.add_constant(year_c_future)
proj = ols.predict(X_future)
proj = np.clip(proj, 0.0, 1.0)
pred_summary = (ols.get_prediction(X_future)
                .summary_frame(alpha=0.05))
forecast_df = pd.DataFrame({
    'Year': FORECAST_YEARS,
    'Projected_Index': proj.round(4),
    'CI_Lower': (pred_summary['obs_ci_lower']
                 .clip(0, 1).round(4)),
    'CI_Upper': (pred_summary['obs_ci_upper']
                 .clip(0, 1).round(4)),
})
print(f"\nPrestige projection (10-year horizon):")
print(forecast_df.to_string(index=False))

##### 5. Export results
annual_prestige.to_csv(
    "out_G_langua_presti_annual_index.csv",
    index=False)
domain_summary.to_csv(
    "out_G_langua_presti_domain_summary.csv",
    index=False)
forecast_df.to_csv(
    "out_G_langua_presti_forecast.csv",
    index=False)
pd.DataFrame([{
    'Slope_Per_Year': round(ols.params[1], 6),
    'Slope_Per_Decade': round(slope_per_decade, 4),
    'R2': round(ols.rsquared, 4),
    'p_value': round(ols.pvalues[1], 6),
    'Significant_p05': ols.pvalues[1] < 0.05
}]).to_csv(
    "out_G_langua_presti_trend_model.csv",
    index=False)
print("\nLanguage prestige analysis completed.")
print("Output files:")
print("  out_G_langua_presti_annual_index.csv")
print("  out_G_langua_presti_domain_summary.csv")
print("  out_G_langua_presti_trend_model.csv")
print("  out_G_langua_presti_forecast.csv")
