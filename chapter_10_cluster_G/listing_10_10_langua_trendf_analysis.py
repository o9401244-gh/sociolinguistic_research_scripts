import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing)
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Load processed time series (output of Listing 10.9)
df = pd.read_csv(
    "out_G_langua_trendf_processed.csv",
    encoding='utf-8')
print(f"Time series: {len(df)} observations, "
      f"{df['Year'].min():.0f}--"
      f"{df['Year'].max():.0f}")
series = df['Variant_Freq'].values
years  = df['Year'].values

##### 1. Stationarity test (ADF)
adf_result = adfuller(series, autolag='AIC')
print(f"\nADF test: statistic = "
      f"{adf_result[0]:.4f}, "
      f"p = {adf_result[1]:.4f}")
is_stationary = adf_result[1] < 0.05
print(f"Series {'is' if is_stationary else 'is NOT'} "
      f"stationary at p < 0.05")
d = 0 if is_stationary else 1

##### 2. Train/test split (last 20% as test window)
n_test  = max(3, int(len(series) * 0.2))
n_train = len(series) - n_test
train   = series[:n_train]
test    = series[n_train:]
train_years = years[:n_train]
test_years  = years[n_train:]
print(f"\nTrain: {n_train} obs "
      f"({years[0]:.0f}--{train_years[-1]:.0f}), "
      f"Test: {n_test} obs "
      f"({test_years[0]:.0f}--{test_years[-1]:.0f})")

##### 3. Fit ARIMA model
arima = None
arima_rmse = np.inf
try:
    arima = ARIMA(train, order=(1, d, 1)).fit()
    arima_forecast = arima.forecast(steps=n_test)
    arima_rmse = np.sqrt(
        ((test - arima_forecast) ** 2).mean())
    arima_mae  = np.abs(
        test - arima_forecast).mean()
    print(f"\nARIMA(1,{d},1) test RMSE: "
          f"{arima_rmse:.4f}, MAE: {arima_mae:.4f}")
except Exception as e:
    print(f"ARIMA fitting failed: {e}")

##### 4. Fit exponential smoothing (Holt linear trend)
ets = None
ets_rmse = np.inf
try:
    ets = ExponentialSmoothing(
        train, trend='add',
        seasonal=None).fit(optimized=True)
    ets_forecast = ets.forecast(steps=n_test)
    ets_rmse = np.sqrt(
        ((test - ets_forecast) ** 2).mean())
    ets_mae  = np.abs(
        test - ets_forecast).mean()
    print(f"ETS (Holt) test RMSE: "
          f"{ets_rmse:.4f}, MAE: {ets_mae:.4f}")
except Exception as e:
    print(f"ETS fitting failed: {e}")

##### 5. Select best model and generate 10-year
# forecast
HORIZON = 10
if arima_rmse <= ets_rmse and arima is not None:
    best_model_name = 'ARIMA'
    best_model = ARIMA(
        series, order=(1, d, 1)).fit()
    fc_result = best_model.get_forecast(
        steps=HORIZON)
    fc_mean  = fc_result.predicted_mean
    fc_ci    = fc_result.conf_int(alpha=0.05)
    fc_lower = fc_ci.iloc[:, 0].values
    fc_upper = fc_ci.iloc[:, 1].values
elif ets is not None:
    best_model_name = 'ETS'
    best_model = ExponentialSmoothing(
        series, trend='add',
        seasonal=None).fit(optimized=True)
    fc_mean = best_model.forecast(steps=HORIZON)
    resid_std = best_model.resid.std()
    fc_lower = fc_mean - 1.96 * resid_std
    fc_upper = fc_mean + 1.96 * resid_std
else:
    print("Warning: No model available for "
          "forecasting.")
    fc_mean = fc_lower = fc_upper = np.full(
        HORIZON, np.nan)
    best_model_name = 'None'

fc_years  = list(range(
    int(years[-1]) + 1,
    int(years[-1]) + HORIZON + 1))
fc_mean   = np.clip(fc_mean, 0.0, 1.0)
fc_lower  = np.clip(fc_lower, 0.0, 1.0)
fc_upper  = np.clip(fc_upper, 0.0, 1.0)
forecast_df = pd.DataFrame({
    'Year':      fc_years,
    'Forecast':  fc_mean.round(4),
    'CI_Lower':  fc_lower.round(4),
    'CI_Upper':  fc_upper.round(4),
    'Model':     best_model_name
})
print(f"\n--- {best_model_name} Forecast "
      f"(10-year horizon) ---")
print(forecast_df.to_string(index=False))

##### 6. Export results
df[['Year', 'Variant_Freq']].to_csv(
    "out_G_langua_trendf_series.csv", index=False)
pd.DataFrame([{
    'ARIMA_RMSE': (round(arima_rmse, 4)
                   if arima_rmse != np.inf
                   else None),
    'ETS_RMSE':   (round(ets_rmse, 4)
                   if ets_rmse != np.inf
                   else None),
    'Best_Model': best_model_name
}]).to_csv(
    "out_G_langua_trendf_model_comparison.csv",
    index=False)
forecast_df.to_csv(
    "out_G_langua_trendf_forecast.csv", index=False)
print("\nLanguage trend forecasting analysis completed.")
print("Output files:")
print("  out_G_langua_trendf_series.csv")
print("  out_G_langua_trendf_model_comparison.csv")
print("  out_G_langua_trendf_forecast.csv")
