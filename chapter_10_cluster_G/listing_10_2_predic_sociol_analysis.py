import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# Load processed dataset (output of Listing 10.1)
df = pd.read_csv(
    "out_G_predic_sociol_processed.csv",
    encoding='utf-8')
print(f"Observations loaded: {len(df)}")
print(f"Year range: {df['Year'].min()} to "
      f"{df['Year'].max()}")

##### 1. Assemble feature columns
feature_cols = ['Year_Centered']
for c in ['Age_Centered', 'Education_Num']:
    if c in df.columns:
        feature_cols.append(c)
feature_cols += [
    c for c in df.columns
    if c.startswith(('Gender_', 'Comm_'))]
feature_cols = [
    c for c in feature_cols if c in df.columns]
print(f"Feature columns: {feature_cols}")

##### 2. Prepare model-ready dataset
model_df = df[
    feature_cols + ['Outcome_Binary']].dropna()
X = model_df[feature_cols].values
y = model_df['Outcome_Binary'].values
print(f"Model-ready observations: {len(model_df)}")

# Train/test split (80/20, stratified where possible)
min_class = pd.Series(y).value_counts().min()
if min_class < 2:
    print("Warning: insufficient observations per class "
          "for stratified split. Falling back to "
          "non-stratified split. Results should be "
          "interpreted with caution.")
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=0.2, random_state=42))
else:
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=0.2,
            random_state=42, stratify=y))
print(f"Training set: {len(X_train)}, "
      f"Test set: {len(X_test)}")

# Scale features for logistic regression
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Recover year mean for forecasting
year_mean = df['Year'].mean()

# Guard for single-class training or test sets
if (len(np.unique(y_train)) < 2
        or len(np.unique(y_test)) < 2):
    print("Warning: training or test set contains only "
          "one class. Model fitting skipped. Use a "
          "larger dataset for meaningful model "
          "evaluation.")
else:
    ##### 3. Fit logistic regression model
    lr = LogisticRegression(
        max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    y_prob_lr = lr.predict_proba(X_test_s)[:, 1]
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr  = f1_score(
        y_test, y_pred_lr, zero_division=0)
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    print(f"\n--- Logistic Regression (Holdout) ---")
    print(f"Accuracy: {acc_lr:.4f}, "
          f"F1: {f1_lr:.4f}, AUC: {auc_lr:.4f}")
    print(classification_report(
        y_test, y_pred_lr, zero_division=0))

    # Coefficient table
    coef_df = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': lr.coef_[0].round(4),
        'Odds_Ratio': np.exp(
            lr.coef_[0]).round(4),
    }).sort_values('Coefficient', ascending=False)
    print("\nLogistic Regression Coefficients:")
    print(coef_df.to_string(index=False))

    # Cross-validation AUC
    from sklearn.model_selection import KFold
    n_splits = min(5, len(X_train))
    if n_splits < 2:
        print("Warning: training set too small for "
              "cross-validation. Skipping CV AUC.")
    else:
        cv = KFold(
            n_splits=n_splits,
            shuffle=True, random_state=42)
        cv_auc_lr = cross_val_score(
            LogisticRegression(
                max_iter=1000, random_state=42),
            X_train_s, y_train,
            cv=cv, scoring='roc_auc')
        print(f"{n_splits}-fold CV AUC (train): "
              f"{cv_auc_lr.mean():.4f} +/- "
              f"{cv_auc_lr.std():.4f}")

    ##### 4. Fit random forest model
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf  = f1_score(
        y_test, y_pred_rf, zero_division=0)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    print(f"\n--- Random Forest (Holdout) ---")
    print(f"Accuracy: {acc_rf:.4f}, "
          f"F1: {f1_rf:.4f}, AUC: {auc_rf:.4f}")
    print(classification_report(
        y_test, y_pred_rf, zero_division=0))

    # Feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': (rf.feature_importances_
                       .round(4)),
    }).sort_values('Importance', ascending=False)
    print("\nRandom Forest Feature Importances:")
    print(importance_df.to_string(index=False))

    ##### 5. Model comparison summary
    comparison = pd.DataFrame([
        {'Model': 'Logistic_Regression',
         'Accuracy': round(acc_lr, 4),
         'F1': round(f1_lr, 4),
         'AUC': round(auc_lr, 4)},
        {'Model': 'Random_Forest',
         'Accuracy': round(acc_rf, 4),
         'F1': round(f1_rf, 4),
         'AUC': round(auc_rf, 4)},
    ])
    print(f"\n--- Model Comparison ---")
    print(comparison.to_string(index=False))

    ##### 6. Forecast future variant probabilities
    FORECAST_YEARS = list(range(
        int(df['Year'].max()) + 1,
        int(df['Year'].max()) + 11))
    feature_means = model_df[feature_cols].mean()
    forecast_rows = []
    for yr in FORECAST_YEARS:
        row = feature_means.copy()
        row['Year_Centered'] = yr - year_mean
        forecast_rows.append(row.values)
    X_future   = np.array(forecast_rows)
    X_future_s = scaler.transform(X_future)
    prob_lr_future = lr.predict_proba(
        X_future_s)[:, 1]
    prob_rf_future = rf.predict_proba(
        X_future)[:, 1]
    forecast_df = pd.DataFrame({
        'Year': FORECAST_YEARS,
        'Prob_LR': prob_lr_future.round(4),
        'Prob_RF': prob_rf_future.round(4),
        'Prob_Ensemble': (
            (prob_lr_future + prob_rf_future) / 2
        ).round(4),
    })
    print(f"\n--- Forecast: Probability of Variant ---")
    print(forecast_df.to_string(index=False))

    ##### 7. Export all results
    comparison.to_csv(
        "out_G_predic_sociol_model_comparison.csv",
        index=False)
    coef_df.to_csv(
        "out_G_predic_sociol_lr_coefficients.csv",
        index=False)
    importance_df.to_csv(
        "out_G_predic_sociol_rf_importance.csv",
        index=False)
    forecast_df.to_csv(
        "out_G_predic_sociol_forecast.csv",
        index=False)
    pd.DataFrame({
        'Observation_ID': (
            model_df.index[len(X_train):]),
        'y_true': y_test,
        'y_pred_lr': y_pred_lr,
        'y_prob_lr': y_prob_lr.round(4),
        'y_pred_rf': y_pred_rf,
        'y_prob_rf': y_prob_rf.round(4),
    }).to_csv(
        "out_G_predic_sociol_holdout_predictions.csv",
        index=False)
    print("\nPredictive sociolinguistic modeling "
          "analysis completed.")
    print("Output files:")
    print("  out_G_predic_sociol_model_comparison.csv")
    print("  out_G_predic_sociol_lr_coefficients.csv")
    print("  out_G_predic_sociol_rf_importance.csv")
    print("  out_G_predic_sociol_forecast.csv")
    print("  out_G_predic_sociol_holdout_predictions.csv")
