import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder)
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load choice data (output of Listing 10.15)
df = pd.read_csv(
    "out_G_langua_choice_processed.csv",
    encoding='utf-8')

##### 1. Encode outcome and features
variant_labels = sorted(
    df['Chosen_Variant'].str.strip().unique())
le = LabelEncoder()
df['Outcome_Enc'] = le.fit_transform(
    df['Chosen_Variant'].str.strip())
print(f"Variant encoding: "
      f"{dict(zip(le.classes_, le.transform(le.classes_)))}")
n_classes = len(variant_labels)

# Context columns (already processed by 10.15)
context_cols = [
    c for c in df.columns
    if c.startswith('Context_')]
for col in context_cols:
    if df[col].dtype == object:
        df[col] = (df[col].str.strip()
                   .str.lower())
        c_dummies = pd.get_dummies(
            df[col], prefix=col,
            drop_first=True, dtype=int)
        df = pd.concat([df, c_dummies], axis=1)

# Collect all feature columns
feature_cols = ['Year_Centered']
for c in ['Age_Centered', 'Education_Num']:
    if c in df.columns:
        feature_cols.append(c)
feature_cols += [
    c for c in df.columns
    if c.startswith(('Gender_', 'Context_'))]
feature_cols = [
    c for c in feature_cols if c in df.columns]
print(f"Features: {feature_cols}")

##### 2. Fit multinomial logistic regression
model_data = df[
    feature_cols + ['Outcome_Enc']].dropna()
X = model_data[feature_cols].values
y = model_data['Outcome_Enc'].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=(y if n_classes <= 10 else None))
scaler  = StandardScaler()
X_tr_s  = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)
model = LogisticRegression(
    multi_class='multinomial',
    max_iter=1000, random_state=42,
    solver='lbfgs')
model.fit(X_tr_s, y_tr)
y_pred = model.predict(X_te_s)
holdout_acc = accuracy_score(y_te, y_pred)
holdout_f1  = f1_score(
    y_te, y_pred, average='weighted',
    zero_division=0)
print(f"\nHoldout accuracy: {holdout_acc:.4f}, "
      f"weighted F1: {holdout_f1:.4f}")

##### 3. Constraint effect table
# (coefficients per variant)
coef_records = []
classes = le.classes_
if model.coef_.shape[0] == 1:
    for feat, coef in zip(
            feature_cols, model.coef_[0]):
        coef_records.append({
            'Variant': classes[1],
            'Feature': feat,
            'Coefficient': round(coef, 4),
            'Odds_Ratio': round(
                np.exp(coef), 4),
        })
else:
    for i, cls in enumerate(
            classes[1:], start=1):
        for feat, coef in zip(
                feature_cols,
                model.coef_[i-1]):
            coef_records.append({
                'Variant': cls,
                'Feature': feat,
                'Coefficient': round(coef, 4),
                'Odds_Ratio': round(
                    np.exp(coef), 4),
            })
coef_df = pd.DataFrame(coef_records)
print(f"\n--- Constraint Effects ---")
print(coef_df.to_string(index=False))

##### 4. Simulate community-level trajectories
year_mean = df['Year'].mean()
FUTURE_YEARS = list(range(
    int(df['Year'].max()) + 1,
    int(df['Year'].max()) + 11))
feature_means = model_data[feature_cols].mean()
sim_rows = []
for yr in FUTURE_YEARS:
    row = feature_means.copy()
    row['Year_Centered'] = yr - year_mean
    sim_rows.append(row.values)
X_sim = scaler.transform(np.array(sim_rows))
probs = model.predict_proba(X_sim)
sim_df = pd.DataFrame(probs, columns=classes)
sim_df.insert(0, 'Year', FUTURE_YEARS)
print(f"\n--- Simulated Variant Probabilities ---")
print(sim_df.to_string(index=False))

##### 5. Export results
coef_df.to_csv(
    "out_G_langua_choice_constraint_effects.csv",
    index=False)
sim_df.to_csv(
    "out_G_langua_choice_simulation.csv",
    index=False)
pd.DataFrame([{
    'Holdout_Accuracy': round(holdout_acc, 4),
    'Holdout_F1_Weighted': round(holdout_f1, 4),
    'N_Variants': n_classes,
    'N_Observations': len(model_data),
}]).to_csv(
    "out_G_langua_choice_model_fit.csv",
    index=False)
print("\nLanguage choice modeling analysis completed.")
print("Output files:")
print("  out_G_langua_choice_constraint_effects.csv")
print("  out_G_langua_choice_simulation.csv")
print("  out_G_langua_choice_model_fit.csv")
