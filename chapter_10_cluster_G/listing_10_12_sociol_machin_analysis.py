import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Load dataset (output of Listing 10.11)
df = pd.read_csv(
    "out_G_sociol_machin_processed.csv",
    encoding='utf-8')
print(f"Observations: {len(df)}, "
      f"Class balance: "
      f"{df['Outcome_Binary'].value_counts().to_dict()}")

##### 1. Assemble feature columns
feature_cols = ['Year_Centered']
for c in ['Age_Centered', 'Education_Num']:
    if c in df.columns:
        feature_cols.append(c)
feature_cols += [
    c for c in df.columns
    if c.startswith(('Gender_', 'Comm_',
                     'Feature_'))]
feature_cols = [
    c for c in feature_cols if c in df.columns]
print(f"Features: {feature_cols}")
all_features = feature_cols

##### 2. Prepare model-ready arrays
model_df = df[
    feature_cols + ['Outcome_Binary']
].dropna(subset=feature_cols + ['Outcome_Binary'])
X = model_df[feature_cols].values
y = model_df['Outcome_Binary'].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y)
scaler  = StandardScaler()
X_tr_s  = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

##### 3. Logistic regression baseline
lr = LogisticRegression(
    max_iter=1000, random_state=42)
lr.fit(X_tr_s, y_tr)
results = {}
y_pred_lr = lr.predict(X_te_s)
y_prob_lr = lr.predict_proba(X_te_s)[:, 1]
results['LR'] = {
    'Accuracy': round(
        accuracy_score(y_te, y_pred_lr), 4),
    'F1': round(f1_score(
        y_te, y_pred_lr, zero_division=0), 4),
    'AUC': round(
        roc_auc_score(y_te, y_prob_lr), 4),
}

##### 4. Random forest with grid search
rf_grid = GridSearchCV(
    RandomForestClassifier(
        random_state=42, n_jobs=-1),
    param_grid={
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]},
    cv=5, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_tr, y_tr)
rf_best = rf_grid.best_estimator_
print(f"Best RF params: {rf_grid.best_params_}")
y_pred_rf = rf_best.predict(X_te)
y_prob_rf = rf_best.predict_proba(X_te)[:, 1]
results['RF'] = {
    'Accuracy': round(
        accuracy_score(y_te, y_pred_rf), 4),
    'F1': round(f1_score(
        y_te, y_pred_rf, zero_division=0), 4),
    'AUC': round(
        roc_auc_score(y_te, y_prob_rf), 4),
}

##### 5. Gradient boosting with grid search
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid={
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]},
    cv=5, scoring='roc_auc', n_jobs=-1)
gb_grid.fit(X_tr, y_tr)
gb_best = gb_grid.best_estimator_
print(f"Best GB params: {gb_grid.best_params_}")
y_pred_gb = gb_best.predict(X_te)
y_prob_gb = gb_best.predict_proba(X_te)[:, 1]
results['GB'] = {
    'Accuracy': round(
        accuracy_score(y_te, y_pred_gb), 4),
    'F1': round(f1_score(
        y_te, y_pred_gb, zero_division=0), 4),
    'AUC': round(
        roc_auc_score(y_te, y_prob_gb), 4),
}
comparison_df = (pd.DataFrame(results).T
                 .reset_index())
comparison_df.columns = [
    'Model', 'Accuracy', 'F1', 'AUC']
print(f"\n--- Model Comparison ---")
print(comparison_df.to_string(index=False))

##### 6. Feature importances (best tree-based model)
best_tree = (
    gb_best
    if results['GB']['AUC'] >= results['RF']['AUC']
    else rf_best)
importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': (best_tree.feature_importances_
                   .round(4)),
}).sort_values('Importance', ascending=False)
print(f"\nFeature Importances "
      f"({type(best_tree).__name__}):")
print(importance_df.head(10).to_string(index=False))

##### 7. Subgroup-level evaluation (by Community)
subgroup_records = []
comm_col = (
    'Community'
    if 'Community' in model_df.columns
    else None)
if (comm_col
        and model_df[comm_col].nunique() >= 2):
    test_idx  = model_df.index[len(X_tr):]
    test_meta = (
        model_df.loc[test_idx, comm_col].values
        if len(test_idx) == len(y_te)
        else None)
    if test_meta is not None:
        for grp in np.unique(test_meta):
            mask = test_meta == grp
            if mask.sum() < 5:
                continue
            grp_acc = accuracy_score(
                y_te[mask], y_pred_gb[mask])
            grp_f1  = f1_score(
                y_te[mask], y_pred_gb[mask],
                zero_division=0)
            try:
                grp_auc = roc_auc_score(
                    y_te[mask], y_prob_gb[mask])
            except ValueError:
                grp_auc = np.nan
            subgroup_records.append({
                'Community': grp,
                'N_Test': int(mask.sum()),
                'Accuracy': round(grp_acc, 4),
                'F1': round(grp_f1, 4),
                'AUC': round(grp_auc, 4),
            })
if subgroup_records:
    subgroup_df = pd.DataFrame(subgroup_records)
    print(f"\n--- Subgroup Performance "
          f"(Community) ---")
    print(subgroup_df.to_string(index=False))
else:
    subgroup_df = pd.DataFrame()

##### 8. Export results
comparison_df.to_csv(
    "out_G_sociol_machin_model_comparison.csv",
    index=False)
importance_df.to_csv(
    "out_G_sociol_machin_feature_importance.csv",
    index=False)
if not subgroup_df.empty:
    subgroup_df.to_csv(
        "out_G_sociol_machin_subgroup.csv",
        index=False)
print("\nSociolinguistic machine learning analysis "
      "completed.")
print("Output files:")
print("  out_G_sociol_machin_model_comparison.csv")
print("  out_G_sociol_machin_feature_importance.csv")
print("  out_G_sociol_machin_subgroup.csv")
