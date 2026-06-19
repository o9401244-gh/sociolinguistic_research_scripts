import pandas as pd
import numpy as np
import re

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster C block)
df = pd.read_csv("out_C_sociol_interv_preprocessed.csv",
                 encoding='utf-8')
print(f"Records loaded: {len(df)}")

##### 1. Normalize and validate interview module labels
if 'Module' in df.columns:
    df['Module'] = df['Module'].str.strip().str.lower()
    valid_modules = {
        'community', 'personal', 'narrative',
        'danger-of-death', 'language', 'other'}
    unrecognized = ~df['Module'].isin(valid_modules)
    if unrecognized.any():
        n = unrecognized.sum()
        vals = (df.loc[unrecognized, 'Module']
                .unique().tolist())
        print(f"Warning: {n} turns with unrecognized "
              f"Module values: {vals}")
else:
    print("Warning: Module column not found. Interview "
          "module structure will not be preserved.")

##### 2. Flag turns containing metalinguistic commentary
# Metalinguistic turns are those in which the speaker
# explicitly reflects on, evaluates, or theorizes about
# language use. They form a distinct analytic data stream
# for attitude and ideology analysis.
meta_patterns = [
    r"\bI\s+(always|never|usually|sometimes)\s+say\b",
    r"\bwe\s+(always|never|usually|sometimes)\s+say\b",
    r"\bpeople\s+(here|there|in\s+\w+)\s+say\b",
    r"\b(correct|wrong|proper|improper|appropriate)\b",
    r"\b(accent|dialect|language)\b.{0,60}"
    r"\b(good|bad|right|wrong)\b",
    r"\bI\s+don't\s+(like|use)\b",
    r"\bthey\s+(say|call|use)\b",
]
combined = '|'.join(meta_patterns)
content_col = 'Turn_Content'
if content_col in df.columns:
    df['Metalinguistic_Flag'] = df[content_col].apply(
        lambda x: 1 if re.search(
            combined, str(x), re.IGNORECASE) else 0)
    n_meta = df['Metalinguistic_Flag'].sum()
    print(f"Turns flagged as metalinguistic: {n_meta}")
else:
    print(f"Warning: {content_col} column not found. "
          "Skipping metalinguistic flagging.")

##### 3. Initialize binary Code_ columns for
# variation-analytic coding of elicited speech
# (adapt variable names to research context)
for col in ['Code_Target_Variant',
            'Code_Vernacular_Form',
            'Code_Standard_Form',
            'Code_Code_Switch',
            'Code_Narrative_Peak']:
    df[col] = np.nan

##### 4. Save coding-ready dataset
df = df.drop(columns=['Analytical_Category',
             'Code_Feature_A', 'Code_Feature_B'],
             errors='ignore')
df.to_csv("out_C_sociol_interv_processed.csv", index=False)
print(f"Records saved: {len(df)}")
print("Sociolinguistic interviews data processing "
      "completed.")
