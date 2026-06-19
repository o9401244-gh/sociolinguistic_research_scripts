import pandas as pd
import numpy as np
import re

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster A block)
df = pd.read_csv("out_A_inclus_langua_preprocessed.csv",
                 encoding='utf-8')
print(f"Records loaded: {len(df)}")

##### 1. Flag potentially marked or exclusionary patterns
# Define flagging patterns for nomination and
# categorization review (adapt to research context)
flag_patterns = [
    r"\bthe\s+\w+s\b",
    r"\bnon[-\s]\w+",
    r"\babnormal\b",
    r"\bdeviant\b",
    r"\bproblem\w*\b",
    r"\bburden\w*\b",
    r"\billegal\s+\w+",
    r"\bundocumented\s+\w+",
]
combined_pattern = '|'.join(flag_patterns)
df['Flagged'] = df['Text_Unit'].apply(
    lambda x: bool(re.search(
        combined_pattern, str(x), re.IGNORECASE)))
flagged_count = df['Flagged'].sum()
print(f"Units flagged for nomination/categorization "
      f"review: {flagged_count}")

##### 2. Initialize coding columns for the three audit axes

# Nomination and categorization axis
df['Nomination_Category'] = np.nan
# e.g., generic plural, criminalization,
#        deficit framing, neutral

# Representation and salience axis
df['Salience'] = np.nan
# e.g., foregrounded, backgrounded, absent

# Normalization and markedness axis
df['Markedness'] = np.nan
# e.g., marked, unmarked

##### 3. Encode genre as categorical variable
# (required for audit-axis analysis by genre
# in the analysis script)
if 'Genre' in df.columns:
    df['Genre_Code'] = pd.Categorical(df['Genre']).codes

##### 4. Save flagged units for qualitative follow-up
flagged_cols = ['Unit_ID', 'Text_Unit']
for col in ['Source', 'Genre', 'Social_Actor']:
    if col in df.columns:
        flagged_cols.append(col)

flagged_units = df[df['Flagged'] == True][flagged_cols]
flagged_units.to_csv(
    "out_A_inclus_langua_flagged_units.csv", index=False)
print(f"Flagged units saved: {len(flagged_units)} records.")

##### 5. Save processed dataset
df = df.drop(columns=['Analytical_Category',
             'Code_Feature_A', 'Code_Feature_B'],
             errors='ignore')
df.to_csv("out_A_inclus_langua_processed.csv", index=False)
print("Inclusive language audit data processing completed.")
