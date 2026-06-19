import pandas as pd
import numpy as np

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster A block)
df = pd.read_csv("out_A_repres_discou_preprocessed.csv",
                 encoding='utf-8')
print(f"Records loaded: {len(df)}")

##### 1. Initialize representational coding columns
# These columns are left as NaN for manual analyst coding.

# Representational strategy dimension
df['Representational_Strategy'] = np.nan
# e.g., nomination, predication, perspectivization,
#        intensification, mitigation

# Social actor dimension
df['Social_Actor'] = np.nan
# e.g., the social actor or group represented in the unit

# Agency dimension
df['Agency'] = np.nan
# e.g., agentive, passive

# Framing category dimension
df['Framing_Category'] = np.nan
# e.g., foregrounding, backgrounding, omission

##### 2. Initialize binary Code_ columns for
# co-occurrence analysis (adapt names to research context)
for col in ['Code_Nomination', 'Code_Predication',
            'Code_Perspectivization',
            'Code_Intensification', 'Code_Mitigation']:
    df[col] = np.nan

##### 3. Save coding-ready dataset
df = df.drop(columns=['Analytical_Category',
             'Code_Feature_A', 'Code_Feature_B'],
             errors='ignore')
df.to_csv("out_A_repres_discou_processed.csv", index=False)
print(f"Records saved: {len(df)}")
print("Representational discourse analysis data processing "
      "completed.")
