import pandas as pd
import numpy as np

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster C block)
df = pd.read_csv("out_C_intera_analys_preprocessed.csv",
                 encoding='utf-8')
print(f"Records loaded: {len(df)}")

##### 1. Flag potential transition-relevance places
# A TRP is flagged where the inter-turn gap falls within
# the threshold range suggesting smooth turn transition.
# Adapt TRP_MIN and TRP_MAX to your data's timing properties.
TRP_MIN = -0.2  # overlap threshold (seconds)
TRP_MAX =  0.5  # gap threshold (seconds)

if 'Gap' in df.columns:
    df['TRP_Flagged'] = df['Gap'].apply(
        lambda x: 1 if pd.notna(x)
        and TRP_MIN <= x <= TRP_MAX else 0)
    print(f"Transition-relevance places flagged: "
          f"{df['TRP_Flagged'].sum()}")
else:
    print("Warning: Gap column not found. Ensure "
          "Cluster C preprocessing block has run.")

##### 2. Initialize coding columns for the four
# conversation-analytic dimensions.
# All columns left as NaN for manual analyst coding.

# Turn-taking dimension
df['Turn_Type'] = np.nan
# e.g., self-select, other-select, overlap-onset

# Sequence organization dimension
df['Sequence_Position'] = np.nan
# e.g., FPP, SPP, pre-expansion, insert-expansion,
#        post-expansion
df['Pair_Type'] = np.nan
# e.g., question-answer, request-grant,
#        offer-accept, greeting-greeting

# Repair dimension
df['Repair_Initiation'] = np.nan
# e.g., self-initiated, other-initiated, none
df['Repair_Completion'] = np.nan
# e.g., self-completed, other-completed, none

# Participant alignment dimension
df['Alignment'] = np.nan
# e.g., preferred, dispreferred,
#        affiliative, disaffiliative

##### 3. Save coding-ready dataset
df.to_csv("out_C_intera_analys_processed.csv", index=False)
print(f"Records saved: {len(df)}")
print("Interactional analysis data processing completed.")
