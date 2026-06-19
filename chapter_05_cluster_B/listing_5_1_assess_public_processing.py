import pandas as pd

# Load preprocessed dataset (output of Prerequisite
# Preprocessing Script, Listing 3.1, Cluster B block)
df = pd.read_csv("out_B_assess_public_preprocessed.csv",
                 encoding='utf-8')
print(f"Records loaded: {len(df)}")

##### Extract open-ended responses for qualitative analysis
if 'Open_Response' in df.columns:
    open_ended = df[['Open_Response']].dropna()
    open_ended.to_csv(
        "out_B_assess_public_open_responses.csv",
        index=False)
    print(f"Open-ended responses saved: "
          f"{len(open_ended)} records.")
else:
    print("Warning: Open_Response column not found. "
          "Skipping qualitative extraction.")

##### Save processed dataset
df.to_csv("out_B_assess_public_processed.csv", index=False)
print("Public attitudes data processing completed.")
