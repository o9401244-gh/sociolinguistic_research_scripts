import pandas as pd
import numpy as np
import re
from collections import Counter

# Load processed corpus and user stats
# (output of Listing 8.5)
df    = pd.read_csv(
    "out_E_social_media_processed.csv",
    encoding='utf-8')
users = pd.read_csv(
    "out_E_social_media_user_stats.csv",
    encoding='utf-8')

# Parse timestamps if present
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'], errors='coerce')
    df['Year']  = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month
print(f"Posts loaded: {len(df)}")
print(f"Users loaded: {len(users)}")

##### MODULE A: Neologism Detection
# A.1 Define baseline and target time windows.
# Adjust year boundaries to match the corpus date range.
if ('Year' in df.columns
        and df['Year'].notna().sum() > 0):
    year_min  = int(df['Year'].min())
    year_max  = int(df['Year'].max())
    midpoint  = year_min + (year_max - year_min) // 2
    baseline_df = df[df['Year'] <= midpoint]
    target_df   = df[df['Year'] > midpoint]
    print(f"\nNeologism detection windows:")
    print(f"  Baseline: {year_min}--{midpoint} "
          f"({len(baseline_df)} posts)")
    print(f"  Target:   {midpoint+1}--{year_max} "
          f"({len(target_df)} posts)")
else:
    split = len(df) // 2
    baseline_df = df.iloc[:split]
    target_df   = df.iloc[split:]
    print("No Year column: splitting corpus by row "
          "order for neologism detection.")

# A.2 Build token frequency distributions
def get_token_freq(texts, min_length=3):
    """Return a Counter of tokens, excluding very
    short tokens."""
    tokens = []
    for text in texts.dropna():
        tokens.extend([
            t for t in str(text).lower().split()
            if (len(t) >= min_length
                and re.match(
                    r'^[a-zA-Z\u00C0-\u024F#]+$',
                    t))])
    return Counter(tokens)

baseline_freq  = get_token_freq(
    baseline_df['Text_Normalized'])
target_freq    = get_token_freq(
    target_df['Text_Normalized'])
baseline_total = sum(baseline_freq.values())
target_total   = sum(target_freq.values())

# A.3 Compute relative frequency change and flag
# neologism candidates
MIN_TARGET_FREQ      = 10
MIN_FREQ_PER_MILLION = 50
FREQ_RATIO_THRESHOLD = 3.0
neologism_candidates = []
for token, target_count in target_freq.items():
    target_norm = (
        (target_count / target_total) * 1_000_000
        if target_total > 0 else 0)
    if target_count < MIN_TARGET_FREQ:
        continue
    if target_norm < MIN_FREQ_PER_MILLION:
        continue
    baseline_count = baseline_freq.get(token, 0)
    if baseline_count == 0:
        freq_ratio = np.inf
    else:
        baseline_norm = (
            (baseline_count / baseline_total)
            * 1_000_000
            if baseline_total > 0 else 0)
        freq_ratio = (target_norm / baseline_norm
                      if baseline_norm > 0
                      else np.inf)
    if freq_ratio >= FREQ_RATIO_THRESHOLD:
        neologism_candidates.append({
            'Token': token,
            'Baseline_Count': baseline_count,
            'Target_Count': target_count,
            'Target_Freq_Per_Million': round(
                target_norm, 2),
            'Freq_Ratio': (round(freq_ratio, 2)
                           if freq_ratio != np.inf
                           else None),
            'New_In_Target': baseline_count == 0,
        })
neologism_df = (pd.DataFrame(neologism_candidates)
                .sort_values('Target_Count',
                             ascending=False))
print(f"\n--- Neologism Candidates Detected: "
      f"{len(neologism_df)} ---")
if not neologism_df.empty:
    print(neologism_df.head(20).to_string(index=False))

##### MODULE B: Code-Switching Analysis
n_total = len(df)
n_cs    = df['Code_Switch_Flag'].sum()
cs_rate_overall = round(n_cs / n_total, 4)
print(f"\n--- Code-Switching Analysis ---")
print(f"Overall code-switching rate: "
      f"{cs_rate_overall:.4f} "
      f"({n_cs} of {n_total} posts)")

cs_by_platform = pd.DataFrame()
if 'Platform' in df.columns:
    cs_by_platform = df.groupby('Platform').agg(
        N_Posts=('Post_ID', 'count'),
        N_CS_Posts=('Code_Switch_Flag', 'sum')
    ).reset_index()
    cs_by_platform['CS_Rate'] = (
        cs_by_platform['N_CS_Posts']
        / cs_by_platform['N_Posts']).round(4)
    print(f"\nCode-switching rate by platform:")
    print(cs_by_platform.to_string(index=False))

cs_by_month = pd.DataFrame()
if ('Year' in df.columns
        and 'Month' in df.columns):
    cs_by_month = df.groupby(['Year', 'Month']).agg(
        N_Posts=('Post_ID', 'count'),
        N_CS_Posts=('Code_Switch_Flag', 'sum')
    ).reset_index()
    cs_by_month['CS_Rate'] = (
        cs_by_month['N_CS_Posts']
        / cs_by_month['N_Posts']).round(4)

cs_by_location = pd.DataFrame()
if ('Location' in df.columns
        and df['Location'].notna().sum() > 0):
    cs_by_location = df.groupby('Location').agg(
        N_Posts=('Post_ID', 'count'),
        N_CS_Posts=('Code_Switch_Flag', 'sum')
    ).reset_index()
    cs_by_location['CS_Rate'] = (
        cs_by_location['N_CS_Posts']
        / cs_by_location['N_Posts']).round(4)
    cs_by_location = cs_by_location.sort_values(
        'CS_Rate', ascending=False)
    print(f"\nCode-switching rate by location "
          f"(top 10):")
    print(cs_by_location.head(10).to_string(
        index=False))

cs_by_user = users[[
    'User_ID', 'N_Posts',
    'N_CS_Posts', 'CS_Rate']].copy()
high_cs_users = cs_by_user[
    (cs_by_user['CS_Rate'] > 0.5)
    & (cs_by_user['N_Posts'] >= 5)
].sort_values('CS_Rate', ascending=False)
print(f"\nHigh-switching users "
      f"(CS rate > 0.5, >= 5 posts): "
      f"{len(high_cs_users)}")

##### MODULE C: Language Diffusion Analysis
diffusion_records = []
if (not neologism_df.empty
        and 'Timestamp' in df.columns):
    top_neologisms = neologism_df.head(50)[
        'Token'].tolist()
    for token in top_neologisms:
        token_pattern = re.compile(
            r'\b' + re.escape(token) + r'\b',
            re.IGNORECASE)
        token_mask = df['Text_Normalized'].str.contains(
            token_pattern, regex=True, na=False)
        token_posts = df[token_mask].copy()
        if token_posts.empty:
            continue
        token_posts = token_posts.sort_values(
            'Timestamp')
        first_appearance = (token_posts['Timestamp']
                            .min())
        n_users = token_posts['User_ID'].nunique()
        n_posts = len(token_posts)
        n_locations = (
            token_posts['Location'].nunique()
            if ('Location' in token_posts.columns
                and token_posts['Location']
                    .notna().sum() > 0)
            else None)
        diffusion_records.append({
            'Token': token,
            'First_Appearance': first_appearance,
            'N_Posts': n_posts,
            'N_Unique_Users': n_users,
            'N_Locations': n_locations,
        })
diffusion_df = pd.DataFrame(diffusion_records)
if not diffusion_df.empty:
    diffusion_df = diffusion_df.sort_values(
        'First_Appearance').reset_index(drop=True)
    print(f"\n--- Language Diffusion: "
          f"{len(diffusion_df)} tokens tracked ---")
    print(diffusion_df.head(10).to_string(index=False))

##### Save all outputs
neologism_df.to_csv(
    "out_E_social_media_neologisms.csv", index=False)
cs_summary = pd.DataFrame([{
    'Total_Posts': n_total,
    'CS_Posts': int(n_cs),
    'CS_Rate_Overall': cs_rate_overall}])
cs_summary.to_csv(
    "out_E_social_media_cs_summary.csv", index=False)
if not cs_by_platform.empty:
    cs_by_platform.to_csv(
        "out_E_social_media_cs_by_platform.csv",
        index=False)
if not cs_by_month.empty:
    cs_by_month.to_csv(
        "out_E_social_media_cs_by_month.csv",
        index=False)
if not cs_by_location.empty:
    cs_by_location.to_csv(
        "out_E_social_media_cs_by_location.csv",
        index=False)
if not diffusion_df.empty:
    diffusion_df.to_csv(
        "out_E_social_media_diffusion.csv",
        index=False)
print("\nSocial media language analysis data analysis "
      "completed.")
print("Output files:")
print("  out_E_social_media_neologisms.csv")
print("  out_E_social_media_cs_summary.csv")
print("  out_E_social_media_cs_by_platform.csv")
print("  out_E_social_media_cs_by_month.csv")
print("  out_E_social_media_cs_by_location.csv")
print("  out_E_social_media_diffusion.csv")
