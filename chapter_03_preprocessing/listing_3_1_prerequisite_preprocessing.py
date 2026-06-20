import pandas as pd
import numpy as np
import re

# =============================================================
# PREREQUISITE PREPROCESSING SCRIPT
# Sociolinguistic Research: Facilitating Analysis with AI
# =============================================================
# This script uses a plain numbered menu to collect all
# required configuration from the user at runtime. No part
# of the script needs to be edited manually before running.
#
# This design is intentional. A more sophisticated interface
# (e.g., graphical menus, third-party prompt libraries) would
# require additional dependencies and obscure the underlying
# logic. A plain numbered menu works in any terminal, any
# Python environment, and any operating system.
# =============================================================

# =============================================================
# BEFORE YOU BEGIN
# =============================================================
# 1. Run this script from the folder that contains your raw
#    data file. Output files are saved to the same folder.
#
# 2. For methods requiring additional input files (Social
#    Network Analysis, Language Practice Network Mapping,
#    Evaluating Language Interventions), prepare those files
#    in the same folder before running the cluster-level
#    Processing Script. The menu will remind you of the
#    required filenames and columns.
#
# 3. If your method is in Cluster G, the preprocessing block
#    already computes Outcome_Binary from your Variant column.
#    You do not need to create this column manually.
#
# 4. Required packages beyond pandas and numpy:
#    pip install matplotlib scipy statsmodels scikit-learn
#    Not all packages are needed for every method; install
#    as prompted by ImportError when running cluster scripts.
# =============================================================


# -------------------------------------------------------------
# SECTION 1 — METHOD REGISTRY
# -------------------------------------------------------------
# Each entry defines one method's preprocessing profile:
#   'label'      : display name shown in the submenu
#   'id_col'     : default primary ID column name
#   'content_col': default primary content column name
#   'output'     : default output filename for preprocessed.csv
#   'extra'      : list of additional required input files
#                  (empty list if none)
#   'extra_note' : plain-language note about each extra file
# -------------------------------------------------------------

METHODS = {
    'A': [
        {
            'label':       'Representational Discourse Analysis',
            'id_col':      'Unit_ID',
            'content_col': 'Text_Unit',
            'output':      'out_A_repres_discou_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
        {
            'label':       'Rhetorical Strategy Analysis',
            'id_col':      'Unit_ID',
            'content_col': 'Text_Unit',
            'output':      'out_A_rhetor_strate_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
        {
            'label':       'Inclusive Language Audit',
            'id_col':      'Unit_ID',
            'content_col': 'Text_Unit',
            'output':      'out_A_inclus_langua_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
    ],
    'B': [
        {
            'label':       'Assessing Public Attitudes',
            'id_col':      'Respondent_ID',
            'content_col': 'Attitude_Q1',
            'output':      'out_B_assess_public_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
        {
            'label':       'Documenting Language Attitudes',
            'id_col':      'Respondent_ID',
            'content_col': 'Attitude_Q1',
            'output':      'out_B_docume_langua_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
    ],
    'C': [
        {
            'label':       'Interactional Analysis',
            'id_col':      'Turn_ID',
            'content_col': 'Turn_Content',
            'output':      'out_C_intera_analys_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
        {
            'label':       'Sociolinguistic Interviews',
            'id_col':      'Turn_ID',
            'content_col': 'Turn_Content',
            'output':      'out_C_sociol_interv_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
    ],
    'D': [
        {
            'label':       'Geolinguistic Mapping',
            'id_col':      'Site_ID',
            'content_col': 'Feature',
            'output':      'out_D_geolin_mappin_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
        {
            'label':       'Social Network Analysis',
            'id_col':      'Node_ID',
            'content_col': 'Feature',
            'output':      'out_D_social_networ_preprocessed.csv',
            'extra':       ['sna_edges_raw.csv'],
            'extra_note':  [
                'sna_edges_raw.csv  — raw edge list file with columns '
                'Node_A, Node_B, Tie_Type, and optionally Tie_Strength. '
                'Node_A and Node_B values must match the Node_ID values '
                'in your main data file exactly (case-sensitive). '
                'This file is NOT produced by this script; you must '
                'prepare it separately before running the Data Processing '
                'Script (Listing 7.3).'
            ],
        },
    ],
    'E': [
        {
            'label':       'Sociolinguistic Corpus Analysis',
            'id_col':      'Text_ID',
            'content_col': 'Text_Content',
            'output':      'out_E_sociol_corpus_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
        {
            'label':       'Causal Analysis of Language Variation',
            'id_col':      'Text_ID',
            'content_col': 'Text_Content',
            'output':      'out_E_causal_analys_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
        {
            'label':       'Social Media Language Analysis',
            'id_col':      'Text_ID',
            'content_col': 'Text_Content',
            'output':      'out_E_social_media_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
    ],
    'F': [
        {
            'label':       'Evaluating Language Interventions',
            'id_col':      'Participant_ID',
            'content_col': 'Group',
            'output':      'out_F_evalua_langua_preprocessed.csv',
            'extra':       ['intervention_outcomes_raw.csv'],
            'extra_note':  [
                'intervention_outcomes_raw.csv  — longitudinal outcomes '
                'file with columns Participant_ID, Time (e.g., Pre/Post), '
                'and one or more outcome score columns. Participant_ID '
                'values must match those in your main data file exactly. '
                'This file is NOT produced by this script; you must '
                'prepare it separately before running the Data Processing '
                'Script (Listing 9.1).'
            ],
        },
    ],
    'G': [
        {
            'label':       'Predictive Sociolinguistic Modeling',
            'id_col':      'Observation_ID',
            'content_col': 'Variant',
            'output':      'out_G_predic_sociol_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
        {
            'label':       'Computational Simulation of Language Change',
            'id_col':      'Observation_ID',
            'content_col': 'Variant',
            'output':      'out_G_comput_simula_preprocessed.csv',
            'extra':       ['simulation_params.csv'],
            'extra_note':  [
                'simulation_params.csv  — parameter configuration file '
                'with columns N_Agents, N_Steps, Init_Freq_A, '
                'Prestige_Bias, Network_Density, N_Simulations. '
                'This is the PRIMARY input for this method; '
                'preprocessed.csv is not used. Prepare this file '
                'before running Listing 10.3.'
            ],
        },
        {
            'label':       'Language Change Longitudinal Tracking',
            'id_col':      'Observation_ID',
            'content_col': 'Variant',
            'output':      'out_G_langua_change_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
        {
            'label':       'Language Prestige Analysis',
            'id_col':      'Observation_ID',
            'content_col': 'Variant',
            'output':      'out_G_langua_presti_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
        {
            'label':       'Language Trend Forecasting',
            'id_col':      'Observation_ID',
            'content_col': 'Variant',
            'output':      'out_G_langua_trendf_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
        {
            'label':       'Sociolinguistic Machine Learning',
            'id_col':      'Observation_ID',
            'content_col': 'Variant',
            'output':      'out_G_sociol_machin_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
        {
            'label':       'Sociolinguistic Trend Forecasting',
            'id_col':      'Observation_ID',
            'content_col': 'Variant',
            'output':      'out_G_sociol_trendf_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
        {
            'label':       'Language Choice Modeling',
            'id_col':      'Observation_ID',
            'content_col': 'Variant',
            'output':      'out_G_langua_choice_preprocessed.csv',
            'extra':       [],
            'extra_note':  [],
        },
        {
            'label':       'Language Practice Network Mapping',
            'id_col':      'Observation_ID',
            'content_col': 'Variant',
            'output':      'out_G_langua_practi_preprocessed.csv',
            'extra':       ['network_edges_raw.csv'],
            'extra_note':  [
                'network_edges_raw.csv  — raw edge list file with columns '
                'Speaker_A, Speaker_B, and Tie_Type. Speaker_A and '
                'Speaker_B values must match the Speaker_ID values in '
                'your main data file exactly (case-sensitive). '
                'This file is NOT produced by this script; you must '
                'prepare it separately before running the Data Processing '
                'Script (Listing 10.17).'
            ],
        },
    ],
}

CLUSTER_LABELS = {
    'A': 'Textual Discourse Processing',
    'B': 'Survey & Attitudinal Measurement',
    'C': 'Observational & Interactional Field Analysis',
    'D': 'Spatial & Network Structural Analysis',
    'E': 'Corpus-Based Pattern Analysis',
    'F': 'Intervention & Program Evaluation',
    'G': 'Predictive & Simulation Modeling',
}


# -------------------------------------------------------------
# SECTION 2 — MENU HELPERS
# -------------------------------------------------------------

def prompt_int(prompt, lo, hi):
    """Prompt until the user enters an integer in [lo, hi]."""
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        print(f"  Please enter a number between {lo} and {hi}.")


def prompt_path(prompt, default):
    """Prompt for a file path; return default if user presses Enter."""
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def prompt_yn(prompt):
    """Prompt for y/n confirmation."""
    while True:
        raw = input(prompt).strip().lower()
        if raw in ('y', 'n'):
            return raw == 'y'
        print("  Please enter y or n.")


# -------------------------------------------------------------
# SECTION 3 — MAIN MENU
# -------------------------------------------------------------

def run_menu():
    print()
    print("=" * 60)
    print("  PREREQUISITE PREPROCESSING SCRIPT")
    print("  Sociolinguistic Research: Facilitating Analysis with AI")
    print("=" * 60)

    # --- Cluster selection ---
    print()
    print("Select your pipeline cluster:")
    cluster_keys = list(CLUSTER_LABELS.keys())
    for i, key in enumerate(cluster_keys, 1):
        print(f"  {i}. Cluster {key} \u2014 {CLUSTER_LABELS[key]}")
    print()
    c_idx = prompt_int("Enter choice (1\u20137): ", 1, 7)
    CLUSTER = cluster_keys[c_idx - 1]

    # --- Method submenu ---
    methods = METHODS[CLUSTER]
    print()
    print(f"Select your method (Cluster {CLUSTER} \u2014 "
          f"{CLUSTER_LABELS[CLUSTER]}):")
    for i, m in enumerate(methods, 1):
        print(f"  {i}. {m['label']}")
    print()
    m_idx = prompt_int(f"Enter choice (1\u2013{len(methods)}): ", 1, len(methods))
    method = methods[m_idx - 1]

    # --- Additional file notice ---
    if method['extra']:
        print()
        print("  NOTE: This method requires the following additional")
        print("  input file(s) that are NOT produced by this script:")
        for note in method['extra_note']:
            print()
            words = note.split()
            line = "    "
            for word in words:
                if len(line) + len(word) + 1 > 62:
                    print(line)
                    line = "    " + word
                else:
                    line += (" " if line.strip() else "") + word
            if line.strip():
                print(line)
        print()
        print("  You do not need that file now. This script only")
        print("  produces the preprocessed node/participant file.")
        print("  Prepare the additional file before running the")
        print("  cluster-level Data Processing Script.")

    # --- File paths ---
    print()
    INPUT_FILE  = prompt_path("Enter input file path  ", "raw_data.csv")
    OUTPUT_FILE = prompt_path("Enter output file path ", method['output'])

    # --- Confirmation ---
    print()
    print("  Configuration confirmed:")
    print(f"    Cluster : {CLUSTER} \u2014 {CLUSTER_LABELS[CLUSTER]}")
    print(f"    Method  : {method['label']}")
    print(f"    Input   : {INPUT_FILE}")
    print(f"    Output  : {OUTPUT_FILE}")
    if method['extra']:
        print(f"    Additional file(s) required later: "
              f"{', '.join(method['extra'])}")
    print()
    if not prompt_yn("Proceed? (y/n): "):
        print("  Cancelled. Re-run the script to start again.")
        return None

    return CLUSTER, method, INPUT_FILE, OUTPUT_FILE


# -------------------------------------------------------------
# SECTION 4 — PREPROCESSING PIPELINE
# -------------------------------------------------------------

def run_preprocessing(CLUSTER, method, INPUT_FILE, OUTPUT_FILE):

    ID_COL      = method['id_col']
    CONTENT_COL = method['content_col']

    # =========================================================
    # UNIVERSAL BLOCK (runs for ALL clusters)
    # =========================================================

    ##### 1. Load with explicit UTF-8 encoding
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    print(f"Records loaded: {len(df)}")

    ##### 2. Deduplication
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        print(f"Duplicate rows removed: {removed}")

    ##### 3. Required field validation
    missing_req = [c for c in [ID_COL, CONTENT_COL]
                   if c not in df.columns]
    if missing_req:
        raise ValueError(f"Required columns missing: {missing_req}")
    df = df.dropna(subset=[ID_COL, CONTENT_COL])
    print(f"Records after required-field filter: {len(df)}")

    ##### 4. Duplicate ID check
    if df[ID_COL].duplicated().any():
        n_dup = df[ID_COL].duplicated().sum()
        print(f"Warning: {n_dup} duplicate {ID_COL} values detected. "
              f"Keeping first occurrence.")
        df = df.drop_duplicates(subset=[ID_COL], keep='first')

    ##### 5. String metadata harmonization
    def smart_title(s):
        return s if s.isupper() else s.title()

    for col in df.select_dtypes(include=['object', 'str']).columns:
        if col == CONTENT_COL:
            continue
        df[col] = df[col].astype(str).str.strip()
        if col != ID_COL:
            df[col] = df[col].apply(smart_title)

    ##### 6. Numeric field coercion
    for col, lo, hi in [('Age', 0, 120), ('Year', 1800, 2100)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            invalid = (df[col].isna() |
                       (df[col] < lo) | (df[col] > hi))
            if invalid.any():
                print(f"Warning: {invalid.sum()} records with invalid "
                      f"{col} values (set to NaN).")

    ##### 7. Remove empty content records
    df[CONTENT_COL] = df[CONTENT_COL].astype(str).str.strip()
    empty = df[CONTENT_COL].str.len() == 0
    if empty.any():
        print(f"Empty {CONTENT_COL} records removed: {empty.sum()}")
        df = df[~empty]

    ##### 8. Encoding anomaly flag
    def non_ascii_ratio(text):
        if not isinstance(text, str) or len(text) == 0:
            return 0.0
        return sum(1 for c in text if ord(c) > 127) / len(text)

    df['Encoding_Flag'] = df[CONTENT_COL].apply(
        lambda x: 1 if non_ascii_ratio(str(x)) > 0.15 else 0)
    n_flagged = df['Encoding_Flag'].sum()
    if n_flagged > 0:
        print(f"Warning: {n_flagged} records flagged for high "
              f"non-ASCII character ratio (>15%).")
    print(f"Universal block complete. Records: {len(df)}")

    # =========================================================
    # CLUSTER-SPECIFIC BLOCKS
    # =========================================================

    if CLUSTER == 'A':
        ##### Cluster A: Textual Discourse Processing
        df[CONTENT_COL] = df[CONTENT_COL].apply(
            lambda x: re.sub(r'\s+', ' ', str(x)))
        df['Unit_Length'] = df[CONTENT_COL].apply(
            lambda x: len(x.split()))
        for col in ['Analytical_Category',
                    'Code_Feature_A', 'Code_Feature_B']:
            df[col] = np.nan
        print("Cluster A block complete.")

    elif CLUSTER == 'B':
        ##### Cluster B: Survey & Attitudinal Measurement
        likert_map = {
            'Strongly disagree': 1, 'Disagree': 2,
            'Neutral': 3, 'Agree': 4, 'Strongly agree': 5
        }
        likert_cols = [c for c in df.columns
                       if c.startswith('Attitude_Q')]
        if not likert_cols:
            print("Warning: no Attitude_Q columns found. "
                  "Check column names.")
        for col in likert_cols:
            df[col] = df[col].map(likert_map)
            unmapped = df[col].isna().sum()
            if unmapped > 0:
                print(f"Warning: {unmapped} unmapped values "
                      f"in {col} after Likert encoding.")
        if likert_cols:
            df[likert_cols] = df[likert_cols].apply(
                lambda x: x.fillna(x.mean()))
            df['Attitude_Score'] = df[likert_cols].mean(axis=1)
            print(f"Composite Attitude_Score computed from "
                  f"{len(likert_cols)} items.")
        print("Cluster B block complete.")

    elif CLUSTER == 'C':
        ##### Cluster C: Observational & Interactional Field Analysis
        if 'Speaker' in df.columns:
            df['Speaker'] = (df['Speaker'].astype(str)
                             .str.strip().str.upper())
        for time_col in ['Start_Time', 'End_Time']:
            if time_col in df.columns:
                df[time_col] = pd.to_numeric(
                    df[time_col], errors='coerce')
        missing_timing = pd.Series(False, index=df.index)
        if 'Start_Time' in df.columns:
            missing_timing |= df['Start_Time'].isna()
        if 'End_Time' in df.columns:
            missing_timing |= df['End_Time'].isna()
        if missing_timing.any():
            print(f"Warning: {missing_timing.sum()} records "
                  f"with missing timing values.")
        if ('Start_Time' in df.columns and
                'End_Time' in df.columns):
            df['Turn_Duration'] = (df['End_Time'] -
                                   df['Start_Time'])
            anomalous = df['Turn_Duration'] <= 0
            if anomalous.any():
                print(f"Warning: {anomalous.sum()} records with "
                      f"anomalous turn durations.")
            df = df.sort_values('Start_Time').reset_index(drop=True)
            df['Gap'] = (df['Start_Time'] -
                         df['End_Time'].shift(1))
            df.loc[0, 'Gap'] = np.nan
        if 'Overlap' in df.columns:
            df['Overlap'] = (pd.to_numeric(df['Overlap'],
                             errors='coerce').fillna(0).astype(int))
        if CONTENT_COL in df.columns:
            df['Turn_Length'] = df[CONTENT_COL].apply(
                lambda x: len(str(x).split()))
        print("Cluster C block complete.")

    elif CLUSTER == 'D':
        ##### Cluster D: Spatial & Network Structural Analysis
        if ('Latitude' in df.columns and
                'Longitude' in df.columns):
            df['Latitude'] = pd.to_numeric(
                df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(
                df['Longitude'], errors='coerce')
            invalid_lat = (df['Latitude'].isna() |
                           (df['Latitude'].abs() > 90))
            invalid_lon = (df['Longitude'].isna() |
                           (df['Longitude'].abs() > 180))
            invalid_coords = invalid_lat | invalid_lon
            if invalid_coords.any():
                print(f"Warning: {invalid_coords.sum()} records "
                      f"with invalid or missing coordinates. "
                      f"Removing.")
                df = df[~invalid_coords]
        print("Cluster D block complete.")

    elif CLUSTER == 'E':
        ##### Cluster E: Corpus-Based Pattern Analysis
        if CONTENT_COL in df.columns:
            df[CONTENT_COL] = df[CONTENT_COL].apply(
                lambda x: re.sub(r'\s+', ' ', str(x)).strip())
        print("Cluster E block complete.")

    elif CLUSTER == 'F':
        ##### Cluster F: Intervention & Program Evaluation
        if 'Group' in df.columns:
            df['Group'] = df['Group'].str.strip().str.lower()
            valid_groups = {'treatment', 'control'}
            invalid = ~df['Group'].isin(valid_groups)
            if invalid.any():
                vals = df.loc[invalid, 'Group'].unique().tolist()
                print(f"Warning: {invalid.sum()} records with "
                      f"unrecognized Group values: {vals}")
        if 'Education' in df.columns:
            df['Education'] = (df['Education'].str.strip()
                               .str.lower())
            edu_map = {'primary': 1, 'secondary': 2, 'tertiary': 3}
            df['Education_Numeric'] = df['Education'].map(edu_map)
            unrecognized = df['Education_Numeric'].isna()
            if unrecognized.any():
                vals = (df.loc[unrecognized, 'Education']
                        .unique().tolist())
                print(f"Warning: {unrecognized.sum()} records with "
                      f"unrecognized Education values: {vals}")
        print("Cluster F block complete.")

    elif CLUSTER == 'G':
        ##### Cluster G: Predictive & Simulation Modeling
        outcome_col = None
        for candidate in ['Variant', 'Outcome_Variant',
                          'Chosen_Variant']:
            if candidate in df.columns:
                outcome_col = candidate
                break
        if outcome_col is not None:
            variants = df[outcome_col].str.strip().unique()
            if len(variants) == 2:
                variant_labels = sorted(variants)
                df['Outcome_Binary'] = (
                    df[outcome_col].str.strip()
                    == variant_labels[1]).astype(int)
                print(f"Outcome_Binary encoded: "
                      f"'{variant_labels[1]}' = 1, "
                      f"'{variant_labels[0]}' = 0")
                print(f"Class distribution:\n"
                      f"{df['Outcome_Binary'].value_counts()}")
            else:
                print(f"Warning: {outcome_col} has {len(variants)} "
                      f"unique values; expected 2. "
                      f"Skipping binary encoding.")
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            year_mean = df['Year'].mean()
            df['Year_Centered'] = df['Year'] - year_mean
            print(f"Year_Centered computed (mean = {year_mean:.1f}).")
        if 'Age' in df.columns:
            age_mean = df['Age'].mean()
            df['Age_Centered'] = df['Age'] - age_mean
            print(f"Age_Centered computed "
                  f"(mean = {age_mean:.2f} years).")
        if 'Education' in df.columns:
            df['Education'] = (df['Education'].str.strip()
                               .str.lower())
            edu_map = {'primary': 1, 'secondary': 2, 'tertiary': 3}
            df['Education_Num'] = df['Education'].map(edu_map)
            unrecognized = df['Education_Num'].isna()
            if unrecognized.any():
                vals = (df.loc[unrecognized, 'Education']
                        .unique().tolist())
                print(f"Warning: {unrecognized.sum()} records with "
                      f"unrecognized Education values: {vals}")
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].str.strip().str.title()
            gender_dummies = pd.get_dummies(
                df['Gender'], prefix='Gender',
                drop_first=True, dtype=int)
            df = pd.concat([df, gender_dummies], axis=1)
            print(f"Gender dummies created: "
                  f"{gender_dummies.columns.tolist()}")
        print("Cluster G block complete.")

    # =========================================================
    # SAVE OUTPUT
    # =========================================================
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Preprocessed file saved: {OUTPUT_FILE}")
    print("Prerequisite preprocessing completed.")


# -------------------------------------------------------------
# SECTION 5 — ENTRY POINT
# -------------------------------------------------------------

if __name__ == '__main__':
    result = run_menu()
    if result is not None:
        CLUSTER, method, INPUT_FILE, OUTPUT_FILE = result
        print()
        run_preprocessing(CLUSTER, method, INPUT_FILE, OUTPUT_FILE)
