import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

# Load processed dataset (output of Listing 4.1)
df = pd.read_csv("out_A_repres_discou_processed.csv")

##### 1. Frequency distribution of representational strategies
print("Frequency Distribution of Representational Strategies:")
strategy_counts = df['Representational_Strategy'].value_counts()
print(strategy_counts)

# Bar chart of strategy frequencies
strategy_counts.plot(kind='bar')
plt.title("Distribution of Representational Strategies")
plt.xlabel("Strategy")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

##### 2. Frequency distribution of social actor categories
print("\nFrequency Distribution of Social Actor Categories:")
actor_counts = df['Social_Actor'].value_counts()
print(actor_counts)

# Bar chart of social actor frequencies
actor_counts.plot(kind='bar')
plt.title("Distribution of Social Actor Categories")
plt.xlabel("Social Actor")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

##### 3. Cross-tabulation: strategy by social actor
crosstab_no_margins = None
if ('Representational_Strategy' in df.columns
        and 'Social_Actor' in df.columns):
    crosstab = pd.crosstab(
        df['Social_Actor'],
        df['Representational_Strategy'],
        margins=True)
    print("\nCross-tabulation: Social Actor by "
          "Representational Strategy:")
    print(crosstab)
    crosstab_no_margins = pd.crosstab(
        df['Social_Actor'],
        df['Representational_Strategy'])
    plt.figure(figsize=(10, 6))
    plt.imshow(crosstab_no_margins, aspect='auto',
               cmap='Blues')
    plt.colorbar(label='Frequency')
    plt.xticks(range(len(crosstab_no_margins.columns)),
               crosstab_no_margins.columns,
               rotation=45, ha='right')
    plt.yticks(range(len(crosstab_no_margins.index)),
               crosstab_no_margins.index)
    plt.title("Heatmap: Social Actor by "
              "Representational Strategy")
    plt.tight_layout()
    plt.show()
else:
    print("Warning: Representational_Strategy or "
          "Social_Actor column not found. "
          "Skipping step 3.")

##### 4. Chi-square test of independence
if crosstab_no_margins is not None:
    chi2, p_val, dof, expected = stats.chi2_contingency(
        crosstab_no_margins)
    print(f"\nChi-Square Test of Independence:")
    print(f"  chi2 = {chi2:.3f}, df = {dof}, "
          f"p = {p_val:.3f}")
else:
    print("Warning: Chi-square test (step 4) skipped — "
          "crosstab from step 3 was not computed.")

##### 5. Agency distribution: agentive vs. passive roles
agency_by_actor = None
if 'Agency' in df.columns:
    print("\nAgency Distribution:")
    agency_counts = df['Agency'].value_counts()
    print(agency_counts)
    agency_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title("Distribution of Agency Roles")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
    if 'Social_Actor' in df.columns:
        agency_by_actor = pd.crosstab(
            df['Social_Actor'], df['Agency'])
        print("\nAgency by Social Actor:")
        print(agency_by_actor)
    else:
        print("Warning: Social_Actor column not found. "
              "Skipping agency by actor crosstab.")
else:
    print("Warning: Agency column not found. "
          "Skipping step 5.")

##### 6. Framing pattern analysis
if 'Framing_Category' in df.columns:
    print("\nFraming Category Distribution:")
    framing_counts = (df['Framing_Category']
                      .value_counts())
    print(framing_counts)
    framing_counts.plot(kind='bar')
    plt.title("Distribution of Framing Categories")
    plt.xlabel("Framing Category")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("Warning: Framing_Category column not found. "
          "Skipping step 6.")

##### 7. Co-occurrence analysis of coding categories
code_columns = [col for col in df.columns
                if col.startswith('Code_')]
if len(code_columns) >= 2:
    print("\nCo-occurrence Counts Among Coding Categories:")
    for col_a, col_b in combinations(code_columns, 2):
        co_occur = (
            (df[col_a] == 1) & (df[col_b] == 1)).sum()
        print(f"  {col_a} + {col_b}: {co_occur}")
else:
    print("Warning: Fewer than 2 Code_ columns found. "
          "Skipping co-occurrence analysis.")

print("Representational discourse analysis completed.")
