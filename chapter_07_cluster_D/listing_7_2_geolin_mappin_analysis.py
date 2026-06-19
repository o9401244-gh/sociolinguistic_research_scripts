import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import MDS
from itertools import combinations

# Load processed datasets (output of Listing 7.1)
freq      = pd.read_csv(
    "out_D_geolin_mappin_frequencies.csv",
    encoding='utf-8')
dominant  = pd.read_csv(
    "out_D_geolin_mappin_dominant.csv",
    encoding='utf-8')
sites     = pd.read_csv(
    "out_D_geolin_mappin_sites.csv",
    encoding='utf-8')
distances = pd.read_csv(
    "out_D_geolin_mappin_distances.csv",
    encoding='utf-8')

site_list    = sorted(sites['Site_ID'].unique())
feature_list = sorted(freq['Feature'].unique())

##### 1. Variant distribution summaries
print("=== Variant Distribution Summary ===")
for feature in feature_list:
    fdata = freq[freq['Feature'] == feature]
    variant_freqs = (fdata.groupby('Variant')['Count']
                     .sum().sort_values(ascending=False))
    total = variant_freqs.sum()
    print(f"\nFeature: {feature} "
          f"(N sites = {fdata['Site_ID'].nunique()})")
    for variant, count in variant_freqs.items():
        print(f"  {variant}: {count} "
              f"({100*count/total:.1f}%)")

# Distribution of dominant variants per feature
print("\n=== Dominant Variant Distribution Per Feature ===")
for feature in feature_list:
    dom = dominant[dominant['Feature'] == feature]
    dom_counts = dom['Dominant_Variant'].value_counts()
    print(f"\nFeature: {feature}")
    for v, c in dom_counts.items():
        print(f"  {v}: {c} sites "
              f"({100*c/len(dom):.1f}%)")

##### 2. Geographic spread of dominant variants
print("\n=== Geographic Spread of Dominant Variants ===")
for feature in feature_list:
    dom = dominant[
        dominant['Feature'] == feature].copy()
    for variant in dom['Dominant_Variant'].unique():
        vdata = dom[dom['Dominant_Variant'] == variant]
        lat_range = (vdata['Site_Lat'].max()
                     - vdata['Site_Lat'].min())
        lon_range = (vdata['Site_Lon'].max()
                     - vdata['Site_Lon'].min())
        print(f"  {feature} / {variant}: "
              f"{len(vdata)} sites, "
              f"lat range={lat_range:.2f}, "
              f"lon range={lon_range:.2f}")

##### 3. Isogloss candidate identification
# For each feature, identify site pairs where dominant
# variants differ and sites are geographically proximate.
# Proximity threshold: adapt to study area (default 100 km).
PROXIMITY_THRESHOLD_KM = 100.0
print(f"\n=== Isogloss Candidates "
      f"(distance <= {PROXIMITY_THRESHOLD_KM} km) ===")
isogloss_records = []
for feature in feature_list:
    dom = (dominant[dominant['Feature'] == feature]
           .set_index('Site_ID')['Dominant_Variant']
           .to_dict())
    proximate = distances[
        distances['Distance_km'] <= PROXIMITY_THRESHOLD_KM]
    for _, row in proximate.iterrows():
        sa, sb = row['Site_A'], row['Site_B']
        if sa in dom and sb in dom:
            if dom[sa] != dom[sb]:
                isogloss_records.append({
                    'Feature': feature,
                    'Site_A': sa,
                    'Variant_A': dom[sa],
                    'Site_B': sb,
                    'Variant_B': dom[sb],
                    'Distance_km': row['Distance_km']})
isoglosses = pd.DataFrame(isogloss_records)
print(f"Total isogloss candidate site pairs: "
      f"{len(isoglosses)}")
if not isoglosses.empty:
    print("\nIsogloss candidates by feature:")
    print(isoglosses.groupby('Feature').size()
          .sort_values(ascending=False))
    isoglosses.to_csv(
        "out_D_geolin_mappin_isogloss_candidates.csv",
        index=False)

##### 4. Isogloss bundle detection
# Site pairs that appear as isogloss candidates across
# multiple features are bundle candidates.
if not isoglosses.empty:
    pair_key = isoglosses[['Site_A', 'Site_B']].apply(
        lambda r: tuple(sorted(
            [r['Site_A'], r['Site_B']])), axis=1)
    isoglosses['Pair_Key'] = pair_key
    bundle_counts = isoglosses.groupby('Pair_Key').size()
    bundles = bundle_counts[
        bundle_counts > 1].sort_values(ascending=False)
    print(f"\n=== Isogloss Bundle Candidates ===")
    print(f"Site pairs with isoglosses across >1 feature: "
          f"{len(bundles)}")
    if not bundles.empty:
        print(bundles.head(20))

##### 5. Aggregate dialectometric distance matrix
# For each pair of sites, compute the proportion of features
# for which the dominant variants differ (Hamming-style
# aggregate distance). Values range from 0 (identical) to 1
# (maximally different across all shared features).
n = len(site_list)
site_index = {s: i for i, s in enumerate(site_list)}
dist_matrix = np.zeros((n, n))
feature_coverage = np.zeros((n, n))
dom_pivot = dominant.pivot(
    index='Site_ID',
    columns='Feature',
    values='Dominant_Variant')
for s1, s2 in combinations(site_list, 2):
    i, j = site_index[s1], site_index[s2]
    if (s1 in dom_pivot.index
            and s2 in dom_pivot.index):
        row1 = dom_pivot.loc[s1]
        row2 = dom_pivot.loc[s2]
        shared = row1.notna() & row2.notna()
        n_shared = shared.sum()
        if n_shared > 0:
            n_differ = (
                row1[shared] != row2[shared]).sum()
            d = n_differ / n_shared
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            feature_coverage[i, j] = n_shared
            feature_coverage[j, i] = n_shared

print("\n=== Dialectometric Distance Matrix ===")
dm_df = pd.DataFrame(
    dist_matrix, index=site_list, columns=site_list)
print(dm_df.round(3))
dm_df.to_csv(
    "out_D_geolin_mappin_dialectometric_matrix.csv")
print(f"\nMean pairwise dialectometric distance: "
      f"{dist_matrix[dist_matrix > 0].mean():.3f}")
print(f"Max pairwise dialectometric distance: "
      f"{dist_matrix.max():.3f}")

##### 6. Multidimensional scaling of dialect distances
if n >= 3:
    mds = MDS(n_components=2,
              dissimilarity='precomputed',
              random_state=42,
              normalized_stress='auto')
    mds_coords = mds.fit_transform(dist_matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(mds_coords[:, 0], mds_coords[:, 1],
               s=60, color='steelblue', zorder=3)
    for i, site in enumerate(site_list):
        ax.annotate(
            site,
            (mds_coords[i, 0], mds_coords[i, 1]),
            fontsize=8, xytext=(4, 4),
            textcoords='offset points')
    ax.set_title("MDS Plot of Dialectometric Distances")
    ax.set_xlabel("MDS Dimension 1")
    ax.set_ylabel("MDS Dimension 2")
    plt.tight_layout()
    plt.savefig("out_D_geolin_mappin_mds.png", dpi=150)
    plt.show()
    mds_df = pd.DataFrame(
        mds_coords,
        index=site_list,
        columns=['MDS_1', 'MDS_2'])
    mds_df.index.name = 'Site_ID'
    mds_df.to_csv(
        "out_D_geolin_mappin_mds_coordinates.csv")

##### 7. Hierarchical clustering of dialect distances
if n >= 3:
    condensed = squareform(dist_matrix)
    linked = linkage(condensed, method='ward')
    fig, ax = plt.subplots(figsize=(max(10, n), 5))
    dendrogram(linked, labels=site_list, ax=ax,
               leaf_rotation=45, leaf_font_size=9)
    ax.set_title(
        "Hierarchical Clustering of Dialect Sites")
    ax.set_xlabel("Site")
    ax.set_ylabel(
        "Aggregate Dialectometric Distance")
    plt.tight_layout()
    plt.savefig(
        "out_D_geolin_mappin_dendrogram.png", dpi=150)
    plt.show()

##### 8. Feature-level geographic variation summary
print("\n=== Feature-Level Geographic Variation ===")
for feature in feature_list:
    dom = dominant[dominant['Feature'] == feature]
    n_variants = dom['Dominant_Variant'].nunique()
    n_sites = len(dom)
    mean_dom_freq = dom['Dominant_Frequency'].mean()
    print(f"  {feature}: {n_variants} dominant "
          f"variant(s) across {n_sites} sites, "
          f"mean dominant frequency = "
          f"{mean_dom_freq:.2f}")

##### 9. Co-occurrence analysis of coding categories
code_columns = [col for col in freq.columns
                if col.startswith('Code_')]
if len(code_columns) >= 2:
    print("\n=== Co-occurrence Counts Among "
          "Coding Categories ===")
    for col_a, col_b in combinations(code_columns, 2):
        co_occur = (
            (freq[col_a] == 1)
            & (freq[col_b] == 1)).sum()
        print(f"  {col_a} + {col_b}: {co_occur}")

print("\nGeolinguistic mapping analysis completed.")
print("Output files:")
print("  out_D_geolin_mappin_isogloss_candidates.csv")
print("  out_D_geolin_mappin_dialectometric_matrix.csv")
print("  out_D_geolin_mappin_mds_coordinates.csv")
print("  out_D_geolin_mappin_mds.png")
print("  out_D_geolin_mappin_dendrogram.png")
