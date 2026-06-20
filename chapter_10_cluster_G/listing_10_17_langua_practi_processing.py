import pandas as pd
import numpy as np

# Load preprocessed node dataset (output of
# Prerequisite Preprocessing Script, Listing 3.1,
# Cluster G block)
nodes = pd.read_csv(
    "out_G_langua_practi_preprocessed.csv",
    encoding='utf-8')
print(f"Node file: {len(nodes)} speakers loaded.")

# Load raw edge file directly (second input file;
# not produced by the preprocessing script —
# prepare this file separately before running)
edges = pd.read_csv(
    "network_edges_raw.csv", encoding='utf-8')
edges = edges.drop_duplicates()
edges = edges.dropna(
    subset=['Speaker_A', 'Speaker_B'])
edges['Speaker_A'] = (
    edges['Speaker_A'].str.strip().str.upper())
edges['Speaker_B'] = (
    edges['Speaker_B'].str.strip().str.upper())

##### 1. Validate Tie_Strength labels
if 'Tie_Strength' in edges.columns:
    edges['Tie_Strength'] = (
        edges['Tie_Strength'].str.strip()
        .str.lower())
    valid = {'strong', 'weak'}
    invalid = ~edges['Tie_Strength'].isin(valid)
    if invalid.any():
        print(f"Warning: {invalid.sum()} edges with "
              f"unrecognized Tie_Strength values.")

##### 2. Remove self-loops
self_loops = (
    edges['Speaker_A'] == edges['Speaker_B'])
if self_loops.any():
    print(f"{self_loops.sum()} self-loop edges "
          f"removed.")
    edges = edges[~self_loops]
print(f"Edge file: {len(edges)} valid ties loaded.")

##### 3. Compute degree centrality
deg_a  = edges.groupby(
    'Speaker_A')['Speaker_B'].nunique()
deg_b  = edges.groupby(
    'Speaker_B')['Speaker_A'].nunique()
degree = deg_a.add(
    deg_b, fill_value=0).reset_index()
degree.columns = ['Speaker_ID', 'Degree']

##### 4. Compute clustering coefficient
# Proportion of a speaker's neighbors who are
# also connected to one another.
neighbor_map = {}
for _, row in edges.iterrows():
    a, b = row['Speaker_A'], row['Speaker_B']
    neighbor_map.setdefault(a, set()).add(b)
    neighbor_map.setdefault(b, set()).add(a)
clustering = {}
for node, neighbors in neighbor_map.items():
    k = len(neighbors)
    if k < 2:
        clustering[node] = 0.0
        continue
    links = sum(
        1 for u in neighbors
        for v in neighbors
        if u < v
        and v in neighbor_map.get(u, set()))
    clustering[node] = links / (k * (k - 1) / 2)
clust_df = pd.DataFrame(
    list(clustering.items()),
    columns=['Speaker_ID', 'Clustering_Coeff'])

##### 5. Compute betweenness centrality
# (approximate via BFS)
all_nodes   = list(neighbor_map.keys())
betweenness = {n: 0.0 for n in all_nodes}
for s in all_nodes:
    visited = {s}
    queue   = [s]
    pred    = {s: []}
    sigma   = {s: 1.0}
    dist    = {s: 0}
    stack   = []
    while queue:
        v = queue.pop(0)
        stack.append(v)
        for w in neighbor_map.get(v, set()):
            if w not in visited:
                visited.add(w)
                queue.append(w)
                dist[w]  = dist[v] + 1
                sigma[w] = 0.0
                pred[w]  = []
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]
                pred[w].append(v)
    delta = {n: 0.0 for n in all_nodes}
    while stack:
        w = stack.pop()
        for v in pred.get(w, []):
            if sigma[w] > 0:
                delta[v] += (
                    sigma[v] / sigma[w]
                    * (1 + delta[w]))
        if w != s:
            betweenness[w] += delta[w]
n = len(all_nodes)
if n > 2:
    norm = (n - 1) * (n - 2)
    betweenness = {
        k: v / norm
        for k, v in betweenness.items()}
between_df = pd.DataFrame(
    list(betweenness.items()),
    columns=['Speaker_ID', 'Betweenness'])

##### 6. Merge network metrics onto node file
nodes = nodes.merge(
    degree, on='Speaker_ID', how='left')
nodes = nodes.merge(
    clust_df, on='Speaker_ID', how='left')
nodes = nodes.merge(
    between_df, on='Speaker_ID', how='left')
nodes['Degree'] = (
    nodes['Degree'].fillna(0).astype(int))
nodes['Clustering_Coeff'] = (
    nodes['Clustering_Coeff'].fillna(0))
nodes['Betweenness'] = (
    nodes['Betweenness'].fillna(0))
print("Network metrics merged: Degree, "
      "Clustering_Coeff, Betweenness.")

##### 7. Save processed outputs
nodes.to_csv(
    "out_G_langua_practi_nodes_processed.csv",
    index=False)
edges.to_csv(
    "out_G_langua_practi_edges_processed.csv",
    index=False)
print("Language practice network mapping data "
      "processing completed.")
print("Output files:")
print("  out_G_langua_practi_nodes_processed.csv  "
      "-- speaker attributes + network metrics")
print("  out_G_langua_practi_edges_processed.csv  "
      "-- validated tie data")
