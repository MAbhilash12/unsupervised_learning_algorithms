import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ==================================
# LOAD DATA
# ==================================

df = pd.read_csv(
    "data/retail_rfm_cleaned.csv"
)

print("Dataset Shape:", df.shape)

# ==================================
# FEATURES
# ==================================

X = df.copy()

# ==================================
# SCALING
# ==================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==================================
# HIERARCHICAL CLUSTERING
# ==================================

n_clusters = 4

model = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric="euclidean",
    linkage="ward"
)

clusters = model.fit_predict(X_scaled)

# ==================================
# ADD CLUSTERS
# ==================================

df["Cluster"] = clusters

# ==================================
# EVALUATION METRICS
# ==================================

silhouette = silhouette_score(
    X_scaled,
    clusters
)

davies = davies_bouldin_score(
    X_scaled,
    clusters
)

calinski = calinski_harabasz_score(
    X_scaled,
    clusters
)

print("\n===== MODEL METRICS =====")

print(
    f"Silhouette Score: {silhouette:.4f}"
)

print(
    f"Davies-Bouldin Score: {davies:.4f}"
)

print(
    f"Calinski-Harabasz Score: {calinski:.4f}"
)

# ==================================
# CLUSTER DISTRIBUTION
# ==================================

print("\n===== CLUSTER COUNTS =====")

print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)

# ==================================
# SAVE SCALER
# ==================================

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# ==================================
# SAVE CLUSTERED DATA
# ==================================

df.to_csv(
    "models/clustered_customers.csv",
    index=False
)

# ==================================
# SAVE METRICS
# ==================================

metrics = {
    "silhouette_score": silhouette,
    "davies_bouldin_score": davies,
    "calinski_harabasz_score": calinski,
    "n_clusters": n_clusters,
    "linkage": "ward"
}

joblib.dump(
    metrics,
    "models/model_metrics.pkl"
)

print("\nFiles Saved Successfully")

print("models/scaler.pkl")
print("models/model_metrics.pkl")
print("models/clustered_customers.csv")