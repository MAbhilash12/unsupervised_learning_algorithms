import pandas as pd
import numpy as np
import joblib

from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ============================================
# LOAD CLEANED DATA
# ============================================

df = pd.read_csv(
    "data/creditcard_cleaned.csv"
)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# ============================================
# FEATURES
# ============================================

X = df.copy()

# ============================================
# DBSCAN PARAMETERS
# ============================================

eps = 1.5
min_samples = 10

# ============================================
# TRAIN MODEL
# ============================================

dbscan = DBSCAN(
    eps=eps,
    min_samples=min_samples
)

clusters = dbscan.fit_predict(X)

# ============================================
# ADD CLUSTER LABELS
# ============================================

df["Cluster"] = clusters

# ============================================
# CLUSTER STATISTICS
# ============================================

n_clusters = len(
    set(clusters)
) - (
    1 if -1 in clusters else 0
)

n_noise = list(clusters).count(-1)

noise_percentage = (
    n_noise / len(clusters)
) * 100

print("\n===== DBSCAN RESULTS =====")

print(
    f"Number of Clusters: {n_clusters}"
)

print(
    f"Noise Points: {n_noise}"
)

print(
    f"Noise Percentage: {noise_percentage:.2f}%"
)

# ============================================
# EVALUATION METRICS
# ============================================

unique_clusters = set(clusters)

if len(unique_clusters) > 1:

    silhouette = silhouette_score(
        X,
        clusters
    )

    davies = davies_bouldin_score(
        X,
        clusters
    )

    calinski = calinski_harabasz_score(
        X,
        clusters
    )

    print("\n===== METRICS =====")

    print(
        f"Silhouette Score: {silhouette:.4f}"
    )

    print(
        f"Davies-Bouldin Score: {davies:.4f}"
    )

    print(
        f"Calinski-Harabasz Score: {calinski:.4f}"
    )

else:

    silhouette = None
    davies = None
    calinski = None

    print(
        "\nMetrics cannot be calculated."
    )

# ============================================
# CLUSTER DISTRIBUTION
# ============================================

print("\n===== CLUSTER COUNTS =====")

print(
    pd.Series(clusters)
    .value_counts()
    .sort_index()
)

# ============================================
# SAVE CLUSTERED DATASET
# ============================================

df.to_csv(
    "models/clustered_creditcard.csv",
    index=False
)

# ============================================
# SAVE METRICS
# ============================================

metrics = {
    "eps": eps,
    "min_samples": min_samples,
    "n_clusters": n_clusters,
    "noise_points": n_noise,
    "noise_percentage": noise_percentage,
    "silhouette_score": silhouette,
    "davies_bouldin_score": davies,
    "calinski_harabasz_score": calinski
}

joblib.dump(
    metrics,
    "models/dbscan_metrics.pkl"
)

# ============================================
# SAVE MODEL PARAMETERS
# ============================================

model_info = {
    "eps": eps,
    "min_samples": min_samples
}

joblib.dump(
    model_info,
    "models/dbscan_model.pkl"
)

# ============================================
# SUCCESS MESSAGE
# ============================================

print("\n===================================")
print("DBSCAN Training Completed")
print("===================================")

print("Saved Files:")

print(
    "models/clustered_creditcard.csv"
)

print(
    "models/dbscan_metrics.pkl"
)

print(
    "models/dbscan_model.pkl"
)