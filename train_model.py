import pandas as pd
import numpy as np
import joblib

from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ==========================================
# LOAD CLEANED DATA
# ==========================================

df = pd.read_csv(
    "data/mall_cleaned.csv"
)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# ==========================================
# FEATURES
# ==========================================

X = df.copy()

# ==========================================
# GMM PARAMETERS
# ==========================================

n_components = 5

# ==========================================
# TRAIN MODEL
# ==========================================

gmm = GaussianMixture(
    n_components=n_components,
    covariance_type="full",
    random_state=42
)

gmm.fit(X)

# ==========================================
# CLUSTER LABELS
# ==========================================

clusters = gmm.predict(X)

df["Cluster"] = clusters

# ==========================================
# CLUSTER PROBABILITIES
# ==========================================

probabilities = gmm.predict_proba(X)

prob_df = pd.DataFrame(
    probabilities,
    columns=[
        f"Cluster_{i}_Probability"
        for i in range(n_components)
    ]
)

# ==========================================
# COMBINE DATA
# ==========================================

final_df = pd.concat(
    [df, prob_df],
    axis=1
)

# ==========================================
# METRICS
# ==========================================

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

# ==========================================
# AIC & BIC
# ==========================================

aic = gmm.aic(X)

bic = gmm.bic(X)

# ==========================================
# OUTPUT
# ==========================================

print("\n========== GMM RESULTS ==========")

print(
    f"Number of Components : {n_components}"
)

print(
    f"Silhouette Score : {silhouette:.4f}"
)

print(
    f"Davies-Bouldin Score : {davies:.4f}"
)

print(
    f"Calinski-Harabasz Score : {calinski:.4f}"
)

print(
    f"AIC : {aic:.2f}"
)

print(
    f"BIC : {bic:.2f}"
)

# ==========================================
# CLUSTER COUNTS
# ==========================================

print("\n========== CLUSTER COUNTS ==========")

print(
    pd.Series(clusters)
    .value_counts()
    .sort_index()
)

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    gmm,
    "models/gmm_model.pkl"
)

# ==========================================
# SAVE METRICS
# ==========================================

metrics = {
    "n_components": n_components,
    "silhouette_score": silhouette,
    "davies_bouldin_score": davies,
    "calinski_harabasz_score": calinski,
    "aic": aic,
    "bic": bic
}

joblib.dump(
    metrics,
    "models/gmm_metrics.pkl"
)

# ==========================================
# SAVE DATASET
# ==========================================

final_df.to_csv(
    "models/gmm_clustered_data.csv",
    index=False
)

# ==========================================
# SAVE CLUSTER ANALYSIS
# ==========================================

cluster_analysis = (
    final_df
    .groupby("Cluster")
    .mean(numeric_only=True)
)

cluster_analysis.to_csv(
    "models/gmm_cluster_analysis.csv"
)

# ==========================================
# SUCCESS MESSAGE
# ==========================================

print("\n===================================")
print("Gaussian Mixture Model Trained")
print("===================================")

print(
    "models/gmm_model.pkl"
)

print(
    "models/gmm_metrics.pkl"
)

print(
    "models/gmm_clustered_data.csv"
)

print(
    "models/gmm_cluster_analysis.csv"
)