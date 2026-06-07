import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/digits_original.csv"
)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop(
    columns=["target"]
)

y = df["target"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# TSNE PARAMETERS
# ==========================================

n_components = 2
perplexity = 30
learning_rate = 200
random_state = 42

# ==========================================
# TSNE TRANSFORMATION
# ==========================================

tsne = TSNE(
    n_components=n_components,
    perplexity=perplexity,
    learning_rate=learning_rate,
    random_state=random_state
)

X_tsne = tsne.fit_transform(
    X_scaled
)

# ==========================================
# CREATE TSNE DATAFRAME
# ==========================================

tsne_df = pd.DataFrame(
    X_tsne,
    columns=[
        "TSNE1",
        "TSNE2"
    ]
)

tsne_df["target"] = y.values

# ==========================================
# CLUSTER CENTERS
# ==========================================

cluster_centers = (
    tsne_df
    .groupby("target")
    [["TSNE1", "TSNE2"]]
    .mean()
)

# ==========================================
# METRICS
# ==========================================

metrics = {
    "n_components": n_components,
    "perplexity": perplexity,
    "learning_rate": learning_rate,
    "samples": len(tsne_df),
    "features": X.shape[1],
    "classes": len(
        tsne_df["target"].unique()
    )
}

# ==========================================
# PRINT RESULTS
# ==========================================

print("\n========== TSNE RESULTS ==========")

print(
    "Samples:",
    metrics["samples"]
)

print(
    "Features:",
    metrics["features"]
)

print(
    "Classes:",
    metrics["classes"]
)

print(
    "Perplexity:",
    perplexity
)

print(
    "Learning Rate:",
    learning_rate
)

# ==========================================
# SAVE TRANSFORMED DATA
# ==========================================

tsne_df.to_csv(
    "models/tsne_transformed.csv",
    index=False
)

# ==========================================
# SAVE CLUSTER CENTERS
# ==========================================

cluster_centers.to_csv(
    "models/tsne_cluster_centers.csv"
)

# ==========================================
# SAVE METRICS
# ==========================================

joblib.dump(
    metrics,
    "models/tsne_metrics.pkl"
)

# ==========================================
# SAVE SCALER
# ==========================================

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# ==========================================
# SUCCESS MESSAGE
# ==========================================

print("\n===================================")
print("t-SNE Transformation Completed")
print("===================================")

print(
    "models/tsne_transformed.csv"
)

print(
    "models/tsne_cluster_centers.csv"
)

print(
    "models/tsne_metrics.pkl"
)

print(
    "models/scaler.pkl"
)