import pandas as pd
import numpy as np
import joblib

from sklearn.decomposition import PCA

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/breast_cancer_cleaned.csv"
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
# PCA MODEL
# ==========================================

n_components = X.shape[1]

pca = PCA(
    n_components=n_components
)

X_pca = pca.fit_transform(X)

# ==========================================
# EXPLAINED VARIANCE
# ==========================================

explained_variance = (
    pca.explained_variance_ratio_
)

cumulative_variance = np.cumsum(
    explained_variance
)

# ==========================================
# METRICS
# ==========================================

metrics = {
    "n_components": n_components,
    "explained_variance_ratio":
    explained_variance,

    "cumulative_variance":
    cumulative_variance,

    "total_variance_retained":
    cumulative_variance[-1]
}

# ==========================================
# LOADINGS
# ==========================================

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[
        f"PC{i}"
        for i in range(
            1,
            n_components + 1
        )
    ],
    index=X.columns
)

# ==========================================
# PCA DATAFRAME
# ==========================================

pca_df = pd.DataFrame(
    X_pca,
    columns=[
        f"PC{i}"
        for i in range(
            1,
            n_components + 1
        )
    ]
)

pca_df["target"] = y.values

# ==========================================
# PRINT RESULTS
# ==========================================

print("\n========== PCA RESULTS ==========")

print(
    "Number of Components:",
    n_components
)

print(
    "Total Variance Retained:",
    round(
        cumulative_variance[-1],
        4
    )
)

print("\nExplained Variance:")

for i,var in enumerate(
    explained_variance,
    start=1
):

    print(
        f"PC{i}: {var:.4f}"
    )

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    pca,
    "models/pca_model.pkl"
)

# ==========================================
# SAVE METRICS
# ==========================================

joblib.dump(
    metrics,
    "models/pca_metrics.pkl"
)

# ==========================================
# SAVE LOADINGS
# ==========================================

loadings.to_csv(
    "models/pca_loadings.csv"
)

# ==========================================
# SAVE PCA DATASET
# ==========================================

pca_df.to_csv(
    "models/pca_transformed.csv",
    index=False
)

# ==========================================
# SAVE VARIANCE TABLE
# ==========================================

variance_df = pd.DataFrame({
    "Component": [
        f"PC{i}"
        for i in range(
            1,
            n_components + 1
        )
    ],

    "Explained Variance":
    explained_variance,

    "Cumulative Variance":
    cumulative_variance
})

variance_df.to_csv(
    "models/pca_variance.csv",
    index=False
)

# ==========================================
# SUCCESS MESSAGE
# ==========================================

print("\n================================")
print("PCA Training Completed")
print("================================")

print(
    "models/pca_model.pkl"
)

print(
    "models/pca_metrics.pkl"
)

print(
    "models/pca_loadings.csv"
)

print(
    "models/pca_variance.csv"
)

print(
    "models/pca_transformed.csv"
)