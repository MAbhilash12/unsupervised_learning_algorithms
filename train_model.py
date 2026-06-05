import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv(
    "data/wholesale_cleaned.csv"
)

X = df.drop(
    columns=["Channel","Region"]
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    kmeans,
    "models/kmeans_model.pkl"
)

print("Model saved")