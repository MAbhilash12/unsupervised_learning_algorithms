import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Wholesale Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#1e3a8a;
}

.subtitle {
    text-align:center;
    color:gray;
    font-size:18px;
}

.metric-box {
    padding:15px;
    border-radius:15px;
    background:#ffffff;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
}

[data-testid="stSidebar"] {
    background-color:#0f172a;
}

[data-testid="stSidebar"] * {
    color:white;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    "<div class='title'>📊 Wholesale Customer Segmentation</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>K-Means Clustering Dashboard</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/wholesale_cleaned.csv")

df = load_data()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Settings")

k = st.sidebar.slider(
    "Select Number of Clusters",
    2,
    10,
    3
)

# -----------------------------
# DATASET OVERVIEW
# -----------------------------
st.header("📁 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Head")
    st.dataframe(df.head())

with col2:
    st.subheader("Dataset Information")

    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Missing Values")
    st.dataframe(
        df.isnull().sum().reset_index().rename(
            columns={"index":"Column",0:"Missing Values"}
        )
    )

st.subheader("Statistical Summary")
st.dataframe(df.describe())

# -----------------------------
# FEATURES
# -----------------------------
X = df.drop(columns=["Channel", "Region"])

# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
st.header("🔥 Correlation Heatmap")

corr = X.corr()

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# HISTOGRAMS
# -----------------------------
st.header("📈 Feature Distributions")

feature = st.selectbox(
    "Select Feature",
    X.columns
)

fig = px.histogram(
    X,
    x=feature,
    nbins=30,
    color_discrete_sequence=["#2563eb"]
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# ELBOW METHOD
# -----------------------------
st.header("📉 Elbow Method")

wcss = []

for i in range(1, 11):

    km = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    km.fit(X_scaled)

    wcss.append(km.inertia_)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=list(range(1,11)),
        y=wcss,
        mode="lines+markers"
    )
)

fig.update_layout(
    title="Elbow Curve",
    xaxis_title="Clusters",
    yaxis_title="WCSS"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# KMEANS
# -----------------------------
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df_clustered = df.copy()
df_clustered["Cluster"] = clusters

# -----------------------------
# METRICS
# -----------------------------
st.header("📊 Evaluation Metrics")

sil = silhouette_score(
    X_scaled,
    clusters
)

db = davies_bouldin_score(
    X_scaled,
    clusters
)

ch = calinski_harabasz_score(
    X_scaled,
    clusters
)

m1, m2, m3 = st.columns(3)

m1.metric(
    "Silhouette Score",
    round(sil,3)
)

m2.metric(
    "Davies-Bouldin Score",
    round(db,3)
)

m3.metric(
    "Calinski-Harabasz Score",
    round(ch,2)
)

# -----------------------------
# PCA
# -----------------------------
st.header("🎯 PCA Cluster Visualization")

pca = PCA(n_components=2)

pca_data = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    pca_data,
    columns=["PC1","PC2"]
)

pca_df["Cluster"] = clusters.astype(str)

fig = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    title="Customer Clusters using PCA",
    color_discrete_sequence=px.colors.qualitative.Bold
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# CLUSTER DISTRIBUTION
# -----------------------------
st.header("📌 Cluster Distribution")

cluster_counts = (
    pd.Series(clusters)
    .value_counts()
    .sort_index()
)

fig = px.bar(
    x=cluster_counts.index,
    y=cluster_counts.values,
    labels={
        "x":"Cluster",
        "y":"Customers"
    },
    color=cluster_counts.index.astype(str)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# CLUSTER CENTERS
# -----------------------------
st.header("🎯 Cluster Centers")

centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=X.columns
)

st.dataframe(centers)

heatmap = px.imshow(
    centers,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Viridis"
)

st.plotly_chart(
    heatmap,
    use_container_width=True
)

# -----------------------------
# CLUSTER ANALYSIS
# -----------------------------
st.header("📊 Cluster Wise Spending Analysis")

analysis = (
    df_clustered
    .groupby("Cluster")
    .mean(numeric_only=True)
)

st.dataframe(analysis)

selected_cluster = st.selectbox(
    "Select Cluster",
    sorted(df_clustered["Cluster"].unique())
)

cluster_data = (
    df_clustered[df_clustered["Cluster"] == selected_cluster]
)

st.write(
    f"Customers in Cluster {selected_cluster}:",
    cluster_data.shape[0]
)

st.dataframe(
    cluster_data.head(20)
)

# -----------------------------
# DOWNLOAD
# -----------------------------
st.header("⬇️ Download Results")

csv = df_clustered.to_csv(
    index=False
)

st.download_button(
    label="Download Clustered Dataset",
    data=csv,
    file_name="clustered_customers.csv",
    mime="text/csv"
)

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    kmeans,
    "models/kmeans_model.pkl"
)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.success(
    "K-Means Clustering Completed Successfully ✅"
)