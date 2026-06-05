import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="DBSCAN Clustering Dashboard",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

.stApp{
    background-color:#0B1120;
}

header[data-testid="stHeader"]{
    background:transparent;
}

[data-testid="stSidebar"]{
    background:#111827;
}

[data-testid="stSidebar"] *{
    color:white;
}

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    background:linear-gradient(
        90deg,
        #10B981,
        #06B6D4
    );
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle{
    text-align:center;
    color:#E5E7EB;
    font-size:18px;
}

.section-title{
    color:white;
    font-size:28px;
    font-weight:bold;
    margin-top:20px;
}

h1,h2,h3,h4,h5,h6,p,label{
    color:white !important;
}

[data-testid="stMetric"]{
    background:#1F2937;
    padding:20px;
    border-radius:15px;
    border:1px solid #10B981;
}

[data-testid="stMetricValue"]{
    color:#10B981 !important;
}

[data-testid="stMetricLabel"]{
    color:white !important;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================

st.markdown(
    "<div class='main-title'> DBSCAN Customer Segmentation</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Density Based Clustering on Credit Card Customers</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ======================================================
# LOAD DATA
# ======================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard_cleaned.csv")

df = load_data()

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("⚙️ DBSCAN Settings")

eps = st.sidebar.slider(
    "Epsilon (eps)",
    0.1,
    5.0,
    1.5,
    0.1
)

min_samples = st.sidebar.slider(
    "Min Samples",
    2,
    50,
    10
)

# ======================================================
# DATASET OVERVIEW
# ======================================================

st.markdown(
    "<div class='section-title'>📁 Dataset Overview</div>",
    unsafe_allow_html=True
)

c1,c2 = st.columns(2)

with c1:
    st.subheader("Dataset Head")
    st.dataframe(df.head())

with c2:
    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

st.subheader("Statistical Summary")
st.dataframe(df.describe())

# ======================================================
# CORRELATION HEATMAP
# ======================================================

st.markdown(
    "<div class='section-title'>🔥 Correlation Heatmap</div>",
    unsafe_allow_html=True
)

corr_fig = px.imshow(
    df.corr(),
    text_auto=False,
    color_continuous_scale="Viridis"
)

st.plotly_chart(
    corr_fig,
    use_container_width=True
)

# ======================================================
# FEATURE DISTRIBUTION
# ======================================================

st.markdown(
    "<div class='section-title'>📈 Feature Distribution</div>",
    unsafe_allow_html=True
)

selected_feature = st.selectbox(
    "Select Feature",
    df.columns
)

hist_fig = px.histogram(
    df,
    x=selected_feature,
    nbins=30,
    color_discrete_sequence=["#10B981"]
)

st.plotly_chart(
    hist_fig,
    use_container_width=True
)

# ======================================================
# DBSCAN MODEL
# ======================================================

model = DBSCAN(
    eps=eps,
    min_samples=min_samples
)

clusters = model.fit_predict(df)

df_clustered = df.copy()
df_clustered["Cluster"] = clusters

# ======================================================
# CLUSTER INFORMATION
# ======================================================

n_clusters = len(
    set(clusters)
) - (
    1 if -1 in clusters else 0
)

noise_points = np.sum(
    clusters == -1
)

noise_percentage = (
    noise_points /
    len(clusters)
) * 100

# ======================================================
# METRICS
# ======================================================

st.markdown(
    "<div class='section-title'>📊 Evaluation Metrics</div>",
    unsafe_allow_html=True
)

m1,m2,m3 = st.columns(3)

if len(set(clusters)) > 1:

    sil = silhouette_score(
        df,
        clusters
    )

    db = davies_bouldin_score(
        df,
        clusters
    )

    ch = calinski_harabasz_score(
        df,
        clusters
    )

    m1.metric(
        "Silhouette Score",
        round(sil,3)
    )

    m2.metric(
        "Davies-Bouldin",
        round(db,3)
    )

    m3.metric(
        "Calinski-Harabasz",
        round(ch,2)
    )

else:

    m1.metric(
        "Silhouette Score",
        "N/A"
    )

    m2.metric(
        "Davies-Bouldin",
        "N/A"
    )

    m3.metric(
        "Calinski-Harabasz",
        "N/A"
    )

# ======================================================
# NOISE ANALYSIS
# ======================================================

st.markdown(
    "<div class='section-title'>🚨 Noise Analysis</div>",
    unsafe_allow_html=True
)

n1,n2,n3 = st.columns(3)

n1.metric(
    "Clusters Found",
    n_clusters
)

n2.metric(
    "Noise Points",
    noise_points
)

n3.metric(
    "Noise %",
    f"{noise_percentage:.2f}%"
)

# ======================================================
# PCA VISUALIZATION
# ======================================================

st.markdown(
    "<div class='section-title'>🎯 PCA Visualization</div>",
    unsafe_allow_html=True
)

pca = PCA(
    n_components=2
)

pca_data = pca.fit_transform(df)

pca_df = pd.DataFrame(
    pca_data,
    columns=["PC1","PC2"]
)

pca_df["Cluster"] = (
    clusters.astype(str)
)

pca_fig = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    title="DBSCAN Clusters",
    color_discrete_sequence=px.colors.qualitative.Bold
)

st.plotly_chart(
    pca_fig,
    use_container_width=True
)

# ======================================================
# CLUSTER DISTRIBUTION
# ======================================================

st.markdown(
    "<div class='section-title'>📌 Cluster Distribution</div>",
    unsafe_allow_html=True
)

cluster_counts = (
    pd.Series(clusters)
    .value_counts()
    .sort_index()
)

bar_fig = px.bar(
    x=cluster_counts.index.astype(str),
    y=cluster_counts.values,
    color=cluster_counts.index.astype(str),
    labels={
        "x":"Cluster",
        "y":"Count"
    }
)

st.plotly_chart(
    bar_fig,
    use_container_width=True
)

# ======================================================
# CLUSTER ANALYSIS
# ======================================================

st.markdown(
    "<div class='section-title'>📊 Cluster Analysis</div>",
    unsafe_allow_html=True
)

analysis = (
    df_clustered
    .groupby("Cluster")
    .mean()
)

st.dataframe(analysis)

# ======================================================
# FILTER CLUSTER
# ======================================================

selected_cluster = st.selectbox(
    "Select Cluster",
    sorted(
        df_clustered["Cluster"]
        .unique()
    )
)

filtered_df = (
    df_clustered[
        df_clustered["Cluster"]
        == selected_cluster
    ]
)

st.write(
    f"Records in Cluster {selected_cluster}:",
    filtered_df.shape[0]
)

st.dataframe(
    filtered_df.head(20)
)

# ======================================================
# DOWNLOAD
# ======================================================

st.markdown(
    "<div class='section-title'>⬇️ Download Results</div>",
    unsafe_allow_html=True
)

st.markdown("""
<style>

/* Buttons */
.stButton button{
    background:linear-gradient(
        90deg,
        #8B5CF6,
        #EC4899
    ) !important;

    color:white !important;

    border:none !important;

    border-radius:12px !important;

    font-weight:bold !important;
}

/* Download Button */
.stDownloadButton button{
    background:linear-gradient(
        90deg,
        #22C55E,
        #16A34A
    ) !important;

    color:white !important;

    border:none !important;

    border-radius:12px !important;

    font-weight:bold !important;
}

</style>
""", unsafe_allow_html=True)


csv = df_clustered.to_csv(
    index=False
)

st.download_button(
    label="Download Clustered Dataset",
    data=csv,
    file_name="clustered_creditcard.csv",
    mime="text/csv"
)



# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.success(
    "DBSCAN Clustering Completed Successfully ✅"
)