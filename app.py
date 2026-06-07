import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from scipy.cluster.hierarchy import linkage, dendrogram

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Hierarchical Clustering Dashboard",
    page_icon="🌳",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* Main Background */
.stApp{
    background-color:#0F172A;
}

/* Make all text white */
html, body, [class*="css"]  {
    color: white;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: white !important;
}

/* Paragraphs */
p {
    color: white !important;
}

/* Labels */
label {
    color: white !important;
}

/* Markdown text */
div[data-testid="stMarkdownContainer"] {
    color: white !important;
}

/* Dataframe text */
[data-testid="stDataFrame"] {
    color: black !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: white !important;
}

[data-testid="stMetricLabel"] {
    color: white !important;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background-color:#111827;
}

[data-testid="stSidebar"] *{
    color:white !important;
}

/* Selectbox */
.stSelectbox label{
    color:white !important;
}

/* Slider */
.stSlider label{
    color:white !important;
}

/* Success box */
[data-testid="stAlert"]{
    color:white !important;
}

/* Main Title */
.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    background:linear-gradient(
        90deg,
        #8B5CF6,
        #EC4899
    );
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

/* Subtitle */
.subtitle{
    text-align:center;
    color:#E2E8F0 !important;
    font-size:18px;
}

/* Section Title */
.section-title{
    color:white !important;
    font-size:28px;
    font-weight:bold;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.markdown(
    "<div class='main-title'>🌳 Hierarchical Customer Segmentation</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Online Retail Customer Segmentation using Agglomerative Clustering</div>",
    unsafe_allow_html=True
)

st.markdown("""
<style>

/* Remove Streamlit Header */
header[data-testid="stHeader"]{
    background: transparent;
}

/* Remove top padding */
.block-container{
    padding-top:1rem;
}

/* Remove toolbar white area */
div[data-testid="stToolbar"]{
    right:2rem;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/retail_rfm_cleaned.csv")

df = load_data()


# ==========================================
# SIDEBAR
# ==========================================

st.markdown("""
<style>

/* Sidebar */
[data-testid="stSidebar"]{
    background: linear-gradient(
        180deg,
        #111827,
        #0F172A
    );
}

/* Sidebar Text */
[data-testid="stSidebar"] *{
    color:white;
}

/* Selectbox Container */
div[data-baseweb="select"] > div{
    background-color:#1E293B !important;
    border:2px solid #8B5CF6 !important;
    border-radius:10px !important;
}

/* Selected Value */
div[data-baseweb="select"] span{
    color:white !important;
    font-weight:bold !important;
}

/* Dropdown Menu */
div[data-baseweb="popover"]{
    background:white !important;
}

/* Dropdown Options */
div[data-baseweb="popover"] *{
    color:black !important;
}

/* Slider Label */
.stSlider label{
    color:white !important;
    font-weight:bold !important;
}

/* Selectbox Label */
.stSelectbox label{
    color:white !important;
    font-weight:bold !important;
}

</style>
""", unsafe_allow_html=True)

# Sidebar Header
st.sidebar.markdown("""
<h2 style="
color:white;
font-size:32px;
font-weight:bold;
margin-bottom:25px;
">
⚙️ Configuration
</h2>
""", unsafe_allow_html=True)

# Linkage Method Label
st.sidebar.markdown("""
<p style="
color:white;
font-size:18px;
font-weight:bold;
margin-bottom:5px;
">
Linkage Method
</p>
""", unsafe_allow_html=True)

linkage_method = st.sidebar.selectbox(
    label="",
    options=[
        "ward",
        "complete",
        "average",
        "single"
    ]
)

# Cluster Label
st.sidebar.markdown("""
<p style="
color:white;
font-size:18px;
font-weight:bold;
margin-top:20px;
margin-bottom:5px;
">
Number of Clusters
</p>
""", unsafe_allow_html=True)

n_clusters = st.sidebar.slider(
    "",
    min_value=2,
    max_value=10,
    value=4
)

# ==================================================
# DATASET OVERVIEW
# ==================================================

st.markdown(
    "<div class='section-title'>📁 Dataset Overview</div>",
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

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

# ==================================================
# CORRELATION HEATMAP
# ==================================================

st.markdown(
    "<div class='section-title'>🔥 Correlation Heatmap</div>",
    unsafe_allow_html=True
)

fig = px.imshow(
    df.corr(),
    text_auto=True,
    color_continuous_scale="Plasma"
)

st.plotly_chart(fig, use_container_width=True)

# ==================================================
# FEATURE DISTRIBUTION
# ==================================================

st.markdown(
    "<div class='section-title'>📈 Feature Distribution</div>",
    unsafe_allow_html=True
)

feature = st.selectbox(
    "Choose Feature",
    df.columns
)

fig = px.histogram(
    df,
    x=feature,
    nbins=30,
    color_discrete_sequence=["#8B5CF6"]
)

st.plotly_chart(fig, use_container_width=True)

# ==================================================
# DENDROGRAM
# ==================================================

st.markdown(
    "<div class='section-title'>🌳 Dendrogram</div>",
    unsafe_allow_html=True
)

sample_size = min(500, len(df))

sample = df.sample(
    sample_size,
    random_state=42
)

linked = linkage(
    sample,
    method=linkage_method
)

fig, ax = plt.subplots(
    figsize=(12,6)
)

dendrogram(
    linked,
    truncate_mode="level",
    p=5,
    ax=ax
)

ax.set_title(
    f"Dendrogram ({linkage_method})"
)

st.pyplot(fig)

# ==================================================
# MODEL
# ==================================================

model = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage=linkage_method
)

clusters = model.fit_predict(df)

df_clustered = df.copy()

df_clustered["Cluster"] = clusters

# ==================================================
# METRICS
# ==================================================

st.markdown(
    "<div class='section-title'>📊 Evaluation Metrics</div>",
    unsafe_allow_html=True
)

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

m1,m2,m3 = st.columns(3)

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
st.markdown("""
<style>

[data-testid="stMetric"]{
    background:#1E293B;
    padding:20px;
    border-radius:15px;
    border:1px solid #8B5CF6;
}

[data-testid="stMetricLabel"]{
    color:#CBD5E1 !important;
}

[data-testid="stMetricValue"]{
    color:#EC4899 !important;
    font-size:32px !important;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# PCA VISUALIZATION
# ==================================================

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

pca_df["Cluster"] = clusters.astype(str)

fig = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    color_discrete_sequence=px.colors.qualitative.Bold,
    title="Customer Clusters"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================================
# CLUSTER DISTRIBUTION
# ==================================================

st.markdown(
    "<div class='section-title'>📌 Cluster Distribution</div>",
    unsafe_allow_html=True
)

cluster_counts = (
    pd.Series(clusters)
    .value_counts()
    .sort_index()
)

fig = px.bar(
    x=cluster_counts.index,
    y=cluster_counts.values,
    color=cluster_counts.index.astype(str)
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================================
# CLUSTER ANALYSIS
# ==================================================

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

selected_cluster = st.selectbox(
    "Select Cluster",
    sorted(df_clustered["Cluster"].unique())
)

st.dataframe(
    df_clustered[
        df_clustered["Cluster"] ==
        selected_cluster
    ].head(20)
)

# ==================================================
# DOWNLOAD
# ==================================================

st.markdown(
    "<div class='section-title'>⬇️ Download Results</div>",
    unsafe_allow_html=True
)

csv = df_clustered.to_csv(
    index=False
)

st.download_button(
    "Download Clustered Dataset",
    csv,
    "hierarchical_clustered_data.csv",
    "text/csv"
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

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.success(
    "Hierarchical Clustering Completed Successfully ✅"
)