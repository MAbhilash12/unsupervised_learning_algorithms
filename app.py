
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Gaussian Mixture Model Dashboard",
    page_icon="🔥",
    layout="wide"
)

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>
.stApp{background:#0A0A0A;}
header[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{background:#111111;}
[data-testid="stSidebar"] *{color:white !important;}

.main-title{
text-align:center;
font-size:48px;
font-weight:bold;
background:linear-gradient(90deg,#FBBF24,#F59E0B,#EA580C);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.subtitle{
text-align:center;
color:#D1D5DB;
font-size:18px;
}

.section-title{
color:#FBBF24;
font-size:28px;
font-weight:bold;
margin-top:20px;
}

h1,h2,h3,h4,h5,h6,p,label,span{
color:white !important;
}

[data-testid="stMetric"]{
background:#1A1A1A;
border:1px solid #FBBF24;
border-radius:12px;
padding:15px;
}

[data-testid="stMetricValue"]{
color:#FBBF24 !important;
}

div[data-baseweb="select"] > div{
background:#1A1A1A !important;
border:1px solid #FBBF24 !important;
}

div[data-baseweb="select"] span{
color:white !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown('<div class="main-title">🔥 Gaussian Mixture Model Clustering</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Soft Customer Segmentation using Mall Customers Dataset</div>', unsafe_allow_html=True)
st.markdown("---")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("data/mall_cleaned.csv")

df = load_data()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("⚙️ Configuration")

n_components = st.sidebar.slider(
    "Number of Components",
    2,10,5
)

covariance_type = st.sidebar.selectbox(
    "Covariance Type",
    ["full","tied","diag","spherical"]
)

# =====================================================
# DATA OVERVIEW
# =====================================================
st.markdown('<div class="section-title">📁 Dataset Overview</div>', unsafe_allow_html=True)

c1,c2 = st.columns(2)

with c1:
    st.subheader("Dataset Head")
    st.dataframe(df.head())

with c2:
    st.subheader("Dataset Information")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])
    st.write("Missing Values")
    st.dataframe(df.isnull().sum())

st.subheader("Statistical Summary")
st.dataframe(df.describe())

# =====================================================
# CORRELATION
# =====================================================
st.markdown('<div class="section-title">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)

fig_corr = px.imshow(
    df.corr(),
    text_auto=True,
    color_continuous_scale="YlOrBr"
)

st.plotly_chart(fig_corr, use_container_width=True)

# =====================================================
# FEATURE DISTRIBUTION
# =====================================================
st.markdown('<div class="section-title">📈 Feature Distribution</div>', unsafe_allow_html=True)

feature = st.selectbox(
    "Select Feature",
    df.columns
)

fig_hist = px.histogram(
    df,
    x=feature,
    nbins=25,
    color_discrete_sequence=["#F59E0B"]
)

st.plotly_chart(fig_hist, use_container_width=True)

# =====================================================
# GMM MODEL
# =====================================================
gmm = GaussianMixture(
    n_components=n_components,
    covariance_type=covariance_type,
    random_state=42
)

gmm.fit(df)

clusters = gmm.predict(df)

probabilities = gmm.predict_proba(df)

df_clustered = df.copy()
df_clustered["Cluster"] = clusters

# =====================================================
# METRICS
# =====================================================
st.markdown('<div class="section-title">📊 Evaluation Metrics</div>', unsafe_allow_html=True)

sil = silhouette_score(df, clusters)
db = davies_bouldin_score(df, clusters)
ch = calinski_harabasz_score(df, clusters)

m1,m2,m3 = st.columns(3)

m1.metric("Silhouette Score", round(sil,3))
m2.metric("Davies-Bouldin Score", round(db,3))
m3.metric("Calinski-Harabasz Score", round(ch,2))

# =====================================================
# AIC BIC
# =====================================================
st.markdown('<div class="section-title">🎯 AIC & BIC</div>', unsafe_allow_html=True)

a1,a2 = st.columns(2)

a1.metric("AIC", round(gmm.aic(df),2))
a2.metric("BIC", round(gmm.bic(df),2))

# =====================================================
# PCA
# =====================================================
st.markdown('<div class="section-title">🎯 PCA Visualization</div>', unsafe_allow_html=True)

pca = PCA(n_components=2)

pca_data = pca.fit_transform(df)

pca_df = pd.DataFrame(
    pca_data,
    columns=["PC1","PC2"]
)

pca_df["Cluster"] = clusters.astype(str)

fig_pca = px.scatter(
    pca_df,
    x="PC1",
    y="PC2",
    color="Cluster",
    title="GMM Clusters",
    color_discrete_sequence=px.colors.qualitative.Bold
)

st.plotly_chart(fig_pca, use_container_width=True)

# =====================================================
# CLUSTER DISTRIBUTION
# =====================================================
st.markdown('<div class="section-title">📌 Cluster Distribution</div>', unsafe_allow_html=True)

counts = pd.Series(clusters).value_counts().sort_index()

fig_bar = px.bar(
    x=counts.index.astype(str),
    y=counts.values,
    color=counts.index.astype(str)
)

st.plotly_chart(fig_bar, use_container_width=True)

# =====================================================
# PROBABILITY TABLE
# =====================================================
st.markdown('<div class="section-title">🧠 Soft Clustering Probabilities</div>', unsafe_allow_html=True)

prob_df = pd.DataFrame(
    probabilities,
    columns=[
        f"Cluster_{i}_Prob"
        for i in range(n_components)
    ]
)

st.dataframe(prob_df.head(20))

# =====================================================
# PROBABILITY HEATMAP
# =====================================================
st.markdown('<div class="section-title">🌡️ Probability Heatmap</div>', unsafe_allow_html=True)

sample_prob = prob_df.head(50)

fig_heat = px.imshow(
    sample_prob.T,
    color_continuous_scale="YlOrRd",
    aspect="auto"
)

st.plotly_chart(fig_heat, use_container_width=True)

# =====================================================
# CLUSTER ANALYSIS
# =====================================================
st.markdown('<div class="section-title">📊 Cluster Analysis</div>', unsafe_allow_html=True)

analysis = df_clustered.groupby("Cluster").mean()

st.dataframe(analysis)

# =====================================================
# FILTER CLUSTER
# =====================================================
selected_cluster = st.selectbox(
    "Select Cluster",
    sorted(df_clustered["Cluster"].unique())
)

filtered = df_clustered[
    df_clustered["Cluster"] == selected_cluster
]

st.write(
    f"Records in Cluster {selected_cluster}:",
    filtered.shape[0]
)

st.dataframe(filtered.head(20))

# =====================================================
# DOWNLOAD
# =====================================================
st.markdown('<div class="section-title">⬇️ Download Results</div>', unsafe_allow_html=True)

final_output = pd.concat(
    [df_clustered, prob_df],
    axis=1
)

csv = final_output.to_csv(index=False)

st.download_button(
    "Download GMM Results",
    csv,
    "gmm_clustered_data.csv",
    "text/csv"
)

st.markdown("""
<style>

/* =======================================
   NORMAL BUTTONS
======================================= */

.stButton button{

    background:linear-gradient(
        90deg,
        #FBBF24,
        #F59E0B,
        #EA580C
    ) !important;

    color:black !important;

    border:none !important;

    border-radius:12px !important;

    font-weight:bold !important;

    transition:0.3s;
}

.stButton button:hover{

    background:linear-gradient(
        90deg,
        #EA580C,
        #F59E0B,
        #FBBF24
    ) !important;

    color:white !important;

    transform:scale(1.02);
}

/* =======================================
   DOWNLOAD BUTTON
======================================= */

.stDownloadButton button{

    background:linear-gradient(
        90deg,
        #FBBF24,
        #F59E0B
    ) !important;

    color:black !important;

    border:none !important;

    border-radius:12px !important;

    font-weight:bold !important;

    transition:0.3s;
}""", unsafe_allow_html=True)
st.success("Gaussian Mixture Model Completed Successfully ✅")
