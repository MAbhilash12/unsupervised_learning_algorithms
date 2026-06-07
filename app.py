import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="PCA Dashboard",
    page_icon="🔵",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main App */
.stApp{
    background:#0B1120;
}

/* Remove Streamlit Header */
header[data-testid="stHeader"]{
    background:transparent;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background:#111827;
}

[data-testid="stSidebar"] *{
    color:white !important;
}

/* Main Title */
.main-title{
    text-align:center;
    font-size:48px;
    font-weight:bold;

    background:linear-gradient(
        90deg,
        #3B82F6,
        #06B6D4
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

/* Subtitle */
.subtitle{
    text-align:center;
    color:#CBD5E1;
    font-size:18px;
}

/* Section Title */
.section-title{
    color:#06B6D4;
    font-size:28px;
    font-weight:bold;
    margin-top:20px;
    margin-bottom:10px;
}

/* Text */
h1,h2,h3,h4,h5,h6,p,label,span{
    color:white !important;
}

/* Metric Cards */
[data-testid="stMetric"]{
    background:#1E293B;
    border:1px solid #3B82F6;
    border-radius:12px;
    padding:15px;
}

[data-testid="stMetricValue"]{
    color:#06B6D4 !important;
}

[data-testid="stMetricLabel"]{
    color:white !important;
}

/* Selectbox */
div[data-baseweb="select"] > div{
    background:#1E293B !important;
    border:1px solid #3B82F6 !important;
}

div[data-baseweb="select"] span{
    color:white !important;
}

/* Download Button */
.stDownloadButton button{

    background:linear-gradient(
        90deg,
        #3B82F6,
        #06B6D4
    ) !important;

    color:white !important;

    border:none !important;

    border-radius:12px !important;

    font-weight:bold !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🔵 Principal Component Analysis Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Dimensionality Reduction using Breast Cancer Dataset</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================================
# LOAD FILES
# ==========================================

@st.cache_data
def load_original_data():
    return pd.read_csv(
        "data/breast_cancer_original.csv"
    )

@st.cache_data
def load_pca_data():
    return pd.read_csv(
        "data/breast_cancer_pca.csv"
    )

@st.cache_data
def load_variance():
    return pd.read_csv(
        "models/pca_variance.csv"
    )

@st.cache_data
def load_loadings():
    return pd.read_csv(
        "models/pca_loadings.csv",
        index_col=0
    )

original_df = load_original_data()

pca_df = load_pca_data()

variance_df = load_variance()

loadings_df = load_loadings()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("⚙️ PCA Controls")

pc_columns = [
    col for col in pca_df.columns
    if col.startswith("PC")
]

pc_x = st.sidebar.selectbox(
    "Select X Component",
    pc_columns,
    index=0
)

pc_y = st.sidebar.selectbox(
    "Select Y Component",
    pc_columns,
    index=1
)

top_features = st.sidebar.slider(
    "Top Contributing Features",
    5,
    20,
    10
)

selected_feature = st.sidebar.selectbox(
    "Feature Distribution",
    original_df.drop(
        columns=["target"]
    ).columns
)

# ==========================================
# DATASET OVERVIEW
# ==========================================

st.markdown(
    '<div class="section-title">📁 Dataset Overview</div>',
    unsafe_allow_html=True
)

c1,c2 = st.columns(2)

with c1:

    st.subheader("Original Dataset")

    st.dataframe(
        original_df.head()
    )

with c2:

    st.subheader("Dataset Information")

    st.write(
        "Rows:",
        original_df.shape[0]
    )

    st.write(
        "Columns:",
        original_df.shape[1]
    )

    st.subheader(
        "Missing Values"
    )

    st.dataframe(
        original_df.isnull()
        .sum()
    )

st.subheader(
    "Statistical Summary"
)

st.dataframe(
    original_df.describe()
)
# ==========================================
# TARGET DISTRIBUTION
# ==========================================

st.markdown(
    '<div class="section-title">🎯 Target Distribution</div>',
    unsafe_allow_html=True
)

target_counts = (
    original_df["target"]
    .value_counts()
    .sort_index()
)

fig_target = px.bar(
    x=["Malignant", "Benign"],
    y=target_counts.values,
    color=["Malignant", "Benign"],
    title="Breast Cancer Class Distribution",
    color_discrete_sequence=[
        "#EF4444",
        "#06B6D4"
    ]
)

fig_target.update_layout(
    plot_bgcolor="#0B1120",
    paper_bgcolor="#0B1120",
    font_color="white"
)

st.plotly_chart(
    fig_target,
    use_container_width=True
)

# ==========================================
# CORRELATION HEATMAP
# ==========================================

st.markdown(
    '<div class="section-title">🔥 Correlation Heatmap</div>',
    unsafe_allow_html=True
)

corr_matrix = original_df.corr()

fig_corr = px.imshow(
    corr_matrix,
    color_continuous_scale="Blues",
    aspect="auto"
)

fig_corr.update_layout(
    height=800,
    plot_bgcolor="#0B1120",
    paper_bgcolor="#0B1120",
    font_color="white"
)

st.plotly_chart(
    fig_corr,
    use_container_width=True
)

# ==========================================
# FEATURE EXPLORER
# ==========================================

st.markdown(
    '<div class="section-title">📈 Feature Explorer</div>',
    unsafe_allow_html=True
)

fig_hist = px.histogram(
    original_df,
    x=selected_feature,
    nbins=30,
    color_discrete_sequence=["#3B82F6"]
)

fig_hist.update_layout(
    plot_bgcolor="#0B1120",
    paper_bgcolor="#0B1120",
    font_color="white"
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)

# ==========================================
# FEATURE VS TARGET
# ==========================================

st.markdown(
    '<div class="section-title">📊 Feature vs Target</div>',
    unsafe_allow_html=True
)

fig_box = px.box(
    original_df,
    x="target",
    y=selected_feature,
    color="target",
    color_discrete_sequence=[
        "#EF4444",
        "#06B6D4"
    ]
)

fig_box.update_layout(
    plot_bgcolor="#0B1120",
    paper_bgcolor="#0B1120",
    font_color="white"
)

st.plotly_chart(
    fig_box,
    use_container_width=True
)

# ==========================================
# FEATURE CORRELATION ANALYSIS
# ==========================================

st.markdown(
    '<div class="section-title">🔍 Top Correlated Features</div>',
    unsafe_allow_html=True
)

corr_target = (
    original_df
    .corr()["target"]
    .drop("target")
    .abs()
    .sort_values(
        ascending=False
    )
)

top_corr = (
    corr_target
    .head(15)
)

fig_corr_target = px.bar(
    x=top_corr.values,
    y=top_corr.index,
    orientation="h",
    color=top_corr.values,
    color_continuous_scale="Blues"
)

fig_corr_target.update_layout(
    plot_bgcolor="#0B1120",
    paper_bgcolor="#0B1120",
    font_color="white",
    title="Top Features Correlated with Target"
)

st.plotly_chart(
    fig_corr_target,
    use_container_width=True
)

# ==========================================
# FEATURE COMPARISON
# ==========================================

st.markdown(
    '<div class="section-title">⚖️ Feature Comparison</div>',
    unsafe_allow_html=True
)

feature_x = st.selectbox(
    "Feature X",
    original_df.drop(
        columns=["target"]
    ).columns,
    key="feature_x"
)

feature_y = st.selectbox(
    "Feature Y",
    original_df.drop(
        columns=["target"]
    ).columns,
    index=1,
    key="feature_y"
)

scatter_fig = px.scatter(
    original_df,
    x=feature_x,
    y=feature_y,
    color=original_df["target"].astype(str),
    color_discrete_sequence=[
        "#EF4444",
        "#06B6D4"
    ]
)

scatter_fig.update_layout(
    plot_bgcolor="#0B1120",
    paper_bgcolor="#0B1120",
    font_color="white"
)

st.plotly_chart(
    scatter_fig,
    use_container_width=True
)
# ==========================================
# EXPLAINED VARIANCE ANALYSIS
# ==========================================

st.markdown(
    '<div class="section-title">📊 Explained Variance Analysis</div>',
    unsafe_allow_html=True
)

fig_variance = px.bar(
    variance_df,
    x="Component",
    y="Explained Variance",
    color="Explained Variance",
    color_continuous_scale="Blues"
)

fig_variance.update_layout(
    title="Explained Variance by Principal Components",
    plot_bgcolor="#0B1120",
    paper_bgcolor="#0B1120",
    font_color="white",
    height=500
)

st.plotly_chart(
    fig_variance,
    use_container_width=True
)

# ==========================================
# CUMULATIVE VARIANCE
# ==========================================

st.markdown(
    '<div class="section-title">📈 Cumulative Variance Curve</div>',
    unsafe_allow_html=True
)

fig_cumulative = go.Figure()

fig_cumulative.add_trace(
    go.Scatter(
        x=variance_df["Component"],
        y=variance_df["Cumulative Variance"],
        mode="lines+markers",
        name="Cumulative Variance"
    )
)

fig_cumulative.add_hline(
    y=0.95,
    line_dash="dash",
    line_color="red",
    annotation_text="95% Variance"
)

fig_cumulative.update_layout(
    title="Cumulative Explained Variance",
    xaxis_title="Principal Components",
    yaxis_title="Variance Retained",
    plot_bgcolor="#0B1120",
    paper_bgcolor="#0B1120",
    font_color="white",
    height=500
)

st.plotly_chart(
    fig_cumulative,
    use_container_width=True
)

# ==========================================
# COMPONENTS REQUIRED FOR 95%
# ==========================================

components_95 = (
    variance_df[
        variance_df["Cumulative Variance"] >= 0.95
    ]
    .index[0]
    + 1
)

# ==========================================
# PCA METRICS CARDS
# ==========================================

st.markdown(
    '<div class="section-title">📌 PCA Metrics</div>',
    unsafe_allow_html=True
)

m1,m2,m3,m4 = st.columns(4)

m1.metric(
    "Total Components",
    len(variance_df)
)

m2.metric(
    "95% Variance Components",
    components_95
)

m3.metric(
    "Variance Retained",
    f"{variance_df['Cumulative Variance'].iloc[-1]*100:.2f}%"
)

m4.metric(
    "Dataset Features",
    original_df.shape[1]-1
)

# ==========================================
# VARIANCE RETENTION TABLE
# ==========================================

st.markdown(
    '<div class="section-title">📋 Variance Retention Table</div>',
    unsafe_allow_html=True
)

variance_display = variance_df.copy()

variance_display["Explained Variance"] = (
    variance_display["Explained Variance"] * 100
).round(2)

variance_display["Cumulative Variance"] = (
    variance_display["Cumulative Variance"] * 100
).round(2)

st.dataframe(
    variance_display,
    use_container_width=True
)

# ==========================================
# TOP PCA COMPONENTS
# ==========================================

st.markdown(
    '<div class="section-title">🏆 Most Important Components</div>',
    unsafe_allow_html=True
)

top_components = (
    variance_df
    .sort_values(
        by="Explained Variance",
        ascending=False
    )
    .head(10)
)

fig_top = px.bar(
    top_components,
    x="Component",
    y="Explained Variance",
    color="Explained Variance",
    color_continuous_scale="Turbo"
)

fig_top.update_layout(
    title="Top Principal Components",
    plot_bgcolor="#0B1120",
    paper_bgcolor="#0B1120",
    font_color="white"
)

st.plotly_chart(
    fig_top,
    use_container_width=True
)

# ==========================================
# VARIANCE DISTRIBUTION PIE CHART
# ==========================================

st.markdown(
    '<div class="section-title">🥧 Variance Distribution</div>',
    unsafe_allow_html=True
)

top5 = variance_df.head(5)

fig_pie = px.pie(
    top5,
    names="Component",
    values="Explained Variance",
    hole=0.4
)

fig_pie.update_layout(
    plot_bgcolor="#0B1120",
    paper_bgcolor="#0B1120",
    font_color="white"
)

st.plotly_chart(
    fig_pie,
    use_container_width=True
)

# ==========================================
# COMPONENT SELECTION INSIGHT
# ==========================================

st.markdown(
    '<div class="section-title">💡 PCA Insight</div>',
    unsafe_allow_html=True
)

st.info(
    f"""
    The first **{components_95} principal components**
    retain approximately **95% of the dataset variance**.

    This means the original feature space can be reduced
    significantly while preserving most of the information.
    """
)
# ==========================================
# PCA SCATTER PLOT
# ==========================================

st.markdown(
    '<div class="section-title">🧠 PCA Scatter Plot</div>',
    unsafe_allow_html=True
)

scatter_fig = px.scatter(
    pca_df,
    x=pc_x,
    y=pc_y,
    color=pca_df["target"].astype(str),
    color_discrete_sequence=[
        "#EF4444",
        "#06B6D4"
    ],
    title=f"{pc_x} vs {pc_y}"
)

scatter_fig.update_layout(
    plot_bgcolor="#0B1120",
    paper_bgcolor="#0B1120",
    font_color="white",
    height=600
)

st.plotly_chart(
    scatter_fig,
    use_container_width=True
)

# ==========================================
# PCA LOADINGS HEATMAP
# ==========================================

st.markdown(
    '<div class="section-title">🌡️ PCA Loadings Heatmap</div>',
    unsafe_allow_html=True
)

heatmap_fig = px.imshow(
    loadings_df,
    color_continuous_scale="RdBu",
    aspect="auto"
)

heatmap_fig.update_layout(
    plot_bgcolor="#0B1120",
    paper_bgcolor="#0B1120",
    font_color="white",
    height=700
)

st.plotly_chart(
    heatmap_fig,
    use_container_width=True
)

# ==========================================
# COMPONENT CONTRIBUTION ANALYSIS
# ==========================================

st.markdown(
    '<div class="section-title">🏆 Feature Contribution Analysis</div>',
    unsafe_allow_html=True
)

selected_pc = st.selectbox(
    "Select Principal Component",
    loadings_df.columns
)

top_features_df = (
    loadings_df[selected_pc]
    .abs()
    .sort_values(
        ascending=False
    )
    .head(top_features)
)

contribution_fig = px.bar(
    x=top_features_df.values,
    y=top_features_df.index,
    orientation="h",
    color=top_features_df.values,
    color_continuous_scale="Turbo"
)

contribution_fig.update_layout(
    title=f"Top Features Contributing to {selected_pc}",
    plot_bgcolor="#0B1120",
    paper_bgcolor="#0B1120",
    font_color="white",
    height=600
)

st.plotly_chart(
    contribution_fig,
    use_container_width=True
)

# ==========================================
# TOP FEATURES TABLE
# ==========================================

st.markdown(
    '<div class="section-title">📋 Top Feature Table</div>',
    unsafe_allow_html=True
)

feature_table = pd.DataFrame({
    "Feature": top_features_df.index,
    "Contribution": top_features_df.values
})

st.dataframe(
    feature_table,
    use_container_width=True
)

# ==========================================
# PCA COMPONENT EXPLORER
# ==========================================

st.markdown(
    '<div class="section-title">🔎 PCA Component Explorer</div>',
    unsafe_allow_html=True
)

selected_component = st.selectbox(
    "Choose PCA Component",
    pc_columns
)

hist_fig = px.histogram(
    pca_df,
    x=selected_component,
    nbins=30,
    color_discrete_sequence=["#06B6D4"]
)

hist_fig.update_layout(
    plot_bgcolor="#0B1120",
    paper_bgcolor="#0B1120",
    font_color="white"
)

st.plotly_chart(
    hist_fig,
    use_container_width=True
)

# ==========================================
# PCA DATASET VIEWER
# ==========================================

st.markdown(
    '<div class="section-title">📄 PCA Dataset Viewer</div>',
    unsafe_allow_html=True
)

rows_to_show = st.slider(
    "Rows to Display",
    10,
    100,
    20
)

st.dataframe(
    pca_df.head(rows_to_show),
    use_container_width=True
)

# ==========================================
# PCA COMPONENT CORRELATION
# ==========================================

st.markdown(
    '<div class="section-title">📊 PCA Component Correlation</div>',
    unsafe_allow_html=True
)

pca_corr = pca_df.corr()

corr_fig = px.imshow(
    pca_corr,
    color_continuous_scale="Blues",
    aspect="auto"
)

corr_fig.update_layout(
    plot_bgcolor="#0B1120",
    paper_bgcolor="#0B1120",
    font_color="white",
    height=700
)

st.plotly_chart(
    corr_fig,
    use_container_width=True
)

# ==========================================
# DOWNLOAD SECTION
# ==========================================

st.markdown(
    '<div class="section-title">⬇️ Download Results</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    pca_csv = pca_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download PCA Dataset",
        data=pca_csv,
        file_name="pca_transformed.csv",
        mime="text/csv"
    )

with col2:

    loadings_csv = (
        loadings_df.to_csv()
    )

    st.download_button(
        label="Download PCA Loadings",
        data=loadings_csv,
        file_name="pca_loadings.csv",
        mime="text/csv"
    )

# ==========================================
# FINAL INSIGHTS
# ==========================================

st.markdown(
    '<div class="section-title">💡 Final Insights</div>',
    unsafe_allow_html=True
)

st.success(
    f"""
    PCA successfully reduced the dimensionality of the dataset.

    Original Features: {original_df.shape[1]-1}

    PCA Components: {len(pc_columns)}

    Variance Retained:
    {variance_df['Cumulative Variance'].iloc[-1]*100:.2f}%
    """
)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <center>
        <h4 style='color:#06B6D4'>
        🔵 PCA Dashboard Completed Successfully
        </h4>
    </center>
    """,
    unsafe_allow_html=True
)