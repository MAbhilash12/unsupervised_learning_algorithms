import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="t-SNE Dashboard",
    page_icon="🟣",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.stApp{
    background:#0F172A;
}

header[data-testid="stHeader"]{
    background:transparent;
}

[data-testid="stSidebar"]{
    background:#1E293B;
}

[data-testid="stSidebar"] *{
    color:white !important;
}

.main-title{
    text-align:center;
    font-size:45px;
    font-weight:bold;

    background:linear-gradient(
        90deg,
        #8B5CF6,
        #EC4899
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.section-title{
    color:#EC4899;
    font-size:26px;
    font-weight:bold;
}

h1,h2,h3,h4,h5,h6,p,label{
    color:white !important;
}

[data-testid="stMetric"]{
    background:#1E293B;
    border:1px solid #8B5CF6;
    border-radius:12px;
    padding:15px;
}

.stDownloadButton button{

    background:linear-gradient(
        90deg,
        #8B5CF6,
        #EC4899
    ) !important;

    color:white !important;

    border:none !important;

    border-radius:10px !important;

    font-weight:bold !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD DATA
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "models/tsne_transformed.csv"
    )

df = load_data()

# =====================================
# HEADER
# =====================================

st.markdown(
    '<div class="main-title">🟣 t-SNE Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("⚙️ Controls")

selected_class = st.sidebar.selectbox(
    "Select Digit",
    sorted(df["target"].unique())
)

# =====================================
# DATASET OVERVIEW
# =====================================

st.markdown(
    '<div class="section-title">📁 Dataset Overview</div>',
    unsafe_allow_html=True
)

c1,c2 = st.columns(2)

with c1:
    st.dataframe(df.head())

with c2:

    st.metric(
        "Rows",
        df.shape[0]
    )

    st.metric(
        "Columns",
        df.shape[1]
    )

    st.metric(
        "Digit Classes",
        df["target"].nunique()
    )

# =====================================
# DIGIT DISTRIBUTION
# =====================================

st.markdown(
    '<div class="section-title">📊 Digit Distribution</div>',
    unsafe_allow_html=True
)

count_df = (
    df["target"]
    .value_counts()
    .sort_index()
)

fig_count = px.bar(
    x=count_df.index,
    y=count_df.values,
    color=count_df.index.astype(str),
    labels={
        "x":"Digit",
        "y":"Count"
    }
)

fig_count.update_layout(
    plot_bgcolor="#0F172A",
    paper_bgcolor="#0F172A",
    font_color="white"
)

st.plotly_chart(
    fig_count,
    use_container_width=True
)

# =====================================
# TSNE SCATTER
# =====================================

st.markdown(
    '<div class="section-title">🧠 t-SNE Visualization</div>',
    unsafe_allow_html=True
)

fig_tsne = px.scatter(
    df,
    x="TSNE1",
    y="TSNE2",
    color=df["target"].astype(str),
    title="Digit Clusters"
)

fig_tsne.update_layout(
    plot_bgcolor="#0F172A",
    paper_bgcolor="#0F172A",
    font_color="white"
)

st.plotly_chart(
    fig_tsne,
    use_container_width=True
)

# =====================================
# CLASS EXPLORER
# =====================================

st.markdown(
    '<div class="section-title">🔍 Class Explorer</div>',
    unsafe_allow_html=True
)

filtered_df = df[
    df["target"] == selected_class
]

fig_class = px.scatter(
    filtered_df,
    x="TSNE1",
    y="TSNE2",
    color=filtered_df["target"].astype(str),
    title=f"Digit {selected_class}"
)

fig_class.update_layout(
    plot_bgcolor="#0F172A",
    paper_bgcolor="#0F172A",
    font_color="white"
)

st.plotly_chart(
    fig_class,
    use_container_width=True
)

st.write(
    f"Samples in Digit {selected_class}:",
    len(filtered_df)
)

# =====================================
# CLUSTER CENTERS
# =====================================

st.markdown(
    '<div class="section-title">🎯 Cluster Centers</div>',
    unsafe_allow_html=True
)

centers = (
    df.groupby("target")
    [["TSNE1","TSNE2"]]
    .mean()
    .reset_index()
)

fig_centers = px.scatter(
    centers,
    x="TSNE1",
    y="TSNE2",
    text="target",
    size_max=30
)

fig_centers.update_traces(
    textposition="top center"
)

fig_centers.update_layout(
    plot_bgcolor="#0F172A",
    paper_bgcolor="#0F172A",
    font_color="white"
)

st.plotly_chart(
    fig_centers,
    use_container_width=True
)

# =====================================
# DATA TABLE
# =====================================

st.markdown(
    '<div class="section-title">📄 Transformed Dataset</div>',
    unsafe_allow_html=True
)

st.dataframe(
    df.head(50)
)

# =====================================
# DOWNLOAD
# =====================================

st.markdown(
    '<div class="section-title">⬇️ Download Dataset</div>',
    unsafe_allow_html=True
)

csv = df.to_csv(
    index=False
)

st.download_button(
    "Download t-SNE Dataset",
    csv,
    "tsne_transformed.csv",
    "text/csv"
)

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.success(
    "t-SNE Visualization Completed Successfully ✅"
)