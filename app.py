import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Isolation Forest Dashboard",
    page_icon="🚨",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.stApp{
    background:#111827;
}

header[data-testid="stHeader"]{
    background:transparent;
}

[data-testid="stSidebar"]{
    background:#1F2937;
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
        #EF4444,
        #F97316
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.section-title{
    color:#F97316;
    font-size:26px;
    font-weight:bold;
}

h1,h2,h3,h4,h5,h6,p,label{
    color:white !important;
}

[data-testid="stMetric"]{
    background:#1F2937;
    border:1px solid #EF4444;
    border-radius:12px;
    padding:15px;
}

[data-testid="stMetricValue"]{
    color:#F97316 !important;
}

.stDownloadButton button{

    background:linear-gradient(
        90deg,
        #EF4444,
        #F97316
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
        "models/anomaly_results.csv"
    )

df = load_data()

# =====================================
# LOAD METRICS
# =====================================

try:
    metrics = joblib.load(
        "models/isolation_metrics.pkl"
    )
except:
    metrics = {}

# =====================================
# HEADER
# =====================================

st.markdown(
    '<div class="main-title">🚨 Isolation Forest Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("⚙️ Controls")

view_option = st.sidebar.radio(
    "View Data",
    [
        "All Records",
        "Only Anomalies",
        "Only Normal"
    ]
)

# =====================================
# DATASET OVERVIEW
# =====================================

st.markdown(
    '<div class="section-title">📁 Dataset Overview</div>',
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

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

# =====================================
# ANOMALY STATS
# =====================================

anomalies = (
    df["Anomaly"] == -1
).sum()

normal = (
    df["Anomaly"] == 1
).sum()

anomaly_percent = (
    anomalies /
    len(df)
) * 100

st.markdown(
    '<div class="section-title">📊 Anomaly Statistics</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

m1.metric(
    "Normal Records",
    normal
)

m2.metric(
    "Anomalies",
    anomalies
)

m3.metric(
    "Anomaly %",
    f"{anomaly_percent:.2f}%"
)

# =====================================
# PIE CHART
# =====================================

st.markdown(
    '<div class="section-title">🥧 Normal vs Anomaly</div>',
    unsafe_allow_html=True
)

pie_df = pd.DataFrame({
    "Type":[
        "Normal",
        "Anomaly"
    ],
    "Count":[
        normal,
        anomalies
    ]
})

fig_pie = px.pie(
    pie_df,
    names="Type",
    values="Count",
    hole=0.4,
    color="Type",
    color_discrete_map={
        "Normal":"#3B82F6",
        "Anomaly":"#EF4444"
    }
)

fig_pie.update_layout(
    paper_bgcolor="#111827",
    font_color="white"
)

st.plotly_chart(
    fig_pie,
    use_container_width=True
)

# =====================================
# SCATTER PLOT
# =====================================

st.markdown(
    '<div class="section-title">📈 Anomaly Visualization</div>',
    unsafe_allow_html=True
)

if (
    "Amount" in df.columns
    and
    "Time" in df.columns
):

    fig_scatter = px.scatter(
        df,
        x="Time",
        y="Amount",
        color=df["Anomaly"]
        .astype(str),
        color_discrete_map={
            "1":"#3B82F6",
            "-1":"#EF4444"
        },
        title="Time vs Amount"
    )

    fig_scatter.update_layout(
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font_color="white"
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )

# =====================================
# FILTER DATA
# =====================================

st.markdown(
    '<div class="section-title">🔍 Data Explorer</div>',
    unsafe_allow_html=True
)

if view_option == "Only Anomalies":

    filtered_df = df[
        df["Anomaly"] == -1
    ]

elif view_option == "Only Normal":

    filtered_df = df[
        df["Anomaly"] == 1
    ]

else:

    filtered_df = df

st.write(
    f"Records Found: {len(filtered_df)}"
)

st.dataframe(
    filtered_df.head(100)
)

# =====================================
# TOP ANOMALIES
# =====================================

st.markdown(
    '<div class="section-title">🚨 Sample Anomalies</div>',
    unsafe_allow_html=True
)

anomaly_df = df[
    df["Anomaly"] == -1
]

st.dataframe(
    anomaly_df.head(20)
)

# =====================================
# DOWNLOAD
# =====================================

st.markdown(
    '<div class="section-title">⬇️ Download Results</div>',
    unsafe_allow_html=True
)

csv = df.to_csv(
    index=False
)

st.download_button(
    "Download Anomaly Results",
    csv,
    "anomaly_results.csv",
    "text/csv"
)

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.success(
    "Isolation Forest Analysis Completed Successfully ✅"
)
