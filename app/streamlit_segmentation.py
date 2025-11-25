import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="Colgate Customer Segmentation", layout="wide")

st.title("👥 Colgate Customer Segmentation Dashboard")
st.markdown("Segmentation using RFM metrics, clustering, and persona insights.")

# -------------------------------------------------------------------------
# PATH MANAGEMENT
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "../data/colgate_customers_clustered.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../models/colgate_kmeans_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "../models/colgate_scaler.pkl")

df = pd.read_csv(DATA_PATH)
kmeans = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Cluster overview
st.subheader("📌 Cluster Distribution")
fig = px.histogram(df, x="cluster", title="Customer Count by Cluster", nbins=10)
st.plotly_chart(fig, use_container_width=True)

# RFM Visualization
st.subheader("📊 RFM Scatter Plot")
fig2 = px.scatter(
    df, x="F", y="M",
    color="cluster",
    size="R",
    hover_data=["customer_id", "preferred_product"],
    title="RFM Segmentation Scatter Plot"
)
st.plotly_chart(fig2, use_container_width=True)

# Persona View
st.subheader("🧬 Cluster Personas")

for c in sorted(df.cluster.unique()):
    cluster_df = df[df.cluster == c]
    st.write(f"### Cluster {c} — Persona Summary")
    st.write(f"- Avg Age: {cluster_df.age.mean():.1f}")
    st.write(f"- Avg Income: {cluster_df.income.mean():.0f}")
    st.write(f"- Preferred Product: {cluster_df.preferred_product.mode()[0]}")
    st.write(f"- Avg Transactions: {cluster_df.transactions_last_year.mean():.1f}")
    st.write(f"- Avg Loyalty: {cluster_df.loyalty_score.mean():.1f}")
    st.markdown("---")
