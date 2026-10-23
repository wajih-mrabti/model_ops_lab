from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "iris_processed.csv"

st.set_page_config(page_title="ModelOps Dashboard", layout="wide")
st.title("ModelOps Dashboard")

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    st.subheader("Processed dataset")
    st.dataframe(df.head(20), use_container_width=True)

    metrics = {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Classes": df["species"].nunique(),
    }
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", metrics["Rows"])
    col2.metric("Columns", metrics["Columns"])
    col3.metric("Classes", metrics["Classes"])
else:
    st.warning("No processed dataset found. Run the preparation step first.")

st.caption("This dashboard is a starting point for monitoring model health, dataset quality, and performance.")
