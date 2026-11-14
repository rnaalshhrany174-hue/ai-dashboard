import streamlit as st
import pandas as pd

df = pd.read_csv("cleaned_data2.csv")

st.title("AI Agent Leaderboard Dashboard")

# ---------- Sidebar: description ----------
st.sidebar.title("About the Data")
st.sidebar.write(
    "This dataset is from the LMArena Agent Leaderboard on Hugging Face. "
    "Each row is one AI model, and its score comes from people voting "
    "on which model did a task better."
)

# ---------- Sidebar: filters ----------
st.sidebar.header("Filters")

orgs = sorted(df["organization"].unique())
selected_orgs = st.sidebar.multiselect("Organization", orgs, default=orgs)

licenses = sorted(df["license"].unique())
selected_licenses = st.sidebar.multiselect("License", licenses, default=licenses)

min_score = float(df["score"].min())
max_score = float(df["score"].max())
score_range = st.sidebar.slider("Score range", min_score, max_score, (min_score, max_score))

# ---------- Apply filters ----------
filtered = df[
    df["organization"].isin(selected_orgs)
    & df["license"].isin(selected_licenses)
    & df["score"].between(score_range[0], score_range[1])
]

st.write(f"Showing {len(filtered)} models")
st.dataframe(filtered)
