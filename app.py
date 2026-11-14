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

if filtered.empty:
    st.warning("No models match the filters. Try changing them.")
    st.stop()

# ---------- Summary statistics ----------
st.header("Summary Statistics")

col1, col2, col3 = st.columns(3)
col1.metric("Number of models", len(filtered))
col2.metric("Average score", f"{filtered['score'].mean():.3f}")
best = filtered.loc[filtered["score"].idxmax(), "model_name"]
col3.metric("Top model", best)

with st.expander("Mean, median and mode"):
    num = filtered.select_dtypes("number")
    stats = pd.DataFrame({
        "mean": num.mean(),
        "median": num.median(),
        "mode": num.mode().iloc[0],
    })
    st.dataframe(stats)

# ---------- Interactive visualizations ----------
st.header("Visualizations")

st.subheader("Average Score by Organization")
avg_by_org = filtered.groupby("organization")["score"].mean().sort_values(ascending=False)
st.bar_chart(avg_by_org)

st.subheader("Top 10 Models by Score")
top10 = filtered.nlargest(10, "score").set_index("model_name")["score"]
st.bar_chart(top10, horizontal=True)

st.subheader("Is a more expensive model a better model?")
cost_data = filtered.dropna(subset=["cost_per_task_p50"])
st.scatter_chart(cost_data, x="cost_per_task_p50", y="score", color="organization")

st.subheader("Number of Models by License")
st.bar_chart(filtered["license"].value_counts())

# ---------- Data preview ----------
st.header("Data Preview")
st.write(f"Showing {len(filtered)} models")
st.dataframe(filtered)
