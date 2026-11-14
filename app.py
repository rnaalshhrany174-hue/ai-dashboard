import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

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

cost_col = next((c for c in df.columns if "cost" in c.lower()), None)
if cost_col:
    st.subheader("Is a more expensive model a better model?")
    cost_data = filtered.dropna(subset=[cost_col])
    st.scatter_chart(cost_data, x=cost_col, y="score", color="organization")

st.subheader("Number of Models by License")
st.bar_chart(filtered["license"].value_counts())

# ---------- Data preview ----------
st.header("Data Preview")
st.write(f"Showing {len(filtered)} models")
st.dataframe(filtered)

# ---------- Prediction model ----------
st.header("Score Prediction")
st.write("A Linear Regression model predicts a model's score from its cost, "
         "output tokens, sessions, and observations.")

features = ["cost_p50", "output_tokens_p50", "session_count", "observation_count"]
X = df[features]
y = df["score"]

@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2").mean()
    final_model = LinearRegression().fit(X, y)
    return final_model, r2, mae, cv

model, r2, mae, cv = train_model()

m1, m2, m3 = st.columns(3)
m1.metric("R2 (test)", f"{r2:.2f}")
m2.metric("MAE", f"{mae:.3f}")
m3.metric("R2 (cross validation)", f"{cv:.2f}")

st.subheader("Try it")
c1, c2 = st.columns(2)
cost = c1.slider("Cost per task", float(X["cost_p50"].min()),
                 float(X["cost_p50"].max()), float(X["cost_p50"].median()))
tokens = c2.slider("Output tokens per task", int(X["output_tokens_p50"].min()),
                   int(X["output_tokens_p50"].max()),
                   int(X["output_tokens_p50"].median()))
c3, c4 = st.columns(2)
sessions = c3.slider("Session count", int(X["session_count"].min()),
                     int(X["session_count"].max()),
                     int(X["session_count"].median()))
obs = c4.slider("Observation count", int(X["observation_count"].min()),
                int(X["observation_count"].max()),
                int(X["observation_count"].median()))

new = pd.DataFrame([[cost, tokens, sessions, obs]], columns=features)
st.metric("Predicted score", f"{model.predict(new)[0]:.3f}")
st.caption("The dataset is small (59 models), so this is only a rough estimate.")
