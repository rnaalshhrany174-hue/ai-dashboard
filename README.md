# AI Agent Leaderboard Dashboard 🤖📊
**Group 6 – Mini Data Insight Dashboard**

🔗 **Live dashboard:** https://group6-dashboard.streamlit.app

## About the Data
The [LMArena Leaderboard Dataset](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset) (agent subset) from Hugging Face.
Each row is one AI model (59 models, 13 columns). People give models real tasks and vote on which one did better; those votes become each model's **score**.

## Project Files
| File | Description |
|---|---|
| `Untitled1.ipynb` | EDA, data cleaning, and prediction model (Google Colab) |
| `cleaned_data2.csv` | The cleaned dataset |
| `app.py` | The Streamlit dashboard |
| `requirements.txt` | Python packages needed |

## Steps
1. **EDA:** rows and columns, data types, missing values, mean / median / mode, and charts
2. **Data Cleaning:**
   - Checked duplicates (none found)
   - Filled missing values with the median (`cost_per_task_p50`: 4, `output_tokens_per_task_p50`: 1)
   - Converted the date column to datetime, count columns to integers, and label columns to category
   - Saved the cleaned data to CSV
3. **Dashboard (Streamlit):**
   - Sidebar: dataset description and filters (organization, license, score range)
   - Main area: summary statistics, interactive charts, and a data preview
4. **Prediction model (optional):** Linear Regression and Random Forest to predict a model's score

## Key Insights
- **Claude Fable 5.2 (Max)** by Anthropic has the highest score.
- **OpenAI** has the most models on the leaderboard, and most models are **proprietary**.
- The average score is close to **0** because scores are relative: models below average get negative scores.
- **Linear Regression** did better than Random Forest (cross-validation R²: 0.43 vs 0.32).
  Both models scored much higher on training data (~0.74), which shows **overfitting**.
  This is expected with only 59 rows.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
