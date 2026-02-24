# 🚢 TitanicBot — Titanic Dataset Chat Agent

A full-stack AI chatbot that analyzes the Titanic dataset using natural language.

**Stack:** FastAPI + LangChain + Claude + Streamlit

---

## 📁 Project Structure

```
titanic_agent/
├── backend.py        # FastAPI + LangChain agent
├── app.py            # Streamlit frontend
├── requirements.txt  # Python dependencies
└── README.md
```

---

## ⚡ Quick Start (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Start the FastAPI backend
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

The Titanic dataset will be auto-downloaded on first run from GitHub.

### 4. Start the Streamlit frontend (new terminal)
```bash
streamlit run app.py --server.port 8501
```

### 5. Open your browser
- **Streamlit UI:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo, set `app.py` as main file
4. Add `ANTHROPIC_API_KEY` in **Secrets** (Settings → Secrets)
5. You'll also need the backend running — see below

**For the backend:** Deploy to Railway, Render, or Fly.io:
```bash
# Example: Railway
railway init
railway up
```
Then set `BACKEND_URL` in Streamlit secrets to your deployed backend URL.

---

## 🚀 Deploy Backend on Render (Free Tier)

1. Create new **Web Service** on [render.com](https://render.com)
2. Connect your GitHub repo
3. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend:app --host 0.0.0.0 --port $PORT`
4. Add env var: `ANTHROPIC_API_KEY=sk-ant-...`
5. Copy the URL, set it as `BACKEND_URL` in Streamlit secrets

---

## 💬 Example Questions

| Question | Tool Used |
|---|---|
| "What percentage of passengers were male?" | `run_query` |
| "Show me a histogram of passenger ages" | `plot_histogram` |
| "What was the average ticket fare?" | `run_query` |
| "How many passengers embarked from each port?" | `plot_bar` |
| "What was the survival rate by gender?" | `plot_survival_by_group` |
| "Show a pie chart of passenger classes" | `plot_pie` |
| "Show me a correlation heatmap" | `plot_correlation_heatmap` |

---

## 🛠️ LangChain Tools

| Tool | Description |
|---|---|
| `get_dataset_info` | Returns dataset shape, columns, nulls, sample rows |
| `run_query` | Executes arbitrary pandas expressions |
| `plot_histogram` | Generates histogram for numeric columns |
| `plot_bar` | Generates bar chart for categorical columns |
| `plot_survival_by_group` | Survival rate grouped by a column |
| `plot_pie` | Pie chart for categorical columns |
| `plot_correlation_heatmap` | Heatmap of numeric correlations |

---

## 🔐 Environment Variables

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Your Claude API key | Required |
| `BACKEND_URL` | URL of FastAPI backend | `http://localhost:8000` |

---

## 📊 Dataset

The Titanic dataset is automatically downloaded from:
`https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`

Or place `titanic.csv` next to `backend.py` for offline use.