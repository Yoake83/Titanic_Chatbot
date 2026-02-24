"""
Titanic Dataset Chat Agent - FastAPI Backend
Uses LangChain with Groq (free) to answer questions about the Titanic dataset.
"""

import os
import json
import base64
import io
import traceback
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Titanic Chat Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Dataset ──────────────────────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def load_titanic() -> pd.DataFrame:
    """Load Titanic dataset — try local file first, then URL."""
    local_path = os.path.join(os.path.dirname(__file__), "titanic.csv")
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    try:
        import urllib.request
        urllib.request.urlretrieve(DATA_URL, local_path)
        return pd.read_csv(local_path)
    except Exception:
        # Fallback: minimal embedded dataset so the app never crashes cold
        raise RuntimeError(
            "Could not load titanic.csv. Place titanic.csv next to backend.py "
            "or ensure internet access so it can be downloaded automatically."
        )

try:
    df = load_titanic()
    print(f"✅ Titanic dataset loaded: {len(df)} rows, {len(df.columns)} columns")
except Exception as e:
    print(f"❌ Dataset load error: {e}")
    df = None

# ── Visualization Helpers ─────────────────────────────────────────────────────
STYLE = {
    "bg": "#0f1117",
    "surface": "#1a1d27",
    "accent": "#e8b86d",
    "accent2": "#6d9ee8",
    "text": "#e8e8f0",
    "grid": "#2a2d3a",
}

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def apply_dark_style(fig, ax_list=None):
    fig.patch.set_facecolor(STYLE["bg"])
    axes = ax_list or fig.get_axes()
    for ax in axes:
        ax.set_facecolor(STYLE["surface"])
        ax.tick_params(colors=STYLE["text"], labelsize=11)
        ax.xaxis.label.set_color(STYLE["text"])
        ax.yaxis.label.set_color(STYLE["text"])
        ax.title.set_color(STYLE["accent"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["grid"])
        ax.grid(True, color=STYLE["grid"], linewidth=0.6, alpha=0.7)

# ── LangChain Tools ───────────────────────────────────────────────────────────

@tool
def get_dataset_info() -> str:
    """Return basic info about the Titanic dataset: shape, columns, dtypes, nulls."""
    if df is None:
        return "Dataset not loaded."
    info = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "sample": df.head(3).to_dict(orient="records"),
    }
    return json.dumps(info, default=str)


@tool
def run_query(query: str) -> str:
    """
    Execute a pandas query/expression on the Titanic DataFrame `df`.
    Use standard pandas syntax. The result is returned as a JSON string.
    Examples:
      - "df['Survived'].value_counts().to_dict()"
      - "df.groupby('Sex')['Survived'].mean().to_dict()"
      - "df['Age'].describe().to_dict()"
      - "df['Sex'].value_counts(normalize=True).mul(100).round(2).to_dict()"
    """
    if df is None:
        return "Dataset not loaded."
    try:
        result = eval(query, {"df": df, "pd": pd, "np": np})  # noqa: S307
        if isinstance(result, pd.DataFrame):
            return result.to_json(orient="records", default_handler=str)
        if isinstance(result, pd.Series):
            return result.to_json(default_handler=str)
        return json.dumps(result, default=str)
    except Exception as exc:
        return f"Query error: {exc}"


@tool
def plot_histogram(column: str, bins: int = 20, title: str = "") -> str:
    """
    Create a histogram for a numeric column and return a base64 PNG.
    Returns JSON: {"image_b64": "...", "description": "..."}.
    Valid numeric columns: Age, Fare, Parch, SibSp, Pclass.
    """
    if df is None:
        return json.dumps({"error": "Dataset not loaded."})
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found."})

    data = df[column].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data, bins=bins, color=STYLE["accent"], edgecolor=STYLE["bg"], linewidth=0.8, alpha=0.9)
    ax.set_xlabel(column, fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(title or f"Distribution of {column}", fontsize=15, fontweight="bold", pad=14)
    apply_dark_style(fig)
    b64 = fig_to_b64(fig)
    plt.close(fig)

    desc = (f"Histogram of {column}: mean={data.mean():.2f}, "
            f"median={data.median():.2f}, std={data.std():.2f}, "
            f"min={data.min():.2f}, max={data.max():.2f}")
    return json.dumps({"image_b64": b64, "description": desc})


@tool
def plot_bar(column: str, title: str = "", normalize: bool = False) -> str:
    """
    Create a bar chart for a categorical column (value counts).
    Returns JSON: {"image_b64": "...", "description": "..."}.
    Valid categorical columns: Sex, Embarked, Pclass, Survived.
    Set normalize=True for percentages.
    """
    if df is None:
        return json.dumps({"error": "Dataset not loaded."})
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found."})

    counts = df[column].value_counts(normalize=normalize).sort_index()
    labels = [str(l) for l in counts.index]
    values = counts.values * (100 if normalize else 1)

    palette = [STYLE["accent"], STYLE["accent2"], "#e86d6d", "#6de8b8", "#b86de8"]
    colors = [palette[i % len(palette)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor=STYLE["bg"], linewidth=0.8, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.1f}{'%' if normalize else ''}",
                ha="center", va="bottom", color=STYLE["text"], fontsize=11)
    ax.set_xlabel(column, fontsize=13)
    ax.set_ylabel("Percentage (%)" if normalize else "Count", fontsize=13)
    ax.set_title(title or f"{column} Distribution", fontsize=15, fontweight="bold", pad=14)
    apply_dark_style(fig)
    b64 = fig_to_b64(fig)
    plt.close(fig)

    desc = f"Bar chart for {column}: " + ", ".join(
        f"{l}={v:.1f}{'%' if normalize else ''}" for l, v in zip(labels, values))
    return json.dumps({"image_b64": b64, "description": desc})


@tool
def plot_survival_by_group(group_column: str, title: str = "") -> str:
    """
    Create a grouped bar chart showing survival rate by a categorical column.
    Returns JSON: {"image_b64": "...", "description": "..."}.
    Valid group columns: Sex, Pclass, Embarked, SibSp, Parch.
    """
    if df is None:
        return json.dumps({"error": "Dataset not loaded."})
    if group_column not in df.columns:
        return json.dumps({"error": f"Column '{group_column}' not found."})

    survival = df.groupby(group_column)["Survived"].mean().mul(100).round(1)
    labels = [str(l) for l in survival.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, survival.values, color=STYLE["accent2"],
                  edgecolor=STYLE["bg"], linewidth=0.8, width=0.5)
    for bar, val in zip(bars, survival.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.1f}%",
                ha="center", va="bottom", color=STYLE["text"], fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_xlabel(group_column, fontsize=13)
    ax.set_ylabel("Survival Rate (%)", fontsize=13)
    ax.set_title(title or f"Survival Rate by {group_column}", fontsize=15, fontweight="bold", pad=14)
    apply_dark_style(fig)
    b64 = fig_to_b64(fig)
    plt.close(fig)

    desc = f"Survival rate by {group_column}: " + ", ".join(
        f"{l}={v:.1f}%" for l, v in zip(labels, survival.values))
    return json.dumps({"image_b64": b64, "description": desc})


@tool
def plot_pie(column: str, title: str = "") -> str:
    """
    Create a pie chart for a categorical column.
    Returns JSON: {"image_b64": "...", "description": "..."}.
    """
    if df is None:
        return json.dumps({"error": "Dataset not loaded."})
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found."})

    counts = df[column].value_counts()
    palette = [STYLE["accent"], STYLE["accent2"], "#e86d6d", "#6de8b8", "#b86de8"]
    colors = [palette[i % len(palette)] for i in range(len(counts))]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts, labels=[str(l) for l in counts.index],
        autopct="%1.1f%%", colors=colors,
        pctdistance=0.78, startangle=140,
        wedgeprops={"edgecolor": STYLE["bg"], "linewidth": 2},
    )
    for t in texts:
        t.set_color(STYLE["text"])
        t.set_fontsize(12)
    for at in autotexts:
        at.set_color(STYLE["bg"])
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_title(title or f"{column} Breakdown", fontsize=15, fontweight="bold",
                 pad=18, color=STYLE["accent"])
    fig.patch.set_facecolor(STYLE["bg"])
    b64 = fig_to_b64(fig)
    plt.close(fig)

    desc = f"Pie chart for {column}: " + ", ".join(
        f"{l}={v}" for l, v in zip(counts.index, counts.values))
    return json.dumps({"image_b64": b64, "description": desc})


@tool
def plot_correlation_heatmap() -> str:
    """
    Create a heatmap of correlations between numeric columns.
    Returns JSON: {"image_b64": "...", "description": "..."}.
    """
    if df is None:
        return json.dumps({"error": "Dataset not loaded."})

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                linewidths=0.5, linecolor=STYLE["grid"],
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 10, "color": STYLE["text"]})
    ax.set_title("Correlation Heatmap", fontsize=15, fontweight="bold",
                 pad=14, color=STYLE["accent"])
    apply_dark_style(fig)
    fig.patch.set_facecolor(STYLE["bg"])
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return json.dumps({"image_b64": b64, "description": "Correlation heatmap of all numeric features."})


# ── Build Agent ───────────────────────────────────────────────────────────────
TOOLS = [
    get_dataset_info,
    run_query,
    plot_histogram,
    plot_bar,
    plot_survival_by_group,
    plot_pie,
    plot_correlation_heatmap,
]

SYSTEM_PROMPT = """You are TitanicBot 🚢, a friendly and insightful data analyst specializing in the Titanic passenger dataset.

## Your Capabilities
- Answer questions about the Titanic dataset using the available tools
- Create beautiful visualizations (histograms, bar charts, pie charts, survival analyses, heatmaps)
- Provide clear, concise, accurate statistics

## Dataset Overview
The Titanic dataset contains 891 passengers with columns:
PassengerId, Survived (0/1), Pclass (1/2/3), Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked (C/Q/S)

## Guidelines
- Always use tools to get accurate data — never guess numbers
- When the user asks for a chart/histogram/plot, use the appropriate visualization tool
- After getting tool results, provide a clear, friendly explanation
- Keep responses concise but insightful — highlight interesting findings
- If a visualization tool returns an image, tell the user you've created it
- Be enthusiastic about data insights!

## Tool Selection Guide
- For "histogram of ages" → use plot_histogram(column="Age")
- For "bar chart of embarkation ports" → use plot_bar(column="Embarked")
- For "pie chart of gender" → use plot_pie(column="Sex")
- For "survival by sex/class" → use plot_survival_by_group(group_column="Sex")
- For statistics (percentages, averages, counts) → use run_query
- For general info → use get_dataset_info
"""

def build_agent():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    
    llm = ChatGroq(
        model="llama3-8b-8192",
        groq_api_key=api_key,
        temperature=0,
        max_tokens=4096,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=True, max_iterations=6)


agent_executor = build_agent()

# ── API Models & Routes ───────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    chat_history: list = []


class ChatResponse(BaseModel):
    text: str
    images: list[str] = []  # list of base64-encoded PNGs
    error: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "dataset_loaded": df is not None,
        "agent_ready": agent_executor is not None,
        "rows": len(df) if df is not None else 0,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if agent_executor is None:
        raise HTTPException(status_code=503, detail="Agent not initialized. Set ANTHROPIC_API_KEY.")
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    # Convert chat history
    lc_history = []
    for msg in req.chat_history:
        if msg.get("role") == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            lc_history.append(AIMessage(content=msg["content"]))

    try:
        result = agent_executor.invoke({
            "input": req.message,
            "chat_history": lc_history,
        })
    except Exception as exc:
        traceback.print_exc()
        return ChatResponse(text="", error=str(exc))

    # Extract text response
    text = result.get("output", "")

    # Collect any images generated by tools during this run
    images = []
    # Scan intermediate steps for tool outputs containing image_b64
    for action, observation in result.get("intermediate_steps", []):
        try:
            obs_data = json.loads(observation)
            if isinstance(obs_data, dict) and "image_b64" in obs_data:
                images.append(obs_data["image_b64"])
        except (json.JSONDecodeError, TypeError):
            pass

    return ChatResponse(text=text, images=images)


@app.get("/dataset/info")
def dataset_info():
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "survived_pct": round(df["Survived"].mean() * 100, 1),
        "male_pct": round((df["Sex"] == "male").mean() * 100, 1),
        "avg_age": round(df["Age"].mean(), 1),
        "avg_fare": round(df["Fare"].mean(), 2),
    }