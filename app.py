"""
Titanic Dataset Chat Agent — Streamlit Frontend
"""

import os
import base64
import json
from io import BytesIO

import streamlit as st
import requests
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TitanicBot 🚢",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Lato:wght@300;400;700&display=swap');

:root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #22263a;
    --accent: #e8b86d;
    --accent2: #6d9ee8;
    --text: #e8e8f0;
    --muted: #888899;
    --border: #2a2d3a;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Lato', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Title */
.titan-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.titan-sub {
    color: var(--muted);
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    margin-bottom: 1.5rem;
}

/* Stat Cards */
.stat-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1.5rem; }
.stat-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 20px;
    min-width: 130px;
    flex: 1;
}
.stat-card .val {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.stat-card .label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* Chat Messages */
.chat-user {
    background: var(--surface2);
    border: 1px solid var(--accent);
    border-radius: 16px 16px 4px 16px;
    padding: 12px 18px;
    margin: 10px 0;
    margin-left: 10%;
    color: var(--text);
    font-size: 0.95rem;
}
.chat-bot {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    margin: 10px 0;
    margin-right: 5%;
    color: var(--text);
    font-size: 0.95rem;
    line-height: 1.65;
}
.chat-bot .bot-label {
    font-size: 0.72rem;
    color: var(--accent);
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* Input area */
[data-testid="stTextInput"] input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Lato', sans-serif !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(232,184,109,0.2) !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, var(--accent), #c89a4d) !important;
    color: var(--bg) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-family: 'Lato', sans-serif !important;
    letter-spacing: 0.04em;
    transition: opacity 0.2s;
}
.stButton button:hover { opacity: 0.88 !important; }

/* Sidebar chips */
.chip {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 0.8rem;
    color: var(--text);
    cursor: pointer;
    margin: 3px 0;
    width: 100%;
    text-align: left;
    transition: border-color 0.2s;
}
.chip:hover { border-color: var(--accent); }

/* Divider */
hr { border-color: var(--border) !important; }

/* Error */
.error-box {
    background: rgba(232, 109, 109, 0.1);
    border: 1px solid #e86d6d;
    border-radius: 10px;
    padding: 12px 16px;
    color: #e86d6d;
    font-size: 0.9rem;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚢 TitanicBot")
    st.markdown("---")

    # API Key
    api_key = st.text_input("Groq API Key", type="password",
                             value=os.environ.get("GROQ_API_KEY", ""),
                             help="Required to power the AI agent")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        st.session_state.api_key_set = True

    st.markdown("---")
    st.markdown("**💡 Example Questions**")

    example_questions = [
        "What percentage of passengers were male?",
        "Show me a histogram of passenger ages",
        "What was the average ticket fare?",
        "How many passengers embarked from each port?",
        "What was the survival rate by gender?",
        "Show a pie chart of passenger classes",
        "What's the survival rate by ticket class?",
        "Show me a correlation heatmap",
        "How many passengers survived vs died?",
        "What was the average age of survivors?",
    ]

    for i, q in enumerate(example_questions):
        if st.button(q, key=f"eq_{i}", use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<span style='color:#888899; font-size:0.75rem;'>"
        "Powered by LangChain + Claude + FastAPI</span>",
        unsafe_allow_html=True,
    )

# ── Helper: call backend ──────────────────────────────────────────────────────
def ask_backend(question: str) -> dict:
    """Send a question to the FastAPI backend and return the response dict."""
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]
    try:
        resp = requests.post(
            f"{BACKEND_URL}/chat",
            json={"message": question, "chat_history": history},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to backend at {BACKEND_URL}. "
                         "Make sure `uvicorn backend:app` is running."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out (>120s). Try a simpler question."}
    except Exception as e:
        return {"error": str(e)}


def get_dataset_stats() -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/dataset/info", timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {}


def b64_to_image(b64_str: str):
    img_bytes = base64.b64decode(b64_str)
    return Image.open(BytesIO(img_bytes))

# ── Main Layout ───────────────────────────────────────────────────────────────
col_main, col_pad = st.columns([3, 0.01])

with col_main:
    st.markdown('<div class="titan-title">🚢 TitanicBot</div>', unsafe_allow_html=True)
    st.markdown('<div class="titan-sub">AI-powered insights into the Titanic passenger dataset</div>',
                unsafe_allow_html=True)

    # ── Stats Row ─────────────────────────────────────────────────────────────
    stats = get_dataset_stats()
    if stats:
        st.markdown(
            f"""<div class="stat-grid">
                <div class="stat-card"><div class="val">{stats.get('shape', [0])[0]}</div><div class="label">Passengers</div></div>
                <div class="stat-card"><div class="val">{stats.get('survived_pct', '?')}%</div><div class="label">Survival Rate</div></div>
                <div class="stat-card"><div class="val">{stats.get('male_pct', '?')}%</div><div class="label">Male</div></div>
                <div class="stat-card"><div class="val">{stats.get('avg_age', '?')}</div><div class="label">Avg Age</div></div>
                <div class="stat-card"><div class="val">${stats.get('avg_fare', '?')}</div><div class="label">Avg Fare</div></div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Chat History ──────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">💬 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            elif msg["role"] == "assistant":
                st.markdown(
                    f'<div class="chat-bot"><div class="bot-label">🤖 TitanicBot</div>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
                # Render images attached to this message
                for img_b64 in msg.get("images", []):
                    try:
                        img = b64_to_image(img_b64)
                        st.image(img, use_container_width=True)
                    except Exception:
                        pass
            elif msg["role"] == "error":
                st.markdown(
                    f'<div class="error-box">⚠️ {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── Input ─────────────────────────────────────────────────────────────────
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Ask anything about the Titanic passengers…",
                label_visibility="collapsed",
                placeholder="e.g. Show me a histogram of passenger ages",
            )
        with col_btn:
            submitted = st.form_submit_button("Send →")

    # Handle example question clicks
    pending = st.session_state.pop("pending_question", None)
    question = pending or (user_input if submitted and user_input.strip() else None)

    if question:
        if not st.session_state.api_key_set and not os.environ.get("GROQ_API_KEY"):
            st.warning("⚠️ Please enter your Groq API Key in the sidebar first.")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})

            with st.spinner("🔍 Analyzing the data…"):
                response = ask_backend(question)

            if "error" in response and response["error"]:
                st.session_state.messages.append({
                    "role": "error",
                    "content": response["error"],
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.get("text", "No response."),
                    "images": response.get("images", []),
                })

            st.rerun()