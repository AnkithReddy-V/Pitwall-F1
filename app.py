import streamlit as st
import re
import base64
import urllib.request
from src.agent import get_agent

st.set_page_config(
    page_title="PitWall · F1 Intelligence",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helper: load image as base64 so it always works ──────────────────
@st.cache_data
def load_image_b64(url: str) -> str:
    """Download image and return as base64 data URI — works in any Streamlit sandbox."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = r.read()
        ext = "jpeg" if url.endswith(".jpg") or "jpg" in url else "png"
        b64 = base64.b64encode(data).decode()
        return f"data:image/{ext};base64,{b64}"
    except Exception:
        return ""

# Pre-load images
HERO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/2019_Formula_One_season_opening_Australian_Grand_Prix_%2847450261772%29.jpg/640px-2019_Formula_One_season_opening_Australian_Grand_Prix_%2847450261772%29.jpg"
SIDE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/2019_Formula_One_tests_Barcelona%2C_Ferrari_SF90%2C_Sebastian_Vettel_%2847403891681%29.jpg/640px-2019_Formula_One_tests_Barcelona%2C_Ferrari_SF90%2C_Sebastian_Vettel_%2847403891681%29.jpg"

hero_b64 = load_image_b64(HERO_URL)
side_b64 = load_image_b64(SIDE_URL)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Barlow+Condensed:wght@600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.stApp {{ background: #f0f2f6; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{ background: #0d0f14 !important; border-right: none; }}
[data-testid="stSidebar"] > div {{ padding: 0 !important; }}

.sb-top {{
    background: #13151e;
    border-bottom: 3px solid #e10600;
    padding: 18px 18px 14px;
}}
.sb-logo-name {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.2rem; font-weight: 800;
    color: #fff; letter-spacing: 4px; line-height: 1;
}}
.sb-logo-tag {{
    font-size: 0.58rem; color: rgba(255,255,255,0.35);
    letter-spacing: 0.22em; text-transform: uppercase; margin-top: 3px;
}}
.sb-car {{
    width: 100%; height: 140px;
    background-image: url('{side_b64}');
    background-size: cover;
    background-position: center 40%;
    display: block;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}}
.sb-section {{
    padding: 12px 18px 5px;
    font-size: 0.57rem; text-transform: uppercase;
    letter-spacing: 0.18em; color: rgba(255,255,255,0.2); font-weight: 700;
}}
.sb-row {{
    display: flex; justify-content: space-between;
    padding: 8px 18px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.79rem;
}}
.sb-row:hover {{ background: rgba(255,255,255,0.03); }}
.sb-lbl {{ color: rgba(255,255,255,0.35); }}
.sb-val {{ color: rgba(255,255,255,0.85); font-weight: 600; }}
.sb-val-r {{ color: #e10600 !important; font-weight: 700; }}

/* ── Hero ── */
.hero-wrap {{
    border-radius: 14px; overflow: hidden;
    display: grid; grid-template-columns: 1fr 1fr;
    height: 210px; margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.14);
}}
.hero-left {{
    background: #0d0f14;
    padding: 28px 30px;
    border-right: 3px solid #e10600;
    display: flex; flex-direction: column; justify-content: center;
}}
.hero-eyebrow {{
    font-size: 0.6rem; text-transform: uppercase;
    letter-spacing: 0.22em; color: #e10600;
    font-weight: 700; margin-bottom: 7px;
}}
.hero-title {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3.4rem; font-weight: 800;
    color: #fff; letter-spacing: 3px; line-height: 1; margin-bottom: 10px;
}}
.hero-desc {{
    font-size: 0.82rem; color: rgba(255,255,255,0.45);
    line-height: 1.7; max-width: 300px;
}}
.hero-tags {{
    display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap;
}}
.hero-tag {{
    font-size: 0.6rem; font-weight: 700; padding: 4px 10px;
    border-radius: 4px; letter-spacing: 0.1em; text-transform: uppercase;
}}
.hero-tag-red {{ background: #e10600; color: white; }}
.hero-tag-outline {{
    background: transparent; color: rgba(255,255,255,0.4);
    border: 1px solid rgba(255,255,255,0.12);
}}
.hero-right {{
    background-image: url('{hero_b64}');
    background-size: cover; background-position: center 35%;
}}
.hero-right-fallback {{
    background: linear-gradient(135deg, #1a0000 0%, #2d1010 50%, #0a0a1a 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 5rem;
}}

/* ── KPI strip ── */
.kpi-grid {{
    display: grid; grid-template-columns: repeat(5,1fr);
    gap: 10px; margin-bottom: 16px;
}}
.kpi-card {{
    background: white; border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border-top: 3px solid #e10600;
    transition: transform .15s, box-shadow .15s;
}}
.kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 14px rgba(0,0,0,0.1); }}
.kpi-lbl {{
    font-size: 0.6rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: #9ca3af;
    font-weight: 700; margin-bottom: 5px;
}}
.kpi-num {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.85rem; font-weight: 700;
    color: #111827; line-height: 1;
}}
.kpi-sub {{ font-size: 0.66rem; color: #d1d5db; margin-top: 2px; }}

/* ── Query card ── */
.qcard {{
    background: white; border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}}
.qcard-lbl {{
    font-size: 0.62rem; text-transform: uppercase;
    letter-spacing: 0.14em; color: #9ca3af;
    font-weight: 700; margin-bottom: 10px;
}}

/* ── Answer card ── */
.acard-header {{
    background: #0d0f14; border-radius: 12px 12px 0 0;
    padding: 14px 20px; display: flex;
    align-items: flex-start; gap: 11px;
    border-bottom: 3px solid #e10600;
}}
.acard-badge {{
    background: #e10600; color: white;
    font-size: 0.58rem; font-weight: 800;
    padding: 4px 9px; border-radius: 4px;
    text-transform: uppercase; letter-spacing: 0.1em;
    white-space: nowrap; margin-top: 1px;
}}
.acard-q {{
    font-size: 0.93rem; font-weight: 600;
    color: white; line-height: 1.4;
}}
.acard-footer {{
    background: #f9fafb;
    border: 1px solid #e5e7eb; border-top: none;
    border-radius: 0 0 12px 12px;
    padding: 9px 20px; display: flex;
    gap: 7px; flex-wrap: wrap;
    margin-bottom: 14px;
}}
.pill {{
    font-size: 0.67rem; padding: 3px 10px;
    border-radius: 20px; font-weight: 500;
}}
.pill-r {{ background:#fff1f0; color:#dc2626; border:1px solid #fca5a5; }}
.pill-b {{ background:#eff6ff; color:#2563eb; border:1px solid #bfdbfe; }}
.pill-p {{ background:#faf5ff; color:#7c3aed; border:1px solid #ddd6fe; }}
.pill-g {{ background:#f3f4f6; color:#4b5563; border:1px solid #e5e7eb; }}

/* ── Section divider ── */
.div-wrap {{
    display:flex; align-items:center;
    gap:10px; margin:16px 0 12px;
}}
.div-line {{ flex:1; height:1px; background:#e5e7eb; }}
.div-lbl {{
    font-size:0.62rem; text-transform:uppercase;
    letter-spacing:0.14em; color:#9ca3af; font-weight:700;
}}

/* ── Streamlit input overrides ── */
.stTextInput > div > div > input {{
    background: #f9fafb !important; border: 1.5px solid #e5e7eb !important;
    border-radius: 8px !important; color: #111827 !important;
    font-size: 0.9rem !important; padding: 10px 14px !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: #e10600 !important;
    box-shadow: 0 0 0 2px rgba(225,6,0,0.1) !important;
    background: white !important;
}}
.stTextInput > div > div > input::placeholder {{ color: #9ca3af !important; }}
.stTextInput label {{ display: none; }}

.stButton > button[kind="primary"] {{
    background: #e10600 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-size: 0.82rem !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    padding: 10px 20px !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: #c20500 !important;
    box-shadow: 0 4px 14px rgba(225,6,0,0.35) !important;
}}

/* Suggestion chips */
[data-testid="stHorizontalBlock"] .stButton > button {{
    background: white !important; border: 1.5px solid #e5e7eb !important;
    color: #374151 !important; font-size: 0.74rem !important;
    font-weight: 500 !important; border-radius: 8px !important;
    padding: 7px 10px !important; text-align: left !important;
    line-height: 1.3 !important; height: auto !important;
    min-height: 42px !important; white-space: normal !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}}
[data-testid="stHorizontalBlock"] .stButton > button:hover {{
    border-color: #e10600 !important; color: #e10600 !important;
    background: #fff8f8 !important;
}}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {{
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.38) !important;
    border-radius: 7px !important; font-size: 0.78rem !important;
    width: 100% !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    border-color: #e10600 !important; color: #e10600 !important;
}}

.stExpander {{
    background: white !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 8px !important;
}}
.streamlit-expanderHeader {{
    background: white !important; font-size: 0.8rem !important;
    color: #4b5563 !important;
}}
.stSpinner > div {{ border-top-color: #e10600 !important; }}
.stMarkdown p {{ color: #111827; }}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "agent" not in st.session_state:
    with st.spinner("Warming up PitWall..."):
        st.session_state.agent = get_agent()

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-top">
        <div class="sb-logo-name">PITWALL</div>
        <div class="sb-logo-tag">Formula 1 Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # Car image — base64 embedded so it always renders
    if side_b64:
        st.markdown(f'<div class="sb-car"></div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="height:140px;background:linear-gradient(135deg,#1a0000,#0a0a1a);
                    display:flex;align-items:center;justify-content:center;font-size:4rem;">
            🏎️
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Database</div>', unsafe_allow_html=True)
    for lbl, val in [
        ("Seasons", "1950 – 2024"), ("Races", "1,102"),
        ("Lap records", "589,081"), ("Drivers", "861"),
        ("Constructors", "212"), ("Pit stops", "11,371"),
    ]:
        st.markdown(f"""
        <div class="sb-row">
            <span class="sb-lbl">{lbl}</span>
            <span class="sb-val">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section">RAG Index</div>', unsafe_allow_html=True)
    for lbl, val in [
        ("Wikipedia pages", "47"), ("Text chunks", "744"),
        ("Schema tables", "12"), ("Glamour drivers", "15"),
    ]:
        st.markdown(f"""
        <div class="sb-row">
            <span class="sb-lbl">{lbl}</span>
            <span class="sb-val sb-val-r">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Stack</div>', unsafe_allow_html=True)
    for lbl, val in [
        ("LLM", "Gemini 2.5 Flash"), ("Vector DB", "FAISS"),
        ("Warehouse", "BigQuery"),
        ("Session queries", str(len(st.session_state.history))),
    ]:
        st.markdown(f"""
        <div class="sb-row">
            <span class="sb-lbl">{lbl}</span>
            <span class="sb-val">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='padding:12px 18px 6px'>", unsafe_allow_html=True)
    if st.button("🗑  Clear conversation"):
        st.session_state.history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:10px 18px 16px;font-size:0.7rem;
                color:rgba(255,255,255,0.2);line-height:2.2;">
        Ankith Reddy Vemula<br>
        MSc Big Data · SFU Burnaby<br>
        <a href="https://github.com/AnkithReddy-V/Pitwall-F1"
           style="color:#e10600;text-decoration:none;">
           GitHub ↗
        </a>
    </div>""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────
hero_right_html = (
    f'<div class="hero-right"></div>'
    if hero_b64
    else '<div class="hero-right hero-right-fallback">🏎️</div>'
)

st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-left">
        <div class="hero-eyebrow">Formula 1 · 1950 – 2024</div>
        <div class="hero-title">PITWALL</div>
        <div class="hero-desc">
            Ask anything about F1 in plain English.<br>
            RAG + Gemini + BigQuery + 589K lap records.
        </div>
        <div class="hero-tags">
            <span class="hero-tag hero-tag-red">● Live</span>
            <span class="hero-tag hero-tag-outline">Gemini 2.5 Flash</span>
            <span class="hero-tag hero-tag-outline">BigQuery</span>
        </div>
    </div>
    {hero_right_html}
</div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────
st.markdown("""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-lbl">Lap records</div>
        <div class="kpi-num">589K</div>
        <div class="kpi-sub">1996 – 2024</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-lbl">Races indexed</div>
        <div class="kpi-num">1,102</div>
        <div class="kpi-sub">75 seasons</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-lbl">Drivers</div>
        <div class="kpi-num">861</div>
        <div class="kpi-sub">All time</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-lbl">RAG chunks</div>
        <div class="kpi-num">744</div>
        <div class="kpi-sub">47 Wikipedia sources</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-lbl">BigQuery tables</div>
        <div class="kpi-num">12</div>
        <div class="kpi-sub">+ Glamour index</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Query card ────────────────────────────────────────────────────────
st.markdown('<div class="qcard">', unsafe_allow_html=True)
st.markdown('<div class="qcard-lbl">Ask PitWall</div>', unsafe_allow_html=True)

SUGGESTIONS = [
    "Who has the most race wins ever?",
    "Fastest pit stop team since 2011?",
    "Highest glamour index, worst performance?",
    "Senna wet weather driving + win stats",
    "Best grid to finish recovery driver?",
    "Most dominant constructor season?",
    "Hamilton vs Schumacher comparison",
    "Which nationality dominates F1?",
]

s_cols = st.columns(4)
for i, s in enumerate(SUGGESTIONS):
    if s_cols[i % 4].button(s, key=f"s_{i}"):
        st.session_state.pending_q = s

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

inp_col, btn_col = st.columns([5, 1])
with inp_col:
    question = st.text_input(
        "q",
        placeholder="Ask anything — history, stats, drivers, teams, glamour index...",
        key="qmain",
    )
with btn_col:
    st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
    ask = st.button("ANALYSE ➜", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Run agent ─────────────────────────────────────────────────────────
active_q = None
if ask and question.strip():
    active_q = question.strip()
elif "pending_q" in st.session_state:
    active_q = st.session_state.pop("pending_q")

if active_q:
    with st.spinner(f"🏎️  Analysing: {active_q[:70]}..."):
        result = st.session_state.agent.ask(active_q)
    st.session_state.history.append({"question": active_q, "result": result})
    st.rerun()

# ── Results ───────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("""
    <div class="div-wrap">
        <div class="div-line"></div>
        <div class="div-lbl">Results</div>
        <div class="div-line"></div>
    </div>
    """, unsafe_allow_html=True)

    for item in reversed(st.session_state.history):
        q      = item["question"]
        result = item["result"]

        # Source pills
        pills = ""
        for src in result["sources"]:
            cls = "pill-r" if "BigQuery" in src else "pill-b" if "Schema" in src else "pill-p"
            pills += f'<span class="pill {cls}">{src}</span>'
        pills += f'<span class="pill pill-g">{result["model"]}</span>'
        pills += f'<span class="pill pill-g">{result["rounds"]} rounds · {len(result["tool_calls"])} calls</span>'

        # Card header
        st.markdown(f"""
        <div class="acard-header">
            <span class="acard-badge">Q</span>
            <span class="acard-q">{q}</span>
        </div>
        """, unsafe_allow_html=True)

        # Answer body — rendered by Streamlit for proper markdown + dark text
        with st.container():
            st.markdown(
                f"""<div style="background:white;padding:16px 20px 4px;
                               border-left:1px solid #e5e7eb;
                               border-right:1px solid #e5e7eb;">
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown(result["answer"])

        # Footer
        st.markdown(
            f'<div class="acard-footer">{pills}</div>',
            unsafe_allow_html=True,
        )

        # SQL + reasoning in two columns
        exp1, exp2 = st.columns(2)
        with exp1:
            if result["sql"]:
                with st.expander("🔍 Generated SQL"):
                    st.code(result["sql"], language="sql")
        with exp2:
            if result["tool_calls"]:
                with st.expander("⚙️ Agent reasoning"):
                    for i, tc in enumerate(result["tool_calls"], 1):
                        clr = {
                            "get_schema_context":    "#2563eb",
                            "get_narrative_context": "#7c3aed",
                            "run_sql":               "#16a34a",
                        }.get(tc["tool"], "#6b7280")
                        st.markdown(
                            f'<span style="color:{clr};font-weight:600;'
                            f'font-size:0.8rem">Step {i} · {tc["tool"]}</span>',
                            unsafe_allow_html=True,
                        )
                        if "question" in tc["args"]:
                            st.caption(tc["args"]["question"])
                        elif "query" in tc["args"]:
                            st.code(tc["args"]["query"][:200], language="sql")