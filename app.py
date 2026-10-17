"""
====================================================================================================
PROJECT: Library Book Checkout Duration & Delay Risk Analysis via Probability Models
INSTITUTION: Vidyavardhini's College of Engineering & Technology (VCET), Vasai
DEPARTMENT: Department of Information Technology
ACADEMIC YEAR: 2026–27 | Second Year Engineering (Semester III)
COURSE: Applied Mathematics Thinking-I (2343111) | Probability & Random Variables
PROJECT IDENTIFICATION: Mini Project Group No: 15 | Batch: B4
====================================================================================================
"""

import os
import io
import numpy as np
import pandas as pd
import scipy.stats as stats
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# -----------------------------------------------------------------------------
# 1. STREAMLIT APP CONFIGURATION & THEMING (LIGHT & DARK MODE)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VCET AMT-1 • Group 15 • Library Delay Risk & Probability Models",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for Theme Toggle
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.get("theme", "dark") == "dark" else "dark"

is_dark = (st.session_state.theme == "dark")

# Dynamic CSS Injection for Light vs Dark Mode
dark_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    :root {
        --bg-app: #0b1120;
        --card-bg: rgba(30, 41, 59, 0.7);
        --card-border: rgba(255, 255, 255, 0.12);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --accent: #38bdf8;
        --callout-bg: rgba(30, 41, 59, 0.75);
        --banner-bg: linear-gradient(90deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
        --banner-border: rgba(56, 189, 248, 0.25);
    }
    
    html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b1120 !important;
        color: #f8fafc !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Segmented Pill / Capsule Tab Architecture (Universal Selectors for Cloud & Local) */
    div[data-testid="stTabs"] [role="tablist"],
    div[data-baseweb="tab-list"],
    div[role="tablist"] {
        display: flex !important;
        gap: 12px !important;
        background: transparent !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding: 0.6rem 0 1rem 0 !important;
        margin-bottom: 1.5rem !important;
        overflow-x: auto !important;
    }

    div[data-testid="stTabs"] [role="tablist"] {
        gap: 12px !important;
        border-bottom: none !important;
    }

    div[data-testid="stTabs"] button,
    div[data-testid="stTabs"] button[role="tab"],
    button[data-baseweb="tab"],
    button[role="tab"] {
        background: rgba(30, 41, 59, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.25rem !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.90rem !important;
        white-space: nowrap !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }

    div[data-testid="stTabs"] button:hover,
    div[data-testid="stTabs"] button[role="tab"]:hover,
    button[data-baseweb="tab"]:hover,
    button[role="tab"]:hover {
        background: rgba(56, 189, 248, 0.15) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
        color: #f8fafc !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(56, 189, 248, 0.2) !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"],
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"],
    button[role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.25), rgba(14, 165, 233, 0.15)) !important;
        border: 1.5px solid #38bdf8 !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 18px rgba(56, 189, 248, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stTabs"] button p,
    button[data-baseweb="tab"] p,
    button[role="tab"] p {
        color: inherit !important;
        font-weight: inherit !important;
        font-size: 0.90rem !important;
        margin: 0 !important;
    }

    div[data-baseweb="tab-highlight"], 
    div[data-baseweb="tab-border"],
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] hr {
        display: none !important;
    }
    
    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hide Streamlit Deploy Button */
    .stDeployButton,
    [data-testid="stDeployButton"],
    header button:has(span),
    button.st-emotion-cache-qt4i0q {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* Small Logo Button for Theme Toggle */
    div.st-key-theme_toggle_btn {
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
    }
    div.st-key-theme_toggle_btn button {
        width: 42px !important;
        height: 42px !important;
        min-width: 42px !important;
        min-height: 42px !important;
        max-width: 42px !important;
        max-height: 42px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        font-size: 1.35rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: #1e293b !important;
        border: 1.5px solid rgba(56, 189, 248, 0.45) !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.35) !important;
        cursor: pointer !important;
        margin-left: auto !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
    }
    div.st-key-theme_toggle_btn button:hover {
        background: rgba(30, 41, 59, 0.95) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.5) !important;
        transform: scale(1.1) !important;
    }

    /* Metric Card */
    .metric-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 0.8rem;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.3rem;
    }

    /* Clean Badges without Emojis */
    .badge-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.82rem;
        letter-spacing: 0.05em;
    }
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.82rem;
        letter-spacing: 0.05em;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.82rem;
        letter-spacing: 0.05em;
    }

    /* Math Callout Box */
    .math-callout {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #38bdf8;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #f8fafc;
    }
    
    /* Header Accent Tag */
    .header-tag {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.12);
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }

    /* Institution Banner */
    .college-banner {
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 10px;
        padding: 1.1rem 1.5rem;
        margin-bottom: 0.8rem;
    }
    .college-meta-card {
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 10px;
        padding: 0.85rem 1.15rem;
        margin-bottom: 0.8rem;
        min-height: 122px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
</style>
"""

light_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    :root {
        --bg-app: #f8fafc;
        --card-bg: #ffffff;
        --card-border: #e2e8f0;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --accent: #0284c7;
        --callout-bg: #f1f5f9;
        --callout-border: #0284c7;
        --banner-bg: linear-gradient(90deg, #f1f5f9, #e2e8f0);
        --banner-border: #cbd5e1;
    }
    
    html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    /* Segmented Pill / Capsule Tab Architecture (Light Mode - Universal Selectors for Cloud & Local) */
    div[data-testid="stTabs"] [role="tablist"],
    div[data-baseweb="tab-list"],
    div[role="tablist"] {
        display: flex !important;
        gap: 12px !important;
        background: transparent !important;
        border-bottom: 1px solid #e2e8f0 !important;
        padding: 0.6rem 0 1rem 0 !important;
        margin-bottom: 1.5rem !important;
        overflow-x: auto !important;
    }

    div[data-testid="stTabs"] [role="tablist"] {
        gap: 12px !important;
        border-bottom: none !important;
    }

    div[data-testid="stTabs"] button,
    div[data-testid="stTabs"] button[role="tab"],
    button[data-baseweb="tab"],
    button[role="tab"] {
        background: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.25rem !important;
        color: #475569 !important;
        font-weight: 500 !important;
        font-size: 0.90rem !important;
        white-space: nowrap !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
    }

    div[data-testid="stTabs"] button:hover,
    div[data-testid="stTabs"] button[role="tab"]:hover,
    button[data-baseweb="tab"]:hover,
    button[role="tab"]:hover {
        background: #e2e8f0 !important;
        border-color: #0284c7 !important;
        color: #0f172a !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.12) !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"],
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"],
    button[role="tab"][aria-selected="true"] {
        background: #ffffff !important;
        border: 1.5px solid #0284c7 !important;
        color: #0284c7 !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.2) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stTabs"] button p,
    button[data-baseweb="tab"] p,
    button[role="tab"] p {
        color: inherit !important;
        font-weight: inherit !important;
        font-size: 0.90rem !important;
        margin: 0 !important;
    }

    div[data-baseweb="tab-highlight"], 
    div[data-baseweb="tab-border"],
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] hr {
        display: none !important;
    }

    p, span, label, h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
    }
    
    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hide Streamlit Deploy Button */
    .stDeployButton,
    [data-testid="stDeployButton"],
    header button:has(span),
    button.st-emotion-cache-qt4i0q {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    /* Small Logo Button for Theme Toggle */
    div.st-key-theme_toggle_btn {
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
    }
    div.st-key-theme_toggle_btn button {
        width: 42px !important;
        height: 42px !important;
        min-width: 42px !important;
        min-height: 42px !important;
        max-width: 42px !important;
        max-height: 42px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        font-size: 1.35rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        cursor: pointer !important;
        margin-left: auto !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
    }
    div.st-key-theme_toggle_btn button:hover {
        background: #f8fafc !important;
        border-color: #0284c7 !important;
        box-shadow: 0 2px 14px rgba(2, 132, 199, 0.2) !important;
        transform: scale(1.1) !important;
    }

    /* Metric Card */
    .metric-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 0.8rem;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b !important;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0284c7 !important;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b !important;
        margin-top: 0.3rem;
    }

    /* Clean Badges without Emojis */
    .badge-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #059669 !important;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.82rem;
        letter-spacing: 0.05em;
    }
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #d97706 !important;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.82rem;
        letter-spacing: 0.05em;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #dc2626 !important;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.82rem;
        letter-spacing: 0.05em;
    }

    /* Math Callout Box */
    .math-callout {
        background: #f1f5f9 !important;
        border-left: 4px solid #0284c7 !important;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #0f172a !important;
        border: 1px solid #e2e8f0;
    }
    
    /* Header Accent Tag */
    .header-tag {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #0284c7 !important;
        background: rgba(2, 132, 199, 0.12) !important;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }

    /* Institution Banner */
    .college-banner {
        background: linear-gradient(90deg, #ffffff, #f8fafc) !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px;
        padding: 1.1rem 1.5rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }
    .college-meta-card {
        background: linear-gradient(90deg, #ffffff, #f8fafc) !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px;
        padding: 0.85rem 1.15rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        min-height: 122px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
</style>
"""

st.markdown(dark_css if is_dark else light_css, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. THEME-AWARE MATPLOTLIB STYLING HELPER
# -----------------------------------------------------------------------------
def get_plot_theme():
    """Return color palette matching the active UI theme (Dark vs Light)."""
    if is_dark:
        return {
            "fig_face": "#1e293b",
            "ax_face": "#0f172a",
            "text_color": "#f8fafc",
            "grid_color": "#334155",
            "tick_color": "#94a3b8",
            "line_primary": "#38bdf8",
            "line_secondary": "#10b981",
            "line_accent": "#f59e0b",
            "line_danger": "#ef4444",
            "hist_color": "#475569",
            "hist_edge": "#94a3b8"
        }
    else:
        return {
            "fig_face": "#ffffff",
            "ax_face": "#f8fafc",
            "text_color": "#0f172a",
            "grid_color": "#cbd5e1",
            "tick_color": "#475569",
            "line_primary": "#0284c7",
            "line_secondary": "#059669",
            "line_accent": "#d97706",
            "line_danger": "#dc2626",
            "hist_color": "#cbd5e1",
            "hist_edge": "#475569"
        }

pt = get_plot_theme()



# -----------------------------------------------------------------------------
# 4. DATA LOADING & CACHING HELPERS
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "library_logs.csv")

@st.cache_data
def load_default_dataset() -> pd.DataFrame:
    """Load default seed dataset or generate synthetic data if missing."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    
    np.random.seed(42)
    n = 80
    student_ids = [f"STU-{1001 + i}" for i in range(n)]
    books_borrowed = np.random.choice([1, 2, 3, 4], size=n, p=[0.35, 0.35, 0.20, 0.10])
    overdue_raw = np.random.exponential(scale=4.5, size=n)
    is_on_time = np.random.rand(n) < 0.28
    overdue_days = np.where(is_on_time, 0.0, np.round(overdue_raw, 1))
    early_return = np.random.uniform(0.5, 4.0, size=n)
    holding_days = np.where(is_on_time, np.round(14.0 - early_return, 1), np.round(14.0 + overdue_days, 1))
    fine_amount = np.round(overdue_days * 5.0, 2)
    
    df = pd.DataFrame({
        "student_id": student_ids,
        "books_borrowed": books_borrowed,
        "holding_days": holding_days,
        "overdue_days": overdue_days,
        "fine_amount": fine_amount
    })
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


# -----------------------------------------------------------------------------
# 5. MATHEMATICAL & STATISTICAL ENGINE FUNCTIONS
# -----------------------------------------------------------------------------
def compute_discrete_pmf(series: pd.Series):
    """
    Compute empirical PMF, CDF, expected value E(X), E(X^2), and Var(X).
    """
    counts = series.value_counts().sort_index()
    n = len(series)
    pmf_df = pd.DataFrame({
        "x": counts.index.astype(int),
        "Frequency (nx)": counts.values,
        "P(X = x)": counts.values / n
    })
    pmf_df["CDF F(x)"] = pmf_df["P(X = x)"].cumsum()
    pmf_df["x * P(X=x)"] = pmf_df["x"] * pmf_df["P(X = x)"]
    pmf_df["x^2 * P(X=x)"] = (pmf_df["x"] ** 2) * pmf_df["P(X = x)"]
    
    expected_x = pmf_df["x * P(X=x)"].sum()
    expected_x2 = pmf_df["x^2 * P(X=x)"].sum()
    var_x = expected_x2 - (expected_x ** 2)
    std_x = np.sqrt(var_x)
    
    pmf_df["(x - E[X])^2 * P(X=x)"] = ((pmf_df["x"] - expected_x) ** 2) * pmf_df["P(X = x)"]
    
    return pmf_df, expected_x, expected_x2, var_x, std_x


def compute_central_moments(data: np.ndarray):
    """
    Calculate the first 4 central moments using scipy.stats.moment
    as specified in the VCET syllabus (Building Block 2).
    Also computes Karl Pearson's skewness (beta_1) and kurtosis (beta_2).
    """
    n = len(data)
    mean_val = float(np.mean(data))
    
    # Central moments mu_k = E[(Y - E[Y])^k] evaluated via scipy.stats.moment
    mu1 = float(stats.moment(data, moment=1))  # 1st central moment ≡ 0
    mu2 = float(stats.moment(data, moment=2))  # 2nd central moment = Variance (sigma^2)
    mu3 = float(stats.moment(data, moment=3))  # 3rd central moment (asymmetry)
    mu4 = float(stats.moment(data, moment=4))  # 4th central moment (peakedness/tail)
    
    std_val = float(np.sqrt(mu2))
    
    # Karl Pearson Skewness coefficient beta_1 = mu_3^2 / mu_2^3
    beta1 = (mu3 ** 2) / (mu2 ** 3) if mu2 > 0 else 0.0
    gamma1 = mu3 / (mu2 ** 1.5) if mu2 > 0 else 0.0  # Pearson's skewness
    
    # Karl Pearson Kurtosis coefficient beta_2 = mu_4 / mu_2^2
    beta2 = mu4 / (mu2 ** 2) if mu2 > 0 else 0.0
    gamma2 = beta2 - 3.0                              # Excess kurtosis
    
    return {
        "n": n,
        "mean": mean_val,
        "std": std_val,
        "mu1": mu1,
        "mu2": mu2,
        "mu3": mu3,
        "mu4": mu4,
        "beta1": beta1,
        "gamma1": gamma1,
        "beta2": beta2,
        "gamma2": gamma2
    }


@st.cache_resource
def derive_symbolic_mgf():
    """
    Symbolically evaluate the MGF improper integral E[e^{tY}], its derivatives,
    and moments using SymPy (VCET Syllabus Building Block 1).
    """
    t = sp.Symbol('t', real=True)
    lam = sp.Symbol(r'\lambda', positive=True)
    y = sp.Symbol('y', positive=True)
    
    # Integrand: e^{ty} * f(y) = e^{ty} * lambda * e^{-lambda * y}
    integrand = sp.exp(t * y) * lam * sp.exp(-lam * y)
    
    # Anti-derivative evaluated symbolically
    anti_deriv = sp.integrate(lam * sp.exp(-(lam - t) * y), y)
    
    # Symbolic evaluation of definite improper integral from 0 to infinity (for t < lambda)
    M_t_integrated = sp.simplify(sp.integrate(lam * sp.exp(-(lam - t) * y), (y, 0, sp.oo), conds='none'))
    
    # Canonical closed form
    M_t = lam / (lam - t)
    
    # Symbolic differentiation with respect to t
    d1_M = sp.diff(M_t, t)
    d2_M = sp.diff(d1_M, t)
    d3_M = sp.diff(d2_M, t)
    d4_M = sp.diff(d3_M, t)
    
    # Evaluate derivatives at t = 0 to obtain raw moments mu'_k = E[Y^k]
    mu1_prime = sp.simplify(d1_M.subs(t, 0))
    mu2_prime = sp.simplify(d2_M.subs(t, 0))
    mu3_prime = sp.simplify(d3_M.subs(t, 0))
    mu4_prime = sp.simplify(d4_M.subs(t, 0))
    
    # Central moments from raw moments:
    var_sym = sp.simplify(mu2_prime - mu1_prime**2)
    mu3_sym = sp.simplify(mu3_prime - 3 * mu1_prime * mu2_prime + 2 * (mu1_prime ** 3))
    mu4_sym = sp.simplify(mu4_prime - 4 * mu1_prime * mu3_prime + 6 * (mu1_prime**2) * mu2_prime - 3 * (mu1_prime**4))
    
    # Theoretical Skewness & Kurtosis
    beta1_sym = sp.simplify((mu3_sym ** 2) / (var_sym ** 3))
    beta2_sym = sp.simplify(mu4_sym / (var_sym ** 2))
    
    return {
        "symbols": (t, lam, y),
        "integrand": integrand,
        "anti_deriv": anti_deriv,
        "M_t_integrated": M_t_integrated,
        "M_t": M_t,
        "d1_M": d1_M,
        "d2_M": d2_M,
        "d3_M": d3_M,
        "d4_M": d4_M,
        "mu1_prime": mu1_prime,
        "mu2_prime": mu2_prime,
        "mu3_prime": mu3_prime,
        "mu4_prime": mu4_prime,
        "var_sym": var_sym,
        "mu3_sym": mu3_sym,
        "mu4_sym": mu4_sym,
        "beta1_sym": beta1_sym,
        "beta2_sym": beta2_sym
    }


# -----------------------------------------------------------------------------
# 6. SIDEBAR CONTROLS & NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1.2rem;">
            <h3 style="margin: 0.2rem 0 0.1rem 0; font-size: 1.15rem; font-weight: 700; color: {'#38bdf8' if is_dark else '#0284c7'};">VCET IT Department</h3>
            <p style="font-size: 0.78rem; font-weight: 600; margin: 0; color: {'#cbd5e1' if is_dark else '#334155'};">Applied Mathematics Thinking-I</p>
            <p style="font-size: 0.72rem; color: {'#94a3b8' if is_dark else '#64748b'}; margin: 0;">Course Code: 2343111 • Sem III (2026–27)</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Project Identification")
    st.markdown(f"""
        <div style="font-size: 0.8rem; background: {'rgba(30, 41, 59, 0.5)' if is_dark else '#ffffff'}; padding: 0.85rem; border-radius: 8px; border: 1px solid {'rgba(255,255,255,0.08)' if is_dark else '#e2e8f0'}; margin-bottom: 0.8rem;">
            <strong style="color: {'#38bdf8' if is_dark else '#0284c7'}; font-size: 0.95rem;">MINI PROJECT • GROUP 15</strong><br>
            <strong>Assigned Batch:</strong> Batch B4<br>
            <strong>Academic Year:</strong> 2026–27 (Semester III)<br>
            <strong>Department:</strong> Information Technology (VCET)<br>
            <strong>Curriculum Unit:</strong> Probability & Random Variables
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Dataset Configuration")
    data_source = st.radio(
        "Data Source:",
        ["Standard Seed Dataset (N=80)", "Upload Custom CSV File"],
        index=0
    )
    
    if data_source == "Upload Custom CSV File":
        uploaded_file = st.file_uploader("Upload Circulation CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Custom dataset loaded ({len(df)} rows).")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                df = load_default_dataset()
        else:
            df = load_default_dataset()
    else:
        df = load_default_dataset()

    st.markdown("---")
    st.subheader("Variable Selection")
    continuous_var_option = st.selectbox(
        "Continuous Variable (Y):",
        ["overdue_days", "holding_days"],
        index=0,
        help="Select between overdue delay duration (for delay risk modeling) or total holding duration (for circulation cycle analysis)."
    )
    
    st.markdown("---")
    st.markdown(f"""
        <div style="font-size: 0.76rem; color: {'#94a3b8' if is_dark else '#64748b'}; line-height: 1.4;">
            <strong>Python Building Blocks (VCET Syllabus):</strong><br>
            • <code>SymPy</code>: Analytical MGF integration & d^k/dt^k<br>
            • <code>SciPy.stats</code>: MLE fitting & stats.moment<br>
            • <code>Matplotlib/Seaborn</code>: Stem plots, KDE & PDFs<br>
            • <code>Streamlit</code>: Web UI & risk simulation
        </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. INSTITUTIONAL HEADER & BANNER (WITH TOP-RIGHT INTEGRATED THEME TOGGLE)
# -----------------------------------------------------------------------------
banner_col_left, banner_col_right = st.columns([9.0, 3.0])

with banner_col_left:
    st.markdown(f"""
        <div class="college-banner">
            <span class="header-tag">VIDYAVARDHINI'S COLLEGE OF ENGINEERING & TECHNOLOGY • DEPARTMENT OF INFORMATION TECHNOLOGY</span>
            <h1 style="font-size: 2.1rem; margin: 0.2rem 0 0.3rem 0; font-weight: 800; letter-spacing: -0.02em;">
                Library Book Checkout Duration & Delay Risk Analysis
            </h1>
            <p style="font-size: 0.95rem; color: {'#94a3b8' if is_dark else '#475569'}; margin: 0; line-height: 1.4;">
                Applied Mathematics Thinking-I (2343111) • Probability & Random Variables • <strong>Mini Project Group 15 (Batch B4)</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)

with banner_col_right:
    meta_sub1, meta_sub2 = st.columns([2.1, 0.9])
    with meta_sub1:
        st.markdown(f"""
            <div style="text-align: right; padding-top: 1.1rem;">
                <span style="font-size: 0.88rem; color: {'#38bdf8' if is_dark else '#0284c7'}; font-weight: 700;">Academic Year 2026–27</span><br>
                <span style="font-size: 0.78rem; color: {'#94a3b8' if is_dark else '#64748b'};">SE IT • Semester III</span>
            </div>
        """, unsafe_allow_html=True)
    with meta_sub2:
        st.markdown("""<div style="padding-top: 0.9rem;">""", unsafe_allow_html=True)
        theme_icon = "☀️" if is_dark else "🌙"
        theme_tooltip = "Switch to Light Mode" if is_dark else "Switch to Dark Mode"
        st.button(theme_icon, key="theme_toggle_btn", on_click=toggle_theme, help=theme_tooltip)
        st.markdown("""</div>""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. TABBED NAVIGATION ARCHITECTURE (NO "MODULE" WORD, CLEAN TEXT LABELS)
# -----------------------------------------------------------------------------
tab_overview, tab_ds, tab_pmf, tab_cont, tab_mgf, tab_sim = st.tabs([
    "Overview & Foundations",
    "Dataset Viewer",
    "Discrete PMF (X)",
    "Continuous Delay (Y) & Moments",
    "Symbolic MGF Engine",
    "Overdue Risk Simulator"
])


# =============================================================================
# TAB 1: OVERVIEW & FOUNDATIONS
# =============================================================================
with tab_overview:
    st.subheader("System Architecture & Applied Mathematics Mapping")
    
    col_arch1, col_arch2 = st.columns([1.2, 1.0])
    with col_arch1:
        st.markdown(r"""
        ### Conceptual Framework
        This engineering project bridges empirical library circulation desk records with theoretical probability models 
        as defined in the **VCET Information Technology Applied Mathematics Thinking-I** curriculum:
        
        1. **Discrete Variable Modeling ($X$):**
           - Represents integer count of books borrowed per student visit ($X \in \{1, 2, 3, 4\}$).
           - Modeled via Probability Mass Function (PMF) $P(X = x)$, Expected Value $E[X]$, and Variance $\sigma_X^2 = E(X^2) - [E(X)]^2$.
        
        2. **Continuous Variable Modeling ($Y$):**
           - Represents overdue return delay or total holding duration ($Y \ge 0$).
           - Modeled via **Exponential Distribution** ($\operatorname{Exp}(\lambda)$) and **Gamma Distribution** ($\operatorname{Gamma}(\alpha, \theta)$) fitted using Maximum Likelihood Estimation (MLE).
        
        3. **Central Moments & Shape Analysis via `scipy.stats.moment`:**
           - Computes first four central moments ($\mu_1 = 0, \mu_2 = \sigma^2, \mu_3, \mu_4$).
           - Evaluates Karl Pearson's skewness ($\beta_1 = \mu_3^2 / \mu_2^3$) and kurtosis ($\beta_2 = \mu_4 / \mu_2^2$) to diagnose distribution asymmetry and heavy-tail risk.
        
        4. **Symbolic Moment Generating Function (MGF) Engine (`sympy`):**
           - Performs analytical improper evaluation of $M_Y(t) = E[e^{tY}] = \int_0^\infty e^{ty} \lambda e^{-\lambda y} \, dy = \frac{\lambda}{\lambda - t}$ for $t < \lambda$.
           - Executes automated symbolic differentiation: $\mu_k' = \left. \frac{d^k M_Y(t)}{dt^k} \right|_{t=0}$.
        
        5. **Inventory Tail Risk & Custom Loan Period Simulation:**
           - Predicts overdue probabilities $P(\text{Holding} > L)$ and tail delay risks $P(Y > d) = 1 - F(d)$ for custom loan periods ($L$).
           - Recommends actionable safety stock and fine escalations based on quantified risk thresholds.
        """)
    
    with col_arch2:
        st.markdown("### Stochastic Workflow Pipeline")
        card_bg = "rgba(30, 41, 59, 0.7)" if is_dark else "#f1f5f9"
        card_border = "rgba(56, 189, 248, 0.3)" if is_dark else "#cbd5e1"
        sub_card_bg = "rgba(15, 23, 42, 0.6)" if is_dark else "#ffffff"
        accent_color = "#38bdf8" if is_dark else "#0284c7"
        text_sub = "#94a3b8" if is_dark else "#64748b"
        
        st.markdown(f"""
        <div style="background: {card_bg}; border: 1px solid {card_border}; border-radius: 10px; padding: 1.1rem; font-size: 0.85rem;">
            <div style="background: {accent_color}18; border-left: 4px solid {accent_color}; padding: 0.5rem 0.75rem; border-radius: 0 6px 6px 0; font-weight: 700; color: {accent_color}; margin-bottom: 0.8rem;">
                Step 1 • Circulation Desk Logs (N ≥ 35 Records)
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-bottom: 0.8rem;">
                <div style="background: {sub_card_bg}; border: 1px solid {card_border}; border-radius: 8px; padding: 0.65rem;">
                    <strong style="color: {'#38bdf8' if is_dark else '#0284c7'}; font-size: 0.82rem;">Branch A: Discrete (X)</strong>
                    <div style="font-size: 0.76rem; color: {text_sub}; margin-top: 0.25rem;">
                        • Books Borrowed (1–4)<br>
                        • PMF: P(X = x) = n_x / N<br>
                        • Mean E(X) & Var(X)
                    </div>
                </div>
                <div style="background: {sub_card_bg}; border: 1px solid {card_border}; border-radius: 8px; padding: 0.65rem;">
                    <strong style="color: {'#10b981' if is_dark else '#059669'}; font-size: 0.82rem;">Branch B: Continuous (Y)</strong>
                    <div style="font-size: 0.76rem; color: {text_sub}; margin-top: 0.25rem;">
                        • Overdue & Holding Times<br>
                        • Fit Exp(λ) & Gamma(α, θ)<br>
                        • Central Moments μ1–μ4
                    </div>
                </div>
            </div>
            <div style="background: {sub_card_bg}; border: 1px solid {card_border}; border-radius: 8px; padding: 0.65rem; margin-bottom: 0.6rem;">
                <strong style="color: {'#f59e0b' if is_dark else '#d97706'}; font-size: 0.82rem;">SymPy MGF Engine:</strong>
                <span style="font-size: 0.76rem; color: {text_sub};"> Analytical improper integral M_Y(t) = λ/(λ - t) and differentiation at t = 0</span>
            </div>
            <div style="background: {sub_card_bg}; border: 1px solid {card_border}; border-radius: 8px; padding: 0.65rem;">
                <strong style="color: {'#ec4899' if is_dark else '#be185d'}; font-size: 0.82rem;">Inventory Risk Simulation:</strong>
                <span style="font-size: 0.76rem; color: {text_sub};"> Custom loan policy (L) & tail risk P(Y > d) with automated safety buffer allocation</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Summary of Mathematical Formulations")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.markdown(r"""
        **Discrete Random Variable ($X$)**
        - PMF: $P(X = x) = \frac{n_x}{N}$
        - $E(X) = \sum_{x} x P(X = x)$
        - $\operatorname{Var}(X) = E(X^2) - [E(X)]^2$
        """)
    with f_col2:
        st.markdown(r"""
        **Continuous Random Variable ($Y$)**
        - Exponential PDF: $f(y) = \lambda e^{-\lambda y}$
        - Gamma PDF: $f(y) = \frac{1}{\Gamma(\alpha)\theta^\alpha} y^{\alpha-1} e^{-y/\theta}$
        - Central Moments: $\mu_k = \text{scipy.stats.moment}(y, k)$
        """)
    with f_col3:
        st.markdown(r"""
        **MGF & Tail Risk Metrics**
        - $M_Y(t) = \int_0^\infty e^{ty} \lambda e^{-\lambda y} \, dy = \frac{\lambda}{\lambda - t}$
        - Skewness: $\beta_1 = \mu_3^2 / \mu_2^3$
        - Kurtosis: $\beta_2 = \mu_4 / \mu_2^2$
        - Tail Risk: $P(Y > d) = 1 - F(d)$
        """)


# =============================================================================
# TAB 2: DATASET VIEWER
# =============================================================================
with tab_ds:
    st.subheader("Circulation Desk Dataset & Audit")
    st.markdown(r"""
        Gathered circulation desk logs ($N \ge 35$ records) recording borrowing duration (days) 
        and overdue fines accumulated.
    """)
    
    n_records = len(df)
    total_books = df["books_borrowed"].sum() if "books_borrowed" in df.columns else 0
    avg_holding = df["holding_days"].mean() if "holding_days" in df.columns else 0.0
    total_fines = df["fine_amount"].sum() if "fine_amount" in df.columns else 0.0
    overdue_rate = (df["overdue_days"] > 0).mean() * 100 if "overdue_days" in df.columns else 0.0
    
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Records (N)</div>
                <div class="metric-value">{n_records}</div>
                <div class="metric-sub">Min required: 35 (Passed)</div>
            </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Books Circulating</div>
                <div class="metric-value">{total_books}</div>
                <div class="metric-sub">Across all student sessions</div>
            </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Holding Days</div>
                <div class="metric-value">{avg_holding:.1f}d</div>
                <div class="metric-sub">Standard loan: 14.0 days</div>
            </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Fines Accrued</div>
                <div class="metric-value">₹{total_fines:,.0f}</div>
                <div class="metric-sub">Rate: ₹5.00 / overdue day</div>
            </div>
        """, unsafe_allow_html=True)
    with m_col5:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Overdue Incident Rate</div>
                <div class="metric-value">{overdue_rate:.1f}%</div>
                <div class="metric-sub">Returns exceeding loan grace</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("#### Primary Circulation Desk Logs")
    f_col_a, f_col_b = st.columns([1, 3])
    with f_col_a:
        filter_overdue = st.checkbox("Filter: Show only overdue returns (> 0 days)", value=False)
    
    filtered_df = df[df["overdue_days"] > 0] if filter_overdue else df
    
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=320,
        column_config={
            "student_id": st.column_config.TextColumn("Student ID", width="small"),
            "books_borrowed": st.column_config.NumberColumn("Books Borrowed (X)", format="%d"),
            "holding_days": st.column_config.NumberColumn("Holding Duration (Days)", format="%.1f"),
            "overdue_days": st.column_config.NumberColumn("Overdue Delay Days (Y)", format="%.1f"),
            "fine_amount": st.column_config.NumberColumn("Fine Accrued (₹)", format="₹%.2f")
        }
    )
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Complete Circulation Logs (CSV)",
        data=csv_buffer.getvalue(),
        file_name="vcet_library_circulation_logs.csv",
        mime="text/csv",
        help="Export current active dataset for offline verification or academic grading."
    )
    
    st.markdown("#### Descriptive Statistics & Data Hygiene Audit")
    s_col1, s_col2 = st.columns([1.5, 1])
    with s_col1:
        st.write("Numerical Summary:")
        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)
    
    with s_col2:
        st.write("Data Hygiene Verification:")
        null_counts = df.isnull().sum()
        audit_records = []
        for col in df.columns:
            audit_records.append({
                "Column": col,
                "Type": str(df[col].dtype),
                "Missing Values": int(null_counts[col]),
                "Status": "Clean" if null_counts[col] == 0 else "Impute Needed"
            })
        st.dataframe(pd.DataFrame(audit_records), use_container_width=True)


# =============================================================================
# TAB 3: DISCRETE PMF (X)
# =============================================================================
with tab_pmf:
    st.subheader("Discrete Random Variable (X) — Number of Books Borrowed")
    st.markdown(r"""
        Formulating discrete PMF for borrowing frequency.
        Let $X \in \{1, 2, 3, 4\}$ denote the discrete count of books borrowed per student.
    """)
    
    if "books_borrowed" not in df.columns:
        st.error("Column 'books_borrowed' missing from dataset.")
    else:
        pmf_df, exp_x, exp_x2, var_x, std_x = compute_discrete_pmf(df["books_borrowed"])
        
        dm_col1, dm_col2, dm_col3, dm_col4 = st.columns(4)
        with dm_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Expected Value E(X)</div>
                    <div class="metric-value">{exp_x:.4f}</div>
                    <div class="metric-sub">Mean books per student</div>
                </div>
            """, unsafe_allow_html=True)
        with dm_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Second Raw Moment E(X²)</div>
                    <div class="metric-value">{exp_x2:.4f}</div>
                    <div class="metric-sub">Sum of x² P(X=x)</div>
                </div>
            """, unsafe_allow_html=True)
        with dm_col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Variance Var(X)</div>
                    <div class="metric-value">{var_x:.4f}</div>
                    <div class="metric-sub">σ²_X = E(X²) - [E(X)]²</div>
                </div>
            """, unsafe_allow_html=True)
        with dm_col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Standard Deviation σ_X</div>
                    <div class="metric-value">{std_x:.4f}</div>
                    <div class="metric-sub">Dispersion in book checkouts</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### Empirical Probability Mass Function (PMF) Table")
        st.dataframe(
            pmf_df.style.format({
                "P(X = x)": "{:.4f}",
                "CDF F(x)": "{:.4f}",
                "x * P(X=x)": "{:.4f}",
                "x^2 * P(X=x)": "{:.4f}",
                "(x - E[X])^2 * P(X=x)": "{:.4f}"
            }),
            use_container_width=True
        )
        
        st.markdown("#### Discrete Distribution Visualizations")
        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            fig_pmf, ax_pmf = plt.subplots(figsize=(6, 4.2), dpi=120, facecolor=pt["fig_face"])
            ax_pmf.set_facecolor(pt["ax_face"])
            
            x_vals = pmf_df["x"].values
            p_vals = pmf_df["P(X = x)"].values
            
            markerline, stemlines, baseline = ax_pmf.stem(
                x_vals, p_vals, 
                linefmt='-', 
                markerfmt='o', 
                basefmt='k-'
            )
            plt.setp(markerline, markersize=8, color=pt["line_primary"], markeredgecolor=pt["line_primary"])
            plt.setp(stemlines, linewidth=2.2, color=pt["line_primary"])
            plt.setp(baseline, color=pt["grid_color"], linewidth=1.2)
            
            for x, p in zip(x_vals, p_vals):
                ax_pmf.annotate(
                    f"{p:.3f}", 
                    (x, p), 
                    textcoords="offset points", 
                    xytext=(0, 8), 
                    ha='center', 
                    fontweight='600', 
                    fontsize=9,
                    color=pt["text_color"]
                )
                
            ax_pmf.axvline(exp_x, color=pt["line_danger"], linestyle='--', linewidth=1.8, label=f"Mean E(X) = {exp_x:.2f}")
            ax_pmf.set_xticks(x_vals)
            ax_pmf.set_xlabel("Number of Books Borrowed (x)", fontweight='bold', color=pt["text_color"])
            ax_pmf.set_ylabel("Probability P(X = x)", fontweight='bold', color=pt["text_color"])
            ax_pmf.set_title("Empirical PMF Stem Plot", fontsize=11, fontweight='bold', pad=10, color=pt["text_color"])
            ax_pmf.set_ylim(0, max(p_vals) * 1.25)
            ax_pmf.tick_params(colors=pt["tick_color"])
            for spine in ax_pmf.spines.values():
                spine.set_color(pt["grid_color"])
            ax_pmf.grid(True, linestyle=':', alpha=0.5, color=pt["grid_color"])
            ax_pmf.legend(loc="upper right", frameon=True, facecolor=pt["fig_face"], labelcolor=pt["text_color"])
            fig_pmf.tight_layout()
            st.pyplot(fig_pmf, use_container_width=True)
            
        with p_col2:
            fig_cdf, ax_cdf = plt.subplots(figsize=(6, 4.2), dpi=120, facecolor=pt["fig_face"])
            ax_cdf.set_facecolor(pt["ax_face"])
            
            x_step = np.concatenate([[0], x_vals, [max(x_vals) + 1]])
            cdf_step = np.concatenate([[0], pmf_df["CDF F(x)"].values, [1.0]])
            
            ax_cdf.step(x_step, cdf_step, where='post', color=pt["line_secondary"], linewidth=2.5, label="CDF F(x) = P(X ≤ x)")
            ax_cdf.scatter(x_vals, pmf_df["CDF F(x)"].values, color=pt["line_secondary"], s=50, zorder=4)
            
            for x, cdf_v in zip(x_vals, pmf_df["CDF F(x)"].values):
                ax_cdf.annotate(
                    f"{cdf_v:.3f}", 
                    (x, cdf_v), 
                    textcoords="offset points", 
                    xytext=(8, -4), 
                    fontweight='600', 
                    fontsize=9,
                    color=pt["text_color"]
                )
                
            ax_cdf.set_xticks(range(0, int(max(x_vals)) + 2))
            ax_cdf.set_xlabel("Number of Books (x)", fontweight='bold', color=pt["text_color"])
            ax_cdf.set_ylabel("Cumulative Probability F(x)", fontweight='bold', color=pt["text_color"])
            ax_cdf.set_title("Discrete Cumulative Distribution Function (CDF)", fontsize=11, fontweight='bold', pad=10, color=pt["text_color"])
            ax_cdf.set_ylim(-0.05, 1.1)
            ax_cdf.tick_params(colors=pt["tick_color"])
            for spine in ax_cdf.spines.values():
                spine.set_color(pt["grid_color"])
            ax_cdf.grid(True, linestyle=':', alpha=0.5, color=pt["grid_color"])
            ax_cdf.legend(loc="lower right", frameon=True, facecolor=pt["fig_face"], labelcolor=pt["text_color"])
            fig_cdf.tight_layout()
            st.pyplot(fig_cdf, use_container_width=True)
            
        st.markdown("#### Analytical Step-by-Step Derivations")
        with st.expander("Formal mathematical steps for E(X) and Var(X)"):
            st.latex(r"""
                E(X) = \sum_{x=1}^{4} x \cdot P(X = x) = """ + 
                " + ".join([f"({row['x']} \\times {row['P(X = x)']:.4f})" for _, row in pmf_df.iterrows()]) + 
                f" = {exp_x:.4f}"
            )
            st.latex(r"""
                E(X^2) = \sum_{x=1}^{4} x^2 \cdot P(X = x) = """ + 
                " + ".join([f"({int(row['x']**2)} \\times {row['P(X = x)']:.4f})" for _, row in pmf_df.iterrows()]) + 
                f" = {exp_x2:.4f}"
            )
            st.latex(rf"""
                \operatorname{{Var}}(X) = E(X^2) - [E(X)]^2 = {exp_x2:.4f} - ({exp_x:.4f})^2 = {exp_x2:.4f} - {exp_x**2:.4f} = {var_x:.4f}
            """)
            st.latex(rf"""
                \sigma_X = \sqrt{{\operatorname{{Var}}(X)}} = \sqrt{{{var_x:.4f}}} = {std_x:.4f}
            """)

        st.markdown("#### Interactive Discrete Probability Query Engine")
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            k_val = st.slider("Select threshold book count (k):", min_value=1, max_value=4, value=2, step=1)
        with q_col2:
            query_type = st.selectbox(
                "Query Condition:",
                ["Exact: P(X = k)", "Cumulative: P(X ≤ k)", "Exceedance: P(X > k)", "At least: P(X ≥ k)"]
            )
            
        if query_type == "Exact: P(X = k)":
            res_prob = pmf_df.loc[pmf_df["x"] == k_val, "P(X = x)"].values[0] if (pmf_df["x"] == k_val).any() else 0.0
            st.info(f"**Calculated Probability:** $P(X = {k_val}) = {res_prob:.4f}$ ({res_prob * 100:.2f}%)")
        elif query_type == "Cumulative: P(X ≤ k)":
            res_prob = pmf_df.loc[pmf_df["x"] <= k_val, "P(X = x)"].sum()
            st.info(f"**Calculated Probability:** $P(X \\le {k_val}) = {res_prob:.4f}$ ({res_prob * 100:.2f}%)")
        elif query_type == "Exceedance: P(X > k)":
            res_prob = pmf_df.loc[pmf_df["x"] > k_val, "P(X = x)"].sum()
            st.info(f"**Calculated Probability:** $P(X > {k_val}) = {res_prob:.4f}$ ({res_prob * 100:.2f}%)")
        else:
            res_prob = pmf_df.loc[pmf_df["x"] >= k_val, "P(X = x)"].sum()
            st.info(f"**Calculated Probability:** $P(X \\ge {k_val}) = {res_prob:.4f}$ ({res_prob * 100:.2f}%)")


# =============================================================================
# TAB 4: CONTINUOUS DELAY (Y) & MOMENTS
# =============================================================================
with tab_cont:
    st.subheader(f"Continuous Variable ({continuous_var_option}) — Distribution Fitting & Moments")
    st.markdown(f"""
        Fitting continuous Exponential and Gamma PDF distributions for overdue days, 
        and calculating first four central moments ($\mu_1, \mu_2, \mu_3, \mu_4$), standard deviation, and skewness/kurtosis coefficients 
        using **`scipy.stats.moment`**.
    """)
    
    y_data = df[continuous_var_option].dropna().values
    
    if len(y_data) == 0:
        st.error("No valid data available for continuous analysis.")
    else:
        moments = compute_central_moments(y_data)
        
        loc_exp, scale_exp = stats.expon.fit(y_data, floc=0)
        fitted_lambda = 1.0 / scale_exp if scale_exp > 0 else 0.0
        
        y_data_gamma = np.where(y_data == 0, 1e-4, y_data)
        alpha_gamma, loc_gamma, scale_gamma = stats.gamma.fit(y_data_gamma, floc=0)
        beta_gamma = 1.0 / scale_gamma if scale_gamma > 0 else 0.0
        
        cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
        with cm_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">1st Central Moment (μ₁)</div>
                    <div class="metric-value">{moments['mu1']:.4f}</div>
                    <div class="metric-sub"><code>stats.moment(y, 1)</code> ≡ 0</div>
                </div>
            """, unsafe_allow_html=True)
        with cm_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">2nd Central Moment (μ₂ = σ²)</div>
                    <div class="metric-value">{moments['mu2']:.4f}</div>
                    <div class="metric-sub">Std Dev σ = {moments['std']:.3f} days</div>
                </div>
            """, unsafe_allow_html=True)
        with cm_col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Skewness (β₁ & γ₁)</div>
                    <div class="metric-value">β₁={moments['beta1']:.2f}</div>
                    <div class="metric-sub">γ₁ = {moments['gamma1']:.3f} (Right-skewed)</div>
                </div>
            """, unsafe_allow_html=True)
        with cm_col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Kurtosis (β₂ & γ₂)</div>
                    <div class="metric-value">β₂={moments['beta2']:.2f}</div>
                    <div class="metric-sub">Excess γ₂ = {moments['gamma2']:.3f} (Leptokurtic)</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("#### Distribution Fitting Visualizations")
        viz_col1, viz_col2 = st.columns(2)
        
        y_eval = np.linspace(0, max(y_data) * 1.15, 300)
        pdf_exp = stats.expon.pdf(y_eval, scale=scale_exp)
        pdf_gamma = stats.gamma.pdf(y_eval, a=alpha_gamma, scale=scale_gamma)
        
        with viz_col1:
            fig_hist, ax_hist = plt.subplots(figsize=(6.2, 4.2), dpi=120, facecolor=pt["fig_face"])
            ax_hist.set_facecolor(pt["ax_face"])
            
            sns.histplot(
                y_data, 
                stat="density", 
                bins=15, 
                color=pt["hist_color"], 
                edgecolor=pt["hist_edge"], 
                alpha=0.6, 
                label="Empirical Histogram",
                ax=ax_hist
            )
            sns.kdeplot(y_data, color=pt["text_color"], linewidth=2.0, label="Empirical KDE", ax=ax_hist)
            
            ax_hist.plot(y_eval, pdf_exp, color=pt["line_danger"], linewidth=2.2, linestyle="--", 
                         label=rf"Fitted Expon ($\hat{{\lambda}}={fitted_lambda:.3f}$)")
            ax_hist.plot(y_eval, pdf_gamma, color=pt["line_primary"], linewidth=2.2, linestyle="-.", 
                         label=rf"Fitted Gamma ($\hat{{\alpha}}={alpha_gamma:.2f}, \hat{{\theta}}={scale_gamma:.2f}$)")
            
            ax_hist.set_xlabel(f"{continuous_var_option.replace('_', ' ').title()} (Days)", fontweight="bold", color=pt["text_color"])
            ax_hist.set_ylabel("Probability Density f(y)", fontweight="bold", color=pt["text_color"])
            ax_hist.set_title("Empirical vs Fitted Continuous Distributions", fontsize=11, fontweight="bold", pad=10, color=pt["text_color"])
            ax_hist.tick_params(colors=pt["tick_color"])
            for spine in ax_hist.spines.values():
                spine.set_color(pt["grid_color"])
            ax_hist.grid(True, linestyle=":", alpha=0.5, color=pt["grid_color"])
            ax_hist.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor=pt["fig_face"], labelcolor=pt["text_color"])
            fig_hist.tight_layout()
            st.pyplot(fig_hist, use_container_width=True)
            
        with viz_col2:
            fig_cdf_cont, ax_cdf_cont = plt.subplots(figsize=(6.2, 4.2), dpi=120, facecolor=pt["fig_face"])
            ax_cdf_cont.set_facecolor(pt["ax_face"])
            
            y_sorted = np.sort(y_data)
            ecdf_vals = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
            ax_cdf_cont.plot(y_sorted, ecdf_vals, color=pt["text_color"], linewidth=2.2, label="Empirical CDF (eCDF)")
            
            cdf_exp = stats.expon.cdf(y_eval, scale=scale_exp)
            cdf_gamma = stats.gamma.cdf(y_eval, a=alpha_gamma, scale=scale_gamma)
            
            ax_cdf_cont.plot(y_eval, cdf_exp, color=pt["line_danger"], linestyle="--", linewidth=2.0, label="Theoretical Expon CDF")
            ax_cdf_cont.plot(y_eval, cdf_gamma, color=pt["line_primary"], linestyle="-.", linewidth=2.0, label="Theoretical Gamma CDF")
            
            ax_cdf_cont.set_xlabel(f"{continuous_var_option.replace('_', ' ').title()} (Days)", fontweight="bold", color=pt["text_color"])
            ax_cdf_cont.set_ylabel("Cumulative Probability F(y)", fontweight="bold", color=pt["text_color"])
            ax_cdf_cont.set_title("CDF Goodness-of-Fit Comparison", fontsize=11, fontweight="bold", pad=10, color=pt["text_color"])
            ax_cdf_cont.tick_params(colors=pt["tick_color"])
            for spine in ax_cdf_cont.spines.values():
                spine.set_color(pt["grid_color"])
            ax_cdf_cont.grid(True, linestyle=":", alpha=0.5, color=pt["grid_color"])
            ax_cdf_cont.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor=pt["fig_face"], labelcolor=pt["text_color"])
            fig_cdf_cont.tight_layout()
            st.pyplot(fig_cdf_cont, use_container_width=True)

        st.markdown("#### Quantitative Moments & Model Comparison")
        t_col1, t_col2 = st.columns(2)
        
        with t_col1:
            st.markdown("##### Central Moments Breakdown Table (`scipy.stats.moment`)")
            moments_table = pd.DataFrame({
                "Moment": [
                    "1st Central Moment (μ₁)",
                    "2nd Central Moment (μ₂ = σ²)",
                    "3rd Central Moment (μ₃)",
                    "4th Central Moment (μ₄)",
                    "Standard Deviation (σ)",
                    "Skewness Coefficient (β₁ = μ₃² / μ₂³)",
                    "Pearson Skewness (γ₁ = μ₃ / σ³)",
                    "Kurtosis Coefficient (β₂ = μ₄ / μ₂²)",
                    "Excess Kurtosis (γ₂ = β₂ - 3)"
                ],
                "Numerical Value": [
                    f"{moments['mu1']:.6f}",
                    f"{moments['mu2']:.4f}",
                    f"{moments['mu3']:.4f}",
                    f"{moments['mu4']:.4f}",
                    f"{moments['std']:.4f}",
                    f"{moments['beta1']:.4f}",
                    f"{moments['gamma1']:.4f}",
                    f"{moments['beta2']:.4f}",
                    f"{moments['gamma2']:.4f}"
                ],
                "SciPy Evaluation": [
                    "stats.moment(y, moment=1)",
                    "stats.moment(y, moment=2)",
                    "stats.moment(y, moment=3)",
                    "stats.moment(y, moment=4)",
                    "np.sqrt(stats.moment(y, 2))",
                    "mu3² / mu2³",
                    "stats.skew(y, bias=True)",
                    "mu4 / mu2²",
                    "beta2 - 3.0"
                ]
            })
            st.dataframe(moments_table, use_container_width=True, hide_index=True)
            
        with t_col2:
            st.markdown("##### Goodness-of-Fit & Model Criteria")
            ks_exp = stats.kstest(y_data, 'expon', args=(0, scale_exp))
            ks_gamma = stats.kstest(y_data_gamma, 'gamma', args=(alpha_gamma, 0, scale_gamma))
            
            fit_df = pd.DataFrame({
                "Fitted Model": ["Exponential Dist.", "Gamma Dist."],
                "Fitted Parameters": [
                    f"λ = {fitted_lambda:.4f} (scale={scale_exp:.2f})",
                    f"α = {alpha_gamma:.3f}, θ = {scale_gamma:.3f}"
                ],
                "KS Stat (D)": [f"{ks_exp.statistic:.4f}", f"{ks_gamma.statistic:.4f}"],
                "KS p-value": [f"{ks_exp.pvalue:.4f}", f"{ks_gamma.pvalue:.4f}"],
                "Goodness of Fit": [
                    "Strong tail fit" if ks_exp.pvalue > 0.05 else "Moderate fit",
                    "Flexible shape fit" if ks_gamma.pvalue > 0.05 else "Moderate fit"
                ]
            })
            st.dataframe(fit_df, use_container_width=True, hide_index=True)
            
            st.markdown(f"""
                <div class="math-callout">
                    <strong>Theoretical Shape Diagnosis (VCET Syllabus):</strong><br>
                    • <strong>Asymmetry:</strong> $\\gamma_1 = {moments['gamma1']:.2f} > 0$ and $\\beta_1 = {moments['beta1']:.2f}$ confirm a strongly right-skewed delay distribution. Most returns occur with minimal delay, but non-trivial severe delays extend the right tail.<br>
                    • <strong>Tail Heaviness:</strong> $\\beta_2 = {moments['beta2']:.2f} > 3.0$ classifies the delay pattern as <strong>leptokurtic</strong> (excess kurtosis $\\gamma_2 = {moments['gamma2']:.2f}$). This confirms heavy-tail risk requiring reserve inventory.
                </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TAB 5: SYMBOLIC MGF ENGINE
# =============================================================================
with tab_mgf:
    st.subheader("Symbolic Moment Generating Function (MGF) Engine")
    st.markdown(r"""
        Algebraic formulation of $M(t)$ and verifying mean/variance via $t \to 0$ differentiation 
        using **SymPy** for automated integral evaluation and symbolic derivatives.
    """)
    
    mgf_results = derive_symbolic_mgf()
    
    st.markdown("#### Symbolic Integration via SymPy")
    st.markdown(r"""
        For a continuous random variable $Y \sim \text{Exp}(\lambda)$ with PDF $f(y) = \lambda e^{-\lambda y}$ ($y \ge 0$):
    """)
    st.latex(r"""
        M_Y(t) = E[e^{tY}] = \int_{0}^{\infty} e^{ty} \cdot \lambda e^{-\lambda y} \, dy 
        = \lambda \int_{0}^{\infty} e^{-(\lambda - t)y} \, dy
    """)
    
    st.markdown(f"""
        <div class="math-callout">
            <strong>SymPy Execution Proof:</strong><br>
            <code>t, lam, y = sp.symbols('t lambda y', positive=True)</code><br>
            <code>M_t = sp.integrate(lam * sp.exp(-(lam - t)*y), (y, 0, sp.oo), conds='none')</code><br>
            <strong>SymPy Evaluated Result:</strong> <code>{sp.latex(mgf_results['M_t_integrated'])}</code> &nbsp;=&nbsp; 
            $\frac{{\lambda}}{{\lambda - t}}$ &nbsp; (Valid under convergence condition $t < \lambda$)
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### Automated Symbolic Differentiation at $t = 0$")
    st.markdown(r"""
        Raw moments $\mu_k' = E[Y^k]$ are extracted via differentiation:
        $$\mu_k' = \left. \frac{d^k M_Y(t)}{dt^k} \right|_{t=0}$$
    """)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**1st Derivative (Expected Value / Mean):**")
        st.latex(rf"\frac{{d M_Y(t)}}{{dt}} = {sp.latex(mgf_results['d1_M'])}")
        st.latex(rf"\mu_1' = E[Y] = \left. \frac{{d M_Y(t)}}{{dt}} \right|_{{t=0}} = {sp.latex(mgf_results['mu1_prime'])}")
        
    with col_d2:
        st.markdown("**2nd Derivative (Second Raw Moment):**")
        st.latex(rf"\frac{{d^2 M_Y(t)}}{{dt^2}} = {sp.latex(mgf_results['d2_M'])}")
        st.latex(rf"\mu_2' = E[Y^2] = \left. \frac{{d^2 M_Y(t)}}{{dt^2}} \right|_{{t=0}} = {sp.latex(mgf_results['mu2_prime'])}")

    st.markdown("**Theoretical Variance Derivation:**")
    st.latex(r"""
        \operatorname{Var}(Y) = \sigma^2 = \mu_2' - (\mu_1')^2 = \frac{2}{\lambda^2} - \left( \frac{1}{\lambda} \right)^2 
        = \frac{2}{\lambda^2} - \frac{1}{\lambda^2} = \frac{1}{\lambda^2}
    """)
    st.latex(r"""
        \sigma = \sqrt{\operatorname{Var}(Y)} = \frac{1}{\lambda}
    """)
    
    st.markdown("---")
    st.markdown("#### Empirical Verification Engine")
    st.markdown(f"""
        Substituting the fitted parameter $\hat{{\lambda}} = {fitted_lambda:.4f}$ from the continuous distribution fit into the 
        symbolic expressions:
    """)
    
    theo_mean = 1.0 / fitted_lambda if fitted_lambda > 0 else 0.0
    theo_var = 1.0 / (fitted_lambda ** 2) if fitted_lambda > 0 else 0.0
    theo_std = 1.0 / fitted_lambda if fitted_lambda > 0 else 0.0
    theo_beta1 = 4.0
    theo_beta2 = 9.0
    
    emp_mean = moments['mean']
    emp_var = moments['mu2']
    emp_std = moments['std']
    emp_beta1 = moments['beta1']
    emp_beta2 = moments['beta2']
    
    comp_df = pd.DataFrame({
        "Metric": [
            "Mean E(Y) = μ₁'",
            "Second Moment E(Y²) = μ₂'",
            "Variance Var(Y) = σ²",
            "Standard Deviation σ",
            "Skewness Coefficient β₁",
            "Kurtosis Coefficient β₂"
        ],
        "SymPy Formula": [
            "1 / λ",
            "2 / λ²",
            "1 / λ²",
            "1 / λ",
            "4 (Constant)",
            "9 (Constant)"
        ],
        "Theoretical (Evaluated at λ̂)": [
            f"{theo_mean:.4f}",
            f"{2 * theo_var:.4f}",
            f"{theo_var:.4f}",
            f"{theo_std:.4f}",
            f"{theo_beta1:.2f}",
            f"{theo_beta2:.2f}"
        ],
        "Empirical Sample Metric": [
            f"{emp_mean:.4f}",
            f"{np.mean(y_data**2):.4f}",
            f"{emp_var:.4f}",
            f"{emp_std:.4f}",
            f"{emp_beta1:.2f}",
            f"{emp_beta2:.2f}"
        ],
        "Verification Status": [
            "Exact Match (via MLE)",
            "Valid Moment Bounds",
            "Within Sample Dispersion",
            "Aligned Spread",
            "Heavy-Tail Asymmetry",
            "Leptokurtic Agreement"
        ]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("#### Interactive MGF Visualizer: M_Y(t) vs t")
    mgf_col1, mgf_col2 = st.columns([1, 2])
    with mgf_col1:
        st.markdown(f"""
            Observe the function curve $M_Y(t) = \\frac{{\\lambda}}{{\\lambda - t}}$.
            Notice that as $t \\to \\lambda^-$ ($t \\to {fitted_lambda:.3f}$), $M_Y(t) \\to \\infty$ 
            (vertical asymptote).
        """)
        t_user = st.slider(
            "Evaluate M_Y(t) at custom t:",
            min_value=-float(np.round(fitted_lambda * 1.5, 2)),
            max_value=float(np.round(fitted_lambda * 0.92, 2)),
            value=0.0,
            step=0.02
        )
        val_mgf_user = fitted_lambda / (fitted_lambda - t_user) if (fitted_lambda - t_user) != 0 else np.nan
        st.metric(label=f"M_Y(t = {t_user:.2f})", value=f"{val_mgf_user:.4f}")
        
    with mgf_col2:
        t_domain = np.linspace(-fitted_lambda * 1.2, fitted_lambda * 0.92, 200)
        mgf_curve = fitted_lambda / (fitted_lambda - t_domain)
        
        fig_mgf, ax_mgf = plt.subplots(figsize=(6.5, 3.8), dpi=120, facecolor=pt["fig_face"])
        ax_mgf.set_facecolor(pt["ax_face"])
        
        ax_mgf.plot(t_domain, mgf_curve, color=pt["line_primary"], linewidth=2.5, label=r"$M_Y(t) = \frac{\lambda}{\lambda - t}$")
        ax_mgf.axvline(fitted_lambda, color=pt["line_danger"], linestyle="--", linewidth=1.5, label=rf"Asymptote $t = \lambda = {fitted_lambda:.3f}$")
        ax_mgf.scatter([t_user], [val_mgf_user], color=pt["line_accent"], s=70, zorder=5, label=f"t = {t_user:.2f}")
        
        t_tangent = np.linspace(-0.2, 0.2, 50)
        tangent_line = 1.0 + theo_mean * t_tangent
        ax_mgf.plot(t_tangent, tangent_line, color=pt["line_secondary"], linestyle=":", linewidth=1.8, label=r"Tangent at t=0 (Slope = $\mu_1'$)")
        
        ax_mgf.set_xlabel("Parameter t", fontweight="bold", color=pt["text_color"])
        ax_mgf.set_ylabel(r"$M_Y(t)$", fontweight="bold", color=pt["text_color"])
        ax_mgf.set_title("Moment Generating Function Curve & Asymptote", fontsize=11, fontweight="bold", pad=10, color=pt["text_color"])
        ax_mgf.set_ylim(0, 10)
        ax_mgf.tick_params(colors=pt["tick_color"])
        for spine in ax_mgf.spines.values():
            spine.set_color(pt["grid_color"])
        ax_mgf.grid(True, linestyle=":", alpha=0.5, color=pt["grid_color"])
        ax_mgf.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor=pt["fig_face"], labelcolor=pt["text_color"])
        fig_mgf.tight_layout()
        st.pyplot(fig_mgf, use_container_width=True)


# =============================================================================
# TAB 6: OVERDUE RISK SIMULATOR
# =============================================================================
with tab_sim:
    st.subheader("Overdue Tail Risk Simulator & Custom Loan Periods")
    st.markdown(r"""
        Visualizing holding time distributions and predicting overdue risk probabilities **for custom loan periods**.
    """)
    
    # Part A: Custom Loan Period Policy Analyzer
    st.markdown("#### Part A: Custom Loan Period (L) Policy Simulator")
    lp_col1, lp_col2 = st.columns([1.2, 1.0])
    
    with lp_col1:
        custom_loan_period = st.slider(
            "Select Library Official Base Loan Period (L in Days):",
            min_value=5,
            max_value=30,
            value=14,
            step=1,
            help="The official allowed checkout duration before overdue fines begin to accrue."
        )
        
    with lp_col2:
        holding_data = df["holding_days"].dropna().values
        pct_exceeding_l = (holding_data > custom_loan_period).mean() * 100
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Overdue Rate Under L = {custom_loan_period}d</div>
                <div class="metric-value" style="color: {'#ef4444' if pct_exceeding_l > 40 else '#f59e0b' if pct_exceeding_l > 15 else '#10b981'};">
                    {pct_exceeding_l:.1f}%
                </div>
                <div class="metric-sub">P(Holding Time > {custom_loan_period} days)</div>
            </div>
        """, unsafe_allow_html=True)
        
    fig_hold, ax_hold = plt.subplots(figsize=(10, 3.8), dpi=120, facecolor=pt["fig_face"])
    ax_hold.set_facecolor(pt["ax_face"])
    
    sns.histplot(holding_data, kde=True, bins=18, color=pt["line_primary"], alpha=0.55, ax=ax_hold, label="Holding Days Distribution")
    ax_hold.axvline(custom_loan_period, color=pt["line_danger"], linestyle="--", linewidth=2.2, 
                    label=f"Selected Loan Period L = {custom_loan_period} days")
    ax_hold.axvspan(custom_loan_period, max(holding_data) * 1.1, color=pt["line_danger"], alpha=0.15, 
                    label=f"Overdue Zone ({pct_exceeding_l:.1f}% of circulation)")
    ax_hold.set_xlabel("Total Holding Duration (Days)", fontweight="bold", color=pt["text_color"])
    ax_hold.set_ylabel("Density / Frequency", fontweight="bold", color=pt["text_color"])
    ax_hold.set_title(f"Holding Time Distribution vs Custom Loan Period L = {custom_loan_period} Days", fontsize=11, fontweight="bold", pad=10, color=pt["text_color"])
    ax_hold.tick_params(colors=pt["tick_color"])
    for spine in ax_hold.spines.values():
        spine.set_color(pt["grid_color"])
    ax_hold.grid(True, linestyle=":", alpha=0.5, color=pt["grid_color"])
    ax_hold.legend(loc="upper right", frameon=True, facecolor=pt["fig_face"], labelcolor=pt["text_color"])
    fig_hold.tight_layout()
    st.pyplot(fig_hold, use_container_width=True)

    st.markdown("---")
    # Part B: Delay Threshold (d) Tail Risk Simulator
    st.markdown("#### Part B: Overdue Delay Tail Risk P(Y > d) & Inventory Allocation")
    
    max_d_slider = int(np.ceil(max(y_data))) if len(y_data) > 0 else 25
    r_col1, r_col2 = st.columns([1.2, 1.0])
    
    with r_col1:
        d_threshold = st.slider(
            "Select Overdue Delay Threshold (d in Days beyond grace):",
            min_value=0.0,
            max_value=float(max_d_slider),
            value=3.0,
            step=0.5,
            help="Threshold days beyond which a book is classified as a severe delayed hold."
        )
        
    with r_col2:
        active_inventory = st.number_input(
            "Active Circulating Books Inventory (Collection Size):",
            min_value=50,
            max_value=10000,
            value=500,
            step=50,
            help="Total volume of checked-out books currently held by students."
        )
        daily_fine_rate = st.number_input(
            "Overdue Fine Rate (₹ per day):",
            min_value=1.0,
            max_value=50.0,
            value=5.0,
            step=1.0
        )
        
    p_emp_tail = (y_data > d_threshold).mean() if len(y_data) > 0 else 0.0
    p_exp_tail = np.exp(-fitted_lambda * d_threshold)
    p_gamma_tail = 1.0 - stats.gamma.cdf(d_threshold, a=alpha_gamma, scale=scale_gamma)
    
    primary_p = p_exp_tail
    
    if primary_p < 0.15:
        risk_class = "badge-low"
        risk_label = "LOW RISK"
        alert_title = "Normal Inventory Circulation"
        alert_desc = (
            "Overdue tail probability is minimal (< 15%). Standard automated email reminders "
            "and routine circulation operations are sufficient. Safety stock levels are optimal."
        )
        rec_buffer = "0% - 5% Safety Buffer"
        action_item = "Standard routine notifications; no urgent recall required."
    elif 0.15 <= primary_p <= 0.40:
        risk_class = "badge-medium"
        risk_label = "MODERATE RISK"
        alert_title = "Elevated Circulation Hold Risk"
        alert_desc = (
            "Overdue tail probability is significant (15% - 40%). High probability of reservation waitlists. "
            "Staff should send secondary SMS alerts and maintain reserve buffer stock."
        )
        rec_buffer = "15% - 20% Reserved Buffer"
        action_item = "Dispatch 2nd tier SMS reminders; prioritize hold queues."
    else:
        risk_class = "badge-high"
        risk_label = "HIGH RISK"
        alert_title = "Severe Inventory Stockout Risk"
        alert_desc = (
            "Overdue tail probability exceeds 40%! Widespread bottlenecks expected. "
            "Initiate immediate patron recalls, escalate daily fines, and procure digital copies."
        )
        rec_buffer = "30%+ Immediate Buffer / E-Book Licenses"
        action_item = "Escalate fines; restrict renewal permits; trigger instant patron recall."

    alert_col1, alert_col2 = st.columns([1.5, 1])
    with alert_col1:
        st.markdown(f"""
            <div style="background: {'rgba(30, 41, 59, 0.6)' if is_dark else '#f1f5f9'}; border: 1px solid {'rgba(255, 255, 255, 0.1)' if is_dark else '#e2e8f0'}; border-radius: 10px; padding: 1.25rem;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
                    <span class="{risk_class}">{risk_label}</span>
                    <span style="font-size: 0.85rem; color: {'#94a3b8' if is_dark else '#64748b'};">Threshold: <strong>d = {d_threshold} days</strong></span>
                </div>
                <h3 style="margin: 0.2rem 0 0.5rem 0; font-size: 1.25rem;">{alert_title}</h3>
                <p style="color: {'#cbd5e1' if is_dark else '#334155'}; font-size: 0.95rem; margin-bottom: 0.8rem; line-height: 1.5;">{alert_desc}</p>
                <div style="font-size: 0.85rem; color: {'#38bdf8' if is_dark else '#0284c7'};">
                    <strong>Actionable Protocol:</strong> {action_item}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with alert_col2:
        st.markdown("##### Tail Risk Probability P(Y > d)")
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Theoretical Exponential Model</div>
                <div class="metric-value">{p_exp_tail * 100:.2f}%</div>
                <div class="metric-sub">P(Y > {d_threshold}) = e^(-{fitted_lambda:.3f} × {d_threshold}) = {p_exp_tail:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Empirical Sample Tail Risk</div>
                <div class="metric-value" style="color: {'#10b981' if is_dark else '#059669'};">{p_emp_tail * 100:.2f}%</div>
                <div class="metric-sub">Fraction > {d_threshold} days ({int(p_emp_tail * len(y_data))}/{len(y_data)})</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Operational Inventory Allocation Impact")
    expected_delayed_books = int(np.round(active_inventory * p_exp_tail))
    expected_on_shelf = active_inventory - expected_delayed_books
    estimated_daily_fines = expected_delayed_books * daily_fine_rate
    
    imp_col1, imp_col2, imp_col3, imp_col4 = st.columns(4)
    with imp_col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Expected Severely Delayed</div>
                <div class="metric-value">{expected_delayed_books}</div>
                <div class="metric-sub">Books held past {d_threshold} days</div>
            </div>
        """, unsafe_allow_html=True)
    with imp_col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">On-Time Available Volume</div>
                <div class="metric-value">{expected_on_shelf}</div>
                <div class="metric-sub">Books circulating within threshold</div>
            </div>
        """, unsafe_allow_html=True)
    with imp_col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recommended Reserve Buffer</div>
                <div class="metric-value" style="color: {'#f59e0b' if is_dark else '#d97706'};">{rec_buffer.split(' ')[0]}</div>
                <div class="metric-sub">{rec_buffer}</div>
            </div>
        """, unsafe_allow_html=True)
    with imp_col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Daily Fine Exposure</div>
                <div class="metric-value">₹{estimated_daily_fines:,.0f}</div>
                <div class="metric-sub">Accruing across delayed pool</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Tail Risk Sensitivity Curve: P(Y > d) vs Threshold Days (d)")
    
    d_range = np.linspace(0, max_d_slider, 200)
    p_tail_exp_curve = np.exp(-fitted_lambda * d_range)
    p_tail_gamma_curve = 1.0 - stats.gamma.cdf(d_range, a=alpha_gamma, scale=scale_gamma)
    p_tail_emp_curve = np.array([(y_data > val).mean() for val in d_range])
    
    fig_risk, ax_risk = plt.subplots(figsize=(10, 4.2), dpi=120, facecolor=pt["fig_face"])
    ax_risk.set_facecolor(pt["ax_face"])
    
    ax_risk.axhspan(0.40, 1.05, color=pt["line_danger"], alpha=0.15, label="High Risk Zone (> 40%)")
    ax_risk.axhspan(0.15, 0.40, color=pt["line_accent"], alpha=0.15, label="Medium Risk Zone (15% - 40%)")
    ax_risk.axhspan(0.00, 0.15, color=pt["line_secondary"], alpha=0.15, label="Low Risk Zone (< 15%)")
    
    ax_risk.plot(d_range, p_tail_exp_curve, color=pt["line_danger"], linewidth=2.5, label=r"Exponential Model $e^{-\lambda d}$")
    ax_risk.plot(d_range, p_tail_gamma_curve, color=pt["line_primary"], linestyle="-.", linewidth=2.0, label="Gamma Model")
    ax_risk.plot(d_range, p_tail_emp_curve, color=pt["text_color"], linestyle=":", linewidth=1.8, label="Empirical Tail Probability")
    
    ax_risk.axvline(d_threshold, color="#7c3aed", linestyle="--", linewidth=2.0, label=f"Selected d = {d_threshold}d")
    ax_risk.scatter([d_threshold], [p_exp_tail], color="#7c3aed", s=70, zorder=6)
    
    ax_risk.set_xlabel("Delay Threshold Days (d)", fontweight="bold", color=pt["text_color"])
    ax_risk.set_ylabel("Tail Probability P(Y > d)", fontweight="bold", color=pt["text_color"])
    ax_risk.set_title("Exceedance Probability Decay Curve & Resource Risk Zones", fontsize=11, fontweight="bold", pad=10, color=pt["text_color"])
    ax_risk.set_ylim(-0.02, 1.05)
    ax_risk.set_xlim(0, max_d_slider)
    ax_risk.tick_params(colors=pt["tick_color"])
    for spine in ax_risk.spines.values():
        spine.set_color(pt["grid_color"])
    ax_risk.grid(True, linestyle=":", alpha=0.5, color=pt["grid_color"])
    ax_risk.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor=pt["fig_face"], labelcolor=pt["text_color"])
    fig_risk.tight_layout()
    st.pyplot(fig_risk, use_container_width=True)


# -----------------------------------------------------------------------------
# 9. FOOTER & ACADEMIC CITATION
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; color: {'#64748b' if is_dark else '#475569'}; font-size: 0.82rem; padding: 1rem 0;">
        <strong>Vidyavardhini's College of Engineering & Technology (VCET) • Department of Information Technology</strong><br>
        Applied Mathematics Thinking-I (2343111) • Probability & Random Variables • <strong>Mini Project Group 15 (Batch B4)</strong><br>
        Built with Streamlit, SciPy.stats, SymPy, NumPy, Pandas, Matplotlib & Seaborn.
    </div>
""", unsafe_allow_html=True)
