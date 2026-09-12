"""
Streamlit Dashboard: Firm Maturity, AI Adoption & Financial Performance
-------------------------------------------------------------------------
A modern, finance-styled dashboard for panel data of publicly listed
service firms (Finland, 2021-2023). Includes:

  Tab 1 - Dashboard & Regression Results   (KPI cards + OLS models)
  Tab 2 - Exploratory Visualizations       (descriptive stats, heatmap, scatter)
  Tab 3 - Methodology & References         (panel-data method note + citations)

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="AI Adoption & Firm Performance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# COLOR PALETTE  (navy / teal — finance & tech feel)
# ============================================================================
NAVY = "#0B2545"
NAVY_LIGHT = "#13315C"
TEAL = "#14B8A6"
TEAL_LIGHT = "#5EEAD4"
SKY = "#38BDF8"
AMBER = "#F59E0B"
SLATE = "#64748B"
BG = "#F4F6F9"
CARD_BG = "#FFFFFF"

COLOR_SEQUENCE = [NAVY, TEAL, SKY, AMBER, "#8B5CF6", "#EF4444"]
DIVERGING_SCALE = [[0.0, NAVY], [0.5, "#FFFFFF"], [1.0, TEAL]]

px.defaults.color_discrete_sequence = COLOR_SEQUENCE
px.defaults.template = "plotly_white"

# ============================================================================
# CUSTOM CSS — modern finance-dashboard look
# ============================================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background-color: {BG};
    }}

    /* Header banner */
    .app-header {{
        background: linear-gradient(120deg, {NAVY} 0%, {NAVY_LIGHT} 55%, {TEAL} 160%);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(11, 37, 69, 0.25);
    }}
    .app-header h1 {{
        color: #FFFFFF;
        font-weight: 800;
        font-size: 1.9rem;
        margin-bottom: 0.3rem;
    }}
    .app-header p {{
        color: #D8E4F2;
        font-size: 0.98rem;
        margin: 0;
    }}
    .app-badge {{
        display: inline-block;
        background: rgba(20, 184, 166, 0.18);
        color: {TEAL_LIGHT};
        border: 1px solid rgba(94, 234, 212, 0.4);
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 0.7rem;
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid #E5EAF1;
        border-left: 4px solid {TEAL};
        border-radius: 14px;
        padding: 1rem 1.2rem 0.8rem 1.2rem;
        box-shadow: 0 2px 10px rgba(11, 37, 69, 0.05);
    }}
    div[data-testid="stMetricValue"] {{
        color: {NAVY};
        font-weight: 800;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {SLATE};
        font-weight: 600;
    }}

    /* Section headers */
    h2, h3 {{
        color: {NAVY};
        font-weight: 700;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #E9EEF5;
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        font-weight: 600;
        color: {SLATE};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {NAVY} !important;
        color: #FFFFFF !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {NAVY};
    }}
    section[data-testid="stSidebar"] * {{
        color: #E5EAF1 !important;
    }}
    section[data-testid="stSidebar"] .stSelectbox div, 
    section[data-testid="stSidebar"] .stMultiSelect div {{
        color: {NAVY} !important;
    }}

    /* Reference cards */
    .ref-card {{
        background-color: {CARD_BG};
        border: 1px solid #E5EAF1;
        border-left: 4px solid {NAVY};
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 2px 8px rgba(11, 37, 69, 0.04);
    }}
    .ref-card b {{ color: {NAVY}; }}

    .method-card {{
        background-color: {CARD_BG};
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        border: 1px solid #E5EAF1;
        box-shadow: 0 2px 10px rgba(11, 37, 69, 0.05);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# HEADER
# ============================================================================
st.markdown(
    """
    <div class="app-header">
        <div class="app-badge">SERVICE FIRMS · 2021–2023 PANEL DATA · PUBLIC DEMO</div>
        <h1>📈 Firm Maturity, AI Adoption &amp; Financial Performance</h1>
        <p>An interactive research dashboard linking AI adoption and organizational maturity
        to ROA, ROE and financial risk — built for panel-data analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# DATA LOADING
# ============================================================================
DEFAULT_PATH = "sample_data.csv"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-looking columns (thousands separators, quotes) to floats."""
    for col in df.columns:
        if col in ("Company", "Unit") :
            continue
        if df[col].dtype == object:
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace('"', "", regex=False)
                .str.strip()
            )
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().mean() > 0.5:
                df[col] = converted
    return df


@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = clean_columns(df)
    df = coerce_numeric(df)
    return df


st.sidebar.markdown("### 📁 Data source")
uploaded = st.sidebar.file_uploader(
    "Upload your own CSV (optional)",
    type=["csv"],
    help=(
        "Use this to explore your own private/licensed dataset (e.g. an ORBIS export). "
        "It is processed only in this browser session's memory and is never saved to "
        "the app's repository or disk."
    ),
)

df = None
using_sample = False
if uploaded is not None:
    df = load_data(uploaded)
    st.sidebar.success("Your file was loaded for this session only.")
else:
    try:
        df = load_data(DEFAULT_PATH)
        using_sample = True
        st.sidebar.info("Showing built-in **sample/demo data** (synthetic).")
    except FileNotFoundError:
        st.sidebar.warning(f"No `{DEFAULT_PATH}` found next to the app. Please upload a CSV file.")

if df is None:
    st.stop()

if using_sample:
    st.info(
        "🧪 **This is a public demo running on synthetically generated sample data** — "
        "not the real, licensed ORBIS dataset used in the underlying thesis. Company names, "
        "financials and ratios below are fabricated for demonstration purposes only, though "
        "they are generated to broadly mirror the same statistical relationships discussed "
        "in the study (AI adoption, firm maturity, ROA/ROE, risk). "
        "To analyze your own data, use the uploader in the sidebar — your file stays private "
        "to your browser session and is never stored on the server.",
        icon="🧪",
    )

numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

st.sidebar.markdown("---")
st.sidebar.write(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")
with st.sidebar.expander("Preview raw data"):
    st.dataframe(df.head(10))

st.sidebar.markdown("---")
st.sidebar.caption(
    "Palette: navy (#0B2545) + teal (#14B8A6) — designed for financial / "
    "management-style dashboards."
)


def fit_ols(data: pd.DataFrame, y: str, xs: list, cov_type_choice: str):
    model_df = data[[y] + xs + (["Unit"] if "Unit" in data.columns else [])].dropna()
    if model_df.empty or len(xs) == 0:
        return None, None
    X = sm.add_constant(model_df[xs])
    Y = model_df[y]
    if cov_type_choice == "HC1 (robust)":
        res = sm.OLS(Y, X).fit(cov_type="HC1")
    elif cov_type_choice == "cluster by Unit/Company" and "Unit" in model_df.columns:
        res = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": model_df["Unit"]})
    else:
        res = sm.OLS(Y, X).fit()
    return res, model_df


# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3 = st.tabs(
    ["📊  Dashboard & Regression Results", "🔍  Exploratory Visualizations", "📚  Methodology & References"]
)

# ----------------------------------------------------------------------------
# TAB 1 — DASHBOARD & REGRESSION RESULTS
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Key indicators")

    n_companies = df["Company"].nunique() if "Company" in df.columns else df.shape[0]
    n_obs = df.shape[0]
    avg_roa = df["ROA"].mean() if "ROA" in df.columns else np.nan
    avg_roe = df["ROE"].mean() if "ROE" in df.columns else np.nan
    ai_adoption_rate = (df["AI"] > 0).mean() * 100 if "AI" in df.columns else np.nan

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Companies", f"{n_companies}")
    k2.metric("Firm-year observations", f"{n_obs}")
    k3.metric("Average ROA", f"{avg_roa:.2f}%" if pd.notna(avg_roa) else "—")
    k4.metric("Average ROE", f"{avg_roe:.2f}%" if pd.notna(avg_roe) else "—")
    k5.metric("AI adoption rate", f"{ai_adoption_rate:.0f}%" if pd.notna(ai_adoption_rate) else "—")

    st.markdown("")
    st.subheader("Regression models (OLS)")
    st.markdown(
        f"By default, each model regresses a financial outcome on **AI adoption** and "
        f"**firm maturity**, controlling for **Age** and **Size**:\n\n"
        f"`Outcome = β₀ + β₁·AI + β₂·Maturity + β₃·Age + β₄·Size + ε`"
    )

    reg_col1, reg_col2, reg_col3 = st.columns([1, 1.4, 1])
    with reg_col1:
        outcome_var = st.selectbox(
            "Dependent variable (Y)",
            options=[c for c in ["ROA", "ROE", "Risk"] if c in numeric_cols] or numeric_cols,
        )
    with reg_col2:
        default_predictors = [c for c in ["AI", "Maturity", "Age", "Size"] if c in numeric_cols and c != outcome_var]
        predictors = st.multiselect(
            "Independent variables (X)",
            options=[c for c in numeric_cols if c != outcome_var],
            default=default_predictors,
        )
    with reg_col3:
        cov_type = st.selectbox(
            "Standard errors",
            options=["nonrobust", "HC1 (robust)", "cluster by Unit/Company"],
            index=1,
        )

    if predictors:
        result, used_df = fit_ols(df, outcome_var, predictors, cov_type)
        if result is not None:
            st.markdown(f"**Model:** `{outcome_var} ~ {' + '.join(predictors)}`")
            coef_table = pd.DataFrame(
                {
                    "coef": result.params,
                    "std err": result.bse,
                    "t / z": result.tvalues,
                    "p-value": result.pvalues,
                    "CI lower (95%)": result.conf_int()[0],
                    "CI upper (95%)": result.conf_int()[1],
                }
            )

            def _highlight_sig(row):
                return ["background-color: rgba(20,184,166,0.12)" if row["p-value"] < 0.05 else "" for _ in row]

            st.dataframe(coef_table.style.apply(_highlight_sig, axis=1).format("{:.4f}"), use_container_width=True)
            st.caption(
                f"N = {int(result.nobs)}  |  R² = {result.rsquared:.3f}  |  "
                f"Adj. R² = {result.rsquared_adj:.3f}  |  highlighted rows = p < 0.05"
            )
            with st.expander("Show full statsmodels summary"):
                st.text(result.summary().as_text())
        else:
            st.warning("Not enough non-missing observations to fit this model.")
    else:
        st.info("Select at least one independent variable.")

    st.markdown("---")
    st.subheader("AI & Maturity effects across ROA, ROE and Risk")
    base_predictors = [c for c in ["AI", "Maturity", "Age", "Size"] if c in numeric_cols]
    outcomes = [c for c in ["ROA", "ROE", "Risk"] if c in numeric_cols]

    summary_rows = []
    sub_tabs = st.tabs(outcomes) if outcomes else []
    for sub_tab, out_var in zip(sub_tabs, outcomes):
        with sub_tab:
            xs = [p for p in base_predictors if p != out_var]
            res, _ = fit_ols(df, out_var, xs, cov_type)
            if res is not None:
                st.text(res.summary().as_text())
                for var in ["AI", "Maturity"]:
                    if var in res.params.index:
                        summary_rows.append(
                            {
                                "Outcome": out_var,
                                "Predictor": var,
                                "Coefficient": res.params[var],
                                "p-value": res.pvalues[var],
                                "Significant (p<0.05)": "Yes" if res.pvalues[var] < 0.05 else "No",
                            }
                        )
            else:
                st.warning(f"Could not fit model for {out_var}.")

    if summary_rows:
        st.markdown("**Quick comparison of AI / Maturity effects:**")
        st.dataframe(
            pd.DataFrame(summary_rows).style.format({"Coefficient": "{:.3f}", "p-value": "{:.3f}"}),
            use_container_width=True,
        )

# ----------------------------------------------------------------------------
# TAB 2 — EXPLORATORY VISUALIZATIONS
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Descriptive statistics")
    st.dataframe(df[numeric_cols].describe().T.style.format("{:.3f}"), use_container_width=True)

    st.markdown("---")
    st.subheader("Correlation matrix")
    default_corr_vars = [c for c in ["Age", "Size", "Maturity", "AI", "ROA", "ROE", "Risk"] if c in numeric_cols]
    corr_vars = st.multiselect(
        "Variables to include",
        options=numeric_cols,
        default=default_corr_vars if default_corr_vars else numeric_cols,
        key="corr_vars",
    )
    if len(corr_vars) >= 2:
        corr_matrix = df[corr_vars].corr(method="pearson")
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale=DIVERGING_SCALE,
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Pearson Correlation Matrix",
        )
        fig_corr.update_layout(height=500, font=dict(family="Inter"), title_font_color=NAVY)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Select at least two variables to draw the heatmap.")

    st.markdown("---")
    st.subheader("Interactive scatter plot")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        x_var = st.selectbox("X axis", options=numeric_cols, index=numeric_cols.index("AI") if "AI" in numeric_cols else 0)
    with sc2:
        y_var = st.selectbox("Y axis", options=numeric_cols, index=numeric_cols.index("ROA") if "ROA" in numeric_cols else 1)
    with sc3:
        color_var = st.selectbox(
            "Color by (optional)",
            options=["(none)"] + [c for c in df.columns if c not in numeric_cols] + numeric_cols,
            index=0,
        )
    with sc4:
        trendline = st.checkbox("Add OLS trendline", value=True)

    plot_df = df.dropna(subset=[x_var, y_var])
    color_arg = None if color_var == "(none)" else color_var

    fig_scatter = px.scatter(
        plot_df,
        x=x_var,
        y=y_var,
        color=color_arg,
        hover_name="Company" if "Company" in plot_df.columns else None,
        hover_data=["Year"] if "Year" in plot_df.columns else None,
        trendline="ols" if trendline else None,
        title=f"{y_var} vs. {x_var}",
    )
    fig_scatter.update_layout(height=550, font=dict(family="Inter"), title_font_color=NAVY)
    fig_scatter.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    st.plotly_chart(fig_scatter, use_container_width=True)

    if "AI" in df.columns and "Year" in df.columns:
        st.markdown("---")
        st.subheader("AI adoption trend by year")
        trend = df.groupby("Year")["AI"].mean().reset_index()
        fig_trend = px.bar(
            trend, x="Year", y="AI", title="Average AI adoption score by year",
            color_discrete_sequence=[TEAL],
        )
        fig_trend.update_layout(height=380, font=dict(family="Inter"), title_font_color=NAVY)
        st.plotly_chart(fig_trend, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3 — METHODOLOGY & REFERENCES
# ----------------------------------------------------------------------------
with tab3:
    if using_sample:
        st.warning(
            "The methodology below describes the **real research design** of the underlying "
            "thesis. This public demo, however, currently displays **synthetic sample data** "
            "(see the notice at the top of the page) — the licensed ORBIS dataset itself is "
            "not published here.",
            icon="⚠️",
        )

    st.subheader("Methodology")
    st.markdown(
        f"""
        <div class="method-card">
        <p>The dataset covers <b>publicly listed service firms in Finland</b> observed over
        the <b>2021–2023</b> period, structured as an unbalanced firm-year <b>panel</b>
        (multiple observations per company across three years).</p>

        <p>Because the data combine <i>within-firm</i> variation over time with
        <i>between-firm</i> differences, ordinary cross-sectional OLS would ignore
        unobserved, firm-specific heterogeneity (e.g. management quality, culture, or
        legacy systems) that is likely correlated with both AI adoption and financial
        outcomes. To address this, the underlying study estimates <b>panel-data models
        — Fixed Effects (FE) and Random Effects (RE)</b> — which explicitly account for
        this unobserved heterogeneity:</p>

        <ul>
            <li><b>Fixed Effects (FE)</b> removes all time-invariant firm characteristics by
            de-meaning the data, isolating variation <i>within</i> each firm over time.</li>
            <li><b>Random Effects (RE)</b> treats firm-specific effects as random draws
            uncorrelated with the regressors, which allows time-invariant variables
            (such as firm age) to remain in the model and improves estimation efficiency
            when that assumption holds.</li>
        </ul>

        <p>Model choice between FE and RE is typically guided by a
        <b>Hausman specification test</b>; in this study, RE was preferred because the
        research questions require estimating the effect of time-invariant and slow-moving
        variables (firm age, firm size) that FE would otherwise absorb.</p>

        <p>This dashboard reproduces the same logic with transparent, interactive
        <b>OLS regressions</b> (with optional robust or cluster-robust standard errors
        by firm) so that results can be explored dynamically; for the full FE/RE panel
        specification and diagnostics, see the underlying thesis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.subheader("Key references")

    references = [
        (
            "DeAngelo, H., DeAngelo, L., & Stulz, R. M. (2006).",
            "Dividend policy and the earned/contributed capital mix: A test of the life-cycle theory.",
            "Journal of Financial Economics, 81(2), 227–254.",
        ),
        (
            "Dwivedi, Y. K., Hughes, L., et al. (2021).",
            "Artificial Intelligence (AI): Multidisciplinary perspectives on emerging challenges, "
            "opportunities, and agenda for research, practice and policy.",
            "International Journal of Information Management, 57, 102225.",
        ),
        (
            "Wooldridge, J. M. (2010).",
            "Econometric Analysis of Cross Section and Panel Data.",
            "MIT Press.",
        ),
        (
            "Chen, J. (2020).",
            "Artificial Intelligence in Corporate Strategy and Financial Performance.",
            "Journal of Business Research, 108, 75–88.",
        ),
    ]

    for authors, title, source in references:
        st.markdown(
            f"""
            <div class="ref-card">
                <b>{authors}</b> {title} <i>{source}</i>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Reference list limited to core sources on AI adoption, firm maturity/life-cycle "
        "theory, and panel-data methodology cited in the underlying research."
    )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption(
    "Built with Streamlit, Plotly and statsmodels. This dashboard is a research tool — "
    "interpret regression coefficients in light of sample size, panel structure, and "
    "possible endogeneity."
)
