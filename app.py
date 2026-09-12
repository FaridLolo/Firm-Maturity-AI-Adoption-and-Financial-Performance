"""
Streamlit App: Firm Maturity, AI Adoption & Financial Performance
-------------------------------------------------------------------
Reads a panel-data CSV (Finnish service firms), shows descriptive
statistics and a correlation heatmap, runs OLS regression models
(ROA, ROE, Risk ~ AI + Maturity + Age + Size), and lets the user
build interactive scatter plots.

Run with:
    streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Adoption & Firm Performance",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Firm Maturity, AI Adoption & Financial Performance")
st.caption(
    "Interactive dashboard for panel data of publicly listed service firms "
    "(ROA / ROE / Risk vs. AI adoption & firm maturity)."
)

# ----------------------------------------------------------------------
# Data loading helpers
# ----------------------------------------------------------------------
DEFAULT_PATH = "data.csv"

NUMERIC_COLS_HINT = ["Age", "Size", "AI", "ROA", "ROE", "RE", "TA", "Maturity", "Risk"]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names."""
    df.columns = [c.strip() for c in df.columns]
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-looking columns (possibly with thousands separators
    or quoted strings like '70,470,391') into floats."""
    for col in df.columns:
        if col in ("Company", "Unit", "Year") and col != "Year":
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
            # only replace if conversion produced mostly valid numbers
            if converted.notna().mean() > 0.5:
                df[col] = converted
    return df


@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = clean_columns(df)
    df = coerce_numeric(df)
    return df


# ----------------------------------------------------------------------
# Sidebar: data source
# ----------------------------------------------------------------------
st.sidebar.header("1️⃣ Data")
uploaded = st.sidebar.file_uploader("Upload data.csv", type=["csv"])

df = None
if uploaded is not None:
    df = load_data(uploaded)
    st.sidebar.success("File uploaded successfully.")
else:
    try:
        df = load_data(DEFAULT_PATH)
        st.sidebar.info(f"Loaded default file: `{DEFAULT_PATH}`")
    except FileNotFoundError:
        st.sidebar.warning(
            "No `data.csv` found next to the app. Please upload a CSV file."
        )

if df is None:
    st.stop()

# Identify numeric columns actually present
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

st.sidebar.markdown("---")
st.sidebar.write(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")
with st.sidebar.expander("Preview raw data"):
    st.dataframe(df.head(10))

# ----------------------------------------------------------------------
# Section: Descriptive statistics & correlation heatmap
# ----------------------------------------------------------------------
st.header("2️⃣ Descriptive Statistics & Correlation Matrix")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Descriptive statistics")
    st.dataframe(df[numeric_cols].describe().T.style.format("{:.3f}"))

with col2:
    st.subheader("Correlation matrix (heatmap)")
    default_corr_vars = [
        c for c in ["Age", "Size", "Maturity", "AI", "ROA", "ROE", "Risk"] if c in numeric_cols
    ]
    corr_vars = st.multiselect(
        "Variables to include in the correlation matrix",
        options=numeric_cols,
        default=default_corr_vars if default_corr_vars else numeric_cols,
    )
    if len(corr_vars) >= 2:
        corr_matrix = df[corr_vars].corr(method="pearson")
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Pearson Correlation Matrix",
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Select at least two variables to draw the heatmap.")

# ----------------------------------------------------------------------
# Section: Regression models
# ----------------------------------------------------------------------
st.header("3️⃣ Regression Models (OLS)")

st.markdown(
    "By default, each model regresses a financial outcome on **AI adoption** "
    "and **firm maturity**, with **Age** and **Size** as control variables:\n\n"
    "`Outcome = β0 + β1·AI + β2·Maturity + β3·Age + β4·Size + ε`\n\n"
    "You can change the outcome, predictors, and standard-error type below."
)

reg_col1, reg_col2, reg_col3 = st.columns([1, 1.4, 1])

with reg_col1:
    outcome_var = st.selectbox(
        "Dependent variable (Y)",
        options=[c for c in ["ROA", "ROE", "Risk"] if c in numeric_cols]
        or numeric_cols,
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

run_all = st.checkbox(
    "Also show ROA, ROE and Risk models together (using AI + Maturity + Age + Size)",
    value=True,
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
        res = sm.OLS(Y, X).fit(
            cov_type="cluster", cov_kwds={"groups": model_df["Unit"]}
        )
    else:
        res = sm.OLS(Y, X).fit()
    return res, model_df


if predictors:
    result, used_df = fit_ols(df, outcome_var, predictors, cov_type)
    if result is not None:
        st.subheader(f"Model: {outcome_var} ~ {' + '.join(predictors)}")
        st.text(result.summary().as_text())

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
        st.dataframe(coef_table.style.format("{:.4f}"))
        st.caption(
            f"N = {int(result.nobs)}  |  R² = {result.rsquared:.3f}  |  "
            f"Adj. R² = {result.rsquared_adj:.3f}"
        )
    else:
        st.warning("Not enough non-missing observations to fit this model.")
else:
    st.info("Select at least one independent variable.")

if run_all:
    st.markdown("---")
    st.subheader("Summary: AI & Maturity effects across ROA, ROE and Risk")
    base_predictors = [c for c in ["AI", "Maturity", "Age", "Size"] if c in numeric_cols]
    outcomes = [c for c in ["ROA", "ROE", "Risk"] if c in numeric_cols]

    summary_rows = []
    tabs = st.tabs(outcomes) if outcomes else []
    for tab, out_var in zip(tabs, outcomes):
        with tab:
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
        st.dataframe(pd.DataFrame(summary_rows).style.format({"Coefficient": "{:.3f}", "p-value": "{:.3f}"}))

# ----------------------------------------------------------------------
# Section: Interactive scatter plot
# ----------------------------------------------------------------------
st.header("4️⃣ Interactive Scatter Plot")

sc1, sc2, sc3, sc4 = st.columns([1, 1, 1, 1])
with sc1:
    x_var = st.selectbox("X axis", options=numeric_cols, index=numeric_cols.index("AI") if "AI" in numeric_cols else 0)
with sc2:
    y_var = st.selectbox(
        "Y axis",
        options=numeric_cols,
        index=numeric_cols.index("ROA") if "ROA" in numeric_cols else 1,
    )
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
fig_scatter.update_layout(height=550)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
st.caption(
    "Built with Streamlit, Plotly and statsmodels. "
    "This dashboard is a research tool — interpret regression coefficients "
    "in light of sample size, panel structure, and possible endogeneity."
)
