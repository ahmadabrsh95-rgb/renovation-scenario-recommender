import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================
FILE_PATH = "trade_off_summary.xlsx"
SHEET_NAME = "TradeOff_Summary"
SCENARIOS = ["S01", "S02", "S03"]

INDICATOR_ALTERNATIVES = {
    "energy": ["Total energy"],
    "gwp": ["Total GWP"],
    "overheating": [
        "Mean Overheating [% of Apr-Sep hours]",
        "Mean Overheating [% of Apr–Sep hours]",
        "Mean overheating [% of Apr-Sep hours]",
        "Mean overheating [% of Apr–Sep hours]"
    ],
    "circularity": ["Circularity score"]
}

DIRECTION = {
    "energy": "lower_better",
    "gwp": "lower_better",
    "overheating": "lower_better",
    "circularity": "higher_better"
}


# =========================================================
# HELPERS
# =========================================================
def normalize_text(text):
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()


@st.cache_data
def load_data(file_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    if "Indicator" not in df.columns:
        raise ValueError("The Excel sheet must contain a column named 'Indicator'.")
    df["Indicator"] = df["Indicator"].astype(str).str.strip()
    df["Indicator_clean"] = df["Indicator"].apply(normalize_text)
    return df


def find_indicator_row(df: pd.DataFrame, possible_names):
    candidates = [normalize_text(x) for x in possible_names]

    for cand in candidates:
        exact = df[df["Indicator_clean"] == cand]
        if not exact.empty:
            return exact.iloc[0]

    for cand in candidates:
        contains = df[df["Indicator_clean"].str.contains(re.escape(cand), na=False)]
        if not contains.empty:
            return contains.iloc[0]

    if any("overheating" in cand for cand in candidates):
        mask = (
            df["Indicator_clean"].str.contains("overheating", na=False)
            & df["Indicator_clean"].str.contains("apr", na=False)
            & df["Indicator_clean"].str.contains("sep", na=False)
        )
        row = df.loc[mask]
        if not row.empty:
            return row.iloc[0]

    raise ValueError(f"Could not find indicator row for any of: {possible_names}")


def min_max_score(values, direction):
    values = np.array(values, dtype=float)
    vmin = np.min(values)
    vmax = np.max(values)

    if np.isclose(vmin, vmax):
        return np.ones_like(values, dtype=float)

    if direction == "lower_better":
        return (vmax - values) / (vmax - vmin)
    elif direction == "higher_better":
        return (values - vmin) / (vmax - vmin)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def build_result_table(df: pd.DataFrame, building: str, weights: dict) -> pd.DataFrame:
    rows = {
        key: find_indicator_row(df, INDICATOR_ALTERNATIVES[key])
        for key in INDICATOR_ALTERNATIVES
    }

    raw_data = {}
    for key, row in rows.items():
        vals = []
        for scenario in SCENARIOS:
            col = f"{building}-{scenario}"
            if col not in df.columns:
                raise ValueError(f"Column not found in Excel: {col}")
            vals.append(float(row[col]))
        raw_data[key] = np.array(vals, dtype=float)

    scores = {}
    for key in raw_data:
        scores[key] = min_max_score(raw_data[key], DIRECTION[key])

    total_scores = np.zeros(len(SCENARIOS), dtype=float)
    for key in scores:
        total_scores += weights[key] * scores[key]

    result_df = pd.DataFrame({
        "Scenario": SCENARIOS,
        "Energy use": raw_data["energy"],
        "Total GWP": raw_data["gwp"],
        "Overheating": raw_data["overheating"],
        "Circularity": raw_data["circularity"],
        "Energy score": scores["energy"],
        "GWP score": scores["gwp"],
        "Overheating score": scores["overheating"],
        "Circularity score": scores["circularity"],
        "Weighted total score": total_scores
    })

    result_df = result_df.sort_values("Weighted total score", ascending=False).reset_index(drop=True)
    return result_df


def make_score_plot(result_df: pd.DataFrame, building_label: str):
    plot_df = result_df.sort_values("Scenario").copy()

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(plot_df["Scenario"], plot_df["Weighted total score"])
    ax.set_ylabel("Weighted total score")
    ax.set_title(f"Scenario ranking — {building_label}")
    ax.set_ylim(0, max(1.0, plot_df["Weighted total score"].max() * 1.15))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    for i, val in enumerate(plot_df["Weighted total score"]):
        ax.text(i, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


# =========================================================
# APP
# =========================================================
st.set_page_config(page_title="Renovation Scenario Recommender", layout="wide")

st.title("Renovation Scenario Recommender")
st.write(
    "A simple weighted decision-support tool for comparing renovation scenarios "
    "based on energy use, total GWP, overheating, and circularity."
)

try:
    df = load_data(FILE_PATH, SHEET_NAME)
except Exception as e:
    st.error(f"Could not load Excel file: {e}")
    st.stop()

# Sidebar
st.sidebar.header("Inputs")

building_label = st.sidebar.selectbox(
    "Building type",
    ["High-rise building (HRB)", "Low-rise building (LRB)"]
)
building_code = "HRB" if "HRB" in building_label else "LRB"

st.sidebar.markdown("### Weighting factors")
w_energy = st.sidebar.slider("Energy use", 0, 100, 30, 1)
w_gwp = st.sidebar.slider("Total GWP", 0, 100, 30, 1)
w_over = st.sidebar.slider("Overheating", 0, 100, 20, 1)
w_circ = st.sidebar.slider("Circularity", 0, 100, 20, 1)

raw_weights = {
    "energy": float(w_energy),
    "gwp": float(w_gwp),
    "overheating": float(w_over),
    "circularity": float(w_circ)
}

weight_sum = sum(raw_weights.values())

if weight_sum == 0:
    st.warning("Please set at least one weighting factor above zero.")
    st.stop()

weights = {k: v / weight_sum for k, v in raw_weights.items()}

result_df = build_result_table(df, building_code, weights)
recommended = result_df.iloc[0]["Scenario"]

# Top summary
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Recommendation")
    st.success(f"Recommended renovation scenario: **{recommended}**")

    st.write("**Normalized weights used**")
    st.write(f"- Energy use: {weights['energy']:.3f}")
    st.write(f"- Total GWP: {weights['gwp']:.3f}")
    st.write(f"- Overheating: {weights['overheating']:.3f}")
    st.write(f"- Circularity: {weights['circularity']:.3f}")

with col2:
    st.subheader("Interpretation")
    st.info(
        "The recommended scenario is the best option **given the weighting factors selected**. "
        "This is not an objective best scenario, but the best scenario under the chosen priorities."
    )

# Table
st.subheader("Scenario ranking")
st.dataframe(
    result_df.style.format({
        "Energy use": "{:.2f}",
        "Total GWP": "{:.2f}",
        "Overheating": "{:.2f}",
        "Circularity": "{:.2f}",
        "Energy score": "{:.3f}",
        "GWP score": "{:.3f}",
        "Overheating score": "{:.3f}",
        "Circularity score": "{:.3f}",
        "Weighted total score": "{:.3f}"
    }),
    use_container_width=True
)

# Plot
st.subheader("Weighted total score")
fig = make_score_plot(result_df, building_label)
st.pyplot(fig)

# Optional raw scenario comparison
with st.expander("Show raw indicator values by scenario"):
    raw_cols = ["Scenario", "Energy use", "Total GWP", "Overheating", "Circularity"]
    st.dataframe(
        result_df[raw_cols].sort_values("Scenario").style.format({
            "Energy use": "{:.2f}",
            "Total GWP": "{:.2f}",
            "Overheating": "{:.2f}",
            "Circularity": "{:.2f}",
        }),
        use_container_width=True
    )