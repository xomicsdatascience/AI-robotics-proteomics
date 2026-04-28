from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="Protein Across Drugs",
    layout="wide"
)

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_TTEST_PREFIX = "df_ttest_all"

# =========================
# Helpers
# =========================
@st.cache_data
def load_parquet_parts(data_dir_str: str, prefix: str):
    """
    Load all parquet parts matching:
      {prefix}_part*.parquet
    and concatenate them.
    """
    data_dir = Path(data_dir_str)
    files = sorted(data_dir.glob(f"{prefix}_part*.parquet"))

    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def prepare_ttest_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    required = ["drug", "protein", "log2FC", "p_value", "p_value_adj"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        return pd.DataFrame()

    if "gene_symbol" not in out.columns:
        out["gene_symbol"] = ""

    for c in ["log2FC", "p_value", "p_value_adj", "t_stat", "mean_drug", "mean_DMSO"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["drug"] = out["drug"].astype(str)
    out["protein"] = out["protein"].astype(str)
    out["gene_symbol"] = out["gene_symbol"].fillna("").astype(str)

    return out


def find_matching_rows(df: pd.DataFrame, query: str, search_mode: str) -> pd.DataFrame:
    if df.empty or not str(query).strip():
        return pd.DataFrame()

    q = query.strip().upper()

    if search_mode == "Gene symbol only":
        return df.loc[df["gene_symbol"].str.upper() == q].copy()

    if search_mode == "Protein ID only":
        return df.loc[df["protein"].str.upper() == q].copy()

    sub_gene = df.loc[df["gene_symbol"].str.upper() == q].copy()
    if not sub_gene.empty:
        return sub_gene

    sub_protein = df.loc[df["protein"].str.upper() == q].copy()
    return sub_protein


def classify_significance(sub: pd.DataFrame, fc_thresh: float, fdr_thresh: float) -> pd.DataFrame:
    if sub.empty:
        return sub

    out = sub.copy()
    out = out.dropna(subset=["log2FC", "p_value_adj"])

    if out.empty:
        return out

    out["sig_group"] = "Not significant"
    out.loc[
        (out["log2FC"] >= fc_thresh) & (out["p_value_adj"] < fdr_thresh),
        "sig_group"
    ] = "Up"
    out.loc[
        (out["log2FC"] <= -fc_thresh) & (out["p_value_adj"] < fdr_thresh),
        "sig_group"
    ] = "Down"

    color_map = {
        "Up": "#E64B35",
        "Down": "#3C91E6",
        "Not significant": "#BDBDBD"
    }
    out["dot_color"] = out["sig_group"].map(color_map)

    return out


def filter_rows(
    sub: pd.DataFrame,
    only_sig: bool,
    min_abs_fc: float,
    max_fdr: float | None
) -> pd.DataFrame:
    if sub.empty:
        return sub

    out = sub.copy()
    out = out.loc[out["log2FC"].notna()].copy()

    if max_fdr is not None:
        out = out.loc[out["p_value_adj"].notna() & (out["p_value_adj"] <= max_fdr)].copy()

    if min_abs_fc > 0:
        out = out.loc[out["log2FC"].abs() >= min_abs_fc].copy()

    if only_sig:
        out = out.loc[out["sig_group"] != "Not significant"].copy()

    return out


def order_rows(sub: pd.DataFrame, order_mode: str) -> pd.DataFrame:
    if sub.empty:
        return sub

    out = sub.copy()

    if order_mode == "Drug name":
        out = out.sort_values("drug", ascending=True)
    elif order_mode == "log2FC descending":
        out = out.sort_values("log2FC", ascending=False)
    elif order_mode == "log2FC ascending":
        out = out.sort_values("log2FC", ascending=True)
    elif order_mode == "Adjusted P ascending":
        out = out.sort_values(["p_value_adj", "log2FC"], ascending=[True, False])
    else:
        out["abs_log2FC"] = out["log2FC"].abs()
        out = out.sort_values("abs_log2FC", ascending=False)

    return out


def apply_top_n(sub: pd.DataFrame, top_n_mode: str, top_n: int) -> pd.DataFrame:
    if sub.empty:
        return sub

    out = sub.copy()

    if top_n_mode == "Show all":
        return out
    if top_n_mode == "Top N by |log2FC|":
        return out.reindex(out["log2FC"].abs().sort_values(ascending=False).index).head(top_n)
    if top_n_mode == "Top N upregulated":
        return out.sort_values("log2FC", ascending=False).head(top_n)
    if top_n_mode == "Top N downregulated":
        return out.sort_values("log2FC", ascending=True).head(top_n)

    return out


def make_dot_plot(
    sub: pd.DataFrame,
    query: str,
    rotate_xticks: int,
    height: int,
    title_font_size: int,
    axis_title_font_size: int,
    axis_tick_font_size: int
):
    if sub.empty:
        return None

    plot_df = sub.copy()

    hover_text = (
        "Drug: " + plot_df["drug"].astype(str) +
        "<br>Gene: " + plot_df["gene_symbol"].astype(str) +
        "<br>Protein: " + plot_df["protein"].astype(str) +
        "<br>log2FC: " + plot_df["log2FC"].round(3).astype(str) +
        "<br>adj.P: " + plot_df["p_value_adj"].apply(lambda x: f"{x:.2e}" if pd.notna(x) else "NA")
    )

    if "p_value" in plot_df.columns:
        hover_text += "<br>P-value: " + plot_df["p_value"].apply(lambda x: f"{x:.2e}" if pd.notna(x) else "NA")
    if "t_stat" in plot_df.columns:
        hover_text += "<br>t-stat: " + plot_df["t_stat"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
    if "mean_drug" in plot_df.columns:
        hover_text += "<br>mean_drug: " + plot_df["mean_drug"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
    if "mean_DMSO" in plot_df.columns:
        hover_text += "<br>mean_DMSO: " + plot_df["mean_DMSO"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")

    fig = go.Figure()

    for group in ["Not significant", "Up", "Down"]:
        group_df = plot_df.loc[plot_df["sig_group"] == group].copy()
        if group_df.empty:
            continue

        fig.add_trace(go.Scatter(
            x=group_df["drug"],
            y=group_df["log2FC"],
            mode="markers",
            name=group,
            text=group_df["gene_symbol"].replace("", np.nan).fillna(group_df["protein"]),
            hovertext=hover_text.loc[group_df.index],
            hoverinfo="text",
            marker=dict(size=9, color=group_df["dot_color"], opacity=0.85)
        ))

    fig.add_hline(y=0, line_dash="dash", line_width=1)

    fig.update_layout(
        title=dict(text=f"Protein across drugs: {query}", font=dict(size=title_font_size)),
        xaxis_title="Drug",
        yaxis_title="log2 Fold Change",
        template="simple_white",
        height=height,
        legend_title="Group",
        xaxis=dict(
            tickangle=rotate_xticks,
            title_font=dict(size=axis_title_font_size),
            tickfont=dict(size=axis_tick_font_size)
        ),
        yaxis=dict(
            title_font=dict(size=axis_title_font_size),
            tickfont=dict(size=axis_tick_font_size)
        )
    )

    return fig


# =========================
# Main
# =========================
st.title("Protein Across Drugs")
st.write("Browse how one protein or gene changes across all drugs.")

with st.sidebar:
    st.header("Protein view settings")
    data_dir_input = st.text_input(
        "Data directory",
        value=str(DATA_DIR)
    )
    ttest_prefix = st.text_input(
        "T-test file prefix",
        value=DEFAULT_TTEST_PREFIX
    )

df_ttest = load_parquet_parts(data_dir_input, ttest_prefix)
df_ttest = prepare_ttest_df(df_ttest)

if df_ttest.empty:
    st.error(
        "Cannot load a valid split t-test table. "
        "Please check the folder path and file names like "
        "'df_ttest_all_part1.parquet', 'df_ttest_all_part2.parquet', etc."
    )
    st.stop()

st.caption(f"Loaded rows: {len(df_ttest):,}")

query_col1, query_col2 = st.columns([2, 1])

with query_col1:
    query = st.text_input("Enter gene symbol or protein ID", value="PLK1")

with query_col2:
    search_mode = st.selectbox(
        "Search mode",
        ["Gene symbol preferred", "Gene symbol only", "Protein ID only"]
    )

sub = find_matching_rows(df_ttest, query, search_mode)

if sub.empty:
    st.warning("No matching protein or gene was found in the loaded df_ttest files.")
    st.stop()

controls1, controls2, controls3, controls4 = st.columns(4)

with controls1:
    fc_thresh = st.slider("Significance |log2FC| threshold", 0.0, 2.0, 0.5, 0.1)

with controls2:
    fdr_thresh = st.select_slider(
        "Significance adjusted P threshold",
        options=[0.1, 0.05, 0.01, 0.001],
        value=0.05
    )

with controls3:
    only_sig = st.checkbox("Show only significant drugs", value=False)

with controls4:
    order_mode = st.selectbox(
        "Order drugs by",
        [
            "Absolute log2FC descending",
            "log2FC descending",
            "log2FC ascending",
            "Adjusted P ascending",
            "Drug name"
        ]
    )

controls5, controls6, controls7, controls8 = st.columns(4)

with controls5:
    min_abs_fc = st.slider("Minimum |log2FC| to display", 0.0, 3.0, 0.0, 0.1)

with controls6:
    apply_fdr_filter = st.checkbox("Apply display FDR filter", value=False)

with controls7:
    top_n_mode = st.selectbox(
        "Subset drugs",
        ["Show all", "Top N by |log2FC|", "Top N upregulated", "Top N downregulated"]
    )

with controls8:
    top_n = st.slider("Top N", 5, 100, 30, 1)

max_fdr_display = fdr_thresh if apply_fdr_filter else None

sub = classify_significance(sub, fc_thresh=fc_thresh, fdr_thresh=fdr_thresh)
sub = filter_rows(sub, only_sig=only_sig, min_abs_fc=min_abs_fc, max_fdr=max_fdr_display)
sub = order_rows(sub, order_mode=order_mode)
sub = apply_top_n(sub, top_n_mode=top_n_mode, top_n=top_n)

if sub.empty:
    st.warning("No rows remain after applying the current filters.")
    st.stop()

style1, style2, style3, style4 = st.columns(4)

with style1:
    rotate_xticks = st.slider("X-axis label angle", -90, 90, -60, 5)

with style2:
    fig_height = st.slider("Figure height", 400, 1200, 750, 50)

with style3:
    axis_title_font_size = st.slider("Axis title font size", 10, 36, 20, 1)

with style4:
    axis_tick_font_size = st.slider("Axis tick font size", 8, 28, 12, 1)

title_font_size = 24

fig = make_dot_plot(
    sub=sub,
    query=query,
    rotate_xticks=rotate_xticks,
    height=fig_height,
    title_font_size=title_font_size,
    axis_title_font_size=axis_title_font_size,
    axis_tick_font_size=axis_tick_font_size
)

if fig is not None:
    st.plotly_chart(fig, use_container_width=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Drugs shown", int(len(sub)))
m2.metric("Upregulated", int((sub["sig_group"] == "Up").sum()))
m3.metric("Downregulated", int((sub["sig_group"] == "Down").sum()))
m4.metric("Not significant", int((sub["sig_group"] == "Not significant").sum()))

st.subheader("Result table")

show_cols = [c for c in [
    "drug", "gene_symbol", "protein", "log2FC", "p_value_adj", "p_value",
    "t_stat", "mean_drug", "mean_DMSO", "sig_group"
] if c in sub.columns]

st.dataframe(sub[show_cols], use_container_width=True)

csv_data = sub[show_cols].to_csv(index=False).encode("utf-8")
safe_query = "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in str(query))
st.download_button(
    "Download filtered table CSV",
    data=csv_data,
    file_name=f"{safe_query}_across_drugs.csv",
    mime="text/csv"
)