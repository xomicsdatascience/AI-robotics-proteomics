from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Protein Clustering", layout="wide")
st.info(
    "This network shows proteins with similar response patterns across drugs. "
    "Edges represent Pearson correlation of protein-level changes across the drug panel."
)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = BASE_DIR / "data" / "protein_clustering_results" / "protein_clustering_matrix.csv"
DEFAULT_MEM = BASE_DIR / "data" / "protein_clustering_results" / "protein_clustering_membership.csv"
DEFAULT_SUM = BASE_DIR / "data" / "protein_clustering_results" / "protein_clustering_summary.csv"


@st.cache_data
def load_df(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def clean_display_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "protein" not in out.columns:
        return out

    out["protein"] = out["protein"].astype(str)

    if "gene_symbol" not in out.columns:
        out["gene_symbol"] = ""

    if "display_label" not in out.columns:
        out["display_label"] = out["gene_symbol"]

    out["gene_symbol"] = out["gene_symbol"].fillna("").astype(str)
    out["display_label"] = out["display_label"].fillna("").astype(str)
    out.loc[out["display_label"].str.strip() == "", "display_label"] = out["gene_symbol"]
    out.loc[out["display_label"].str.strip() == "", "display_label"] = out["protein"]
    return out


def get_label_map(mem_df: pd.DataFrame) -> dict:
    if mem_df.empty or "protein" not in mem_df.columns:
        return {}
    sub = clean_display_label(mem_df)
    return dict(zip(sub["protein"].astype(str), sub["display_label"].astype(str)))


def get_cluster_map(mem_df: pd.DataFrame) -> dict:
    if mem_df.empty or "protein" not in mem_df.columns or "cluster" not in mem_df.columns:
        return {}
    sub = mem_df.copy()
    sub["protein"] = sub["protein"].astype(str)
    sub["cluster"] = sub["cluster"].astype(str)
    return dict(zip(sub["protein"], sub["cluster"]))


st.title("Protein Clustering")
st.write("Explore global protein clustering patterns and local protein-to-neighbors correlation networks.")

with st.sidebar:
    st.header("Data input")
    matrix_path = st.text_input("Matrix file", str(DEFAULT_MATRIX))
    mem_path = st.text_input("Membership file", str(DEFAULT_MEM))
    sum_path = st.text_input("Summary file", str(DEFAULT_SUM))

matrix_df = load_df(matrix_path)
mem_df = load_df(mem_path)
sum_df = load_df(sum_path)

if matrix_df.empty or mem_df.empty:
    st.error("Missing required files. Please check the matrix and membership file paths.")
    st.stop()

matrix_df = clean_display_label(matrix_df)
mem_df = clean_display_label(mem_df)

meta_cols = ["protein", "gene_symbol", "display_label", "cluster", "cluster_num"]
drug_cols = [c for c in matrix_df.columns if c not in meta_cols]

if len(drug_cols) == 0:
    st.error("No drug columns were detected in the matrix file.")
    st.stop()

matrix_df[drug_cols] = matrix_df[drug_cols].apply(pd.to_numeric, errors="coerce")

label_map = get_label_map(mem_df)
cluster_map = get_cluster_map(mem_df)

tab1, tab2 = st.tabs(["Overall clustering", "Protein network"])

with tab1:
    st.info(
        "Protein clustering was performed on a filtered subset of proteins to improve robustness and interpretability. "
        "In the current clustering results, proteins were retained based on data completeness, variability across drugs, "
        "and minimum effect size thresholds before clustering."
    )

    with st.expander("Show clustering filter details"):
        st.markdown(
            """
Typical filtering criteria used before clustering included:

- minimum non-missing fraction across drugs
- minimum standard deviation across drugs
- minimum maximum absolute change across drugs

These filters reduce noise from proteins with sparse measurements or minimal variation.
The network tab can be configured separately if you want broader protein coverage.
"""
        )

    st.subheader("Cluster sizes")
    if not sum_df.empty and {"cluster", "n_proteins"}.issubset(sum_df.columns):
        fig_bar = go.Figure(go.Bar(
            x=sum_df["cluster"].astype(str),
            y=pd.to_numeric(sum_df["n_proteins"], errors="coerce")
        ))
        fig_bar.update_layout(
            xaxis_title="Cluster",
            yaxis_title="Number of proteins",
            template="simple_white",
            height=420
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Cluster summary file is missing required columns: cluster, n_proteins.")

    st.subheader("Cluster heatmap")

    if "cluster" not in matrix_df.columns:
        st.warning("The matrix file does not contain a cluster column.")
    else:
        clusters = sorted(matrix_df["cluster"].dropna().astype(str).unique().tolist())
        if len(clusters) == 0:
            st.warning("No clusters detected in the matrix file.")
        else:
            selected_cluster = st.selectbox("Select cluster", clusters)
            show_n = st.slider("Number of proteins to display", 10, 100, 40, 5)

            sub = matrix_df.loc[matrix_df["cluster"].astype(str) == selected_cluster].copy()
            sub["abs_val"] = sub[drug_cols].abs().max(axis=1)
            sub = sub.sort_values("abs_val", ascending=False).head(show_n)

            if sub.empty:
                st.warning("No proteins found for the selected cluster.")
            else:
                heat = sub.set_index("display_label")[drug_cols]
                fig_heat = go.Figure(data=go.Heatmap(
                    z=heat.values,
                    x=heat.columns,
                    y=heat.index,
                    colorscale="RdBu",
                    zmid=0,
                    colorbar=dict(title="Value")
                ))
                fig_heat.update_layout(
                    template="simple_white",
                    height=max(500, int(len(heat) * 12)),
                    xaxis_title="Drugs",
                    yaxis_title="Proteins"
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                st.subheader("Cluster members")
                show_cols = [c for c in ["protein", "gene_symbol", "display_label", "cluster"] if c in sub.columns]
                st.dataframe(sub[show_cols], use_container_width=True)

with tab2:
    st.subheader("Protein-to-neighbors star network")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        query = st.text_input("Protein or gene symbol", "PLK1")
    with c2:
        top_n = st.slider("Top neighbors", 5, 30, 10)
    with c3:
        corr_method = st.selectbox("Correlation", ["pearson", "spearman"])

    query = str(query).strip()
    if query == "":
        st.warning("Please enter a protein or gene symbol.")
        st.stop()

    mem_lookup = mem_df.copy()
    mem_lookup["protein"] = mem_lookup["protein"].astype(str)
    mem_lookup["gene_symbol"] = mem_lookup["gene_symbol"].fillna("").astype(str)
    mem_lookup["display_label"] = mem_lookup["display_label"].fillna("").astype(str)

    sub_gene = mem_lookup.loc[mem_lookup["gene_symbol"].str.upper() == query.upper()].copy()
    if len(sub_gene) > 0:
        protein_id = str(sub_gene.iloc[0]["protein"])
        query_label = str(sub_gene.iloc[0]["display_label"]) if str(sub_gene.iloc[0]["display_label"]).strip() != "" else protein_id
    else:
        protein_id = query
        query_label = label_map.get(protein_id, protein_id)

    mat = matrix_df.copy()
    mat["protein"] = mat["protein"].astype(str)
    mat_num = mat.set_index("protein")[drug_cols]

    if protein_id not in mat_num.index:
        st.warning("Protein not found in the clustering matrix.")
        st.stop()

    corr = mat_num.T.corr(method=corr_method)
    s = corr.loc[protein_id].drop(labels=[protein_id]).sort_values(ascending=False)

    if len(s) == 0:
        st.warning("No valid neighbors were found for this protein.")
        st.stop()

    s = s.head(top_n)
    neighbors = s.index.astype(str).tolist()

    labels = []
    neighbor_clusters = []
    neighbor_hover = []

    for n in neighbors:
        lab = label_map.get(n, n)
        cl = cluster_map.get(n, "NA")
        labels.append(str(lab))
        neighbor_clusters.append(str(cl))
        neighbor_hover.append(
            f"Label: {lab}<br>"
            f"Protein: {n}<br>"
            f"Correlation: {s.loc[n]:.3f}<br>"
            f"Cluster: {cl}"
        )

    theta = np.linspace(0, 2 * np.pi, len(neighbors), endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)

    unique_clusters = sorted(set(neighbor_clusters))
    palette = [
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
        "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#3C91E6"
    ]
    cluster_color_map = {cl: palette[i % len(palette)] for i, cl in enumerate(unique_clusters)}
    node_colors = [cluster_color_map[cl] for cl in neighbor_clusters]

    fig_net = go.Figure()

    for i, n in enumerate(neighbors):
        width_val = 1.5 + 4.5 * float(abs(s.loc[n]))
        fig_net.add_trace(go.Scatter(
            x=[0, x[i]],
            y=[0, y[i]],
            mode="lines",
            line=dict(width=width_val, color="gray"),
            hoverinfo="skip",
            showlegend=False
        ))

    center_cluster = cluster_map.get(protein_id, "NA")
    fig_net.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode="markers+text",
        text=[query_label],
        textposition="top center",
        marker=dict(size=24, color="#D62728"),
        hovertext=[f"Query: {query_label}<br>Protein: {protein_id}<br>Cluster: {center_cluster}"],
        hoverinfo="text",
        showlegend=False
    ))

    fig_net.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(size=18),
        marker=dict(size=16, color=node_colors, line=dict(width=1, color="black")),
        hovertext=neighbor_hover,
        hoverinfo="text",
        showlegend=False
    ))

    fig_net.update_layout(
        template="simple_white",
        height=700,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig_net, use_container_width=True)

    st.subheader("Neighbor table")
    neighbor_df = pd.DataFrame({
        "protein": neighbors,
        "display_label": labels,
        "cluster": neighbor_clusters,
        "correlation": [float(s.loc[n]) for n in neighbors]
    }).sort_values("correlation", ascending=False)

    st.dataframe(neighbor_df, use_container_width=True)
