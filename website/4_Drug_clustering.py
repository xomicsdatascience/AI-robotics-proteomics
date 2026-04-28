
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Drug Clustering", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = BASE_DIR / "data" / "drug_clustering_results" / "drug_clustering_go_fcm_matrix.csv"
DEFAULT_MEM = BASE_DIR / "data" / "drug_clustering_results" / "drug_clustering_go_fcm_membership.csv"
DEFAULT_SUM = BASE_DIR / "data" / "drug_clustering_results" / "drug_clustering_go_fcm_summary.csv"


@st.cache_data
def load_df(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def ensure_string_col(df: pd.DataFrame, col: str, default="") -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        out[col] = default
    out[col] = out[col].fillna(default).astype(str)
    return out


def build_label_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "drug" not in out.columns:
        return out
    out["drug"] = out["drug"].astype(str)
    if "display_label" not in out.columns:
        out["display_label"] = out["drug"]
    out["display_label"] = out["display_label"].fillna("").astype(str)
    out.loc[out["display_label"].str.strip() == "", "display_label"] = out["drug"]
    return out


def get_map(df: pd.DataFrame, key_col: str, val_col: str) -> dict:
    if df.empty or key_col not in df.columns or val_col not in df.columns:
        return {}
    sub = df[[key_col, val_col]].copy()
    sub[key_col] = sub[key_col].astype(str)
    sub[val_col] = sub[val_col].fillna("").astype(str)
    return dict(zip(sub[key_col], sub[val_col]))


st.title("Drug Clustering")
st.write("Explore overall drug clustering results and local drug-to-neighbors correlation networks.")

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

matrix_df = ensure_string_col(matrix_df, "drug")
mem_df = ensure_string_col(mem_df, "drug")
matrix_df = build_label_col(matrix_df)
mem_df = build_label_col(mem_df)

for c in ["cluster", "target", "pathway", "research_area"]:
    matrix_df = ensure_string_col(matrix_df, c)
    mem_df = ensure_string_col(mem_df, c)

meta_cols = [
    "drug", "display_label", "cluster", "cluster_num",
    "target", "pathway", "research_area", "membership_score"
]
value_cols = [c for c in matrix_df.columns if c not in meta_cols]

if len(value_cols) == 0:
    st.error("No pathway columns were detected in the matrix file.")
    st.stop()

matrix_df[value_cols] = matrix_df[value_cols].apply(pd.to_numeric, errors="coerce")

label_map = get_map(mem_df, "drug", "display_label")
cluster_map = get_map(mem_df, "drug", "cluster")
target_map = get_map(mem_df, "drug", "target")
pathway_map = get_map(mem_df, "drug", "pathway")
research_map = get_map(mem_df, "drug", "research_area")

membership_score_map = {}
if "membership_score" in mem_df.columns:
    tmp = mem_df[["drug", "membership_score"]].copy()
    tmp["drug"] = tmp["drug"].astype(str)
    tmp["membership_score"] = pd.to_numeric(tmp["membership_score"], errors="coerce")
    membership_score_map = dict(zip(tmp["drug"], tmp["membership_score"]))

tab1, tab2 = st.tabs(["Overall clustering", "Drug network"])

with tab1:
    st.info(
        "Drug clustering on this page is based on pathway-level signatures. "
        "In the current implementation, clusters are derived from pathway enrichment profiles "
        "and each drug is assigned a dominant cluster based on the highest fuzzy membership."
    )

    with st.expander("Show clustering interpretation notes"):
        st.markdown(
            """
This page summarizes drug clusters inferred from pathway-level perturbation profiles.

Recommended interpretation:
- cluster membership reflects similarity in pathway signatures
- drugs in the same cluster tend to share related response programs
- fuzzy membership scores indicate how strongly each drug belongs to its dominant cluster

The heatmap below displays a subset of drugs within the selected cluster and their pathway-level values.
"""
        )

    st.subheader("Cluster sizes")
    if not sum_df.empty and {"cluster", "n_drugs"}.issubset(sum_df.columns):
        fig_bar = go.Figure(go.Bar(
            x=sum_df["cluster"].astype(str),
            y=pd.to_numeric(sum_df["n_drugs"], errors="coerce")
        ))
        fig_bar.update_layout(
            xaxis_title="Cluster",
            yaxis_title="Number of drugs",
            template="simple_white",
            height=420
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Cluster summary file is missing required columns: cluster, n_drugs.")

    st.subheader("Cluster heatmap")

    if "cluster" not in matrix_df.columns:
        st.warning("The matrix file does not contain a cluster column.")
    else:
        clusters = sorted(matrix_df["cluster"].dropna().astype(str).unique().tolist())
        if len(clusters) == 0:
            st.warning("No clusters detected in the matrix file.")
        else:
            c1, c2 = st.columns([1.2, 1])
            with c1:
                selected_cluster = st.selectbox("Select cluster", clusters)
            with c2:
                show_n = st.slider("Number of drugs to display", 10, 100, 40, 5)

            sub = matrix_df.loc[matrix_df["cluster"].astype(str) == selected_cluster].copy()
            sub["abs_val"] = sub[value_cols].abs().max(axis=1)
            sub = sub.sort_values("abs_val", ascending=False).head(show_n)

            if sub.empty:
                st.warning("No drugs found for the selected cluster.")
            else:
                heat = sub.set_index("display_label")[value_cols]

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
                    xaxis_title="Pathways",
                    yaxis_title="Drugs"
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                st.subheader("Cluster members")
                show_cols = [c for c in [
                    "drug", "display_label", "cluster", "membership_score",
                    "target", "pathway", "research_area"
                ] if c in sub.columns]
                st.dataframe(sub[show_cols], use_container_width=True)

with tab2:
    st.subheader("Drug-to-neighbors star network")

    c1, c2 = st.columns([2, 1])

    drug_options = sorted(matrix_df["drug"].dropna().astype(str).unique().tolist())

    with c1:
        default_drug = "Loratadine"
        default_idx = drug_options.index(default_drug) if default_drug in drug_options else 0

        query = st.selectbox(
            "Select drug",
            drug_options,
            index=default_idx
            )

    with c2:
        top_n = st.slider("Top neighbors", 5, 30, 10)


    query = str(query).strip()
    if query == "":
        st.warning("Please enter a drug name.")
        st.stop()

    mat = matrix_df.copy()
    mat["drug"] = mat["drug"].astype(str)
    mat_num = mat.set_index("drug")[value_cols]

    if query not in mat_num.index:
        st.warning("Drug not found in the clustering matrix.")
        st.stop()

    corr = mat_num.T.corr(method="pearson")
    s = corr.loc[query].drop(labels=[query]).sort_values(ascending=False)

    if len(s) == 0:
        st.warning("No valid neighbors were found for this drug.")
        st.stop()

    s = s.head(top_n)
    neighbors = s.index.astype(str).tolist()

    labels = []
    neighbor_clusters = []
    neighbor_targets = []
    neighbor_pathways = []
    neighbor_research_areas = []
    neighbor_memberships = []
    neighbor_hover = []

    for n in neighbors:
        lab = label_map.get(n, n)
        cl = cluster_map.get(n, "NA")
        tg = target_map.get(n, "NA")
        pw = pathway_map.get(n, "NA")
        ra = research_map.get(n, "NA")
        ms = membership_score_map.get(n, np.nan)

        labels.append(str(lab))
        neighbor_clusters.append(str(cl))
        neighbor_targets.append(str(tg))
        neighbor_pathways.append(str(pw))
        neighbor_research_areas.append(str(ra))
        neighbor_memberships.append(ms)

        membership_text = f"{ms:.3f}" if pd.notna(ms) else "NA"
        neighbor_hover.append(
            f"Drug: {lab}<br>"
            f"Cluster: {cl}<br>"
            f"Target: {tg}<br>"
            f"Pathway: {pw}<br>"
            f"Research area: {ra}<br>"
            f"Membership: {membership_text}<br>"
            f"Correlation: {s.loc[n]:.3f}"
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

    q_label = label_map.get(query, query)
    q_cluster = cluster_map.get(query, "NA")
    q_target = target_map.get(query, "NA")
    q_pathway = pathway_map.get(query, "NA")
    q_research = research_map.get(query, "NA")
    q_membership = membership_score_map.get(query, np.nan)
    q_membership_text = f"{q_membership:.3f}" if pd.notna(q_membership) else "NA"

    fig_net.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode="markers+text",
        text=[q_label],
        textposition="top center",
        marker=dict(size=24, color="#D62728"),
        hovertext=[(
            f"Query drug: {q_label}<br>"
            f"Cluster: {q_cluster}<br>"
            f"Target: {q_target}<br>"
            f"Pathway: {q_pathway}<br>"
            f"Research area: {q_research}<br>"
            f"Membership: {q_membership_text}"
        )],
        hoverinfo="text",
        showlegend=False
    ))

    fig_net.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=15, color=node_colors, line=dict(width=1, color="black")),
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
        "drug": neighbors,
        "display_label": labels,
        "cluster": neighbor_clusters,
        "membership_score": neighbor_memberships,
        "target": neighbor_targets,
        "pathway": neighbor_pathways,
        "research_area": neighbor_research_areas,
        "correlation": [float(s.loc[n]) for n in neighbors]
    }).sort_values("correlation", ascending=False)

    st.dataframe(neighbor_df, use_container_width=True)
