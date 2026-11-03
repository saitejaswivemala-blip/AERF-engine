import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# AERF ENGINE — RESEARCH DASHBOARD
# ============================================================

st.set_page_config(
    page_title="AERF Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_DIR = PROJECT_ROOT / "experiments" / "runs"
RESULT_DIR = PROJECT_ROOT / "experiments" / "results"


# ============================================================
# DATABASE
# ============================================================

def find_latest(folder, pattern):
    if not folder.exists():
        return None

    files = list(folder.glob(pattern))

    if not files:
        return None

    return max(files, key=lambda x: x.stat().st_mtime)


@st.cache_resource
def get_connection(db_path):
    if db_path is None:
        return None

    return sqlite3.connect(str(db_path), check_same_thread=False)


@st.cache_data
def get_tables(db_path):
    connection = get_connection(db_path)

    if connection is None:
        return []

    query = """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """

    return pd.read_sql_query(query, connection)["name"].tolist()


@st.cache_data
def _read_table(table_name, db_path, limit=1000):
    connection = get_connection(db_path)
    tables = get_tables(db_path)

    if connection is None:
        return pd.DataFrame()

    if table_name not in tables:
        return pd.DataFrame()

    try:
        return pd.read_sql_query(
            f'SELECT * FROM "{table_name}" LIMIT {limit}',
            connection,
        )
    except Exception:
        return pd.DataFrame()


def read_table(table_name, limit=1000):
    return _read_table(table_name, str(selected_db_path), limit)


@st.cache_data
def _get_table_columns(table_name, db_path):
    connection = get_connection(db_path)

    if connection is None:
        return []

    try:
        df = pd.read_sql_query(
            f'PRAGMA table_info("{table_name}")',
            connection,
        )

        return df["name"].tolist()
    except Exception:
        return []


def get_table_columns(table_name):
    return _get_table_columns(table_name, str(selected_db_path))


# ============================================================
# RESULT JSON
# ============================================================

def load_result():

    path = find_latest(
        RESULT_DIR,
        "*.json",
    )

    if path is None:
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


result = load_result()


# ============================================================
# HELPERS
# ============================================================

def first_existing(columns, candidates):

    for candidate in candidates:

        if candidate in columns:
            return candidate

    return None


def format_number(value):

    if value is None:
        return "—"

    try:

        value = float(value)

        if value.is_integer():
            return f"{int(value):,}"

        return f"{value:.3f}"

    except Exception:
        return str(value)


def truncate(text, length=180):

    text = str(text)

    if len(text) <= length:
        return text

    return text[:length] + "…"


def result_value(section, key, default=0):

    if not isinstance(result, dict):
        return default

    obj = result.get(section, {})

    if not isinstance(obj, dict):
        return default

    return obj.get(key, default)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 650;
        margin-top: 1rem;
    }

    .claim-card {
        padding: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        background: #ffffff;
    }

    .status-support {
        padding: 5px 10px;
        border-radius: 20px;
        background: #dcfce7;
        color: #166534;
        font-weight: 600;
    }

    .status-contradict {
        padding: 5px 10px;
        border-radius: 20px;
        background: #fee2e2;
        color: #991b1b;
        font-weight: 600;
    }

    .status-neutral {
        padding: 5px 10px;
        border-radius: 20px;
        background: #f3f4f6;
        color: #374151;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "# 🧠 AERF Engine"
)

st.sidebar.caption(
    "Adaptive Evidence Reliability & Reasoning"
)

st.sidebar.divider()

database_files = sorted(DB_DIR.glob("*.db"), key=lambda path: path.name)
database_names = [path.name for path in database_files]
default_database = "experiment_FINAL_2026-09-02.db"
default_index = (
    database_names.index(default_database)
    if default_database in database_names
    else 0
)

selected_database_name = st.sidebar.selectbox(
    "Database",
    database_names,
    index=default_index,
)
selected_db_path = DB_DIR / selected_database_name if database_names else None

conn = get_connection(str(selected_db_path) if selected_db_path else None)
tables = get_tables(str(selected_db_path) if selected_db_path else None)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📄 Scientific Claims",
        "🔗 Evidence Relationships",
        "📊 AERF Reliability",
        "🕸️ Evidence Graph",
        "⏳ Temporal Evolution",
        "🔎 Research Gaps",
        "🧬 Extracted Relations",
        "🗃️ Database Explorer",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Real experiment data • SQLite • AERF Engine"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="main-title">Adaptive Evidence Reliability & Reasoning Engine</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">Multi-document scientific literature analysis, evidence validation and adaptive reliability reasoning.</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # -------------------------
    # MAIN METRICS
    # -------------------------

    papers_df = read_table("papers")
    claims_df = read_table("claims")
    pairs_df = read_table("claim_pairs")
    graph_df = read_table("evidence_graph_edges")
    gaps_df = read_table("research_gaps")
    relations_df = read_table("relations")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Scientific Papers",
            len(papers_df),
        )

    with c2:
        st.metric(
            "Claims",
            len(claims_df),
        )

    with c3:
        st.metric(
            "Evidence Pairs",
            len(pairs_df),
        )

    with c4:
        st.metric(
            "Graph Edges",
            len(graph_df),
        )

    with c5:
        st.metric(
            "Research Gaps",
            len(gaps_df),
        )

    st.divider()

    # -------------------------
    # REAL PIPELINE RESULT
    # -------------------------

    st.subheader("Pipeline Status")

    pipeline = [
        "PDF Ingestion",
        "Text Cleaning & Chunking",
        "Scientific Entity Extraction",
        "Claim Extraction",
        "Relation Extraction",
        "Semantic Embeddings",
        "Candidate Matching",
        "NLI Validation",
        "AERF Reliability Scoring",
        "Adaptive Updating",
        "Evidence Graph",
        "Temporal Evolution",
        "Research Gap Detection",
    ]

    cols = st.columns(4)

    for i, step in enumerate(pipeline):

        with cols[i % 4]:

            st.success(
                f"✓ {step}"
            )

    st.divider()

    # -------------------------
    # VALIDATION SUMMARY
    # -------------------------

    st.subheader("Evidence Validation Summary")

    supports = result_value(
        "phase3",
        "validated_relationships",
        {},
    )

    if isinstance(supports, dict):

        labels = {
            "supports": supports.get("supports", 0),
            "contradicts": supports.get("contradicts", 0),
            "neutral": supports.get("neutral", 0),
            "uncertain": supports.get("uncertain", 0),
        }

        chart_df = pd.DataFrame(
            {
                "Relationship": list(labels.keys()),
                "Count": list(labels.values()),
            }
        )

        st.bar_chart(
            chart_df.set_index("Relationship")
        )

    st.divider()

    # -------------------------
    # ADAPTIVE CHANGES
    # -------------------------

    st.subheader("Adaptive Reliability Activity")

    adaptive_changes = result_value(
        "phase3",
        "adaptive_score_changes",
        0,
    )

    st.metric(
        "Adaptive Score Changes",
        adaptive_changes,
    )

    st.caption(
        "This represents reliability-state changes recorded by the adaptive reasoning layer."
    )


# ============================================================
# CLAIMS
# ============================================================

elif page == "📄 Scientific Claims":

    st.markdown(
        '<div class="section-title">Scientific Claims</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Claims extracted from the processed scientific literature."
    )

    claims = read_table("claims")

    if claims.empty:

        st.warning(
            "No claims available."
        )

    else:

        # -------------------------
        # SEARCH
        # -------------------------

        search = st.text_input(
            "Search claims",
            placeholder="Search scientific claim text...",
        )

        if search:

            claims = claims[
                claims["text"]
                .astype(str)
                .str.contains(
                    search,
                    case=False,
                    na=False,
                )
            ]

        # -------------------------
        # PAPER FILTER
        # -------------------------

        if "paper_id" in claims.columns:

            papers = read_table("papers")

            if not papers.empty:

                paper_id_col = first_existing(
                    papers.columns,
                    ["paper_id", "id"],
                )

                title_col = first_existing(
                    papers.columns,
                    [
                        "title",
                        "paper_title",
                        "name",
                    ],
                )

                if paper_id_col and title_col:

                    paper_map = dict(
                        zip(
                            papers[paper_id_col],
                            papers[title_col],
                        )
                    )

                    claims["Paper"] = claims[
                        "paper_id"
                    ].map(paper_map)

        # -------------------------
        # SUMMARY
        # -------------------------

        st.metric(
            "Claims Found",
            len(claims),
        )

        st.divider()

        # -------------------------
        # CLAIM CARDS
        # -------------------------

        for _, row in claims.head(50).iterrows():

            claim_id = row.get(
                "claim_id",
                "—",
            )

            text = row.get(
                "text",
                "",
            )

            paper = row.get(
                "Paper",
                f"Paper {row.get('paper_id', '—')}",
            )

            with st.container():

                st.markdown(
                    f"**Claim {claim_id}**"
                )

                st.write(text)

                st.caption(
                    f"Source: {paper} • "
                    f"Chunk: {row.get('chunk_id', '—')}"
                )

                st.divider()


# ============================================================
# EVIDENCE RELATIONSHIPS
# ============================================================

elif page == "🔗 Evidence Relationships":

    st.markdown(
        '<div class="section-title">Evidence Relationships</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Candidate claim relationships validated through semantic/NLI reasoning."
    )

    pairs = read_table(
        "claim_pairs"
    )

    claims = read_table(
        "claims"
    )

    if pairs.empty:

        st.warning(
            "No claim relationships available."
        )

    else:

        # -------------------------
        # LABEL SUMMARY
        # -------------------------

        label_col = first_existing(
            pairs.columns,
            [
                "validated_as",
                "nli_label",
                "relation",
            ],
        )

        if label_col:

            counts = (
                pairs[label_col]
                .astype(str)
                .value_counts()
            )

            st.bar_chart(counts)

        st.metric(
            "Relationship Records",
            len(pairs),
        )

        st.divider()

        # -------------------------
        # ADD CLAIM TEXT
        # -------------------------

        if not claims.empty:

            claim_map = dict(
                zip(
                    claims["claim_id"],
                    claims["text"],
                )
            )

            if "claim_a_id" in pairs.columns:

                pairs["Claim A"] = pairs[
                    "claim_a_id"
                ].map(claim_map)

            if "claim_b_id" in pairs.columns:

                pairs["Claim B"] = pairs[
                    "claim_b_id"
                ].map(claim_map)

        # -------------------------
        # FILTER
        # -------------------------

        if label_col:

            selected = st.multiselect(
                "Relationship type",
                sorted(
                    pairs[label_col]
                    .dropna()
                    .astype(str)
                    .unique()
                ),
                default=[],
            )

            if selected:

                pairs = pairs[
                    pairs[label_col]
                    .astype(str)
                    .isin(selected)
                ]

        # -------------------------
        # DISPLAY
        # -------------------------

        display_cols = [
            c for c in [
                "pair_id",
                "claim_a_id",
                "claim_b_id",
                "Claim A",
                "Claim B",
                "cosine_similarity",
                "nli_label",
                "nli_confidence",
                "entity_overlap_score",
                "validated_as",
                "relation_alignment_score",
                "matching_method",
            ]
            if c in pairs.columns
        ]

        st.dataframe(
            pairs[display_cols]
            if display_cols
            else pairs,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# AERF RELIABILITY
# ============================================================

elif page == "📊 AERF Reliability":

    st.markdown(
        '<div class="section-title">AERF Reliability</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Multi-factor evidence reliability assessment."
    )

    scores = read_table(
        "aerf_scores"
    )

    history = read_table(
        "aerf_score_history"
    )

    claims = read_table(
        "claims"
    )

    if scores.empty:

        st.warning(
            "No AERF score records available."
        )

    else:

        # -------------------------
        # SCORE COLUMN
        # -------------------------

        score_col = first_existing(
            scores.columns,
            [
                "score",
                "aerf_score",
                "reliability_score",
            ],
        )

        if score_col:

            scores[score_col] = pd.to_numeric(
                scores[score_col],
                errors="coerce",
            )

            valid_scores = scores[
                score_col
            ].dropna()

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Average Reliability",
                    f"{valid_scores.mean():.3f}",
                )

            with c2:
                st.metric(
                    "Highest",
                    f"{valid_scores.max():.3f}",
                )

            with c3:
                st.metric(
                    "Lowest",
                    f"{valid_scores.min():.3f}",
                )

        st.divider()

        # -------------------------
        # FACTORS
        # -------------------------

        st.subheader(
            "Reliability Dimensions"
        )

        factor_names = [
            "Source",
            "Methodology",
            "Corroboration",
            "Contradiction Pressure",
            "Temporal Relevance",
            "Semantic Confidence",
            "Provenance",
        ]

        st.write(
            "AERF combines evidence characteristics across multiple dimensions."
        )

        factor_cols = []

        for name in factor_names:

            possible = [
                name.lower().replace(" ", "_"),
                name.lower().replace(" ", ""),
            ]

            found = first_existing(
                scores.columns,
                possible,
            )

            if found:
                factor_cols.append(found)

        if factor_cols:

            factor_data = scores[
                factor_cols
            ].apply(
                pd.to_numeric,
                errors="coerce",
            ).mean()

            factor_data.index = [
                x.replace("_", " ").title()
                for x in factor_data.index
            ]

            st.bar_chart(
                factor_data
            )

        st.divider()

        # -------------------------
        # CLAIM MAPPING
        # -------------------------

        if (
            not claims.empty
            and "claim_id" in scores.columns
        ):

            claim_map = dict(
                zip(
                    claims["claim_id"],
                    claims["text"],
                )
            )

            scores["Claim"] = scores[
                "claim_id"
            ].map(claim_map)

        # -------------------------
        # SCORE TABLE
        # -------------------------

        st.subheader(
            "Claim Reliability Scores"
        )

        st.dataframe(
            scores,
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------
        # ADAPTIVE HISTORY
        # -------------------------

        if not history.empty:

            st.divider()

            st.subheader(
                "Adaptive Reliability Changes"
            )

            if (
                "old_score" in history.columns
                and "new_score" in history.columns
            ):

                chart = history[
                    [
                        "old_score",
                        "new_score",
                    ]
                ].copy()

                st.line_chart(
                    chart
                )

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# EVIDENCE GRAPH
# ============================================================

elif page == "🕸️ Evidence Graph":

    st.markdown(
        '<div class="section-title">Evidence Graph</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Structured graph connecting scientific claims, entities and evidence."
    )

    graph = read_table(
        "evidence_graph_edges"
    )

    if graph.empty:

        st.warning(
            "No evidence graph edges available."
        )

    else:

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Graph Edges",
                len(graph),
            )

        with c2:

            if "edge_type" in graph.columns:

                st.metric(
                    "Edge Types",
                    graph["edge_type"]
                    .nunique(),
                )

        st.divider()

        if "edge_type" in graph.columns:

            counts = (
                graph["edge_type"]
                .astype(str)
                .value_counts()
            )

            st.subheader(
                "Edge Distribution"
            )

            st.bar_chart(
                counts
            )

        st.subheader(
            "Graph Records"
        )

        st.dataframe(
            graph,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TEMPORAL EVOLUTION
# ============================================================

elif page == "⏳ Temporal Evolution":

    st.markdown(
        '<div class="section-title">Temporal Evidence Evolution</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Tracks how claim reliability changes as evidence relationships are updated."
    )

    history = read_table(
        "aerf_score_history"
    )

    snapshots = read_table(
        "knowledge_evolution_snapshots"
    )

    if not history.empty:

        st.metric(
            "Reliability Updates",
            len(history),
        )

        if (
            "old_score" in history.columns
            and "new_score" in history.columns
        ):

            chart = history[
                [
                    "old_score",
                    "new_score",
                ]
            ].copy()

            chart.columns = [
                "Previous",
                "Updated",
            ]

            st.line_chart(
                chart
            )

        st.subheader(
            "Reliability Update History"
        )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

    if not snapshots.empty:

        st.divider()

        st.subheader(
            "Knowledge Evolution Snapshots"
        )

        st.metric(
            "Snapshots",
            len(snapshots),
        )

        st.dataframe(
            snapshots,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# RESEARCH GAPS
# ============================================================

elif page == "🔎 Research Gaps":

    st.markdown(
        '<div class="section-title">Research Gaps</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Potential evidence deficiencies identified from the current scientific corpus."
    )

    gaps = read_table(
        "research_gaps"
    )

    if gaps.empty:

        st.warning(
            "No research gaps available."
        )

    else:

        st.metric(
            "Detected Research Gaps",
            len(gaps),
        )

        if "gap_type" in gaps.columns:

            st.subheader(
                "Gap Types"
            )

            counts = (
                gaps["gap_type"]
                .astype(str)
                .value_counts()
            )

            st.bar_chart(
                counts
            )

        st.divider()

        # -------------------------
        # GAP CARDS
        # -------------------------

        description_col = first_existing(
            gaps.columns,
            [
                "description",
                "gap_description",
                "text",
            ],
        )

        for _, row in gaps.head(50).iterrows():

            gap_type = row.get(
                "gap_type",
                "Unknown",
            )

            topic = row.get(
                "topic_key",
                "",
            )

            description = row.get(
                description_col,
                "",
            ) if description_col else ""

            with st.expander(
                f"🔎 {gap_type} — {topic}"
            ):

                st.write(
                    description
                )

                if "related_claim_ids" in row:

                    st.caption(
                        f"Related claims: {row['related_claim_ids']}"
                    )


# ============================================================
# RELATIONS
# ============================================================

elif page == "🧬 Extracted Relations":

    st.markdown(
        '<div class="section-title">Extracted Scientific Relations</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Relations extracted from scientific text using the relation-extraction layer."
    )

    relations = read_table(
        "relations"
    )

    if relations.empty:

        st.warning(
            "No extracted relations available."
        )

    else:

        st.metric(
            "Extracted Relations",
            len(relations),
        )

        if "predicate" in relations.columns:

            st.subheader(
                "Predicates"
            )

            counts = (
                relations["predicate"]
                .astype(str)
                .value_counts()
            )

            st.bar_chart(
                counts
            )

        st.divider()

        st.dataframe(
            relations,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# DATABASE EXPLORER
# ============================================================

elif page == "🗃️ Database Explorer":

    st.markdown(
        '<div class="section-title">Database Explorer</div>',
        unsafe_allow_html=True,
    )

    if conn is None:

        st.error(
            "AERF database could not be opened."
        )

    else:

        st.caption(
            f"Database: {selected_db_path.name if selected_db_path else 'Unknown'}"
        )

        st.metric(
            "Database Tables",
            len(tables),
        )

        st.divider()

        selected_table = st.selectbox(
            "Select table",
            tables,
        )

        data = read_table(
            selected_table,
            1000,
        )

        st.write(
            f"**Rows:** {len(data):,}  |  "
            f"**Columns:** {len(data.columns):,}"
        )

        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AERF Engine • Adaptive Evidence Reliability & Reasoning • "
    "Research prototype"
)