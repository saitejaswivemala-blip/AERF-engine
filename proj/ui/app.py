import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of execution directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import sqlite3
import pandas as pd
import streamlit as st
import shutil

from pipeline.orchestrator import ingest_paper



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





@st.cache_resource(show_spinner="Loading NLP models (first run only)...")
def _load_phase2_models():
    from pipeline.phase2 import Phase2Pipeline
    return Phase2Pipeline(None)


@st.cache_resource(show_spinner="Loading NLI model (first run only)...")
def _load_phase3_models():
    from pipeline.phase3 import Phase3Pipeline
    return Phase3Pipeline(None)


def get_phase2_pipeline(repository):
    p = _load_phase2_models()
    p.repo = repository
    return p


def get_phase3_pipeline(repository):
    p = _load_phase3_models()
    p.repo = repository
    p.scorer.repo = repository
    p.updater.repo = repository
    return p


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
def _read_table(table_name, db_path, limit=None):
    connection = get_connection(db_path)
    tables = get_tables(db_path)

    if connection is None or table_name not in tables:
        return pd.DataFrame()

    try:
        if limit is None:
            query = f'SELECT * FROM "{table_name}"'
            return pd.read_sql_query(query, connection)

        query = f'SELECT * FROM "{table_name}" LIMIT ?'
        return pd.read_sql_query(
            query,
            connection,
            params=(int(limit),),
        )

    except Exception:
        return pd.DataFrame()


def read_table(table_name, limit=None):
    return _read_table(
        table_name,
        str(selected_db_path),
        limit=limit,
    )


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



sessions_dir = DB_DIR / "sessions"
session_db_files = list(sessions_dir.glob("*.db")) if sessions_dir.exists() else []
database_files = sorted(list(DB_DIR.glob("*.db")) + session_db_files, key=lambda path: path.name)

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

selected_db_path = next((p for p in database_files if p.name == selected_database_name), None) if database_names else None



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

        "📄 New Analysis",

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

# NEW ANALYSIS

# ============================================================



elif page == "📄 New Analysis":

    st.markdown(
        '<div class="section-title">New Analysis — Isolated Multi-Paper Sessions</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Upload and analyze research papers in an isolated, session-scoped environment. "
        "Results are strictly scoped to the papers uploaded in this session, leaving the "
        "original 11-paper database and corpus untouched."
    )

    from pipeline.session_manager import (
        _fmt_time,
        create_session,
        generate_session_id,
        get_session,
        get_session_repository,
        get_session_results,
        list_sessions,
        run_session_analysis,
    )

    # ----------------------------------------------------
    # SESSION SELECTOR & HISTORY
    # ----------------------------------------------------
    all_sessions = list_sessions()
    NEW_SESSION_OPTION = "➕ Start New Analysis Session"
    session_options = [NEW_SESSION_OPTION]
    session_id_map = {NEW_SESSION_OPTION: None}

    for s in all_sessions:
        sid = s["session_id"]
        paper_cnt = s.get("paper_count", len(s.get("papers", [])))
        status = str(s.get("status", "completed")).lower()
        status_icon = "✅" if status == "completed" else ("⏳" if status == "processing" else ("⏸️" if status == "stopped" else "❌"))
        dt_str = s.get("created_at", "")[:16].replace("T", " ")
        name_str = f" — {s['name']}" if s.get("name") and s["name"] != sid else ""
        lbl = f"{status_icon} {sid}{name_str} ({paper_cnt} paper{'s' if paper_cnt != 1 else ''}) — {dt_str}"
        session_options.append(lbl)
        session_id_map[lbl] = sid

    # Ensure state defaults
    if "session_view_mode" not in st.session_state:
        st.session_state["session_view_mode"] = "new"
    if "active_session_id" not in st.session_state:
        st.session_state["active_session_id"] = None

    curr_mode = st.session_state.get("session_view_mode", "new")
    active_sid = st.session_state.get("active_session_id")

    curr_index = 0
    if curr_mode == "view" and active_sid:
        for idx, opt in enumerate(session_options):
            if session_id_map.get(opt) == active_sid:
                curr_index = idx
                break

    col_sel, col_act = st.columns([4, 1])
    with col_sel:
        selected_lbl = st.selectbox(
            "Select Analysis Session",
            options=session_options,
            index=curr_index,
            key="session_picker_widget",
            help="Switch between historical analysis sessions or start a new one.",
        )
    with col_act:
        st.write("")
        st.write("")
        if st.button("➕ New Session", key="btn_create_new_session_top", use_container_width=True):
            st.session_state["session_view_mode"] = "new"
            st.session_state["active_session_id"] = None
            st.rerun()

    # Determine mode based on selection
    chosen_sid = session_id_map.get(selected_lbl)
    if selected_lbl == NEW_SESSION_OPTION or chosen_sid is None:
        if st.session_state.get("session_view_mode") != "new" or st.session_state.get("active_session_id") is not None:
            st.session_state["session_view_mode"] = "new"
            st.session_state["active_session_id"] = None
            st.rerun()
        is_new_mode = True
    else:
        if st.session_state.get("session_view_mode") != "view" or st.session_state.get("active_session_id") != chosen_sid:
            st.session_state["session_view_mode"] = "view"
            st.session_state["active_session_id"] = chosen_sid
            st.rerun()
        is_new_mode = False

    # ----------------------------------------------------
    # MODE 1: UPLOAD & START NEW SESSION FORM
    # ----------------------------------------------------
    if is_new_mode:
        st.subheader("Start New Analysis Session")
        st.write("Upload 1 or more PDF research papers to process in an isolated session.")

        session_name_input = st.text_input(
            "Session Name (Optional)",
            key="new_session_name_input",
            placeholder="e.g. Subglacial hydrology comparison, Quantum optics review",
        )

        uploaded_files = st.file_uploader(
            "Upload PDF research papers",
            type=["pdf"],
            accept_multiple_files=True,
            key="research_pdf_uploader",
        )

        if uploaded_files:
            if len(uploaded_files) == 1:
                st.info(
                    "ℹ️ **Note on Comparison**: You have selected 1 PDF file. "
                    "Multi-paper comparison requires at least two papers. "
                    "You can proceed to extract claims and baseline reliability for this paper, "
                    "or select additional PDF files to enable cross-paper comparison."
                )
            else:
                st.success(f"✅ {len(uploaded_files)} PDF files selected for isolated comparative analysis.")

            upload_dir = PROJECT_ROOT / "experiments" / "uploads" / "sessions"
            upload_dir.mkdir(parents=True, exist_ok=True)

            file_details = []
            for f in uploaded_files:
                file_details.append({
                    "Filename": f.name,
                    "Size": f"{len(f.getvalue()) / 1024:.1f} KB",
                    "Ready": "✅ Ready",
                })
            st.dataframe(pd.DataFrame(file_details), use_container_width=True, hide_index=True)

        # Check for resumable sessions (interrupted or stopped)
        incomplete_sessions = [
            s for s in all_sessions if s.get("status") in ("processing", "stopped", "failed")
        ]
        if incomplete_sessions:
            st.markdown("---")
            st.markdown("#### 🔄 Resumable Sessions (Interrupted / Checkpoint Available)")
            for inc in incomplete_sessions[:3]:
                sid = inc["session_id"]
                s_name = inc.get("name", sid)
                s_dir = (PROJECT_ROOT / "experiments" / "uploads" / "sessions") / sid
                pdfs = list(s_dir.glob("*.pdf")) if s_dir.exists() else []
                r_col1, r_col2 = st.columns([3, 1])
                with r_col1:
                    st.write(f"**{s_name}** (`{sid}`)")
                    st.caption(f"Status: `{inc.get('status')}` • {len(pdfs)} PDF file(s) saved in session folder")
                with r_col2:
                    if pdfs and st.button(f"▶️ Resume", key=f"btn_resume_{sid}", use_container_width=True):
                        st.session_state["resuming_session_id"] = sid
                        st.session_state["resuming_pdfs"] = [str(p) for p in pdfs]
                        st.session_state["resuming_name"] = s_name
                        st.rerun()
            st.markdown("---")

        analyze_btn = st.button(
            "🚀 Analyze Uploaded Papers",
            type="primary",
            use_container_width=True,
            key="btn_analyze_uploaded_papers",
        )

        is_resuming = "resuming_session_id" in st.session_state
        if analyze_btn or is_resuming:
            if not is_resuming and not uploaded_files:
                st.warning("⚠️ Please upload at least one PDF research paper before starting the analysis. (Multi-paper comparison requires at least two papers.)")
            else:
                progress_container = st.container()
                with progress_container:
                    st.markdown("### ⚙️ Live Processing Pipeline")
                    progress_bar = st.progress(0)
                    status_box = st.info("Initializing analysis...")
                    metric_cols = st.columns(4)
                    metric_paper = metric_cols[0].empty()
                    metric_chunks = metric_cols[1].empty()
                    metric_claims = metric_cols[2].empty()
                    metric_time = metric_cols[3].empty()

                    metric_paper.metric("Paper", "—")
                    metric_chunks.metric("Progress", "0 / 0")
                    metric_claims.metric("Claims Extracted", "0")
                    metric_time.metric("Elapsed / ETA", "0s")

                try:
                    if is_resuming:
                        active_sid = st.session_state.pop("resuming_session_id")
                        saved_paths = st.session_state.pop("resuming_pdfs")
                        st.session_state.pop("resuming_name", None)
                        session_repo = get_session_repository(active_sid)
                    else:
                        active_sid = generate_session_id()
                        sess_name = session_name_input.strip() or f"Session ({len(uploaded_files)} papers)"
                        session_upload_dir = (PROJECT_ROOT / "experiments" / "uploads" / "sessions") / active_sid
                        session_upload_dir.mkdir(parents=True, exist_ok=True)
                        saved_paths = []
                        for f in uploaded_files:
                            sp = session_upload_dir / f.name
                            with open(sp, "wb") as out_f:
                                out_f.write(f.getvalue())
                            saved_paths.append(sp)

                        _, session_repo = create_session(
                            session_id=active_sid,
                            name=sess_name,
                        )

                    def update_progress(info):
                        if isinstance(info, dict):
                            pct = info.get("percent", info.get("overall_pct", 0))
                            progress_bar.progress(min(100, max(0, pct)))
                            msg = info.get("message", "Processing...")
                            status_box.info(f"**{msg}**")
                            stage = info.get("stage", "")
                            if stage == "phase1":
                                metric_paper.metric("File", f"{info.get('file_index', 1)} / {info.get('total_files', 1)}")
                                metric_chunks.metric("Stage", "PDF Ingestion")
                                metric_claims.metric("Filename", info.get("filename", "")[:18])
                                metric_time.metric("Status", "Parsing...")
                            elif stage == "phase2":
                                metric_paper.metric("Current Paper", f"{info.get('paper_idx', 1)} / {info.get('total_papers', 1)}")
                                metric_chunks.metric("Chunks", f"{info.get('processed_chunks', 0)} / {info.get('total_chunks', 0)}")
                                metric_claims.metric("Claims Extracted", f"{info.get('claims_found', 0)}")
                                metric_time.metric("Elapsed / ETA", f"{_fmt_time(info.get('elapsed_s', 0))} / ~{_fmt_time(info.get('eta_s', 0))}")
                            elif stage == "phase3":
                                metric_paper.metric("Stage", "Reasoning")
                                metric_chunks.metric("Pairs", f"{info.get('current_pair', 0)} / {info.get('total_pairs', 0)}")
                                metric_claims.metric("Valid Pairs", f"{info.get('matched', 0)}")
                                metric_time.metric("Elapsed / ETA", f"{_fmt_time(info.get('elapsed_s', 0))} / ~{_fmt_time(info.get('eta_s', 0))}")
                        else:
                            pct = int(info)
                            progress_bar.progress(min(100, max(0, pct)))

                    p2 = get_phase2_pipeline(session_repo)
                    p3 = get_phase3_pipeline(session_repo)

                    run_session_analysis(
                        session_id=active_sid,
                        pdf_paths=saved_paths,
                        phase2_pipeline=p2,
                        phase3_pipeline=p3,
                        progress_callback=update_progress,
                        repo=session_repo,
                    )

                    st.cache_data.clear()
                    st.session_state["session_view_mode"] = "view"
                    st.session_state["active_session_id"] = active_sid
                    status_box.success("✅ Analysis completed successfully!")
                    st.rerun()

                except InterruptedError as ie:
                    status_box.warning(f"⏸️ Analysis paused / stopped: {ie}. All progress has been checkpointed in the session database.")
                except Exception as exc:
                    status_box.error(f"❌ Analysis failed: {exc}")
                    st.exception(exc)

    # ----------------------------------------------------
    # MODE 2: DISPLAY SESSION RESULTS VIEW
    # ----------------------------------------------------
    else:
        active_session_id = st.session_state.get("active_session_id")
        results = get_session_results(active_session_id)
        sess_info = results.get("session_info", {})
        summary = results.get("summary", {})
        papers = results.get("papers", [])
        claims = results.get("claims", [])
        pairs = results.get("pairs", [])
        scores = results.get("scores", [])
        graph_edges = results.get("graph_edges", [])
        gaps = results.get("gaps", [])

        # Strict Session Isolation Notice
        st.info(
            "🔒 **Strict Session Isolation Active**: Analysis and comparisons are strictly limited to the "
            "papers uploaded in this session. Historical claims, scores, and evidence relationships from the "
            "original 11-paper database or other sessions are not included."
        )

        # Header & Status
        hdr_col1, hdr_col2 = st.columns([3, 1])
        with hdr_col1:
            st.markdown(f"### 🔬 Session Results: `{active_session_id}`")
            created_dt = sess_info.get("created_at", "")[:19].replace("T", " ")
            st.caption(f"**Name:** {sess_info.get('name', 'Analysis Session')} • **Created:** {created_dt}")
        with hdr_col2:
            status_val = str(sess_info.get("status", "completed")).lower()
            if status_val == "failed":
                st.error("Status: FAILED")
            elif status_val == "processing":
                st.warning("Status: PROCESSING")
            else:
                st.success("Status: COMPLETED")
            if st.button("➕ New Session", key="btn_new_session_from_results", use_container_width=True):
                st.session_state["session_view_mode"] = "new"
                st.session_state["active_session_id"] = None
                st.rerun()

        if sess_info.get("status") in ("failed", "stopped") and sess_info.get("error_message"):
            st.error(f"**Session Failure Message:** {sess_info['error_message']}")

        # Uploaded Papers overview
        st.markdown("#### 📄 Uploaded Papers in This Session")
        if papers:
            paper_rows = []
            for p in papers:
                p_claims = [c for c in claims if c.get("paper_id") == p["paper_id"]]
                paper_rows.append({
                    "Paper ID": p["paper_id"],
                    "Filename": p["source_filename"],
                    "Title": p.get("title") or "—",
                    "Year": p.get("publication_year") or "—",
                    "Pages": p.get("page_count") or "—",
                    "Extracted Claims": len(p_claims),
                })
            st.dataframe(pd.DataFrame(paper_rows), use_container_width=True, hide_index=True)

        # Multi-paper check
        if len(papers) < 2:
            st.warning(
                "⚠️ **Multi-paper comparison requires at least two papers.**\n\n"
                f"Currently, only {len(papers)} paper ({papers[0]['source_filename'] if papers else 'none'}) "
                "is in this session. Single-paper analysis provides claim extraction, named entities, relations, "
                "and baseline reliability scoring. To enable cross-paper comparative analysis, claim-level "
                "agreement/disagreement, and cross-document evidence evaluation, start a new session with at least two papers."
            )

        st.divider()

        # Summary KPI metric cards
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Uploaded Papers", summary.get("total_papers", 0))
        k2.metric("Extracted Claims", summary.get("total_claims", 0))
        k3.metric("Evidence Pairs", summary.get("total_pairs", 0))
        k4.metric("Mean Reliability", f"{summary.get('mean_score', 0.0):.3f}")
        k5.metric("Graph Edges", summary.get("total_graph_edges", 0))
        k6.metric("Research Gaps", summary.get("total_gaps", 0))

        st.divider()

        # Tabs for detailed scoped results
        tab_compare, tab_claims, tab_pairs, tab_scores, tab_graph, tab_gaps = st.tabs([
            "⚔️ Direct Paper Comparison",
            f"📄 Extracted Claims ({len(claims)})",
            f"🔗 Evidence Relationships ({len(pairs)})",
            f"📊 AERF Reliability ({len(scores)})",
            f"🕸️ Evidence Graph ({len(graph_edges)})",
            f"🔎 Research Gaps ({len(gaps)})",
        ])

        with tab_compare:
            st.markdown("##### ⚔️ Direct Multi-Paper Comparative Analysis")
            if len(papers) < 2:
                st.info(
                    "ℹ️ **Multi-paper comparison requires at least two papers.**\n\n"
                    "Upload 2 or more research papers in a session to compare their claims, "
                    "detect agreements and contradictions, and evaluate cross-paper evidence coverage."
                )
            else:
                st.caption(
                    "Compare claims and evidence between any two uploaded papers in this session. "
                    "All findings are strictly derived from the selected documents."
                )
                paper_options = [p["source_filename"] for p in papers]
                col_pa, col_pb = st.columns(2)
                with col_pa:
                    paper_a_name = st.selectbox(
                        "Select Paper A",
                        options=paper_options,
                        index=0,
                        key="compare_paper_a_select",
                    )
                with col_pb:
                    default_b_idx = 1 if len(paper_options) > 1 else 0
                    paper_b_name = st.selectbox(
                        "Select Paper B",
                        options=paper_options,
                        index=default_b_idx,
                        key="compare_paper_b_select",
                    )

                if paper_a_name == paper_b_name:
                    st.warning("⚠️ Please select two distinct papers to perform cross-paper comparison.")
                else:
                    paper_a_obj = next(p for p in papers if p["source_filename"] == paper_a_name)
                    paper_b_obj = next(p for p in papers if p["source_filename"] == paper_b_name)
                    pid_a = paper_a_obj["paper_id"]
                    pid_b = paper_b_obj["paper_id"]

                    claims_a = [c for c in claims if c.get("paper_id") == pid_a]
                    claims_b = [c for c in claims if c.get("paper_id") == pid_b]

                    # Map claims to scores & page numbers
                    score_map = {s["claim_id"]: s for s in scores}
                    page_map = {c["claim_id"]: c.get("page_number", "—") for c in claims}

                    # Filter cross-paper pairs strictly between A and B
                    ab_pairs = [
                        p for p in pairs
                        if (p.get("paper_a_id") == pid_a and p.get("paper_b_id") == pid_b)
                        or (p.get("paper_a_id") == pid_b and p.get("paper_b_id") == pid_a)
                    ]

                    sup_pairs = [p for p in ab_pairs if p.get("validated_as") == "supports"]
                    con_pairs = [p for p in ab_pairs if p.get("validated_as") == "contradicts"]
                    neu_pairs = [p for p in ab_pairs if p.get("validated_as") == "neutral"]

                    # Metric cards for comparative alignment
                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mc1.metric(f"Claims in {paper_a_name[:18]}...", len(claims_a))
                    mc2.metric(f"Claims in {paper_b_name[:18]}...", len(claims_b))
                    mc3.metric("🟢 Supporting Matches", len(sup_pairs))
                    mc4.metric("🔴 Contradicting Matches", len(con_pairs))
                    agree_denom = len(sup_pairs) + len(con_pairs)
                    agree_pct = f"{(len(sup_pairs) / agree_denom * 100):.1f}%" if agree_denom > 0 else "N/A"
                    mc5.metric("Agreement Rate", agree_pct)

                    st.markdown("---")

                    # Section: Claim-Level Alignment & Evidence Attribution
                    st.markdown("#### 📑 Claim-Level Alignment & Source Attribution")
                    st.write(
                        f"Evaluate how claims in **{paper_a_name}** are supported, contradicted, or uncorroborated by **{paper_b_name}** (and vice-versa)."
                    )

                    align_tab_a, align_tab_b, align_tab_summary = st.tabs([
                        f"Claims in {paper_a_name[:24]} → Evidence in {paper_b_name[:24]}",
                        f"Claims in {paper_b_name[:24]} → Evidence in {paper_a_name[:24]}",
                        "📊 Comparative Evidence Summary",
                    ])

                    def render_claim_alignment_view(source_claims, source_name, target_pid, target_name):
                        if not source_claims:
                            st.info(f"No claims extracted from {source_name}.")
                            return

                        for idx, sc in enumerate(source_claims, start=1):
                            cid = sc["claim_id"]
                            c_text = sc.get("text", "")
                            page_num = sc.get("page_number", "—")
                            sc_score = score_map.get(cid, {})
                            score_val = float(sc_score.get("score") or 0.0)
                            fb = sc_score.get("factor_breakdown") or {}

                            # Find matching pairs with target paper
                            matching_pairs = []
                            for p in ab_pairs:
                                if p.get("claim_a_id") == cid and p.get("paper_b_id") == target_pid:
                                    matching_pairs.append({
                                        "opposing_claim_id": p.get("claim_b_id"),
                                        "opposing_text": p.get("claim_b_text"),
                                        "opposing_page": page_map.get(p.get("claim_b_id"), "—"),
                                        "status": p.get("validated_as"),
                                        "confidence": float(p.get("nli_confidence") or 0.0),
                                        "similarity": float(p.get("cosine_similarity") or 0.0),
                                    })
                                elif p.get("claim_b_id") == cid and p.get("paper_a_id") == target_pid:
                                    matching_pairs.append({
                                        "opposing_claim_id": p.get("claim_a_id"),
                                        "opposing_text": p.get("claim_a_text"),
                                        "opposing_page": page_map.get(p.get("claim_a_id"), "—"),
                                        "status": p.get("validated_as"),
                                        "confidence": float(p.get("nli_confidence") or 0.0),
                                        "similarity": float(p.get("cosine_similarity") or 0.0),
                                    })

                            has_sup = any(m["status"] == "supports" for m in matching_pairs)
                            has_con = any(m["status"] == "contradicts" for m in matching_pairs)
                            status_badge = "🟢 SUPPORTED" if has_sup else ("🔴 CONTRADICTED" if has_con else ("⚪ NEUTRAL MATCH" if matching_pairs else "⚠️ INSUFFICIENT EVIDENCE"))

                            with st.expander(f"**[{status_badge}]** Claim {cid} (p. {page_num}) — Score: `{score_val:.4f}` | {c_text[:90]}...", expanded=False):
                                st.markdown(f"**Full Claim Text:**\n> {c_text}")
                                st.caption(f"**Source Document:** `{source_name}` • **Page:** {page_num} • **Claim ID:** `{cid}`")

                                # Factor breakdown
                                st.markdown("**AERF Reliability Factors (Session-Scoped):**")
                                f_cols = st.columns(7)
                                f_cols[0].metric("Source", f"{float(fb.get('source') or 0):.2f}")
                                f_cols[1].metric("Method", f"{float(fb.get('methodology') or 0):.2f}")
                                f_cols[2].metric("Corrob.", f"{float(fb.get('corroboration') or 0):.2f}")
                                f_cols[3].metric("Contrad.", f"{float(fb.get('contradiction') or 0):.2f}")
                                f_cols[4].metric("Temporal", f"{float(fb.get('temporal') or 0):.2f}")
                                f_cols[5].metric("Semantic", f"{float(fb.get('semantic') or 0):.2f}")
                                f_cols[6].metric("Proven.", f"{float(fb.get('provenance') or 0):.2f}")

                                st.markdown(f"**Evidence Alignment with `{target_name}`:**")
                                if matching_pairs:
                                    for m in matching_pairs:
                                        rel_icon = "🟢" if m["status"] == "supports" else ("🔴" if m["status"] == "contradicts" else "⚪")
                                        st.markdown(
                                            f"- {rel_icon} **{m['status'].upper()}** by `{target_name}` (Page {m['opposing_page']}, Claim `{m['opposing_claim_id']}`):\n"
                                            f"  > *\"{m['opposing_text']}\"*\n"
                                            f"  *(NLI Confidence: {m['confidence']:.2f} | Cosine Similarity: {m['similarity']:.2f})*"
                                        )
                                else:
                                    st.warning(
                                        f"⚠️ **Insufficient Evidence in `{target_name}`**: "
                                        f"No claim in `{target_name}` met the semantic similarity or NLI criteria to corroborate or contradict this claim. "
                                        f"This claim is uncorroborated within this two-paper session."
                                    )

                    with align_tab_a:
                        render_claim_alignment_view(claims_a, paper_a_name, pid_b, paper_b_name)

                    with align_tab_b:
                        render_claim_alignment_view(claims_b, paper_b_name, pid_a, paper_a_name)

                    with align_tab_summary:
                        st.markdown(f"##### Direct Evidence Matrix: `{paper_a_name}` ↔ `{paper_b_name}`")
                        if ab_pairs:
                            cmp_rows = []
                            for p in ab_pairs:
                                cmp_rows.append({
                                    "Pair ID": p.get("pair_id"),
                                    "Validated As": str(p.get("validated_as", "")).upper(),
                                    "Confidence": f"{float(p.get('nli_confidence') or 0):.2f}",
                                    "Cosine Sim": f"{float(p.get('cosine_similarity') or 0):.2f}",
                                    f"Claim in {paper_a_name[:20]}": p.get("claim_a_text") if p.get("paper_a_id") == pid_a else p.get("claim_b_text"),
                                    f"Claim in {paper_b_name[:20]}": p.get("claim_b_text") if p.get("paper_a_id") == pid_a else p.get("claim_a_text"),
                                })
                            st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
                        else:
                            st.info(
                                f"No cross-paper claim pairs between `{paper_a_name}` and `{paper_b_name}` "
                                f"exceeded the semantic similarity gate (threshold >= 0.55). Both papers discuss distinct claims or topics."
                            )

                    st.markdown("---")
                    st.markdown("#### ⚖️ Limitations & Uncertainty in Direct Comparison")
                    claims_a_with_match = set()
                    claims_b_with_match = set()
                    for p in ab_pairs:
                        if p.get("paper_a_id") == pid_a:
                            claims_a_with_match.add(p.get("claim_a_id"))
                            claims_b_with_match.add(p.get("claim_b_id"))
                        else:
                            claims_a_with_match.add(p.get("claim_b_id"))
                            claims_b_with_match.add(p.get("claim_a_id"))

                    uncorr_a = len(claims_a) - len(claims_a_with_match)
                    uncorr_b = len(claims_b) - len(claims_b_with_match)

                    st.markdown(
                        f"- **Evidence Coverage**: {len(claims_a_with_match)} of {len(claims_a)} claims in *{paper_a_name}* "
                        f"have direct evidence matches in *{paper_b_name}* ({uncorr_a} uncorroborated).\n"
                        f"- **Evidence Coverage**: {len(claims_b_with_match)} of {len(claims_b)} claims in *{paper_b_name}* "
                        f"have direct evidence matches in *{paper_a_name}* ({uncorr_b} uncorroborated).\n"
                        f"- **Missing / Insufficient Evidence**: Claims without cross-paper matches represent either unique contributions, differing experimental conditions, or domain gaps between these two specific papers. AERF Engine never hallucinates or imports external evidence.\n"
                        f"- **Session Scope**: Analysis is limited strictly to papers uploaded in this session."
                    )

        with tab_claims:
            st.markdown("##### Scientific Claims Extracted in This Session")
            if claims:
                filter_paper = st.selectbox(
                    "Filter by Paper",
                    options=["All Papers"] + [p["source_filename"] for p in papers],
                    key="session_claims_paper_filter",
                )
                search_q = st.text_input("Search claims by keyword", key="session_claims_search")

                claims_filtered = claims
                if filter_paper != "All Papers":
                    claims_filtered = [c for c in claims_filtered if c.get("source_filename") == filter_paper]
                if search_q.strip():
                    q_lower = search_q.strip().lower()
                    claims_filtered = [c for c in claims_filtered if q_lower in c.get("text", "").lower()]

                st.write(f"Showing **{len(claims_filtered)}** of **{len(claims)}** claims:")
                claims_display = []
                for c in claims_filtered:
                    claims_display.append({
                        "Claim ID": c.get("claim_id"),
                        "Paper": c.get("source_filename"),
                        "Page": c.get("page_number", "—"),
                        "Claim Text": c.get("text"),
                    })
                st.dataframe(pd.DataFrame(claims_display), use_container_width=True, hide_index=True)
            else:
                st.info("No claims were extracted for the papers in this session.")

        with tab_pairs:
            st.markdown("##### Cross-Paper Evidence Relationships")
            if len(papers) <= 1:
                st.info(
                    "ℹ️ **Single-Paper Session**: 1 PDF was uploaded. Claims and baseline "
                    "reliability scores have been extracted without requiring cross-paper comparisons."
                )
            elif pairs:
                p_sup = summary.get("supports_count", 0)
                p_con = summary.get("contradicts_count", 0)
                p_neu = summary.get("neutral_count", 0)
                st.write(f"**Relationship Summary:** 🟢 Supports: {p_sup} | 🔴 Contradicts: {p_con} | ⚪ Neutral: {p_neu}")

                pairs_display = []
                for p in pairs:
                    pairs_display.append({
                        "Pair ID": p.get("pair_id"),
                        "Paper A": p.get("paper_a_filename"),
                        "Claim A": p.get("claim_a_text"),
                        "Paper B": p.get("paper_b_filename"),
                        "Claim B": p.get("claim_b_text"),
                        "NLI Label": p.get("nli_label"),
                        "Confidence": f"{float(p.get('nli_confidence') or 0):.2f}",
                        "Cosine Sim": f"{float(p.get('cosine_similarity') or 0):.2f}",
                        "Validated As": p.get("validated_as"),
                    })
                st.dataframe(pd.DataFrame(pairs_display), use_container_width=True, hide_index=True)
            else:
                st.info(
                    "No cross-paper claim pairs met the semantic similarity threshold "
                    "between the papers in this session."
                )

        with tab_scores:
            st.markdown("##### AERF Reliability Scores (Session-Scoped)")
            if scores:
                scores_display = []
                for s in scores:
                    fb = s.get("factor_breakdown") or {}
                    scores_display.append({
                        "Claim ID": s.get("claim_id"),
                        "Paper": s.get("source_filename"),
                        "Claim Text": s.get("claim_text"),
                        "AERF Score": f"{float(s.get('score') or 0):.4f}",
                        "Source": f"{float(fb.get('source') or 0):.2f}",
                        "Methodology": f"{float(fb.get('methodology') or 0):.2f}",
                        "Corroboration": f"{float(fb.get('corroboration') or 0):.2f}",
                        "Contradiction": f"{float(fb.get('contradiction') or 0):.2f}",
                        "Temporal": f"{float(fb.get('temporal') or 0):.2f}",
                        "Semantic": f"{float(fb.get('semantic') or 0):.2f}",
                        "Provenance": f"{float(fb.get('provenance') or 0):.2f}",
                    })
                st.dataframe(pd.DataFrame(scores_display), use_container_width=True, hide_index=True)
            else:
                st.info("No reliability scores recorded for this session.")

        with tab_graph:
            st.markdown("##### Evidence Graph (Session-Scoped)")
            if graph_edges:
                st.write(f"**Total Graph Edges:** {len(graph_edges)}")
                edges_display = []
                for e in graph_edges:
                    edges_display.append({
                        "Edge ID": e.get("edge_id"),
                        "Source Type": e.get("source_node_type"),
                        "Source ID": e.get("source_node_id"),
                        "Target Type": e.get("target_node_type"),
                        "Target ID": e.get("target_node_id"),
                        "Edge Type": e.get("edge_type"),
                        "Weight": f"{float(e.get('weight') or 0):.2f}",
                    })
                st.dataframe(pd.DataFrame(edges_display), use_container_width=True, hide_index=True)
            else:
                st.info("No graph edges recorded for this session.")

        with tab_gaps:
            st.markdown("##### Research Gaps (Session-Scoped)")
            if gaps:
                st.write(f"**Total Gaps Detected:** {len(gaps)}")
                gaps_display = []
                for g in gaps:
                    gaps_display.append({
                        "Gap ID": g.get("gap_id"),
                        "Topic": g.get("topic_key"),
                        "Gap Type": g.get("gap_type"),
                        "Description": g.get("description"),
                        "Related Claim IDs": str(g.get("related_claim_ids")),
                    })
                st.dataframe(pd.DataFrame(gaps_display), use_container_width=True, hide_index=True)
            else:
                st.info("No research gaps detected among the claims in this session.")






# ============================================================

# FOOTER

# ============================================================



st.divider()



st.caption(

    "AERF Engine • Adaptive Evidence Reliability & Reasoning • "

    "Research prototype"

)