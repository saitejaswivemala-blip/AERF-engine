import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.session_manager import run_session_analysis, get_session_results

def main():
    session_id = "session_20261026_182922_b04beb"
    pdf_dir = Path(__file__).parent.parent / "experiments" / "uploads" / "sessions" / session_id
    pdf_paths = list(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs in {pdf_dir}: {[p.name for p in pdf_paths]}")
    
    print(f"Starting analysis for session: {session_id}")
    result = run_session_analysis(session_id, pdf_paths=pdf_paths)
    print(f"Analysis result status: {result.get('status')}")
    details = get_session_results(session_id)
    summary = details.get("summary", {})
    print(f"Papers parsed: {summary.get('total_papers')}")
    print(f"Claims extracted: {summary.get('total_claims')}")
    print(f"Evidence pairs: {summary.get('total_pairs')}")
    print(f"Supports: {summary.get('supports_count')}, Contradicts: {summary.get('contradicts_count')}, Neutral: {summary.get('neutral_count')}")
    print(f"Graph edges: {summary.get('total_graph_edges')}, Gaps: {summary.get('total_gaps')}")
    if details.get('session_info', {}).get('error_message'):
        print(f"Error: {details['session_info']['error_message']}")

if __name__ == "__main__":
    main()
