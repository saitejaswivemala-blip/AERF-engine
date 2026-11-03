from __future__ import annotations

import pytest


class TestUISessionStateTransitions:
    """
    Regression and state-transition tests for the Streamlit New Analysis session UI.
    Verifies that the new-session form remains visible, handles transitions between
    form mode and results view mode, and prevents selectbox state locking.
    """

    NEW_SESSION_OPTION = "➕ Start New Analysis Session"

    def _setup_session_map(self, sessions_list):
        session_options = [self.NEW_SESSION_OPTION]
        session_id_map = {self.NEW_SESSION_OPTION: None}
        for s in sessions_list:
            sid = s["session_id"]
            paper_cnt = s.get("paper_count", len(s.get("papers", [])))
            status = str(s.get("status", "completed")).lower()
            status_icon = "✅" if status == "completed" else ("⏳" if status == "processing" else "❌")
            dt_str = s.get("created_at", "")[:16].replace("T", " ")
            name_str = f" — {s['name']}" if s.get("name") and s["name"] != sid else ""
            lbl = f"{status_icon} {sid}{name_str} ({paper_cnt} papers) — {dt_str}"
            session_options.append(lbl)
            session_id_map[lbl] = sid
        return session_options, session_id_map

    def _evaluate_view_mode(self, session_state, session_options, session_id_map):
        """Simulates the state synchronization logic in ui/app.py before rendering."""
        # Ensure state defaults
        if "session_view_mode" not in session_state:
            session_state["session_view_mode"] = "new"

        if session_state.get("session_view_mode") == "new":
            session_state["session_selector_key"] = self.NEW_SESSION_OPTION
            session_state["active_session_id"] = None
        else:
            active_sid = session_state.get("active_session_id")
            matched_opt = None
            if active_sid:
                for opt, sid in session_id_map.items():
                    if sid == active_sid:
                        matched_opt = opt
                        break
            if matched_opt:
                session_state["session_selector_key"] = matched_opt
            elif "session_selector_key" not in session_state or session_state["session_selector_key"] not in session_options:
                session_state["session_selector_key"] = session_options[0]

        is_new_mode = (
            session_state.get("session_view_mode") == "new"
            or not session_state.get("active_session_id")
            or session_state.get("session_selector_key") == self.NEW_SESSION_OPTION
        )
        return is_new_mode

    def _on_new_session_clicked(self, session_state):
        session_state["session_view_mode"] = "new"
        session_state["active_session_id"] = None
        session_state["session_selector_key"] = self.NEW_SESSION_OPTION

    def _on_session_selected(self, session_state, selected_option, session_id_map):
        session_state["session_selector_key"] = selected_option
        if selected_option == self.NEW_SESSION_OPTION or selected_option is None:
            session_state["session_view_mode"] = "new"
            session_state["active_session_id"] = None
        else:
            sid = session_id_map.get(selected_option)
            if sid:
                session_state["session_view_mode"] = "view"
                session_state["active_session_id"] = sid
            else:
                session_state["session_view_mode"] = "new"
                session_state["active_session_id"] = None

    def test_initial_page_load_displays_new_session_form(self):
        """On initial visit, the form must be visible and in 'new' mode."""
        session_state = {}
        sample_sessions = [{"session_id": "session_123", "status": "completed", "papers": [1]}]
        options, id_map = self._setup_session_map(sample_sessions)

        is_new_mode = self._evaluate_view_mode(session_state, options, id_map)

        assert is_new_mode is True
        assert session_state["session_view_mode"] == "new"
        assert session_state["active_session_id"] is None
        assert session_state["session_selector_key"] == self.NEW_SESSION_OPTION

    def test_clicking_new_session_button_opens_form_from_saved_results(self):
        """
        Regression test: When viewing an existing session, clicking '+ New Session'
        must switch to 'new' mode and make the form visible.
        """
        sample_sessions = [
            {"session_id": "session_alpha", "status": "completed", "papers": [1]},
            {"session_id": "session_beta", "status": "failed", "papers": [2]},
        ]
        options, id_map = self._setup_session_map(sample_sessions)

        # Simulate currently viewing session_alpha
        session_state = {
            "session_view_mode": "view",
            "active_session_id": "session_alpha",
        }
        # First ensure viewing mode evaluates to False (results view active)
        is_new_mode_before = self._evaluate_view_mode(session_state, options, id_map)
        assert is_new_mode_before is False

        # User clicks "+ New Session"
        self._on_new_session_clicked(session_state)

        # Re-evaluate for rerun
        is_new_mode_after = self._evaluate_view_mode(session_state, options, id_map)

        assert is_new_mode_after is True
        assert session_state["session_view_mode"] == "new"
        assert session_state["active_session_id"] is None
        assert session_state["session_selector_key"] == self.NEW_SESSION_OPTION

    def test_form_remains_visible_on_stream_reruns_during_typing(self):
        """Simulate user typing session name and uploading files across multiple reruns."""
        sample_sessions = [{"session_id": "session_past", "status": "completed", "papers": [1]}]
        options, id_map = self._setup_session_map(sample_sessions)

        session_state = {
            "session_view_mode": "new",
            "new_session_name_input": "Hydrology Study 2026",
        }

        # First rerun
        is_new_1 = self._evaluate_view_mode(session_state, options, id_map)
        assert is_new_1 is True

        # Second rerun (e.g. file uploaded)
        session_state["research_pdf_uploader"] = ["dummy.pdf"]
        is_new_2 = self._evaluate_view_mode(session_state, options, id_map)
        assert is_new_2 is True
        assert session_state["new_session_name_input"] == "Hydrology Study 2026"

    def test_selecting_existing_session_only_opens_saved_results(self):
        """Selecting an existing session in the dropdown opens its results and does not break navigation."""
        sample_sessions = [
            {"session_id": "session_001", "status": "completed", "papers": [1, 2]},
            {"session_id": "session_002", "status": "failed", "papers": []},
        ]
        options, id_map = self._setup_session_map(sample_sessions)

        session_state = {"session_view_mode": "new"}

        # User selects session_001 from dropdown
        target_option = [opt for opt in options if "session_001" in opt][0]
        self._on_session_selected(session_state, target_option, id_map)

        is_new = self._evaluate_view_mode(session_state, options, id_map)
        assert is_new is False
        assert session_state["active_session_id"] == "session_001"
        assert session_state["session_view_mode"] == "view"

        # User selects dropdown back to "➕ Start New Analysis Session"
        self._on_session_selected(session_state, self.NEW_SESSION_OPTION, id_map)
        is_new_back = self._evaluate_view_mode(session_state, options, id_map)
        assert is_new_back is True
        assert session_state["active_session_id"] is None
        assert session_state["session_view_mode"] == "new"

    def test_failed_session_does_not_crash_selector(self):
        """Verify that failed sessions appear cleanly in options without raising exceptions."""
        sample_sessions = [
            {"session_id": "session_err", "status": "failed", "papers": [], "error_message": "Corrupted PDF"},
        ]
        options, id_map = self._setup_session_map(sample_sessions)

        failed_opt = [opt for opt in options if "session_err" in opt][0]
        assert "❌" in failed_opt
        assert id_map[failed_opt] == "session_err"
