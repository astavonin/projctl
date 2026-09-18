"""Tests for projctl.handlers.loader module."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

from projctl.config import Config
from projctl.exceptions import PlatformError
from projctl.formatters import format_user as _format_user, format_users as _format_users
from projctl.formatters import print_mr_comments
from projctl.handlers.loader import (
    ENVELOPE_FIELDS,
    NOTE_RECORD_FIELDS,
    THREAD_RECORD_FIELDS,
    TicketLoader,
    _build_json_note,
    _fold_thread,
    _parse_note_timestamp,
)
from projctl.utils.gitlab_identity import CURRENT_USER_QUERY


class TestTicketLoaderInit:
    """Test TicketLoader initialization."""

    def test_init(self, new_config_path: Path) -> None:
        """Loader initializes correctly."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        assert loader.config == config
        assert loader.group == "test/group"


class TestParseReference:
    """Test reference parsing."""

    def test_parse_issue_number(self, new_config_path: Path) -> None:
        """Parse issue number reference."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        ref_type, ref_id, project = loader.parse_reference("#123")

        assert ref_type == "issue"
        assert ref_id == "123"

    def test_parse_epic_reference(self, new_config_path: Path) -> None:
        """Parse epic reference with & prefix."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        ref_type, ref_id, project = loader.parse_reference("&21")

        assert ref_type == "epic"
        assert ref_id == "21"

    def test_parse_milestone_reference(self, new_config_path: Path) -> None:
        """Parse milestone reference with % prefix."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        ref_type, ref_id, project = loader.parse_reference("%123")

        assert ref_type == "milestone"
        assert ref_id == "123"

    def test_parse_mr_reference(self, new_config_path: Path) -> None:
        """Parse MR reference with ! prefix."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        ref_type, ref_id, project = loader.parse_reference("!134")

        assert ref_type == "mr"
        assert ref_id == "134"

    def test_parse_issue_url(self, new_config_path: Path) -> None:
        """Parse issue URL."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        url = "https://gitlab.example.com/group/project/-/issues/123"
        ref_type, ref_id, project = loader.parse_reference(url)

        assert ref_type == "issue"
        assert ref_id == "123"
        assert project == "group/project"

    def test_parse_epic_url(self, new_config_path: Path) -> None:
        """Parse epic URL."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        url = "https://gitlab.example.com/groups/test/-/epics/21"
        ref_type, ref_id, project = loader.parse_reference(url)

        assert ref_type == "epic"
        assert ref_id == "21"

    def test_parse_mr_url(self, new_config_path: Path) -> None:
        """Parse MR URL."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        url = "https://gitlab.example.com/group/project/-/merge_requests/134"
        ref_type, ref_id, project = loader.parse_reference(url)

        assert ref_type == "mr"
        assert ref_id == "134"

    def test_parse_plain_number(self, new_config_path: Path) -> None:
        """Parse plain number as issue."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        ref_type, ref_id, project = loader.parse_reference("123")

        assert ref_type == "issue"
        assert ref_id == "123"

    def test_parse_invalid_reference(self, new_config_path: Path) -> None:
        """Invalid reference raises ValueError."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        with pytest.raises(ValueError, match="Invalid reference"):
            loader.parse_reference("invalid")


class TestLoadIssue:
    """Test issue loading."""

    @patch("subprocess.run")
    def test_load_issue_success(
        self, mock_run: Mock, new_config_path: Path, mock_glab_issue_view: str
    ) -> None:
        """Load issue successfully."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        mock_run.return_value = Mock(stdout=mock_glab_issue_view, stderr="", returncode=0)

        result = loader.load_issue("#1")

        assert result is not None
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_load_issue_with_project(
        self, mock_run: Mock, new_config_path: Path, mock_glab_issue_view: str
    ) -> None:
        """Load issue with specific project."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        mock_run.return_value = Mock(stdout=mock_glab_issue_view, returncode=0)

        result = loader.load_issue("#1", project="group/project")

        # Verify project was encoded and passed in the API endpoint
        call_args = mock_run.call_args[0][0]
        joined = " ".join(call_args)
        # The project path is URL-encoded (/ → %2F) in the API endpoint
        assert "group%2Fproject" in joined or "group/project" in joined

    @patch("subprocess.run")
    def test_load_issue_command_failure(self, mock_run: Mock, new_config_path: Path) -> None:
        """Issue loading failure raises PlatformError."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["glab", "issue", "view"], stderr="Error loading issue"
        )

        with pytest.raises(PlatformError, match="Command failed"):
            loader.load_issue("#1")


class TestLoadEpic:
    """Test epic loading."""

    @patch("subprocess.run")
    def test_load_epic_success(
        self, mock_run: Mock, new_config_path: Path, mock_glab_epic_view: str
    ) -> None:
        """Load epic successfully."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        graphql_no_assignees = json.dumps(
            {
                "data": {
                    "group": {
                        "workItem": {"widgets": [{"type": "ASSIGNEES", "assignees": {"nodes": []}}]}
                    }
                }
            }
        )
        # Call order: REST epic data, GraphQL assignees
        mock_run.side_effect = [
            Mock(stdout=mock_glab_epic_view, returncode=0),
            Mock(stdout=graphql_no_assignees, returncode=0),
        ]

        result = loader.load_epic("&21")

        assert result is not None


class TestLoadMR:
    """Test MR loading."""

    @patch("subprocess.run")
    def test_load_mr_success(
        self, mock_run: Mock, new_config_path: Path, mock_glab_mr_view: str
    ) -> None:
        """Load MR successfully."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        mock_run.return_value = Mock(stdout=mock_glab_mr_view, returncode=0)

        result = loader.load_mr("!134")

        assert result is not None
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_load_mr_with_project(
        self, mock_run: Mock, new_config_path: Path, mock_glab_mr_view: str
    ) -> None:
        """Load MR with specific project (project kwarg accepted without error)."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        mock_run.return_value = Mock(stdout=mock_glab_mr_view, returncode=0)

        result = loader.load_mr("!134", project="group/project")

        # Verify the command was executed (project kwarg does not raise)
        mock_run.assert_called_once()


class TestLoadMilestone:
    """Test milestone loading."""

    @patch("subprocess.run")
    def test_load_milestone_success(self, mock_run: Mock, new_config_path: Path) -> None:
        """Load milestone successfully."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        milestone_data = {
            "id": 123,
            "iid": 1,
            "title": "v1.0",
            "state": "active",
            "description": "Milestone description",
        }

        # Call order when default_group is set:
        # 1. GET groups/{group}/milestones?per_page=100  (iid→id lookup)
        # 2. GET groups/{group}/milestones/{id}          (milestone data)
        # 3. GET groups/{group}/milestones/{id}/issues   (issues list)
        milestones_list = [milestone_data]
        mock_run.side_effect = [
            Mock(stdout=json.dumps(milestones_list), returncode=0),
            Mock(stdout=json.dumps(milestone_data), returncode=0),
            Mock(stdout="[]", returncode=0),
        ]

        result = loader.load_milestone("%1")

        assert result is not None


class TestFormatting:
    """Test output formatting."""

    @patch("subprocess.run")
    def test_markdown_output_format(
        self, mock_run: Mock, new_config_path: Path, mock_glab_issue_view: str, capsys
    ) -> None:
        """Issue output is formatted as markdown."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        mock_run.return_value = Mock(stdout=mock_glab_issue_view, returncode=0)

        loader.load_issue("#1")

        captured = capsys.readouterr()
        # Verify actual heading with the IID from mock data
        assert "# Issue #1: Test Issue" in captured.out

    @patch("subprocess.run")
    def test_includes_metadata(
        self, mock_run: Mock, new_config_path: Path, mock_glab_issue_view: str, capsys
    ) -> None:
        """Output includes issue metadata with correct values from mock data."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        mock_run.return_value = Mock(stdout=mock_glab_issue_view, returncode=0)

        loader.load_issue("#1")

        captured = capsys.readouterr()
        assert "**State:** opened" in captured.out
        assert "**Labels:**" in captured.out
        assert "`type::feature`" in captured.out


class TestFormatUser:
    """Test _format_user and _format_users helpers."""

    def test_format_user_with_name_and_username(self) -> None:
        """Full user dict renders as 'Name (@username)'."""
        user = {"name": "Alex Stavonin", "username": "alex.stavonin"}
        assert _format_user(user) == "Alex Stavonin (@alex.stavonin)"

    def test_format_user_username_only(self) -> None:
        """User with no name falls back to username."""
        user = {"username": "alex.stavonin"}
        assert _format_user(user) == "alex.stavonin (@alex.stavonin)"

    def test_format_user_name_only(self) -> None:
        """User with name but no username renders without (@...) suffix."""
        user = {"name": "Alice"}
        assert _format_user(user) == "Alice"

    def test_format_user_empty_dict(self) -> None:
        """Empty dict returns '?'."""
        assert _format_user({}) == "?"

    def test_format_users_multiple(self) -> None:
        """Multiple users are comma-separated."""
        users = [
            {"name": "Alice", "username": "alice"},
            {"name": "Bob", "username": "bob"},
        ]
        assert _format_users(users) == "Alice (@alice), Bob (@bob)"

    def test_format_users_empty(self) -> None:
        """Empty list returns empty string."""
        assert _format_users([]) == ""


class TestGetStatusHistory:
    """Test _get_status_history."""

    def test_returns_chronological_order(self, new_config_path: Path) -> None:
        """Notes are reversed from newest-first to oldest-first."""
        notes = [
            {
                "system": True,
                "body": "set status to **Done**",
                "created_at": "2026-03-25T10:00:00Z",
            },
            {
                "system": True,
                "body": "set status to **In progress**",
                "created_at": "2026-03-10T08:00:00Z",
            },
            {
                "system": True,
                "body": "set status to **To do**",
                "created_at": "2026-03-01T09:00:00Z",
            },
        ]
        config = Config(new_config_path)
        loader = TicketLoader(config)

        with patch.object(loader, "_run_glab_command", return_value=json.dumps(notes)):
            history = loader._get_status_history(1403, 22)

        assert [h["status"] for h in history] == ["To do", "In progress", "Done"]
        assert history[0]["timestamp"] == "2026-03-01T09:00:00Z"

    def test_ignores_non_system_notes(self, new_config_path: Path) -> None:
        """Non-system notes are skipped."""
        notes = [
            {
                "system": False,
                "body": "set status to **Done**",
                "created_at": "2026-03-25T10:00:00Z",
            },
            {
                "system": True,
                "body": "set status to **In progress**",
                "created_at": "2026-03-10T08:00:00Z",
            },
        ]
        config = Config(new_config_path)
        loader = TicketLoader(config)

        with patch.object(loader, "_run_glab_command", return_value=json.dumps(notes)):
            history = loader._get_status_history(1403, 22)

        assert len(history) == 1
        assert history[0]["status"] == "In progress"

    def test_returns_empty_on_error(self, new_config_path: Path) -> None:
        """PlatformError returns empty list without raising."""
        config = Config(new_config_path)
        loader = TicketLoader(config)

        with patch.object(loader, "_run_glab_command", side_effect=PlatformError("fail")):
            history = loader._get_status_history(1403, 22)

        assert history == []


class TestComputeTiming:
    """Test _compute_timing."""

    def test_in_progress_then_done(self, new_config_path: Path) -> None:
        """Normal flow: To do → In progress → Done."""
        history = [
            {"status": "To do", "timestamp": "2026-03-01T09:00:00Z"},
            {"status": "In progress", "timestamp": "2026-03-10T08:00:00Z"},
            {"status": "Done", "timestamp": "2026-03-25T10:00:00Z"},
        ]
        loader = TicketLoader(Config(new_config_path))
        result = loader._compute_timing(history)

        assert result["current_status"] == "Done"
        assert result["start_date"] == "2026-03-10T08:00:00Z"
        assert result["end_date"] == "2026-03-25T10:00:00Z"
        assert result["is_rejected"] is False

    def test_todo_to_done_no_in_progress(self, new_config_path: Path) -> None:
        """To do → Done directly: start_date is None, end_date is set."""
        history = [
            {"status": "To do", "timestamp": "2026-03-01T09:00:00Z"},
            {"status": "Done", "timestamp": "2026-03-25T10:00:00Z"},
        ]
        loader = TicketLoader(Config(new_config_path))
        result = loader._compute_timing(history)

        assert result["current_status"] == "Done"
        assert result["start_date"] is None
        assert result["end_date"] == "2026-03-25T10:00:00Z"
        assert result["is_rejected"] is False

    def test_duplicate_is_rejected(self, new_config_path: Path) -> None:
        """Duplicate status: no dates, is_rejected True."""
        history = [
            {"status": "To do", "timestamp": "2026-03-01T09:00:00Z"},
            {"status": "Duplicate", "timestamp": "2026-03-30T08:00:00Z"},
        ]
        loader = TicketLoader(Config(new_config_path))
        result = loader._compute_timing(history)

        assert result["current_status"] == "Duplicate"
        assert result["start_date"] is None
        assert result["end_date"] is None
        assert result["is_rejected"] is True

    def test_wont_do_is_rejected(self, new_config_path: Path) -> None:
        """Won't do status: no dates, is_rejected True."""
        history = [
            {"status": "To do", "timestamp": "2026-03-01T09:00:00Z"},
            {"status": "Won't do", "timestamp": "2026-04-01T08:00:00Z"},
        ]
        loader = TicketLoader(Config(new_config_path))
        result = loader._compute_timing(history)

        assert result["is_rejected"] is True
        assert result["start_date"] is None
        assert result["end_date"] is None

    def test_cycled_back_uses_first_in_progress_and_last_done(self, new_config_path: Path) -> None:
        """To do → In progress → To do → In progress → Done: first start, last end."""
        history = [
            {"status": "To do", "timestamp": "2026-03-01T09:00:00Z"},
            {"status": "In progress", "timestamp": "2026-03-10T08:00:00Z"},
            {"status": "To do", "timestamp": "2026-03-15T09:00:00Z"},
            {"status": "In progress", "timestamp": "2026-03-17T08:00:00Z"},
            {"status": "Done", "timestamp": "2026-03-25T10:00:00Z"},
        ]
        loader = TicketLoader(Config(new_config_path))
        result = loader._compute_timing(history)

        assert result["start_date"] == "2026-03-10T08:00:00Z"
        assert result["end_date"] == "2026-03-25T10:00:00Z"

    def test_empty_history(self, new_config_path: Path) -> None:
        """Empty history returns all None fields."""
        loader = TicketLoader(Config(new_config_path))
        result = loader._compute_timing([])

        assert result["current_status"] is None
        assert result["start_date"] is None
        assert result["end_date"] is None
        assert result["is_rejected"] is False


class TestDeriveEpicDates:
    """Test _derive_epic_dates."""

    def test_all_done_with_in_progress(self) -> None:
        """All non-rejected issues done: returns earliest start, latest end."""
        issues = [
            {
                "timing": {
                    "is_rejected": False,
                    "start_date": "2026-03-10T08:00:00Z",
                    "end_date": "2026-03-20T10:00:00Z",
                }
            },
            {
                "timing": {
                    "is_rejected": False,
                    "start_date": "2026-03-05T09:00:00Z",
                    "end_date": "2026-03-25T10:00:00Z",
                }
            },
        ]
        result = TicketLoader._derive_epic_dates(issues)

        assert result["start_date"] == "2026-03-05T09:00:00Z"
        assert result["end_date"] == "2026-03-25T10:00:00Z"

    def test_any_unfinished_clears_end_date(self) -> None:
        """Any non-rejected issue without end_date → epic end is None."""
        issues = [
            {
                "timing": {
                    "is_rejected": False,
                    "start_date": "2026-03-10T08:00:00Z",
                    "end_date": "2026-03-20T10:00:00Z",
                }
            },
            {
                "timing": {
                    "is_rejected": False,
                    "start_date": "2026-03-12T08:00:00Z",
                    "end_date": None,
                }
            },
        ]
        result = TicketLoader._derive_epic_dates(issues)

        assert result["start_date"] == "2026-03-10T08:00:00Z"
        assert result["end_date"] is None

    def test_rejected_issues_excluded(self) -> None:
        """Rejected issues do not affect dates."""
        issues = [
            {"timing": {"is_rejected": True, "start_date": None, "end_date": None}},
            {
                "timing": {
                    "is_rejected": False,
                    "start_date": "2026-03-10T08:00:00Z",
                    "end_date": "2026-03-20T10:00:00Z",
                }
            },
        ]
        result = TicketLoader._derive_epic_dates(issues)

        assert result["start_date"] == "2026-03-10T08:00:00Z"
        assert result["end_date"] == "2026-03-20T10:00:00Z"

    def test_todo_to_done_no_start(self) -> None:
        """Issues without start_date (To do → Done) don't contribute to epic start."""
        issues = [
            {
                "timing": {
                    "is_rejected": False,
                    "start_date": None,
                    "end_date": "2026-03-20T10:00:00Z",
                }
            },
            {
                "timing": {
                    "is_rejected": False,
                    "start_date": "2026-03-10T08:00:00Z",
                    "end_date": "2026-03-25T10:00:00Z",
                }
            },
        ]
        result = TicketLoader._derive_epic_dates(issues)

        assert result["start_date"] == "2026-03-10T08:00:00Z"
        assert result["end_date"] == "2026-03-25T10:00:00Z"

    def test_all_issues_no_start_dates(self) -> None:
        """All issues went To do → Done: epic start is None."""
        issues = [
            {
                "timing": {
                    "is_rejected": False,
                    "start_date": None,
                    "end_date": "2026-03-20T10:00:00Z",
                }
            },
            {
                "timing": {
                    "is_rejected": False,
                    "start_date": None,
                    "end_date": "2026-03-25T10:00:00Z",
                }
            },
        ]
        result = TicketLoader._derive_epic_dates(issues)

        assert result["start_date"] is None
        assert result["end_date"] == "2026-03-25T10:00:00Z"

    def test_empty_issues(self) -> None:
        """No issues → both dates None."""
        result = TicketLoader._derive_epic_dates([])

        assert result["start_date"] is None
        assert result["end_date"] is None

    def test_all_rejected(self) -> None:
        """All issues rejected → both dates None."""
        issues = [
            {"timing": {"is_rejected": True, "start_date": None, "end_date": None}},
            {"timing": {"is_rejected": True, "start_date": None, "end_date": None}},
        ]
        result = TicketLoader._derive_epic_dates(issues)

        assert result["start_date"] is None
        assert result["end_date"] is None


class TestParseNoteTimestamp:
    """Direct tests for the Z-suffix shim `_parse_note_timestamp()` relies on.

    `datetime.fromisoformat()` has accepted a bare 'Z' natively since 3.11, but
    `pyproject.toml` declares `requires-python = ">=3.9"` — see C3 and the
    precedent this adopts, test_timelog.py's
    TestParseSpentAt.test_z_suffix_is_rewritten_to_an_explicit_offset_before_parsing.
    """

    def test_z_suffix_is_rewritten_to_an_explicit_offset_before_parsing(self) -> None:
        """Asserted on the string that actually reaches fromisoformat(), not just the
        parsed result — an output-only assertion cannot tell "shim ran" apart from
        "shim absent, the 3.11+ native parser handled it anyway"."""
        with patch("projctl.handlers.loader.datetime") as mock_datetime:
            mock_datetime.fromisoformat.side_effect = datetime.fromisoformat

            _parse_note_timestamp("2026-08-07T09:00:00Z")

        mock_datetime.fromisoformat.assert_called_once_with("2026-08-07T09:00:00+00:00")


class TestLoadMrComments:
    """Test MR comment loading, including thread identity."""

    _RAW_DISCUSSIONS = [
        {
            "id": "thread-abc123",
            "notes": [
                {
                    "id": 9001,
                    "system": False,
                    "body": "Major: this assertion is not scoped to the build job",
                    "resolvable": True,
                    "resolved": False,
                    "author": {"name": "Reviewer One"},
                    "position": {"new_path": "ci/audit.py", "new_line": 42},
                    "created_at": "2026-08-07T09:00:00Z",
                },
                # An author reply in the same thread — the shape that distinguishes a
                # per-thread id from a per-note one.
                {
                    "id": 9004,
                    "system": False,
                    "body": "Scoped it to the build job",
                    "resolvable": True,
                    "resolved": False,
                    "author": {"name": "Author"},
                    "position": {"new_path": "ci/audit.py", "new_line": 42},
                    "created_at": "2026-08-07T09:10:00Z",
                },
            ],
        },
        {
            "id": "thread-def456",
            "notes": [
                {
                    "id": 9002,
                    "system": True,
                    "body": "assigned to @someone",
                    "author": {"name": "GitLab"},
                },
                {
                    "id": 9003,
                    "system": False,
                    "body": "General remark, not on a diff line",
                    "resolvable": False,
                    "resolved": False,
                    "author": {"name": "Reviewer Two"},
                    "created_at": "2026-08-07T09:05:00Z",
                },
            ],
        },
    ]

    def _load_full(self, new_config_path: Path, discussions: Any = None) -> Dict[str, Any]:
        """Run load_mr_comments() against the subprocess boundary and return the envelope."""
        loader = TicketLoader(Config(new_config_path))
        payload = self._RAW_DISCUSSIONS if discussions is None else discussions
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                Mock(stdout=json.dumps({"iid": 235}), returncode=0),
                Mock(stdout=json.dumps(payload), returncode=0),
            ]
            return loader.load_mr_comments("235")

    def _load(self, new_config_path: Path) -> list:
        return self._load_full(new_config_path)["comments"]

    def test_each_comment_carries_its_enclosing_thread_id(self, new_config_path: Path) -> None:
        """`resolve:`/`replies:` in a review YAML take the discussion id, not the note
        id, so dropping it makes the loaded output unusable for resolving a thread."""
        comments = self._load(new_config_path)

        assert [c["discussion_id"] for c in comments] == [
            "thread-abc123",
            "thread-abc123",
            "thread-def456",
        ]

    def test_all_notes_in_one_thread_share_that_thread_id(self, new_config_path: Path) -> None:
        """Every note in a discussion carries the discussion's id, not its own.

        A per-note id would pass any single-note fixture yet make a reply to the
        second note POST to /discussions/<note_id>, which GitLab 404s.
        """
        comments = self._load(new_config_path)
        first, reply = comments[0], comments[1]

        assert first["discussion_id"] == reply["discussion_id"] == "thread-abc123"
        assert first["id"] != reply["id"]

    def test_note_id_and_thread_id_stay_distinct(self, new_config_path: Path) -> None:
        """The two ids address different resources; one must never stand in for the
        other, since the discussions endpoint rejects a note id."""
        comments = self._load(new_config_path)

        assert comments[0]["id"] == 9001
        assert comments[0]["discussion_id"] == "thread-abc123"

    def test_system_notes_are_still_filtered_out(self, new_config_path: Path) -> None:
        """Threading data must not resurrect the system notes the loader excludes."""
        comments = self._load(new_config_path)

        assert len(comments) == 3
        assert all("assigned to" not in c["body"] for c in comments)

    def test_thread_id_defaults_to_empty_when_absent(self, new_config_path: Path) -> None:
        """A discussion without an id must not raise — the note is still worth showing."""
        comments = self._load_full(
            new_config_path, [{"notes": [{"id": 1, "system": False, "body": "hi"}]}]
        )["comments"]

        assert comments[0]["discussion_id"] == ""

    def test_explicit_null_thread_id_becomes_empty_string(self, new_config_path: Path) -> None:
        """A JSON null must normalize to "" like a missing key does.

        dict.get(key, "") returns None for an explicit null, which would violate the
        str type load_mr_comments() documents for discussion_id.
        """
        comments = self._load_full(
            new_config_path,
            [{"id": None, "notes": [{"id": 1, "system": False, "body": "hi"}]}],
        )["comments"]

        assert comments[0]["discussion_id"] == ""

    def test_comment_dict_shape(self, new_config_path: Path) -> None:
        """Full contract of one comment dict.

        TestLoadMrComments is the only coverage load_mr_comments() has anywhere in
        the suite, so any key not asserted here can drift or disappear silently.
        """
        comments = self._load(new_config_path)

        assert comments[0] == {
            "id": 9001,
            "discussion_id": "thread-abc123",
            "author": "Reviewer One",
            "body": "Major: this assertion is not scoped to the build job",
            "resolvable": True,
            "resolved": False,
            "file_path": "ci/audit.py",
            "line": 42,
            "created_at": "2026-08-07T09:00:00Z",
        }

    def test_positionless_note_emits_empty_file_path_and_line(self, new_config_path: Path) -> None:
        """A note with no `position` yields empty strings, not placeholders.

        The formatter's regression fixture for the thread-id defect hard-codes
        file_path="" / line="" as the shape a top-level note takes, so that guard
        is only as good as this producer-side assertion.
        """
        comments = self._load(new_config_path)

        assert comments[2] == {
            "id": 9003,
            "discussion_id": "thread-def456",
            "author": "Reviewer Two",
            "body": "General remark, not on a diff line",
            "resolvable": False,
            "resolved": False,
            "file_path": "",
            "line": "",
            "created_at": "2026-08-07T09:05:00Z",
        }

    def test_discussions_are_fetched_with_the_note_list_argv(self, new_config_path: Path) -> None:
        """`mr note list` is what returns discussion objects; `mr view` returns none.

        The mock replays responses by call order, so every other test here passes
        whatever subcommand is issued — this is the only assertion pinning it.
        """
        loader = TicketLoader(Config(new_config_path))
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                Mock(stdout=json.dumps({"iid": 235}), returncode=0),
                Mock(stdout=json.dumps(self._RAW_DISCUSSIONS), returncode=0),
            ]
            loader.load_mr_comments("235")

        assert mock_run.call_args_list[1].args[0] == [
            "glab",
            "mr",
            "note",
            "list",
            "235",
            "--output",
            "json",
        ]

    def test_envelope_carries_mr_metadata(self, new_config_path: Path) -> None:
        """print_mr_info() does `print_mr(data["mr"])`, so an empty envelope is not
        enough — the metadata itself has to survive."""
        result = self._load_full(new_config_path)

        assert result["mr"] == {"iid": 235}

    def test_print_mr_info_includes_comment_thread_id(self, new_config_path: Path, capsys) -> None:
        """Composition test for the path the CLI actually runs: load_mr_comments()
        feeding print_mr_info(..., with_comments=True). The two halves are tested in
        isolation elsewhere, which cannot catch the producer and consumer drifting
        apart — e.g. a key rename on one side with only that side's tests updated.

        Covers the unresolvable thread too: that is the shape the recorded observed
        failure was about, and it is otherwise guarded only by a hand-built fixture.
        """
        data = self._load_full(new_config_path)
        TicketLoader(Config(new_config_path)).print_mr_info(data, with_comments=True)

        out = capsys.readouterr().out
        assert "thread: thread-abc123" in out
        assert "thread: thread-def456" in out

    def test_print_mr_info_omits_comments_when_not_requested(
        self, new_config_path: Path, capsys
    ) -> None:
        """Without --comments the listing must not appear.

        `cli.py` passes the flag straight through, so a gate that ignores it would
        print review comments on every plain `projctl load mr N`.
        """
        data = self._load_full(new_config_path)
        TicketLoader(Config(new_config_path)).print_mr_info(data, with_comments=False)

        out = capsys.readouterr().out
        assert "thread: thread-abc123" not in out
        assert "Review Comments" not in out

    def test_load_mr_comments_propagates_platform_error(self, new_config_path: Path) -> None:
        """Pins the method's error contract against its declared `Raises`: unlike
        _get_status_history, which swallows PlatformError and returns [], this method
        must propagate it rather than silently returning partial data."""
        loader = TicketLoader(Config(new_config_path))
        with patch.object(loader, "_run_glab_command", side_effect=PlatformError("fail")):
            with pytest.raises(PlatformError):
                loader.load_mr_comments("235")

    def test_explicit_null_author_does_not_raise(self, new_config_path: Path) -> None:
        """An explicit JSON null `author` (not an absent key) must degrade to
        "Unknown" via the same guarded `note.get("author") or {}` form
        _build_json_note() already uses, not raise AttributeError."""
        discussions = [
            {
                "id": "d1",
                "notes": [
                    {
                        "id": 1,
                        "system": False,
                        "body": "claim",
                        "author": None,
                        "resolvable": True,
                        "resolved": False,
                        "created_at": "2026-08-07T09:00:00Z",
                    }
                ],
            }
        ]

        comments = self._load_full(new_config_path, discussions)["comments"]

        assert comments[0]["author"] == "Unknown"

    def test_explicit_null_position_fields_yield_empty_file_path_and_line(
        self, new_config_path: Path
    ) -> None:
        """An explicit JSON null `new_path`/`new_line` (not an absent
        `position`) must degrade to "" the same way an absent position
        does — `position.get(key, "")` returns None for an explicit null,
        the same trap _build_json_note() already guards on the --json
        path."""
        discussions = [
            {
                "id": "d1",
                "notes": [
                    {
                        "id": 1,
                        "system": False,
                        "body": "claim",
                        "author": {"name": "Reviewer"},
                        "resolvable": True,
                        "resolved": False,
                        "position": {
                            "new_path": None,
                            "new_line": None,
                            "old_path": "ci/audit.py",
                        },
                        "created_at": "2026-08-07T09:00:00Z",
                    }
                ],
            }
        ]

        comments = self._load_full(new_config_path, discussions)["comments"]

        assert comments[0]["file_path"] == ""
        assert comments[0]["line"] == ""

    def test_explicit_null_position_object_yields_empty_file_path_and_line(
        self, new_config_path: Path
    ) -> None:
        """An explicit JSON null `position` (the whole object, not just its
        fields) must degrade the same way an absent `position` key does —
        `note.get("position", {})` would return None here instead of the
        guarded `{}` default, raising on the next `.get("new_path", ...)`.
        The --json fold's twin already covers this; the Markdown path did
        not."""
        discussions = [
            {
                "id": "d1",
                "notes": [
                    {
                        "id": 1,
                        "system": False,
                        "body": "claim",
                        "author": {"name": "Reviewer"},
                        "resolvable": True,
                        "resolved": False,
                        "position": None,
                        "created_at": "2026-08-07T09:00:00Z",
                    }
                ],
            }
        ]

        comments = self._load_full(new_config_path, discussions)["comments"]

        assert comments[0]["file_path"] == ""
        assert comments[0]["line"] == ""

    def test_explicit_null_display_name_normalizes_to_unknown(self, new_config_path: Path) -> None:
        """A null-valued `author.name` (not an absent `author` object) must
        also degrade to "Unknown" — `author.get("name", "Unknown")` returns
        None for an explicit null."""
        discussions = [
            {
                "id": "d1",
                "notes": [
                    {
                        "id": 1,
                        "system": False,
                        "body": "claim",
                        "author": {"name": None, "username": "reviewer"},
                        "resolvable": True,
                        "resolved": False,
                        "created_at": "2026-08-07T09:00:00Z",
                    }
                ],
            }
        ]

        comments = self._load_full(new_config_path, discussions)["comments"]

        assert comments[0]["author"] == "Unknown"

    def test_explicit_null_created_at_becomes_empty_string(self, new_config_path: Path) -> None:
        """A JSON null `created_at` (not an absent key) must normalize to ""
        the same way load_mr_comments_json()'s note builder already does —
        `note.get("created_at", "")` returns None for an explicit null."""
        discussions = [
            {
                "id": "d1",
                "notes": [
                    {
                        "id": 1,
                        "system": False,
                        "body": "claim",
                        "author": {"name": "Reviewer"},
                        "resolvable": True,
                        "resolved": False,
                        "created_at": None,
                    }
                ],
            }
        ]

        comments = self._load_full(new_config_path, discussions)["comments"]

        assert comments[0]["created_at"] == ""


# Shared fixtures for the load mr --comments --json cases below.
# `author` carries both keys with distinct values so a display-name fallback
# substituted for `author_username` is observable rather than masked by a
# fixture that never exercises the fallback branch.
_MR_VIEW_JSON: Dict[str, Any] = {
    "iid": 235,
    "title": "Ref #42: fix the thing",
    "web_url": "https://gitlab.example.com/group/project/-/merge_requests/235",
    "author": {"username": "author.user", "name": "Author Display Name"},
    "source_project_id": 10,
    "target_project_id": 20,
    "source_branch": "feature/42-fix",
    "target_branch": "main",
}


def _current_user_graphql(username: str = "astavonin") -> str:
    return json.dumps({"data": {"currentUser": {"username": username}}})


def _note(
    note_id: int,
    body: str,
    *,
    username: Optional[str] = "reviewer",
    name: str = "Reviewer",
    resolvable: bool = True,
    resolved: bool = False,
    created_at: str = "2026-08-07T09:00:00Z",
    file_path: Optional[str] = "ci/audit.py",
    line: int = 42,
) -> Dict[str, Any]:
    """Build one raw glab note object for a --json fold fixture.

    ``username=None`` omits the author's ``username`` key entirely (the "no
    username in the payload" shape); ``file_path=None`` omits ``position``
    entirely (a general, non-inline note).
    """
    author: Dict[str, str] = {"name": name}
    if username is not None:
        author["username"] = username
    note: Dict[str, Any] = {
        "id": note_id,
        "system": False,
        "body": body,
        "resolvable": resolvable,
        "resolved": resolved,
        "author": author,
        "created_at": created_at,
    }
    if file_path is not None:
        note["position"] = {"new_path": file_path, "new_line": line}
    return note


def _discussion(discussion_id: Optional[str], notes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build one raw glab discussion object; ``discussion_id=None`` omits ``id`` entirely."""
    discussion: Dict[str, Any] = {"notes": notes}
    if discussion_id is not None:
        discussion["id"] = discussion_id
    return discussion


def _load_json_full(
    new_config_path: Path,
    discussions: list,
    mr_data: Optional[Dict[str, Any]] = None,
    viewer: str = "astavonin",
) -> Dict[str, Any]:
    """Run load_mr_comments_json() against the subprocess boundary and return the payload."""
    loader = TicketLoader(Config(new_config_path))
    mr_payload = mr_data if mr_data is not None else dict(_MR_VIEW_JSON)
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            Mock(stdout=json.dumps(mr_payload), returncode=0),
            Mock(stdout=json.dumps(discussions), returncode=0),
            Mock(stdout=_current_user_graphql(viewer), returncode=0),
        ]
        return loader.load_mr_comments_json("235")


class TestBuildJsonNote:
    """Direct calls to _build_json_note(), bypassing _iter_filtered_notes().

    A null body reaching the fold is already filtered out upstream (the
    same way an empty one is), so this function's own guard is otherwise
    never exercised with one — calling it directly is what makes that
    guard's own correctness observable rather than dead code.
    """

    def test_null_body_normalizes_to_empty_string(self) -> None:
        """Asserts the whole record: this fixture omits `author`, `position`,
        `resolvable` and `resolved`, so it is the only site exercising all
        four defaults — an assertion on `body` alone leaves the other three
        unpinned."""
        note = {"id": 1, "body": None, "created_at": "2026-08-07T09:00:00Z"}

        result = _build_json_note("d1", note)

        assert result == {
            "id": 1,
            "discussion_id": "d1",
            "author": "Unknown",
            "author_username": "",
            "body": "",
            "resolvable": False,
            "resolved": False,
            "file_path": "",
            "line": "",
            "created_at": "2026-08-07T09:00:00Z",
        }


class TestLoadMrCommentsJsonExposure:
    """The `--json` payload's shape: envelope, viewer, and the Markdown path untouched."""

    def test_json_folds_one_thread_per_discussion(self, new_config_path: Path) -> None:
        """One thread record per discussion_id, each carrying its note records, and
        `viewer` on the envelope."""
        discussions = [
            _discussion("d1", [_note(1, "claim one")]),
            _discussion("d2", [_note(2, "claim two")]),
        ]

        result = _load_json_full(new_config_path, discussions)

        assert {t["discussion_id"] for t in result["threads"]} == {"d1", "d2"}
        assert all(len(t["notes"]) == 1 for t in result["threads"])
        assert result["mr"]["viewer"] == "astavonin"

    def test_note_without_username_key_yields_empty_author_username(
        self, new_config_path: Path
    ) -> None:
        """A display-name fallback in a username field is the mismatch the field
        exists to prevent — an absent username must stay empty, never "Unknown"."""
        note = _note(1, "claim", username=None)

        result = _load_json_full(new_config_path, [_discussion("d1", [note])])

        note_record = result["threads"][0]["notes"][0]
        assert note_record["author_username"] == ""
        assert note_record["author"] == "Reviewer"

    def test_explicit_null_username_normalizes_to_empty_string(self, new_config_path: Path) -> None:
        """An explicit JSON null `author.username` (not an absent key) must
        normalize to "" the same way an absent key does — `.get(key, "")`
        returns None for an explicit null, which would let the note bypass
        the unidentifiable_author skip arm and be adjudicated as a
        stranger's claim."""
        note = _note(1, "claim")
        note["author"]["username"] = None

        result = _load_json_full(new_config_path, [_discussion("d1", [note])])

        thread = result["threads"][0]
        assert thread["notes"][0]["author_username"] == ""
        assert thread["skip"] is True
        assert thread["skip_reason"] == "unidentifiable_author"

    def test_note_with_explicit_null_position_fields_yields_empty_strings(
        self, new_config_path: Path
    ) -> None:
        """A comment on a deleted line carries `position: {new_path: null,
        new_line: null, old_path: ...}` — an explicit null, not an absent
        position. Both must degrade to "", the same shape absence gets,
        never None."""
        note = _note(1, "claim")
        note["position"] = {"new_path": None, "new_line": None, "old_path": "ci/audit.py"}

        result = _load_json_full(new_config_path, [_discussion("d1", [note])])

        note_record = result["threads"][0]["notes"][0]
        assert note_record["file_path"] == ""
        assert note_record["line"] == ""

    def test_explicit_null_author_object_normalizes_to_unknown(self, new_config_path: Path) -> None:
        """An explicit JSON null `author` (the whole object, not just its
        `username`) must degrade the same way an absent `author` key does —
        `note.get("author", {})` would return None here instead of the
        guarded `{}` default, raising on the next `.get("name", ...)`."""
        note = _note(1, "claim")
        note["author"] = None

        result = _load_json_full(new_config_path, [_discussion("d1", [note])])

        note_record = result["threads"][0]["notes"][0]
        assert note_record["author"] == "Unknown"
        assert note_record["author_username"] == ""

    def test_explicit_null_display_name_normalizes_to_unknown(self, new_config_path: Path) -> None:
        """A null-valued `author.name` (not an absent `author` object) must
        also degrade to "Unknown" — `author.get("name", "Unknown")` returns
        None for an explicit null, the field left behind when every other
        field on this builder was hardened."""
        note = _note(1, "claim")
        note["author"]["name"] = None

        result = _load_json_full(new_config_path, [_discussion("d1", [note])])

        assert result["threads"][0]["notes"][0]["author"] == "Unknown"

    def test_explicit_null_created_at_normalizes_to_empty_string_and_skips_as_unusable(
        self, new_config_path: Path
    ) -> None:
        """A null-valued `created_at` (not an absent key) must degrade to ""
        like every other normalized field, and the fold's own
        unusable-created_at arm catches the empty result."""
        note = _note(1, "claim")
        note["created_at"] = None

        result = _load_json_full(new_config_path, [_discussion("d1", [note])])

        thread = result["threads"][0]
        assert thread["notes"][0]["created_at"] == ""
        assert thread["skip"] is True
        assert thread["skip_reason"] == "unusable_created_at"

    def test_explicit_null_position_object_yields_empty_file_path_and_line(
        self, new_config_path: Path
    ) -> None:
        """An explicit JSON null `position` (the whole object, not just its
        fields) must degrade the same way an absent `position` key does —
        `note.get("position", {})` would return None here instead of the
        guarded `{}` default, raising on the next `.get("new_path", ...)`."""
        note = _note(1, "claim")
        note["position"] = None

        result = _load_json_full(new_config_path, [_discussion("d1", [note])])

        note_record = result["threads"][0]["notes"][0]
        assert note_record["file_path"] == ""
        assert note_record["line"] == ""

    def test_plain_load_mr_comments_has_no_viewer_and_no_graphql_call(
        self, new_config_path: Path
    ) -> None:
        """Without --json, load_mr_comments() must not gain a GraphQL call or a
        viewer key — the exposure is opt-in."""
        loader = TicketLoader(Config(new_config_path))
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                Mock(stdout=json.dumps(_MR_VIEW_JSON), returncode=0),
                Mock(stdout=json.dumps([_discussion("d1", [_note(1, "x")])]), returncode=0),
            ]
            result = loader.load_mr_comments("235")

        assert "viewer" not in result["mr"]
        assert mock_run.call_count == 2

    def test_null_current_user_raises_platform_error(self, new_config_path: Path) -> None:
        """A `currentUser: null` GraphQL response is a hard error, never an empty
        or falsy viewer that would make every skip arm silently pass."""
        loader = TicketLoader(Config(new_config_path))
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                Mock(stdout=json.dumps(_MR_VIEW_JSON), returncode=0),
                Mock(stdout=json.dumps([_discussion("d1", [_note(1, "x")])]), returncode=0),
                Mock(stdout=json.dumps({"data": {"currentUser": None}}), returncode=0),
            ]
            with pytest.raises(PlatformError, match="currentUser returned null"):
                loader.load_mr_comments_json("235")

    def test_markdown_syntax_body_round_trips_unchanged_through_both_paths(
        self, new_config_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A body carrying `## ` headings, a `thread: <id>` line, a `|`, and
        surrounding whitespace strips identically on both paths — the --json
        fold and the Markdown path are compared directly to each other, not
        each to a literal that could mask one side drifting from the other."""
        tricky_body = "  ## Heading\nthread: fake-id\n| col |  \n"
        stripped_body = tricky_body.strip()
        discussions = [_discussion("d1", [_note(1, tricky_body)])]

        json_result = _load_json_full(new_config_path, discussions)
        json_body = json_result["threads"][0]["notes"][0]["body"]
        assert json_body == stripped_body
        assert json_result["threads"][0]["claim"] == stripped_body

        md_loader = TicketLoader(Config(new_config_path))
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                Mock(stdout=json.dumps(_MR_VIEW_JSON), returncode=0),
                Mock(stdout=json.dumps(discussions), returncode=0),
            ]
            md_data = md_loader.load_mr_comments("235")
        print_mr_comments(md_data["comments"])

        md_body = md_data["comments"][0]["body"]
        assert md_body == json_body
        assert md_body in capsys.readouterr().out

    def test_envelope_carries_exactly_the_eight_projected_fields(
        self, new_config_path: Path
    ) -> None:
        """The envelope under --json carries all eight projected fields and no
        other key, so the gate reads a guarded projection rather than whatever
        `glab` happened to emit."""
        result = _load_json_full(new_config_path, [_discussion("d1", [_note(1, "x")])])

        assert set(result["mr"].keys()) == set(ENVELOPE_FIELDS)

    def test_envelope_carries_every_projected_field_with_its_real_value(
        self, new_config_path: Path
    ) -> None:
        """Every non-`viewer` envelope field is asserted against a
        populated, distinctly-valued fixture — a wrong source-key mapping
        (source_project_id/target_project_id swapped, source_branch/
        target_branch swapped, or a display-name fallback substituted for
        author_username) must fail this even though every field is
        individually present."""
        result = _load_json_full(new_config_path, [_discussion("d1", [_note(1, "x")])])

        envelope = result["mr"]
        assert envelope["author_username"] == "author.user"
        assert envelope["source_project_id"] == 10
        assert envelope["target_project_id"] == 20
        assert (
            envelope["web_url"] == "https://gitlab.example.com/group/project/-/merge_requests/235"
        )
        assert envelope["title"] == "Ref #42: fix the thing"
        assert envelope["source_branch"] == "feature/42-fix"
        assert envelope["target_branch"] == "main"

    def test_envelope_author_username_empty_when_username_key_absent(
        self, new_config_path: Path
    ) -> None:
        """An absent `username` must stay empty, never fall back to the
        author's display `name` — the fallback the field exists to
        prevent (see the note-level equivalent in TestLoadMrComments)."""
        mr_data = dict(_MR_VIEW_JSON)
        mr_data["author"] = {"name": "Author Display Name"}

        result = _load_json_full(
            new_config_path, [_discussion("d1", [_note(1, "x")])], mr_data=mr_data
        )

        assert result["mr"]["author_username"] == ""

    def test_resolve_viewer_graphql_argv_carries_the_current_user_query(
        self, new_config_path: Path
    ) -> None:
        """_resolve_viewer()'s GraphQL argv is asserted in full — matching
        the precedent set by this file's own
        test_discussions_are_fetched_with_the_note_list_argv and by
        test_timelog.py's current-user-query assertion."""
        loader = TicketLoader(Config(new_config_path))
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                Mock(stdout=json.dumps(_MR_VIEW_JSON), returncode=0),
                Mock(stdout=json.dumps([_discussion("d1", [_note(1, "x")])]), returncode=0),
                Mock(stdout=_current_user_graphql("astavonin"), returncode=0),
            ]
            loader.load_mr_comments_json("235")

        assert mock_run.call_args_list[2].args[0] == [
            "glab",
            "api",
            "graphql",
            "-f",
            f"query={CURRENT_USER_QUERY}",
        ]

    def test_bang_prefixed_reference_reaches_glab_normalized(self, new_config_path: Path) -> None:
        """Every other --json test in this module passes an already-bare
        "235", so none of them exercise _normalize_mr_ref() — a "!235"
        reference must still reach glab as the bare "235"."""
        loader = TicketLoader(Config(new_config_path))
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                Mock(stdout=json.dumps(_MR_VIEW_JSON), returncode=0),
                Mock(stdout=json.dumps([_discussion("d1", [_note(1, "x")])]), returncode=0),
                Mock(stdout=_current_user_graphql("astavonin"), returncode=0),
            ]
            loader.load_mr_comments_json("!235")

        assert mock_run.call_args_list[0].args[0] == [
            "glab",
            "mr",
            "view",
            "235",
            "--output",
            "json",
        ]

    def test_envelope_defaults_when_mr_payload_omits_every_projected_field(
        self, new_config_path: Path
    ) -> None:
        """A minimal `mr view` payload (missing author/project ids/branches/etc.)
        still yields every envelope key, degraded to "" or None rather than
        raising or omitting the key."""
        result = _load_json_full(new_config_path, [], mr_data={"iid": 235})

        envelope = result["mr"]
        assert envelope["author_username"] == ""
        assert envelope["source_project_id"] is None
        assert envelope["target_project_id"] is None
        assert envelope["web_url"] == ""
        assert envelope["title"] == ""
        assert envelope["source_branch"] == ""
        assert envelope["target_branch"] == ""

    def test_envelope_author_username_empty_when_author_object_is_null(
        self, new_config_path: Path
    ) -> None:
        """An explicit JSON null `author` (the whole object) must degrade
        the same way an absent `author` key does — `mr_data.get("author",
        {})` would return None here instead of the guarded `{}` default,
        raising on the next `.get("username")`."""
        mr_data = dict(_MR_VIEW_JSON)
        mr_data["author"] = None

        result = _load_json_full(new_config_path, [], mr_data=mr_data)

        assert result["mr"]["author_username"] == ""

    def test_envelope_string_fields_normalize_explicit_nulls_to_empty_string(
        self, new_config_path: Path
    ) -> None:
        """Each of the five projected string fields must distinguish an
        explicit JSON null from an absent key — `.get(key, "")` returns
        None for the former, which the sibling
        test_envelope_defaults_when_mr_payload_omits_every_projected_field
        (absent keys) cannot observe."""
        mr_data = {
            "iid": 235,
            "title": None,
            "web_url": None,
            "author": {"username": None},
            "source_project_id": 10,
            "target_project_id": 20,
            "source_branch": None,
            "target_branch": None,
        }

        result = _load_json_full(new_config_path, [], mr_data=mr_data)

        envelope = result["mr"]
        assert envelope["author_username"] == ""
        assert envelope["web_url"] == ""
        assert envelope["title"] == ""
        assert envelope["source_branch"] == ""
        assert envelope["target_branch"] == ""

    def test_no_discussions_yields_empty_threads_list(self, new_config_path: Path) -> None:
        """An MR with no discussions at all folds to an empty thread list, not an
        error or a missing key."""
        result = _load_json_full(new_config_path, [])

        assert result["threads"] == []

    def test_blank_body_note_is_filtered_out_of_the_fold(self, new_config_path: Path) -> None:
        """A note whose body is empty (or whitespace-only) is filtered the same
        way load_mr_comments() already filters it, leaving the thread with only
        its remaining note."""
        blank = _note(1, "   ", username="reviewer", created_at="2026-08-07T09:00:00Z")
        real = _note(2, "actual claim", username="reviewer", created_at="2026-08-07T10:00:00Z")

        result = _load_json_full(new_config_path, [_discussion("d1", [blank, real])])

        assert [n["id"] for n in result["threads"][0]["notes"]] == [2]


class TestLoadMrCommentsJsonErrors:
    """Error propagation for load_mr_comments_json(), beyond the null-viewer case above."""

    def test_platform_error_from_transport_propagates(self, new_config_path: Path) -> None:
        """A glab failure fetching the MR/discussions is not swallowed."""
        loader = TicketLoader(Config(new_config_path))
        with patch.object(loader, "_run_glab_command", side_effect=PlatformError("fail")):
            with pytest.raises(PlatformError):
                loader.load_mr_comments_json("235")

    def test_note_missing_id_raises_key_error(self, new_config_path: Path) -> None:
        """`note["id"]` is read without a default — an id-less note payload is a
        loader bug worth a loud KeyError, not a silently dropped record."""
        broken_note = {
            "system": False,
            "body": "x",
            "author": {"username": "reviewer"},
            "resolvable": True,
            "resolved": False,
            "created_at": "2026-08-07T09:00:00Z",
        }
        with pytest.raises(KeyError):
            _load_json_full(new_config_path, [_discussion("d1", [broken_note])])


class TestLoadMrCommentsJsonKeyContract:
    """Every emitted thread and note record carries exactly its owning tuple's keys."""

    def test_thread_record_keys_equal_thread_record_fields(self, new_config_path: Path) -> None:
        result = _load_json_full(new_config_path, [_discussion("d1", [_note(1, "x")])])

        assert set(result["threads"][0].keys()) == set(THREAD_RECORD_FIELDS)

    def test_note_record_keys_equal_note_record_fields(self, new_config_path: Path) -> None:
        result = _load_json_full(new_config_path, [_discussion("d1", [_note(1, "x")])])

        assert set(result["threads"][0]["notes"][0].keys()) == set(NOTE_RECORD_FIELDS)


class TestThreadFoldAndSkip:
    """The fold (§5.1's thread-record table) and the five-arm skip predicate."""

    def test_notes_returned_newest_first_are_still_folded_chronologically(
        self, new_config_path: Path
    ) -> None:
        """`claim` and the skip arm both read the chronologically **last** note —
        an implementation trusting transport order instead would read the
        transport-last note (the reviewer's), missing the viewer's later reply."""
        notes = [
            _note(2, "our later reply", username="astavonin", created_at="2026-08-07T10:00:00Z"),
            _note(1, "original claim", username="reviewer", created_at="2026-08-07T09:00:00Z"),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["skip"] is True
        assert thread["skip_reason"] == "answered_by_viewer"
        assert thread["claim"] == "original claim"

    def test_claim_tracks_chronological_order_even_with_two_non_viewer_notes(
        self, new_config_path: Path
    ) -> None:
        """The sibling above has only one non-viewer note, so a claim loop
        reading transport order instead of chronological order still finds
        it last either way. Two reviewer notes, newest first in transport,
        are what makes the two orders disagree on which one is "last"."""
        notes = [
            _note(1, "later claim", username="reviewer", created_at="2026-08-07T11:00:00Z"),
            _note(2, "earlier claim", username="reviewer", created_at="2026-08-07T09:00:00Z"),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["claim"] == "later claim"
        assert thread["created_at"] == "2026-08-07T11:00:00Z"

    def test_tied_created_at_breaks_tie_by_ascending_note_id(self, new_config_path: Path) -> None:
        """Two notes sharing one created_at order by ascending note id, so
        "last" is single-valued on a tie."""
        notes = [
            _note(20, "second", created_at="2026-08-07T09:00:00Z"),
            _note(10, "first", created_at="2026-08-07T09:00:00Z"),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        ordered_ids = [n["id"] for n in result["threads"][0]["notes"]]
        assert ordered_ids == [10, 20]

    def test_second_discussions_second_note_groups_into_its_own_thread(
        self, new_config_path: Path
    ) -> None:
        """The multi-note discussion here is the *second* one, not the first —
        a `_group_notes()` index bug that always points a new discussion at
        group 0 would fold this note into the first discussion's thread
        instead of its own."""
        discussions = [
            _discussion("d1", [_note(1, "solo")]),
            _discussion(
                "d2",
                [
                    _note(2, "first in d2", created_at="2026-08-07T09:00:00Z"),
                    _note(3, "second in d2", created_at="2026-08-07T10:00:00Z"),
                ],
            ),
        ]

        result = _load_json_full(new_config_path, discussions)

        threads_by_id = {t["discussion_id"]: t for t in result["threads"]}
        assert [n["id"] for n in threads_by_id["d1"]["notes"]] == [1]
        assert [n["id"] for n in threads_by_id["d2"]["notes"]] == [2, 3]

    def test_thread_skipped_when_the_second_note_lacks_created_at(
        self, new_config_path: Path
    ) -> None:
        """Every existing unusable-created_at fixture puts the bad timestamp
        at index 0 — a check narrowed to only the first note would still
        pass them all. This one puts it second."""
        notes = [
            _note(1, "reviewer claim", username="reviewer", created_at="2026-08-07T09:00:00Z"),
            _note(2, "our own note", username="astavonin", created_at=""),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["skip"] is True
        assert thread["skip_reason"] == "unusable_created_at"

    def test_resolve_viewer_returns_the_actual_graphql_username_not_a_hardcoded_default(
        self, new_config_path: Path
    ) -> None:
        """Every other --json test in this module feeds "astavonin" on the
        GraphQL side, so a `_resolve_viewer()` that discards the extractor's
        result and returns a hardcoded "astavonin" would still pass them
        all. A distinct username, driving an answered_by_viewer skip
        decision, is what makes the actually-resolved value observable."""
        notes = [
            _note(1, "claim", username="someone-else", created_at="2026-08-07T09:00:00Z"),
            _note(2, "our reply", username="distinct-viewer-42", created_at="2026-08-07T10:00:00Z"),
        ]

        result = _load_json_full(
            new_config_path, [_discussion("d1", notes)], viewer="distinct-viewer-42"
        )

        thread = result["threads"][0]
        assert result["mr"]["viewer"] == "distinct-viewer-42"
        assert thread["skip"] is True
        assert thread["skip_reason"] == "answered_by_viewer"

    def test_thread_resolvable_when_any_note_resolvable(self, new_config_path: Path) -> None:
        notes = [_note(1, "a", resolvable=False), _note(2, "b", resolvable=True)]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        assert result["threads"][0]["resolvable"] is True

    def test_thread_not_resolved_when_any_resolvable_note_unresolved(
        self, new_config_path: Path
    ) -> None:
        notes = [
            _note(1, "a", resolvable=True, resolved=True),
            _note(2, "b", resolvable=True, resolved=False),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        assert result["threads"][0]["resolved"] is False

    def test_thread_resolved_and_skipped_when_sole_resolvable_note_resolved(
        self, new_config_path: Path
    ) -> None:
        notes = [_note(1, "a", resolvable=True, resolved=True)]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["resolved"] is True
        assert thread["skip"] is True
        assert thread["skip_reason"] == "resolved"

    def test_thread_with_no_resolvable_note_is_not_skipped(self, new_config_path: Path) -> None:
        """The vacuous-truth case: no resolvable note means `resolved` must be
        False, not True — otherwise every general MR note would read as
        already resolved and drop out of the quorum."""
        notes = [_note(1, "general remark", resolvable=False, resolved=False)]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["resolvable"] is False
        assert thread["resolved"] is False
        assert thread["skip"] is False

    def test_thread_skipped_when_last_note_is_viewers_own(self, new_config_path: Path) -> None:
        notes = [
            _note(1, "claim", username="reviewer", created_at="2026-08-07T09:00:00Z"),
            _note(2, "our reply", username="astavonin", created_at="2026-08-07T10:00:00Z"),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["skip"] is True
        assert thread["skip_reason"] == "answered_by_viewer"

    def test_thread_not_skipped_when_reviewer_answers_back_after_our_reply(
        self, new_config_path: Path
    ) -> None:
        """A middle note is ours, the last is not — the reviewer answered our
        reply, so the thread is still owed a verdict on their follow-up."""
        notes = [
            _note(1, "claim", username="reviewer", created_at="2026-08-07T09:00:00Z"),
            _note(2, "our reply", username="astavonin", created_at="2026-08-07T10:00:00Z"),
            _note(3, "reviewer follow-up", username="reviewer", created_at="2026-08-07T11:00:00Z"),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["skip"] is False
        assert thread["claim"] == "reviewer follow-up"

    def test_thread_not_skipped_when_only_first_note_is_ours(self, new_config_path: Path) -> None:
        notes = [
            _note(1, "our opening remark", username="astavonin", created_at="2026-08-07T09:00:00Z"),
            _note(2, "reviewer claim", username="reviewer", created_at="2026-08-07T10:00:00Z"),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        assert result["threads"][0]["skip"] is False

    def test_thread_not_skipped_on_display_name_collision_with_viewer(
        self, new_config_path: Path
    ) -> None:
        """The answered-arm comparison reads `author_username` alone — a display
        name that happens to equal the viewer's username must not fire it."""
        note = _note(1, "claim", username="impersonator", name="astavonin")

        result = _load_json_full(new_config_path, [_discussion("d1", [note])], viewer="astavonin")

        assert result["threads"][0]["skip"] is False

    def test_claim_note_position_and_timestamp_carry_onto_the_thread(
        self, new_config_path: Path
    ) -> None:
        notes = [
            _note(1, "claim", file_path="ci/audit.py", line=42, created_at="2026-08-07T09:00:00Z")
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["file_path"] == "ci/audit.py"
        assert thread["line"] == 42
        assert thread["created_at"] == "2026-08-07T09:00:00Z"

    def test_claim_note_without_position_leaves_path_and_line_empty_but_keeps_timestamp(
        self, new_config_path: Path
    ) -> None:
        note = _note(1, "general claim", created_at="2026-08-07T09:00:00Z", file_path=None)

        result = _load_json_full(new_config_path, [_discussion("d1", [note])])

        thread = result["threads"][0]
        assert thread["file_path"] == ""
        assert thread["line"] == ""
        assert thread["created_at"] == "2026-08-07T09:00:00Z"

    def test_claim_note_provenance_survives_a_later_viewer_reply(
        self, new_config_path: Path
    ) -> None:
        """file_path, line and created_at come from the claim note — the
        last note not authored by viewer — not from whichever note sorts
        chronologically last. A single-note thread can't tell the two
        apart; this multi-note fixture, where the viewer's reply carries
        different values, can."""
        notes = [
            _note(
                1,
                "reviewer claim",
                username="reviewer",
                file_path="ci/audit.py",
                line=42,
                created_at="2026-08-07T09:00:00Z",
            ),
            _note(
                2,
                "our reply",
                username="astavonin",
                file_path="ci/other.py",
                line=99,
                created_at="2026-08-07T10:00:00Z",
            ),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["file_path"] == "ci/audit.py"
        assert thread["line"] == 42
        assert thread["created_at"] == "2026-08-07T09:00:00Z"

    def test_thread_skipped_when_every_note_is_the_viewers(self, new_config_path: Path) -> None:
        notes = [
            _note(1, "our note", username="astavonin", created_at="2026-08-07T09:00:00Z"),
            _note(2, "our other note", username="astavonin", created_at="2026-08-07T10:00:00Z"),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["skip"] is True
        assert thread["skip_reason"] == "answered_by_viewer"
        assert thread["claim"] == ""
        assert thread["file_path"] == ""
        assert thread["line"] == ""
        assert thread["created_at"] == ""

    def test_resolved_arm_wins_over_answered_arm_when_both_apply(
        self, new_config_path: Path
    ) -> None:
        """On the commonest thread shape — resolved, and we had the last word —
        the resolved arm fires first and owns the one skip_reason value."""
        notes = [
            _note(
                1,
                "claim",
                username="reviewer",
                resolvable=True,
                resolved=True,
                created_at="2026-08-07T09:00:00Z",
            ),
            _note(
                2,
                "ack",
                username="astavonin",
                resolvable=True,
                resolved=True,
                created_at="2026-08-07T10:00:00Z",
            ),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["skip"] is True
        assert thread["skip_reason"] == "resolved"

    def test_thread_skipped_when_last_note_has_empty_author_username(
        self, new_config_path: Path
    ) -> None:
        notes = [
            _note(1, "claim", username="reviewer", created_at="2026-08-07T09:00:00Z"),
            _note(2, "follow-up", username=None, created_at="2026-08-07T10:00:00Z"),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["skip"] is True
        assert thread["skip_reason"] == "unidentifiable_author"

    def test_thread_not_skipped_when_only_a_middle_note_has_empty_author_username(
        self, new_config_path: Path
    ) -> None:
        notes = [
            _note(1, "mystery note", username=None, created_at="2026-08-07T09:00:00Z"),
            _note(2, "reviewer claim", username="reviewer", created_at="2026-08-07T10:00:00Z"),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        assert result["threads"][0]["skip"] is False

    def test_two_blank_discussion_id_notes_become_two_separate_skipped_threads(
        self, new_config_path: Path
    ) -> None:
        """A blank discussion_id groups nothing — each note is minted on its own
        note id rather than being folded into one thread of unrelated claims."""
        discussions = [_discussion(None, [_note(1, "a"), _note(2, "b")])]

        result = _load_json_full(new_config_path, discussions)

        threads = result["threads"]
        assert len(threads) == 2
        assert {t["discussion_id"] for t in threads} == {"1", "2"}
        assert all(t["skip"] and t["skip_reason"] == "blank_discussion_id" for t in threads)

    def test_thread_skipped_when_a_non_claim_note_lacks_created_at(
        self, new_config_path: Path
    ) -> None:
        """The unusable-created_at arm reads every note, not just the claim
        note — a bad timestamp on an unrelated note still fires it, since
        ordering the thread at all depends on every note parsing."""
        notes = [
            _note(1, "our own note", username="astavonin", created_at=""),
            _note(2, "reviewer claim", username="reviewer", created_at="2026-08-07T10:00:00Z"),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["skip"] is True
        assert thread["skip_reason"] == "unusable_created_at"

    def test_claim_is_the_parseable_note_even_when_the_unusable_one_sorts_last(
        self, new_config_path: Path
    ) -> None:
        """Both notes here are the reviewer's — neither is the viewer's own —
        so the claim tracks whichever note the sort places last. The sort
        floor an unusable created_at is pinned to exists precisely to keep
        that note from winning the "last" position over one that actually
        parsed."""
        notes = [
            _note(1, "unparseable note", username="reviewer", created_at=""),
            _note(2, "reviewer claim", username="reviewer", created_at="2026-08-07T09:00:00Z"),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["skip"] is True
        assert thread["skip_reason"] == "unusable_created_at"
        assert thread["claim"] == "reviewer claim"
        assert thread["created_at"] == "2026-08-07T09:00:00Z"

    def test_thread_skipped_when_created_at_does_not_parse(self, new_config_path: Path) -> None:
        """A present-but-unparseable created_at fires the same arm as an empty
        one — rejected on parsing, not on emptiness."""
        notes = [_note(1, "claim", created_at="not-a-timestamp")]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["skip"] is True
        assert thread["skip_reason"] == "unusable_created_at"

    def test_thread_skipped_when_created_at_has_no_utc_offset(self, new_config_path: Path) -> None:
        """A naive ISO datetime ("2026-08-07T09:00:00", no offset) parses
        under fromisoformat() but can't be compared against an aware
        sibling or the sort floor — treated as unusable, the same as an
        unparseable value, rather than silently assumed UTC (see C3)."""
        notes = [_note(1, "claim", created_at="2026-08-07T09:00:00")]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["skip"] is True
        assert thread["skip_reason"] == "unusable_created_at"

    def test_fold_does_not_raise_when_naive_and_aware_created_at_are_mixed(
        self, new_config_path: Path
    ) -> None:
        """A naive created_at alongside an aware one in the same thread must
        not raise TypeError during sort — the naive value is unusable
        rather than compared against its aware sibling."""
        notes = [
            _note(1, "claim", created_at="2026-08-07T09:00:00"),
            _note(2, "reply", created_at="2026-08-07T10:00:00Z"),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        assert thread["skip"] is True
        assert thread["skip_reason"] == "unusable_created_at"

    def test_thread_with_explicit_non_utc_offset_created_at_is_usable(
        self, new_config_path: Path
    ) -> None:
        """A parseable, offset-bearing timestamp that isn't Z-suffixed
        (+02:00) is usable and orders correctly against a Z-suffixed
        sibling — comparing aware datetimes at different offsets must not
        raise."""
        notes = [
            _note(1, "our note", username="astavonin", created_at="2026-08-07T09:00:00Z"),
            _note(
                2,
                "reviewer claim",
                username="reviewer",
                created_at="2026-08-07T13:00:00+02:00",
            ),
        ]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        thread = result["threads"][0]
        # 13:00+02:00 == 11:00Z, later than 09:00Z, so the reviewer's note
        # sorts last — not skipped, and its offset-bearing value is kept.
        assert thread["skip"] is False
        assert thread["created_at"] == "2026-08-07T13:00:00+02:00"


class TestSkipArmPrecedence:
    """Each adjacent pair of §5.1's five skip arms, evaluated in order — the
    first arm to fire owns skip_reason. A fixture satisfying two adjacent
    arms at once is the only way a swapped-order mutant becomes observable."""

    def test_resolved_wins_over_blank_discussion_id(self, new_config_path: Path) -> None:
        discussions = [_discussion(None, [_note(1, "claim", resolvable=True, resolved=True)])]

        result = _load_json_full(new_config_path, discussions)

        assert result["threads"][0]["skip_reason"] == "resolved"

    def test_blank_discussion_id_wins_over_unusable_created_at(self, new_config_path: Path) -> None:
        discussions = [_discussion(None, [_note(1, "claim", created_at="not-a-timestamp")])]

        result = _load_json_full(new_config_path, discussions)

        assert result["threads"][0]["skip_reason"] == "blank_discussion_id"

    def test_unusable_created_at_wins_over_answered_by_viewer(self, new_config_path: Path) -> None:
        """A single note authored by viewer, with an unparseable created_at,
        satisfies both the unusable-timestamp arm and the answered-by-viewer
        arm (trivially, as the thread's only/last note) — the earlier arm
        must own skip_reason."""
        notes = [_note(1, "our note", username="astavonin", created_at="not-a-timestamp")]

        result = _load_json_full(new_config_path, [_discussion("d1", notes)])

        assert result["threads"][0]["skip_reason"] == "unusable_created_at"

    def test_answered_by_viewer_wins_over_unidentifiable_author(self) -> None:
        """Only constructible with an empty viewer, which the public load
        path can never produce — extract_current_username() hard-errors on
        one before the fold runs — so this calls _fold_thread() directly to
        pin the order these two arms are checked in."""
        note = {
            "id": 1,
            "discussion_id": "d1",
            "author": "Nobody",
            "author_username": "",
            "body": "x",
            "resolvable": False,
            "resolved": False,
            "file_path": "",
            "line": "",
            "created_at": "2026-08-07T09:00:00Z",
        }

        thread = _fold_thread([note], viewer="")

        assert thread["skip"] is True
        assert thread["skip_reason"] == "answered_by_viewer"


class TestLoadMrCommentsJsonIntegration:
    """One recorded-transport round trip covering the whole --json payload at once."""

    def test_full_payload_carries_every_owning_tuples_keys_and_nothing_else(
        self, new_config_path: Path
    ) -> None:
        """stdout parses as JSON; the envelope, every thread record, and every
        note record inside one carry every member of their own owning tuple in
        loader.py and nothing else; `viewer` equals the GraphQL fixture's
        username. A builder that dropped `skip`, dropped `author_username`
        from a note, or omitted an envelope field would fail these exact
        equality checks rather than emitting a record a consumer reads as
        falsy."""
        discussions = [
            _discussion(
                "d1",
                [
                    _note(1, "first claim", username="reviewer", created_at="2026-08-07T09:00:00Z"),
                    _note(
                        2,
                        "reviewer follow-up",
                        username="reviewer",
                        created_at="2026-08-07T10:00:00Z",
                    ),
                ],
            ),
            _discussion("d2", [_note(3, "general remark", resolvable=False, file_path=None)]),
        ]

        result = _load_json_full(new_config_path, discussions, viewer="astavonin")

        assert set(result.keys()) == {"mr", "threads"}
        assert set(result["mr"].keys()) == set(ENVELOPE_FIELDS)
        assert result["mr"]["viewer"] == "astavonin"
        assert len(result["threads"]) == 2
        for thread in result["threads"]:
            assert set(thread.keys()) == set(THREAD_RECORD_FIELDS)
            for note in thread["notes"]:
                assert set(note.keys()) == set(NOTE_RECORD_FIELDS)
