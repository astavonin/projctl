"""Tests for projctl.handlers.loader module."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from projctl.config import Config
from projctl.exceptions import PlatformError
from projctl.formatters import format_user as _format_user, format_users as _format_users
from projctl.handlers.loader import TicketLoader


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
